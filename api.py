# api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
import time
import threading
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

from etl import run_etl_pipeline

# Memory optimizations for free tier (512MB limit)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Reduces memory usage
torch.set_num_threads(1)  # Limit CPU threads to reduce memory

# Global flag for model loading state
model_loading = False
model_load_error = None
sentence_model = None
model_loading_lock = threading.Lock()

# -----------------------------
# Load artifacts ONCE at startup (before FastAPI app)
# -----------------------------
print("🔹 Loading artifacts...")
artifacts = run_etl_pipeline()

df = artifacts["data"]
meta_sim = artifacts["meta_sim"]
desc_embeddings = artifacts["desc_embeddings"]

def load_model_background():
    """Load the SentenceTransformer model in background thread."""
    global sentence_model, model_loading, model_load_error
    try:
        with model_loading_lock:
            if sentence_model is None and not model_loading:
                model_loading = True
                print("🔹 Loading SentenceTransformer model in background...")
                with torch.no_grad():  # Disable gradients to save memory
                    sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                    sentence_model.eval()
                print("✅ Model loaded successfully in background!")
                model_loading = False
    except Exception as e:
        model_loading = False
        model_load_error = str(e)
        print(f"❌ Error loading model: {e}")

# -----------------------------
# Initialize API
# -----------------------------
app = FastAPI(
    title="Movie Recommendation API",
    description="Hybrid TF-IDF + Embedding Movie Recommender",
    version="1.0"
)

@app.on_event("startup")
async def startup_event():
    """Pre-load model in background after service starts."""
    # Give the service a few seconds to fully start before loading model
    def delayed_load():
        time.sleep(5)  # Wait 5 seconds after startup
        load_model_background()
    
    # Start background thread to load model
    thread = threading.Thread(target=delayed_load, daemon=True)
    thread.start()
    print("🔹 Scheduled background model loading (will start in 5 seconds)...")

# -----------------------------
# CORS Middleware (for Hugging Face Streamlit frontend)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific Hugging Face app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_sentence_model():
    """Get the SentenceTransformer model, loading if necessary."""
    global sentence_model, model_loading, model_load_error
    
    # If model is already loaded, return it
    if sentence_model is not None:
        return sentence_model
    
    # If model is currently loading, wait a bit
    if model_loading:
        wait_time = 0
        max_wait = 30  # Wait up to 30 seconds
        while model_loading and wait_time < max_wait:
            time.sleep(1)
            wait_time += 1
            if sentence_model is not None:
                return sentence_model
    
    # If model failed to load, try loading synchronously (last resort)
    if model_load_error:
        print(f"⚠️ Previous load failed: {model_load_error}. Retrying...")
        model_load_error = None
    
    # Try loading now if still None
    if sentence_model is None:
        print("🔹 Loading SentenceTransformer model on-demand...")
        try:
            with model_loading_lock:
                if sentence_model is None:  # Double-check after acquiring lock
                    with torch.no_grad():
                        sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                        sentence_model.eval()
                    print("✅ Model loaded successfully on-demand!")
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=503, detail=f"Model loading failed. Please try again in a moment. Error: {error_msg}")
    
    return sentence_model

title_to_idx = {
    title.lower(): idx
    for idx, title in enumerate(df["Title"].tolist())
}

# -----------------------------
# Request schemas
# -----------------------------
class TitleRequest(BaseModel):
    title: str
    top_n: int = 6
    alpha: float = 0.45

class QueryRequest(BaseModel):
    query: str
    top_n: int = 6

# -----------------------------
# Helper: Hybrid recommendation
# -----------------------------
def hybrid_recommend(idx: int, alpha: float, top_n: int):
    # compute embedding similarity ONLY for one movie
    emb_sim = cosine_similarity(
        desc_embeddings[idx].reshape(1, -1),
        desc_embeddings
    )[0]

    hybrid_sim = alpha * meta_sim[idx] + (1 - alpha) * emb_sim

    top_idx = np.argsort(hybrid_sim)[::-1]
    top_idx = [i for i in top_idx if i != idx][:top_n]

    return df.iloc[top_idx]

# -----------------------------
# API Endpoints
# -----------------------------
@app.post("/recommend/by-title")
def recommend_by_title(req: TitleRequest):
    key = req.title.strip().lower()

    if key not in title_to_idx:
        raise HTTPException(status_code=404, detail="Movie title not found")

    idx = title_to_idx[key]
    recs = hybrid_recommend(idx, req.alpha, req.top_n)

    return {
        "source_movie": df.loc[idx, "Title"],
        "recommendations": recs[
            ["Title", "Genre", "Rating", "Votes", "Description"]
        ].to_dict(orient="records")
    }

@app.post("/recommend/by-query")
def recommend_by_query(req: QueryRequest):
    """Get recommendations based on natural language query."""
    try:
        # Get model (will load if not already loaded)
        model = get_sentence_model()
        
        # Encode with memory-efficient settings
        with torch.no_grad():
            q_emb = model.encode(
                [req.query], 
                convert_to_numpy=True, 
                show_progress_bar=False,
                batch_size=1,
                normalize_embeddings=True
            )
        
        # Ensure float32 for memory efficiency
        if q_emb.dtype == np.float64:
            q_emb = q_emb.astype(np.float32)
        
        # Compute similarity
        sims = cosine_similarity(q_emb, desc_embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:req.top_n]

        recs = df.iloc[top_idx]

        return {
            "query": req.query,
            "recommendations": recs[
                ["Title", "Genre", "Rating", "Votes", "Description"]
            ].to_dict(orient="records")
        }
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health")
def health():
    """Health check endpoint."""
    global sentence_model, model_loading, model_load_error
    status = {
        "status": "ok",
        "model_loaded": sentence_model is not None,
        "model_loading": model_loading
    }
    if model_load_error:
        status["model_load_error"] = model_load_error
    return status
