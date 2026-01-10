# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

from etl import run_etl_pipeline

# Memory optimizations for free tier (512MB limit)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Reduces memory usage
torch.set_num_threads(1)  # Limit CPU threads to reduce memory

# -----------------------------
# Initialize API
# -----------------------------
app = FastAPI(
    title="Movie Recommendation API",
    description="Hybrid TF-IDF + Embedding Movie Recommender",
    version="1.0"
)

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

# -----------------------------
# Load artifacts ONCE at startup
# -----------------------------
print("🔹 Loading artifacts...")
artifacts = run_etl_pipeline()

df = artifacts["data"]
meta_sim = artifacts["meta_sim"]
desc_embeddings = artifacts["desc_embeddings"]

# Lazy load SentenceTransformer model (only when needed for query endpoint)
# This saves ~100-200MB memory on startup for free tier
sentence_model = None

def get_sentence_model():
    """Lazy load the SentenceTransformer model only when needed."""
    global sentence_model
    if sentence_model is None:
        print("🔹 Loading SentenceTransformer model (lazy load)...")
        with torch.no_grad():  # Disable gradients to save memory
            sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            sentence_model.eval()
        print("✅ Model loaded successfully!")
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
    # Lazy load model on first use (saves memory on startup)
    model = get_sentence_model()
    # Encode with memory-efficient settings
    with torch.no_grad():
        q_emb = model.encode([req.query], convert_to_numpy=True, show_progress_bar=False)
    # Ensure float32 for memory efficiency
    if q_emb.dtype == np.float64:
        q_emb = q_emb.astype(np.float32)
    sims = cosine_similarity(q_emb, desc_embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:req.top_n]

    recs = df.iloc[top_idx]

    return {
        "query": req.query,
        "recommendations": recs[
            ["Title", "Genre", "Rating", "Votes", "Description"]
        ].to_dict(orient="records")
    }

@app.get("/health")
def health():
    return {"status": "ok"}
