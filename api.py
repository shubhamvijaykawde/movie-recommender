# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from etl import run_etl_pipeline

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

# Load SentenceTransformer model ONCE (critical for memory optimization)
print("🔹 Loading SentenceTransformer model...")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded successfully!")

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
    # Use the pre-loaded model (no reload on each request)
    q_emb = sentence_model.encode([req.query])
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
