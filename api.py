# api.py (Render backend — NO TRANSFORMERS)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from etl import run_etl_pipeline

# -----------------------------
# Load artifacts ONCE
# -----------------------------
print("🔹 Loading artifacts...")
artifacts = run_etl_pipeline()

df = artifacts["data"]
meta_sim = artifacts["meta_sim"]
desc_embeddings = artifacts["desc_embeddings"].astype(np.float32)

title_to_idx = {
    title.lower(): idx
    for idx, title in enumerate(df["Title"].tolist())
}

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Movie Recommendation API",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Schemas
# -----------------------------
class TitleRequest(BaseModel):
    title: str
    top_n: int = 6
    alpha: float = 0.45

class EmbeddingQueryRequest(BaseModel):
    embedding: list[float]
    top_n: int = 6

# -----------------------------
# Helpers
# -----------------------------
def hybrid_recommend(idx: int, alpha: float, top_n: int):
    emb_sim = cosine_similarity(
        desc_embeddings[idx].reshape(1, -1),
        desc_embeddings
    )[0]

    hybrid_sim = alpha * meta_sim[idx] + (1 - alpha) * emb_sim
    top_idx = np.argsort(hybrid_sim)[::-1]
    top_idx = [i for i in top_idx if i != idx][:top_n]

    return df.iloc[top_idx]

# -----------------------------
# Endpoints
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

@app.post("/recommend/by-embedding")
def recommend_by_embedding(req: EmbeddingQueryRequest):
    q_emb = np.array(req.embedding, dtype=np.float32).reshape(1, -1)

    sims = cosine_similarity(q_emb, desc_embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:req.top_n]
    recs = df.iloc[top_idx]

    return {
        "recommendations": recs[
            ["Title", "Genre", "Rating", "Votes", "Description"]
        ].to_dict(orient="records")
    }

@app.get("/health")
def health():
    return {"status": "ok"}
