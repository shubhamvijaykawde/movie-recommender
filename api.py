# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from etl import run_etl_pipeline


# -------------------------
# App
# -------------------------
app = FastAPI(
    title="Movie Recommendation API",
    description="Hybrid TF-IDF + Embedding Movie Recommender",
    version="1.0"
)


# -------------------------
# Load artifacts ONCE
# -------------------------
artifacts = run_etl_pipeline("IMDB-Movie-Data.csv")

df = artifacts["data"]
tfidf_matrix = artifacts["tfidf_matrix"]
desc_embeddings = artifacts["desc_embeddings"]

title_to_idx = {t.lower(): i for i, t in enumerate(df["Title"].tolist())}

query_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)


# -------------------------
# Schemas
# -------------------------
class TitleRequest(BaseModel):
    title: str
    top_n: int = 6
    alpha: float = 0.45


class QueryRequest(BaseModel):
    query: str
    top_n: int = 6


# -------------------------
# Helpers
# -------------------------
def hybrid_recommend(idx: int, alpha: float, top_n: int):
    meta_scores = tfidf_matrix[idx]
    meta_sim = cosine_similarity(meta_scores, tfidf_matrix)[0]

    desc_sim = cosine_similarity(
        desc_embeddings[idx].reshape(1, -1),
        desc_embeddings
    )[0]

    hybrid = alpha * meta_sim + (1 - alpha) * desc_sim
    top_idx = hybrid.argsort()[::-1][1: top_n + 1]

    return df.iloc[top_idx]


# -------------------------
# Endpoints
# -------------------------
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
    q_emb = query_model.encode([req.query])
    sims = cosine_similarity(q_emb, desc_embeddings)[0]
    top_idx = sims.argsort()[::-1][:req.top_n]

    recs = df.iloc[top_idx]
    return {
        "query": req.query,
        "recommendations": recs[
            ["Title", "Genre", "Rating", "Votes", "Description"]
        ].to_dict(orient="records")
    }
