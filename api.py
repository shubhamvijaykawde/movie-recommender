from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from etl import run_etl_pipeline

# --------------------------------------------------
# Initialize app
# --------------------------------------------------
app = FastAPI(
    title="Movie Recommendation API",
    description="Hybrid TF-IDF + Embedding Movie Recommender",
    version="1.0"
)

# --------------------------------------------------
# Load ETL artifacts ONCE at startup
# --------------------------------------------------
artifacts = run_etl_pipeline("IMDB-Movie-Data.csv")

df = artifacts["data"]
meta_sim = artifacts["meta_sim"]
desc_sim = artifacts["desc_sim"]
desc_embeddings = artifacts["desc_embeddings"]

title_to_idx = {t.lower(): i for i, t in enumerate(df["Title"].tolist())}

query_model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# Request schemas
# --------------------------------------------------
class TitleRequest(BaseModel):
    title: str
    top_n: int = 6
    alpha: float = 0.45

class QueryRequest(BaseModel):
    query: str
    top_n: int = 6

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def hybrid_recommend(idx: int, alpha: float, top_n: int):
    hybrid_sim = alpha * meta_sim + (1 - alpha) * desc_sim
    sims = hybrid_sim[idx]
    top_idx = np.argsort(sims)[::-1]
    top_idx = [i for i in top_idx if i != idx][:top_n]
    return df.iloc[top_idx]

# --------------------------------------------------
# API endpoints
# --------------------------------------------------
@app.post("/recommend/by-title")
def recommend_by_title(req: TitleRequest):
    key = req.title.strip().lower()

    if key not in title_to_idx:
        raise HTTPException(status_code=404, detail="Movie title not found")

    idx = title_to_idx[key]
    recs = hybrid_recommend(idx, req.alpha, req.top_n)

    return {
        "source_movie": df.loc[idx, "Title"],
        "recommendations": recs[[
            "Title", "Genre", "Rating", "Votes", "Description"
        ]].to_dict(orient="records")
    }


@app.post("/recommend/by-query")
def recommend_by_query(req: QueryRequest):
    q_emb = query_model.encode([req.query])
    sims = cosine_similarity(q_emb, desc_embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:req.top_n]

    recs = df.iloc[top_idx]
    return {
        "query": req.query,
        "recommendations": recs[
            ["Title", "Genre", "Rating", "Votes", "Description"]
        ].to_dict(orient="records")
    }
