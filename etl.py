# etl.py
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# -----------------------------
# 1. Extract
# -----------------------------
def extract_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

# -----------------------------
# 2. Transform
# -----------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Revenue (Millions)"] = df["Revenue (Millions)"].fillna(df["Revenue (Millions)"].median())
    df["Metascore"] = df["Metascore"].fillna(df["Metascore"].median())

    text_cols = ["Title", "Genre", "Description", "Director", "Actors"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    df["Genre"] = df["Genre"].str.lower()
    df["Director"] = df["Director"].str.lower()
    return df

def build_metadata_field(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["metadata"] = (
        df["Genre"].fillna("") + " " +
        df["Director"].fillna("") + " " +
        df["Actors"].fillna("")
    )
    return df

# -----------------------------
# 3. Load (precomputed artifacts)
# -----------------------------
def load_artifacts():
    """
    Load precomputed artifacts (pickle files) for memory-efficient startup.
    """
    with open("data.pkl", "rb") as f:
        df = pickle.load(f)
    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    with open("meta_sim.pkl", "rb") as f:
        meta_sim = pickle.load(f)
    with open("desc_embeddings.pkl", "rb") as f:
        desc_embeddings = pickle.load(f)
    with open("desc_sim.pkl", "rb") as f:
        desc_sim = pickle.load(f)
    return {
        "data": df,
        "tfidf_matrix": tfidf_matrix,
        "tfidf_vectorizer": tfidf_vectorizer,
        "meta_sim": meta_sim,
        "desc_embeddings": desc_embeddings,
        "desc_sim": desc_sim
    }

# -----------------------------
# 4. Precompute helper (run locally only)
# -----------------------------
def precompute_artifacts(csv_path="IMDB-Movie-Data.csv"):
    """
    Run locally to precompute embeddings and similarity matrices.
    Save as .pkl files for Render deployment.
    """
    df = extract_data(csv_path)
    df = clean_data(df)
    df = build_metadata_field(df)

    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1,2), stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["metadata"])
    meta_sim = cosine_similarity(tfidf_matrix)

    # Embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    desc_embeddings = model.encode(df["Description"].tolist(), batch_size=64, show_progress_bar=True)
    desc_sim = cosine_similarity(desc_embeddings)

    # Save all
    with open("data.pkl", "wb") as f:
        pickle.dump(df, f)
    with open("tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    with open("meta_sim.pkl", "wb") as f:
        pickle.dump(meta_sim, f)
    with open("desc_embeddings.pkl", "wb") as f:
        pickle.dump(desc_embeddings, f)
    with open("desc_sim.pkl", "wb") as f:
        pickle.dump(desc_sim, f)
    print("✅ All artifacts precomputed and saved.")

# -----------------------------
# 5. ETL entrypoint for API
# -----------------------------
def run_etl_pipeline(csv_path=None):
    """
    Load precomputed artifacts for production (Render).
    """
    return load_artifacts()
