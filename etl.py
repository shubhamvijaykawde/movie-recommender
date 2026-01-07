# etl.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


# -------------------------
# Extract
# -------------------------
def extract_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


# -------------------------
# Transform
# -------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fix pandas FutureWarning (NO inplace)
    df["Revenue (Millions)"] = df["Revenue (Millions)"].fillna(
        df["Revenue (Millions)"].median()
    )
    df["Metascore"] = df["Metascore"].fillna(
        df["Metascore"].median()
    )

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


# -------------------------
# Feature Engineering
# -------------------------
def build_tfidf_matrix(metadata_series):
    vectorizer = TfidfVectorizer(
        min_df=2,
        ngram_range=(1, 2),
        stop_words="english"
    )
    matrix = vectorizer.fit_transform(metadata_series)
    return matrix, vectorizer


def generate_description_embeddings(
    descriptions,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
):
    model = SentenceTransformer(model_name, device="cpu")
    embeddings = model.encode(
        descriptions,
        batch_size=32,
        show_progress_bar=False
    )
    return embeddings


# -------------------------
# Orchestrator
# -------------------------
def run_etl_pipeline(csv_path: str):
    df = extract_data(csv_path)
    df = clean_data(df)
    df = build_metadata_field(df)

    tfidf_matrix, tfidf_vectorizer = build_tfidf_matrix(df["metadata"])
    desc_embeddings = generate_description_embeddings(df["Description"].tolist())

    return {
        "data": df,
        "tfidf_matrix": tfidf_matrix,
        "tfidf_vectorizer": tfidf_vectorizer,
        "desc_embeddings": desc_embeddings
    }
