import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_data(csv_path: str) -> pd.DataFrame:
    """
    Extract stage: load raw IMDB data from CSV.
    """
    df = pd.read_csv(csv_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and normalize text fields.
    """
    df = df.copy()

    # Fill numeric nulls
    df["Revenue (Millions)"].fillna(df["Revenue (Millions)"].median(), inplace=True)
    df["Metascore"].fillna(df["Metascore"].median(), inplace=True)

    # Normalize text columns
    text_cols = ["Title", "Genre", "Description", "Director", "Actors"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    df["Genre"] = df["Genre"].str.lower()
    df["Director"] = df["Director"].str.lower()

    return df


def build_metadata_field(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine structured metadata into a single text field.
    """
    df = df.copy()

    df["metadata"] = (
        df["Genre"].fillna("") + " " +
        df["Director"].fillna("") + " " +
        df["Actors"].fillna("")
    )

    return df


def generate_description_embeddings(
    df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Generate semantic embeddings from movie descriptions.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        df["Description"].tolist(),
        batch_size=64,
        show_progress_bar=True
    )
    return embeddings


def build_tfidf_matrix(metadata_series):
    """
    Build TF-IDF feature matrix from metadata text.
    """
    vectorizer = TfidfVectorizer(
        min_df=2,
        ngram_range=(1, 2),
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(metadata_series)
    return tfidf_matrix, vectorizer


def compute_similarity_matrix(feature_matrix):
    """
    Compute cosine similarity matrix.
    """
    return cosine_similarity(feature_matrix)


def run_etl_pipeline(csv_path: str):
    """
    Full ETL pipeline orchestration.
    """
    # Extract
    df = extract_data(csv_path)

    # Transform
    df = clean_data(df)
    df = build_metadata_field(df)

    # Load
    tfidf_matrix, tfidf_vectorizer = build_tfidf_matrix(df["metadata"])
    meta_sim = compute_similarity_matrix(tfidf_matrix)

    desc_embeddings = generate_description_embeddings(df)
    desc_sim = compute_similarity_matrix(desc_embeddings)

    return {
        "data": df,
        "tfidf_matrix": tfidf_matrix,
        "tfidf_vectorizer": tfidf_vectorizer,
        "meta_sim": meta_sim,
        "desc_embeddings": desc_embeddings,
        "desc_sim": desc_sim
    }