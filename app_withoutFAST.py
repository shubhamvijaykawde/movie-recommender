# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# --------- load precomputed artifacts to speed startup ----------
# For simplicity, the app will recompute embeddings when launched.
# If you prefer faster startup, pickle/model artifacts after running in notebook.

@st.cache_resource
def load_data():
    df = pd.read_csv("IMDB-Movie-Data.csv")
    df.columns = [c.strip() for c in df.columns]
    for col in ["Title","Genre","Description","Director","Actors"]:
        df[col] = df[col].astype(str).str.strip()
    df["Genre"] = df["Genre"].str.lower()
    return df

@st.cache_resource
def init_model_and_embeddings(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    desc_embeddings = model.encode(df["Description"].tolist(), show_progress_bar=False, batch_size=64)
    # metadata tfidf matrix and similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_meta = TfidfVectorizer(min_df=2, ngram_range=(1,2), stop_words="english")
    meta_matrix = tfidf_meta.fit_transform((df["Genre"].fillna("") + " " + df["Director"].fillna("") + " " + df["Actors"].fillna("")))
    meta_sim = cosine_similarity(meta_matrix)
    desc_sim = cosine_similarity(desc_embeddings)
    return model, desc_embeddings, meta_sim, desc_sim

df = load_data()
model, desc_embeddings, meta_sim, desc_sim = init_model_and_embeddings(df)
title_to_idx = {t.lower(): i for i, t in enumerate(df["Title"].tolist())}

st.set_page_config(layout="wide", page_title="LLM-aug Movie Recommender")
st.title("LLM-Augmented Movie Recommender")

mode = st.radio("Search mode:", ["By movie title", "Natural language query"])

alpha = st.slider("Metadata weight (alpha)", 0.0, 1.0, 0.45, 0.05)
top_n = st.slider("Number of recommendations", 3, 12, 6)

def hybrid_recommend_from_idx(idx, alpha, top_n):
    hybrid = alpha * meta_sim + (1 - alpha) * desc_sim
    sims = hybrid[idx]
    top_idx = np.argsort(sims)[::-1]
    top_idx = [i for i in top_idx if i != idx][:top_n]
    return df.iloc[top_idx]

def recommend_by_query(query, top_n):
    q_emb = model.encode([query])
    q_sim = cosine_similarity(q_emb, desc_embeddings)[0]
    top_idx = np.argsort(q_sim)[::-1][:top_n]
    return df.iloc[top_idx]

if mode == "By movie title":
    title_input = st.text_input("Enter a movie title (e.g., Prometheus)")
    if st.button("Recommend"):
        if not title_input:
            st.warning("Please enter a title.")
        else:
            key = title_input.strip().lower()
            if key not in title_to_idx:
                st.error("Title not found. Check spelling or try another movie.")
            else:
                idx = title_to_idx[key]
                recs = hybrid_recommend_from_idx(idx, alpha, top_n)
                st.subheader(f"Recommendations similar to: {df.loc[idx,'Title']}")
                for i, row in recs.iterrows():
                    st.markdown(f"**{row['Title']}** — {row['Genre'].title()}  |  Rating: {row['Rating']}  |  Votes: {int(row['Votes']):,}")
                    # simple explanation
                    src_genres = set(df.loc[idx,"Genre"].split(","))
                    rec_genres = set(str(row["Genre"]).split(","))
                    common_genres = src_genres.intersection(rec_genres)
                    expl = []
                    if common_genres:
                        expl.append("shares genre(s): " + ", ".join([g.title() for g in common_genres]))
                    # show short description
                    st.write(row["Description"][:280] + ("..." if len(row["Description"])>280 else ""))
                    st.write("— " + ("; ".join(expl) if expl else "Based on semantic similarity and metadata."))
                    st.write("---")
else:
    q = st.text_input("Enter a natural-language request (e.g., 'relaxing sci-fi with strong female lead')")
    if st.button("Search"):
        if not q:
            st.warning("Please enter a query.")
        else:
            recs = recommend_by_query(q, top_n)
            st.subheader(f"Results for: \"{q}\"")
            for _, row in recs.iterrows():
                st.markdown(f"**{row['Title']}** — {row['Genre'].title()}  |  Rating: {row['Rating']}  |  Votes: {int(row['Votes']):,}")
                st.write(row["Description"][:300] + ("..." if len(row["Description"])>300 else ""))
                st.write("---")