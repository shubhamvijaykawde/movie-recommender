import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
import numpy as np

# -------------------------
# Config
# -------------------------
API_URL = "https://movie-recommender-6ph3.onrender.com"

st.set_page_config(layout="wide", page_title="LLM-Aug Movie Recommender")
st.title("LLM-Augmented Movie Recommender")

# -------------------------
# Load model ONCE (HF is fine)
# -------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -------------------------
# UI
# -------------------------
mode = st.radio("Search mode:", ["By movie title", "Natural language query"])
top_n = st.slider("Number of recommendations", 3, 12, 6)
alpha = st.slider("Metadata weight (alpha)", 0.0, 1.0, 0.45, 0.05)

# -------------------------
# Helpers
# -------------------------
def display_recommendations(data, source=None):
    if source:
        st.subheader(f"Recommendations similar to: {source}")
    else:
        st.subheader("Recommendations")

    for rec in data:
        st.markdown(
            f"**{rec['Title']}** — {rec['Genre'].title()}  |  "
            f"Rating: {rec['Rating']}  |  Votes: {int(rec['Votes']):,}"
        )
        st.write(rec['Description'][:300] + "...")
        st.write("---")

# -------------------------
# Actions
# -------------------------
if mode == "By movie title":
    title = st.text_input("Enter a movie title")
    if st.button("Recommend"):
        r = requests.post(
            f"{API_URL}/recommend/by-title",
            json={"title": title, "top_n": top_n, "alpha": alpha},
            timeout=20
        )
        if r.ok:
            data = r.json()
            display_recommendations(
                data["recommendations"],
                source=data["source_movie"]
            )
        else:
            st.error(r.text)

else:
    query = st.text_input("Describe the movie you want")
    if st.button("Search"):
        with st.spinner("Encoding query..."):
            emb = model.encode(
                query,
                normalize_embeddings=True
            ).astype(np.float32)

        r = requests.post(
            f"{API_URL}/recommend/by-embedding",
            json={
                "embedding": emb.tolist(),
                "top_n": top_n
            },
            timeout=20
        )

        if r.ok:
            display_recommendations(r.json()["recommendations"])
        else:
            st.error(r.text)
