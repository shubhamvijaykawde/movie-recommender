# app.py (Streamlit frontend calling FastAPI)
import streamlit as st
import requests

# -------------------------
# Config
# -------------------------
API_URL = "http://localhost:8080"  # Replace with deployed URL when live

st.set_page_config(layout="wide", page_title="LLM-aug Movie Recommender")
st.title("LLM-Augmented Movie Recommender")

# -------------------------
# User inputs
# -------------------------
mode = st.radio("Search mode:", ["By movie title", "Natural language query"])
top_n = st.slider("Number of recommendations", 3, 12, 6)
alpha = st.slider("Metadata weight (alpha, for title search)", 0.0, 1.0, 0.45, 0.05)

# -------------------------
# Helper functions
# -------------------------
def fetch_by_title(title, top_n, alpha):
    payload = {
        "title": title,
        "top_n": top_n,
        "alpha": alpha
    }
    try:
        response = requests.post(f"{API_URL}/recommend/by-title", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None

def fetch_by_query(query, top_n):
    payload = {
        "query": query,
        "top_n": top_n
    }
    try:
        response = requests.post(f"{API_URL}/recommend/by-query", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None

def display_recommendations(data, source=None):
    if source:
        st.subheader(f"Recommendations similar to: {source}")
    else:
        st.subheader(f"Recommendations")

    for rec in data:
        st.markdown(f"**{rec['Title']}** — {rec['Genre'].title()}  |  Rating: {rec['Rating']}  |  Votes: {int(rec['Votes']):,}")
        st.write(rec['Description'][:300] + ("..." if len(rec['Description']) > 300 else ""))
        st.write("---")

# -------------------------
# Main interaction
# -------------------------
if mode == "By movie title":
    title_input = st.text_input("Enter a movie title (e.g., Prometheus)")
    if st.button("Recommend"):
        if not title_input:
            st.warning("Please enter a title.")
        else:
            result = fetch_by_title(title_input.strip(), top_n, alpha)
            if result:
                display_recommendations(result['recommendations'], source=result['source_movie'])

else:  # Natural language query
    query_input = st.text_input("Enter a natural-language request (e.g., 'dark philosophical science fiction')")
    if st.button("Search"):
        if not query_input:
            st.warning("Please enter a query.")
        else:
            result = fetch_by_query(query_input.strip(), top_n)
            if result:
                display_recommendations(result['recommendations'])
