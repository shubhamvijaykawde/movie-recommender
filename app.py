# app.py (Streamlit frontend calling FastAPI)
import streamlit as st
import requests
import time

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

def fetch_by_query(query, top_n, max_retries=3, timeout=60):
    """
    Fetch recommendations by query with retry logic for model loading.
    The model loads in background after startup, so first request may take time.
    """
    payload = {
        "query": query,
        "top_n": top_n
    }
    
    for attempt in range(max_retries):
        try:
            with st.spinner(f"Processing your query{' (this may take 10-20 seconds on first request)...' if attempt == 0 else '...'}"):
                response = requests.post(
                    f"{API_URL}/recommend/by-query", 
                    json=payload,
                    timeout=timeout
                )
                response.raise_for_status()
                return response.json()
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                st.warning(f"Request timed out. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                st.error("Request timed out. The model may still be loading. Please wait a moment and try again.")
                return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 502 or e.response.status_code == 503:
                # Service unavailable - model might be loading
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    st.warning(f"Service temporarily unavailable (model may be loading). Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    st.error("Service unavailable. The model is still loading. Please wait 30-60 seconds and try again.")
                    st.info("💡 Tip: Check the API health at /health endpoint to see if model is loaded.")
                    return None
            else:
                st.error(f"API error: {e}")
                if e.response.status_code == 404:
                    st.error("Endpoint not found. Please check your API URL.")
                return None
        except requests.exceptions.ConnectionError:
            st.error("Connection error. Please check if the API is running and the URL is correct.")
            return None
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return None
    
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
