# movie-recommender
# 🎬 LLM-Augmented Movie Recommendation System

A hybrid movie recommendation system combining metadata analysis with semantic search using sentence transformers.

## Features
- Two search modes: By movie title and natural language query
- Hybrid scoring with adjustable weights
- FastAPI backend + Streamlit frontend
- Semantic understanding of movie descriptions

## Quick Start
```bash
pip install -r requirements.txt
pip install -r Fast_requirements.txt
uvicorn api:app --reload --port 8080
streamlit run app.py
