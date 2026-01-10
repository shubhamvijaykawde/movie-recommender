### 🎬 LLM-Augmented Movie Recommendation System

A hybrid movie recommendation system combining metadata-based filtering and semantic search using sentence transformers. Deployable on free-tier cloud services with optimized memory usage.

Features

Dual Search Modes

By Movie Title: Recommend movies based on metadata (genre, director, actors) and semantic similarity

By Natural Language Query: Search using free-text descriptions (e.g., "dark philosophical science fiction")

Hybrid Recommendation Algorithm

Combines TF-IDF metadata similarity and semantic embeddings

Adjustable weighting (alpha) between metadata and semantic similarity

Production-Ready

FastAPI backend for high-performance recommendations

Streamlit frontend deployed on Hugging Face Spaces

Optimized for free-tier memory (CPU-only, pre-computed embeddings)

Architecture
Streamlit App (Hugging Face)
        │
        ▼
FastAPI Backend (Render)
        │
        ▼
Recommendation Engine
- Pre-computed artifacts: embeddings, similarity matrices, metadata

How It Works

Data Preprocessing: Clean and process movie metadata and descriptions

Embedding Generation: SentenceTransformers generate semantic embeddings

Similarity Computation: Pre-computed matrices for fast recommendations

Hybrid Scoring: Combines metadata + semantic similarity with adjustable alpha

Real-time Queries: Fast similarity search using pre-computed artifacts

Tech Stack

Backend: FastAPI, PyTorch, Sentence Transformers, scikit-learn, NumPy, Pandas

Frontend: Streamlit

Deployment: Render (backend), Hugging Face Spaces (frontend)

Installation & Usage

Prerequisites: Python 3.11+, pip

# Clone repository
git clone https://github.com/shubhamvijaykawde/movie-recommender.git
cd llm-augmented-movie-recommender

# Install backend dependencies
pip install -r Fast_requirements.txt

# Install frontend dependencies (if running Streamlit locally)
pip install -r requirements.txt

# Pre-compute artifacts
python -c "from etl import precompute_artifacts; precompute_artifacts()"

# Run backend
uvicorn api:app --reload --port 8080

# Run frontend
streamlit run app.py


API available at http://localhost:8080

Frontend connects to backend via API_URL in app.py

Notes

Pre-computed artifacts: data.pkl, desc_embeddings.pkl, meta_sim.pkl, tfidf_matrix.pkl, tfidf_vectorizer.pkl

Optimized for free-tier deployments: memory-efficient, CPU-only, lazy model loading
## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Dataset**: IMDB Movie Data (1000 movies)
- **Model**: `all-MiniLM-L6-v2` from [sentence-transformers](https://www.sbert.net/)
- **Frameworks**: FastAPI, Streamlit, PyTorch, scikit-learn
- **Hosting**: Render (backend), Hugging Face Spaces (frontend)

---

## 📧 Contact

For questions, suggestions, or issues, please open an issue on GitHub.

---

**⭐ If you find this project helpful, please consider giving it a star!**
