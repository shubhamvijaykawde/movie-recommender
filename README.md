# 🎬 LLM-Augmented Movie Recommendation System

A hybrid movie recommendation system combining **metadata analysis** with **semantic search** using sentence transformers. Perfect for discovering movies based on titles or natural language queries!

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg?logo=python)](https://www.python.org/)

---

## ✨ Features

- 🎯 **Dual Search Modes**
  - Search by movie title for similar recommendations
  - Natural language queries (e.g., "dark philosophical sci-fi")

- 🧠 **Hybrid Algorithm**
  - Combines TF-IDF metadata similarity with semantic embeddings
  - Adjustable weighting between metadata and semantic search
  - Pre-computed similarity matrices for lightning-fast responses

- 🚀 **Production Ready**
  - FastAPI backend with auto-generated OpenAPI docs
  - Streamlit frontend with retry logic and error handling
  - Optimized for free-tier cloud deployment (512MB RAM)

---

## 🏗️ Architecture

```
Frontend (Streamlit)  →  Backend (FastAPI)  →  Recommendation Engine
                              ↓
                     Pre-computed Artifacts
                     (Embeddings, Similarities)
```

---

## 🛠️ Tech Stack

**Backend**: FastAPI, PyTorch, Sentence Transformers, scikit-learn  
**Frontend**: Streamlit, Requests  
**Deployment**: Render (Backend), Hugging Face Spaces (Frontend)

---

## 📦 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/shubhamvijaykawde/movie-recommender.git
cd movie-recommender

# Install dependencies
pip install -r Fast_requirements.txt
pip install -r requirements.txt
```

### 2. Generate Artifacts
run : python etl_offline.py
```bash
python -c "from offline_etl import precompute_artifacts; precompute_artifacts()"
```

This creates the required pickle files (`data.pkl`, `desc_embeddings.pkl`, `meta_sim.pkl`, etc.)

### 3. Run Locally

**Backend:**
```bash
uvicorn api:app --reload --port 8080
```
Visit `http://localhost:8080/docs` for interactive API docs.

**Frontend:**
```bash
streamlit run app.py
```
Update `API_URL` in `app.py` to point to your backend.

---

## 📚 API Endpoints

### `POST /recommend/by-title`
Get similar movies based on a title.
POST /recommend/by-title
{
  "title": "Prometheus",
  "top_n": 5,
  "alpha": 0.45
}

**Parameters:**
- `title` (required): Movie title
- `top_n` (optional): Number of recommendations (default: 6)
- `alpha` (optional): Weight for metadata similarity (0.0-1.0, default: 0.45)

### `POST /recommend/by-query`
Natural language movie search.
POST /recommend/by-query
{
  "query": "psychological thrillers with plot twists",
  "top_n": 5
}

### `GET /health`
Health check endpoint.

---

## 🌐 Deployment

### Backend (Render)

1. Push code to GitHub
2. Connect repo in [Render Dashboard](https://dashboard.render.com/)
3. Configure:
   - **Build Command**: `pip install --upgrade pip && pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r Fast_requirements.txt`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
4. Add environment variable: `PYTHON_VERSION` = `3.11.0`

⚠️ **Important**: Commit all `.pkl` files before deploying!

### Frontend (Hugging Face Spaces)

1. Create a Streamlit Space on Hugging Face
2. Upload `app.py`
3. Update `API_URL` in `app.py` with your Render URL
4. Deploy!

---

## 📁 Project Structure

```
├── api.py                 # FastAPI backend
├── etl.py                 # Data processing pipeline
├── offline_etl.py         # Local embedding precomputation
├── app.py                 # Streamlit frontend
├── Fast_requirements.txt  # Backend dependencies
├── requirements.txt       # Frontend dependencies
├── render.yaml           # Render config
├── IMDB-Movie-Data.csv   # Dataset (1000 movies)
├── *.pkl                 # Pre-computed artifacts
└── *.ipynb               # Analysis notebooks
```

---

## ⚡ Performance

Optimized for free-tier deployment:

Title recommendations: <100ms
Query recommendations: ~500ms
Memory usage: ~250MB (safe for Render free-tier)
Float32 precision saves ~50% memory
Precomputed embeddings & TF-IDF matrices → no runtime heavy computation

---

## 💡 Example Usage

```python
import requests

# By title
resp = requests.post(
    "http://localhost:8080/recommend/by-title",
    json={"title": "The Matrix", "top_n": 5, "alpha": 0.5}
)
print(resp.json()["recommendations"])

# Natural language query
resp = requests.post(
    "http://localhost:8080/recommend/by-query",
    json={"query": "dark sci-fi thrillers", "top_n": 5}
)
print(resp.json()["recommendations"])
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

MIT License - feel free to use this project for learning or building your own recommendation system!

---

## 🙏 Acknowledgments

- **Dataset**: IMDB Movie Data (1000 movies)
- **Model**: `all-MiniLM-L6-v2` from [sentence-transformers](https://www.sbert.net/)
- **Frameworks**: FastAPI, Streamlit, PyTorch, scikit-learn

---

**⭐ Star this repo if you find it helpful!**

For questions or issues, please open a GitHub issue.
