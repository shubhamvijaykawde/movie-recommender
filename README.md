# 🎬 LLM-Augmented Movie Recommendation System

A sophisticated hybrid movie recommendation system that combines traditional metadata-based filtering with modern semantic search using sentence transformers. Deployable on free-tier cloud services with optimized memory usage.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Deployment](#-deployment)
- [Performance Optimizations](#-performance-optimizations)
- [Examples](#-examples)
- [Contributing](#-contributing)

---

## ✨ Features

### 🔍 Dual Search Modes
- **By Movie Title**: Find similar movies based on metadata (genre, director, actors) and semantic similarity
- **Natural Language Query**: Search using free-text descriptions (e.g., "dark philosophical science fiction")

### 🎯 Hybrid Recommendation Algorithm
- **Metadata Similarity (TF-IDF)**: Captures structural similarities (genres, directors, actors)
- **Semantic Similarity (Sentence Embeddings)**: Understands plot and thematic similarities
- **Adjustable Weighting**: Tune the balance between metadata and semantic similarity (alpha parameter)

### 🚀 Production-Ready
- **FastAPI Backend**: High-performance REST API with automatic OpenAPI documentation
- **Streamlit Frontend**: Interactive web interface deployed on Hugging Face Spaces
- **Optimized for Free Tier**: Memory-efficient implementation for cloud deployment (Render free tier compatible)
- **CORS Enabled**: Ready for cross-origin requests from frontend applications

---

## 🏗️ Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  Streamlit App  │────────▶│  FastAPI Backend │────────▶│  Recommendation │
│ (Hugging Face)  │  HTTP   │    (Render)      │         │     Engine      │
└─────────────────┘         └──────────────────┘         └─────────────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │  Pre-computed    │
                            │  Artifacts:      │
                            │  - Embeddings    │
                            │  - Similarities  │
                            │  - Metadata      │
                            └──────────────────┘
```

### How It Works

1. **Data Preprocessing**: Movie metadata and descriptions are processed and vectorized
2. **Embedding Generation**: Sentence transformers create semantic embeddings of movie descriptions
3. **Similarity Computation**: Pre-computed similarity matrices for fast recommendations
4. **Hybrid Scoring**: Combines metadata and semantic similarities with configurable weights
5. **Real-time Queries**: Fast similarity search using pre-computed matrices

---

## 🛠️ Tech Stack

### Backend
- **FastAPI**: Modern Python web framework for building APIs
- **PyTorch**: Deep learning framework (CPU-only for memory efficiency)
- **Sentence Transformers**: Semantic embeddings using `all-MiniLM-L6-v2` model
- **scikit-learn**: TF-IDF vectorization and cosine similarity computation
- **NumPy & Pandas**: Data manipulation and numerical computations

### Frontend
- **Streamlit**: Interactive web application framework
- **Requests**: HTTP client for API communication

### Deployment
- **Render**: Backend API hosting (free tier optimized)
- **Hugging Face Spaces**: Frontend hosting with Streamlit

---

## 📦 Installation

### Prerequisites
- Python 3.11+
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/llm-augmented-movie-recommender.git
cd llm-augmented-movie-recommender
```

### Step 2: Install Dependencies

For local development:
```bash
# Install backend dependencies
pip install -r Fast_requirements.txt

# Install frontend dependencies (if running Streamlit locally)
pip install -r requirements.txt
```

### Step 3: Pre-compute Artifacts

Generate the required pickle files (data, embeddings, similarity matrices):

```bash
python -c "from etl import precompute_artifacts; precompute_artifacts()"
```

This will create:
- `data.pkl`: Processed movie data
- `desc_embeddings.pkl`: Pre-computed sentence embeddings
- `meta_sim.pkl`: Metadata similarity matrix
- `tfidf_matrix.pkl`: TF-IDF vectorization matrix
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer

**Note**: This process takes a few minutes and requires the `IMDB-Movie-Data.csv` file.

---

## 🚀 Usage

### Running Locally

#### Backend (FastAPI)
```bash
uvicorn api:app --reload --port 8080
```

The API will be available at `http://localhost:8080`

- API Documentation: `http://localhost:8080/docs`
- Health Check: `http://localhost:8080/health`

#### Frontend (Streamlit)
```bash
streamlit run app.py
```

Update `API_URL` in `app.py` to point to your FastAPI backend.

### Testing the API

Use the included test script:
```bash
python test_api_local.py
```

Or test manually with curl:

```bash
# Health check
curl http://localhost:8080/health

# Title-based recommendation
curl -X POST http://localhost:8080/recommend/by-title \
  -H "Content-Type: application/json" \
  -d '{"title": "Prometheus", "top_n": 5, "alpha": 0.45}'

# Natural language query
curl -X POST http://localhost:8080/recommend/by-query \
  -H "Content-Type: application/json" \
  -d '{"query": "dark philosophical science fiction", "top_n": 5}'
```

---

## 📚 API Documentation

### Endpoints

#### `POST /recommend/by-title`
Get movie recommendations based on a movie title.

**Request Body:**
```json
{
  "title": "Prometheus",
  "top_n": 6,
  "alpha": 0.45
}
```

**Parameters:**
- `title` (string, required): Movie title (case-insensitive)
- `top_n` (integer, optional): Number of recommendations (default: 6, max: 12)
- `alpha` (float, optional): Weight for metadata similarity (0.0-1.0, default: 0.45)
  - `alpha = 1.0`: Pure metadata similarity
  - `alpha = 0.0`: Pure semantic similarity
  - `alpha = 0.45`: Balanced (45% metadata, 55% semantic)

**Response:**
```json
{
  "source_movie": "Prometheus",
  "recommendations": [
    {
      "Title": "The Martian",
      "Genre": "Adventure, Sci-Fi",
      "Rating": 8.0,
      "Votes": 556097,
      "Description": "An astronaut becomes stranded on Mars..."
    },
    ...
  ]
}
```

#### `POST /recommend/by-query`
Get movie recommendations based on a natural language query.

**Request Body:**
```json
{
  "query": "dark philosophical science fiction",
  "top_n": 6
}
```

**Parameters:**
- `query` (string, required): Natural language description
- `top_n` (integer, optional): Number of recommendations (default: 6)

**Response:**
```json
{
  "query": "dark philosophical science fiction",
  "recommendations": [...]
}
```

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

### Interactive API Documentation

When running locally, visit `http://localhost:8080/docs` for interactive Swagger UI documentation.

---

## 📁 Project Structure

```
llm-augmented-movie-recommender/
│
├── api.py                      # FastAPI backend application
├── etl.py                      # ETL pipeline for data processing
├── app.py                      # Streamlit frontend application
│
├── Fast_requirements.txt       # Backend dependencies
├── requirements.txt            # Frontend dependencies
│
├── render.yaml                 # Render deployment configuration
├── DEPLOYMENT.md               # Detailed deployment guide
├── RENDER_SETUP.txt            # Quick deployment checklist
│
├── test_api_local.py           # API testing script
│
├── IMDB-Movie-Data.csv         # Movie dataset (1000 movies)
│
├── data.pkl                    # Pre-processed movie data (generated)
├── desc_embeddings.pkl         # Pre-computed embeddings (generated)
├── meta_sim.pkl               # Metadata similarity matrix (generated)
├── tfidf_matrix.pkl           # TF-IDF matrix (generated)
├── tfidf_vectorizer.pkl       # TF-IDF vectorizer (generated)
│
├── MovieRecommendation.ipynb   # Jupyter notebook with analysis
├── FurtherAnalysis.ipynb       # Additional analysis notebook
│
└── README.md                   # This file
```

---

## 🌐 Deployment

### Quick Deployment Guide

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

#### Backend (Render)

1. **Push code to GitHub**
2. **Connect to Render**: Go to [Render Dashboard](https://dashboard.render.com/) → New → Web Service
3. **Configure service**:
   - Repository: Your GitHub repo
   - Build Command: `pip install --upgrade pip && pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r Fast_requirements.txt`
   - Start Command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
   - Plan: Free
4. **Add environment variables** (optional):
   - `PYTHON_VERSION`: `3.11.0`
5. **Deploy!** Wait 5-10 minutes for first deployment

#### Frontend(Hugging Face Spaces)

1. **Create a Hugging Face Space** with Streamlit SDK
2. **Upload `app.py`** to your Space
3. **Update `API_URL`** in `app.py` to your Render API URL
4. **Commit and push** - Hugging Face auto-deploys

### Pre-computed Artifacts

⚠️ **Important**: Before deploying, ensure all pickle files are generated and committed to your repository:

```bash
python -c "from etl import precompute_artifacts; precompute_artifacts()"
git add *.pkl
git commit -m "Add pre-computed artifacts"
git push
```

---

## ⚡ Performance Optimizations

This project is optimized for free-tier cloud deployment (512MB RAM limit):

### Memory Optimizations
- ✅ **Lazy Model Loading**: SentenceTransformer loads only when needed (~100-200MB saved)
- ✅ **Float32 Precision**: All arrays use float32 instead of float64 (~50% memory savings)
- ✅ **CPU-Only PyTorch**: Smaller installation footprint
- ✅ **Pre-computed Similarities**: No runtime computation overhead
- ✅ **PyTorch Threading**: Limited to single thread to reduce memory usage

### Performance Characteristics
- **Startup Time**: ~5-10 seconds (without model), ~15-20 seconds (with model)
- **Title Recommendations**: <100ms (uses pre-computed matrices)
- **Query Recommendations**: ~500ms (includes model encoding)
- **Memory Usage**: ~250-350MB (within 512MB free tier limit)

---

## 💡 Examples

### Example 1: Title-Based Recommendation

```python
import requests

response = requests.post(
    "http://localhost:8080/recommend/by-title",
    json={
        "title": "The Matrix",
        "top_n": 5,
        "alpha": 0.5  # 50% metadata, 50% semantic
    }
)

recommendations = response.json()["recommendations"]
for movie in recommendations:
    print(f"{movie['Title']} - {movie['Genre']} (Rating: {movie['Rating']})")
```

**Output:**
```
The Matrix Reloaded - Action, Sci-Fi (Rating: 7.2)
Inception - Action, Sci-Fi, Thriller (Rating: 8.8)
Blade Runner 2049 - Drama, Mystery, Sci-Fi (Rating: 8.0)
Ex Machina - Drama, Sci-Fi, Thriller (Rating: 7.7)
Interstellar - Adventure, Drama, Sci-Fi (Rating: 8.6)
```

### Example 2: Natural Language Query

```python
response = requests.post(
    "http://localhost:8080/recommend/by-query",
    json={
        "query": "psychological thrillers with plot twists",
        "top_n": 5
    }
)

recommendations = response.json()["recommendations"]
for movie in recommendations:
    print(f"{movie['Title']} - {movie['Rating']}/10")
```

**Output:**
```
Shutter Island - 8.2/10
Gone Girl - 8.1/10
Inception - 8.8/10
The Prestige - 8.5/10
Fight Club - 8.8/10
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r Fast_requirements.txt
pip install -r requirements.txt

# Run tests
python test_api_local.py
```

---

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
