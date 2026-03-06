# 20 Newsgroups — Semantic Search with Fuzzy Clustering and Semantic Cache

A semantic search system over the [20 Newsgroups dataset](https://archive.uci.edu/dataset/113/twenty+newsgroups) that combines:

1. **Vector embeddings + ChromaDB retrieval**
2. **Fuzzy C-Means clustering**
3. **Semantic cache for paraphrased query detection**
4. **FastAPI backend**
5. **Streamlit frontend**
6. **Docker containerization**
7. **One-command system launcher**

The system recognises **semantically similar queries** and serves cached results without recomputing retrieval.

---

## System Architecture

```
User Query
│
▼
Streamlit Frontend
│
▼
FastAPI Backend
│
▼
Semantic Cache
│
▼
Query Engine
│
▼
ChromaDB Vector Search
│
▼
Relevant Documents
```

Offline pipeline:

```
Dataset
│
▼
Cleaning
│
▼
Embeddings
│
▼
Vector DB Ingestion
│
▼
Fuzzy Clustering
│
▼
Saved Artifacts
```

---

## Project Structure

```
project/
│
├── start.py                     # One-command launcher
│
├── src/
│   ├── __init__.py
│   ├── corpus_prep.py           # Corpus cleaning + embeddings
│   ├── fuzzy_clustering.py      # Fuzzy C-Means clustering
│   ├── semantic_cache.py        # Custom semantic cache
│   ├── query_engine.py          # Query embedding + retrieval
│   └── main.py                  # FastAPI API service
│
├── frontend/
│   └── app.py                   # Streamlit frontend UI
│
├── scripts/
│   ├── download_dataset.py
│   └── run_pipeline.py          # Full preprocessing pipeline
│
├── tests/
│   └── test_cache.py
│
├── data/                        # Generated artifacts
│   ├── 20_newsgroups/
│   ├── embeddings.npy
│   ├── corpus_meta.parquet
│   ├── pca_model.joblib
│   ├── fcm_centers.npy
│   ├── membership_matrix.npy
│   └── chroma_db/
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Quick Start (Recommended)

Run the entire system with one command:

```bash
python start.py
```

The launcher will:

1. Install dependencies
2. Run the preprocessing pipeline if needed
3. Start the FastAPI backend
4. Wait for backend readiness
5. Launch the Streamlit frontend

---

## Access the System

| Service     | URL                        |
|-------------|----------------------------|
| API         | http://localhost:8000      |
| API Docs    | http://localhost:8000/docs |
| Frontend UI | http://localhost:8501      |

---

## Manual Setup

### 1. Create virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python scripts/download_dataset.py
```

Downloads the dataset to `data/20_newsgroups`.

### 3. Run Preprocessing Pipeline

```bash
python scripts/run_pipeline.py --dataset-path data/20_newsgroups
```

Optional flags:

```
--skip-k-selection
--skip-embedding
```

### 4. Start API

```bash
uvicorn src.main:app --reload
```

### 5. Start Frontend

```bash
streamlit run frontend/app.py
```

---

## API Reference

### POST /query

Embed a query, check semantic cache, retrieve results.

**Request**

```json
{
  "query": "What are the arguments about gun control?"
}
```

**Response — cache miss**

```json
{
  "query": "...",
  "cache_hit": false,
  "result": {
    "retrieved_documents": [...],
    "dominant_cluster": 3
  }
}
```

**Response — cache hit**

```json
{
  "cache_hit": true,
  "matched_query": "...",
  "similarity_score": 0.91
}
```

### GET /cache/stats

Returns cache statistics.

### DELETE /cache

Flushes the cache.

### GET /health

Health check endpoint.

---

## Design Decisions

### Embedding Model

`all-MiniLM-L6-v2` produces 384-dimensional embeddings. Chosen for its small memory footprint, fast inference, and strong semantic similarity performance.

### Vector Store

ChromaDB was chosen for persistent storage, simple setup, cosine similarity search, and metadata filtering.

### Fuzzy Clustering

Instead of assigning each document to a single cluster, Fuzzy C-Means assigns **membership probabilities**:

```
cluster 3  → 0.72
cluster 11 → 0.18
cluster 7  → 0.10
```

This reflects the fact that documents may span multiple topics.

### Semantic Cache

Traditional cache maps `query string → result`. The semantic cache maps `query embedding → similar cached query`:

1. Embed incoming query
2. Predict cluster memberships
3. Identify relevant cache entries
4. Compute cosine similarity
5. Return cached result if similarity exceeds threshold (default: `0.85`)

### Cache Lookup Optimization

Instead of an O(n) full cache scan, **cluster-gated search** reduces comparisons to O(n/k). For 10,000 cache entries across 15 clusters, this yields ~667 comparisons per lookup.

---

## Testing

```bash
pytest tests/
```

Tests cover cache hits, cache misses, paraphrase detection, LRU eviction, and concurrency safety.

---

## Docker

**Build and run**

```bash
docker build -t newsgroups-search .
docker run -p 8000:8000 -p 8501:8501 -v $(pwd)/data:/app/data newsgroups-search
```

**Docker Compose**

```bash
docker-compose up --build   # Start
docker-compose up -d        # Detached
docker-compose down         # Stop
```

---

## Environment Variables

```bash
cp .env.example .env
```

| Variable                   | Default          |
|----------------------------|------------------|
| CHROMA_PERSIST_DIR         | data/chroma_db   |
| CLUSTERING_ARTIFACTS_DIR   | data             |
| CACHE_SIMILARITY_THRESHOLD | 0.85             |
| CACHE_MAX_SIZE             | 10000            |

---

## Estimated Runtime (CPU)

| Step             | Time       |
|------------------|------------|
| Dataset download | 1–2 min    |
| Cleaning         | 1–2 min    |
| Embedding        | 20–40 min  |
| Vector ingestion | 3–5 min    |
| Clustering       | 5–15 min   |
| **Total**        | 30–60 min  |

With GPU: ~5–10 minutes.

---

## Future Improvements

- FAISS ANN search
- Redis distributed cache
- GPU embeddings
- Query analytics dashboard
- Cluster visualization

---

## Author

**Bhumit Goyal**  
B.Tech Computer Science Engineering, VIT Vellore