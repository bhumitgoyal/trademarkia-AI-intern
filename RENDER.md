# Deploying This Project on Render

This project should be deployed as **two Render web services**:

1. `semantic-search-api` (FastAPI)
2. `semantic-search-ui` (Streamlit)

## 1) Push your repo to GitHub

Render deploys from a GitHub repository, so commit and push this project first.

## 2) Create the API service (`semantic-search-api`)

In Render dashboard:

1. `New` -> `Web Service`
2. Connect your repo
3. Use these settings:

- Runtime: `Python 3`
- Build Command:
  ```bash
  pip install --upgrade pip && pip install -r requirements.txt
  ```
- Start Command:
  ```bash
  python scripts/download_dataset.py --output-dir data && if [ ! -f data/pca_model.joblib ] || [ ! -f data/fcm_centers.npy ] || [ ! -f data/chroma_db/chroma.sqlite3 ]; then python scripts/run_pipeline.py --dataset-path data/20_newsgroups --skip-k-selection; fi && uvicorn src.main:app --host 0.0.0.0 --port $PORT
  ```

Set environment variables:

- `CHROMA_PERSIST_DIR=data/chroma_db`
- `CLUSTERING_ARTIFACTS_DIR=data`
- `CACHE_SIMILARITY_THRESHOLD=0.85`
- `CACHE_MAX_SIZE=10000`

After deploy, note your API URL, for example:
`https://semantic-search-api.onrender.com`

Health check URL:
`https://semantic-search-api.onrender.com/health`

## 3) Create the UI service (`semantic-search-ui`)

Create another Render Web Service from the same repo:

- Runtime: `Python 3`
- Build Command:
  ```bash
  pip install --upgrade pip && pip install -r requirements.txt
  ```
- Start Command:
  ```bash
  streamlit run frontend/app.py --server.address 0.0.0.0 --server.port $PORT
  ```

Set environment variable:

- `API_BASE_URL=https://semantic-search-api.onrender.com`
  - Replace with your real API service URL.

Open the UI service URL and run queries.

## Notes

- First API startup can be slow because it downloads dataset and builds embeddings/index.
- Free instances may sleep and have slower cold starts.
- If startup exceeds limits, move to a paid instance type.
