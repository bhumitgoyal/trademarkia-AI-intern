
FROM python:3.11-slim


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHROMA_PERSIST_DIR=/app/data/chroma_db \
    CLUSTERING_ARTIFACTS_DIR=/app/data \
    CACHE_SIMILARITY_THRESHOLD=0.85 \
    CACHE_MAX_SIZE=10000


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


COPY src/ ./src/
COPY scripts/ ./scripts/
COPY frontend/ ./frontend/
COPY start.py ./start.py


RUN mkdir -p /app/data


EXPOSE 8000
EXPOSE 8501


HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1


CMD ["python", "start.py"]