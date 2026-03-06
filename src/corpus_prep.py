import os
import re
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

NEWSGROUP_CATEGORIES = [
    "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x",
    "misc.forsale", "rec.autos", "rec.motorcycles", "rec.sport.baseball",
    "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med",
    "sci.space", "soc.religion.christian", "talk.politics.guns",
    "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc",
]

MIN_BODY_LENGTH = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "newsgroups"
EMBEDDING_BATCH_SIZE = 128

def clean_article(raw_text: str) -> str:
    lines = raw_text.split("\n")
    
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            header_end = i + 1
            break
    
    body_lines=lines[header_end:]
    sig_start = len(body_lines)
    for i, line in enumerate(body_lines):
        if line.strip() == "--" or line.strip() == "-- ":
            sig_start = i
            break
    body_lines=body_lines[:sig_start]
    
    cleaned=[]
    for line in body_lines:
        stripped = line.strip()

        
        if stripped.startswith(">"):
            continue

        if len(stripped.split()) < 3:
            continue

        if re.match(r'^(From|Subject|Date|Lines|Message-ID|Path):', stripped):
            continue

        cleaned.append(stripped)
    
    result = " ".join(cleaned)
    
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

def load_corpus(dataset_path: str) -> str:
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path '{dataset_path}' does not exist. "
            "Download from: https://archive.ics.uci.edu/dataset/113/twenty+newsgroups"
        )
    
    records=[]
    logger.info(f"Loading corpus from: {dataset_path}")
    
    for category in NEWSGROUP_CATEGORIES:
        category_path = dataset_path / category


        if not category_path.exists():
            logger.warning(f"Category directory not found: {category_path}, skipping")
            continue

        article_files = list(category_path.iterdir())
        logger.info(f"  {category}: {len(article_files)} articles")

        for article_file in article_files:
            if not article_file.is_file():
                continue

            try:

                raw_text = article_file.read_text(encoding="latin-1")
            except Exception as e:
                logger.debug(f"    Could not read {article_file}: {e}")
                continue

            clean_text = clean_article(raw_text)

            if len(clean_text) < MIN_BODY_LENGTH:
                continue

            records.append({
                "doc_id": f"{category}/{article_file.name}",
                "newsgroup": category,
                "raw_text": raw_text,
                "clean_text": clean_text,
            })

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} articles after cleaning (from ~20,000 raw)")
    return df

def generate_embeddings(
    texts: list[str],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Embedding {len(texts)} documents in batches of {batch_size}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
        convert_to_numpy=True,
    )

    logger.info(f"Embedding complete. Shape: {embeddings.shape}")
    return embeddings

def setup_chroma_client(persist_dir: str) -> chromadb.PersistentClient:
    """
    Initialize a persistent ChromaDB client.
    Data is written to disk so it survives API restarts.
    """
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    return client

def ingest_to_vector_db(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    persist_dir: str,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 512,
) -> chromadb.Collection:
    
    client = setup_chroma_client(persist_dir)


    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection '{collection_name}'")
    except Exception:
        pass  # Collection didn't exist, fine

    collection = client.create_collection(
        name=collection_name,

        metadata={"hnsw:space": "cosine"},
    )

    n_docs = len(df)
    logger.info(f"Ingesting {n_docs} documents into ChromaDB...")


    for start in tqdm(range(0, n_docs, batch_size), desc="Ingesting batches"):
        end = min(start + batch_size, n_docs)
        batch_df = df.iloc[start:end]
        batch_embeddings = embeddings[start:end]

        collection.add(
            ids=batch_df["doc_id"].tolist(),
            embeddings=batch_embeddings.tolist(),
            documents=batch_df["clean_text"].tolist(),
            metadatas=[
                {
                    "newsgroup": row["newsgroup"],
                    "doc_id": row["doc_id"],
                }
                for _, row in batch_df.iterrows()
            ],
        )

    logger.info(f"Ingestion complete. Collection '{collection_name}' has {collection.count()} documents.")
    return collection

def save_corpus_metadata(df: pd.DataFrame, output_path: str) -> None:

    df_meta = df[["doc_id", "newsgroup", "clean_text"]].copy()
    df_meta.to_parquet(output_path, index=False)
    logger.info(f"Corpus metadata saved to: {output_path}")
    
def save_embeddings(embeddings: np.ndarray, output_path: str) -> None:

    np.save(output_path, embeddings)
    logger.info(f"Embeddings saved to: {output_path} (shape: {embeddings.shape})")
    
def run_ingestion(
    dataset_path: str,
    output_dir: str = "data",
    chroma_persist_dir: str = "data/chroma_db",
) -> tuple[pd.DataFrame, np.ndarray]:
    
    os.makedirs(output_dir, exist_ok=True)


    df = load_corpus(dataset_path)


    embeddings = generate_embeddings(df["clean_text"].tolist())


    ingest_to_vector_db(df, embeddings, persist_dir=chroma_persist_dir)


    save_corpus_metadata(df, os.path.join(output_dir, "corpus_meta.parquet"))
    save_embeddings(embeddings, os.path.join(output_dir, "embeddings.npy"))

    return df, embeddings

if __name__ == "__main__":
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/20_newsgroups"
    run_ingestion(dataset_path=dataset_path)
