import os
import numpy as np
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import skfuzzy as fuzz
import joblib
from loguru import logger

from .corpus_prep import EMBEDDING_MODEL, COLLECTION_NAME

class QueryEngine:
    def __init__(
        self,
        chroma_persist_dir: str,
        clustering_artifacts_dir: str,
        embedding_model: str = EMBEDDING_MODEL,
        n_results: int = 5,
        fcm_fuzziness: float = 1.5,
    ):
        self.n_results = n_results
        self.fcm_fuzziness = fcm_fuzziness

        logger.info("Initializing QueryEngine...")

        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)

        logger.info(f"Connecting to ChromaDB at: {chroma_persist_dir}")
        client = chromadb.PersistentClient(
            path=chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_collection(COLLECTION_NAME)
        logger.info(f"ChromaDB collection '{COLLECTION_NAME}': {self.collection.count()} documents")

        logger.info(f"Loading clustering artifacts from: {clustering_artifacts_dir}")
        self.pca_model = joblib.load(
            os.path.join(clustering_artifacts_dir, "pca_model.joblib")
        )
        self.fcm_centers = np.load(
            os.path.join(clustering_artifacts_dir, "fcm_centers.npy")
        )
        self.n_clusters = self.fcm_centers.shape[0]
        logger.info(f"FCM model loaded: {self.n_clusters} clusters")

        logger.info("QueryEngine ready.")

    def embed_query(self, query_text: str) -> np.ndarray:

        embedding = self.encoder.encode(
            query_text,
            normalize_embeddings=True,  
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)

    def predict_cluster_memberships(self, embedding: np.ndarray) -> np.ndarray:

        reduced = self.pca_model.transform(embedding.reshape(1, -1))  

        norm = np.linalg.norm(reduced)
        if norm > 0:
            reduced = reduced / norm

       
        u_pred, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            test_data=reduced.T,        
            cntr_trained=self.fcm_centers, 
            m=self.fcm_fuzziness,
            error=0.005,
            maxiter=300,
        )


        memberships = u_pred[:, 0].astype(np.float64)


        memberships = np.clip(memberships, 0, 1)
        memberships /= memberships.sum()

        return memberships

    def retrieve_documents(
        self,
        query_embedding: np.ndarray,
        n_results: Optional[int] = None,
        filter_newsgroup: Optional[str] = None,
    ) -> list[dict]:
       
        k = n_results or self.n_results

        where_filter = None
        if filter_newsgroup:
            where_filter = {"newsgroup": {"$eq": filter_newsgroup}}

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

        docs = []
        for i in range(len(results["ids"][0])):
            docs.append({
                "doc_id": results["ids"][0][i],
                "newsgroup": results["metadatas"][0][i].get("newsgroup", "unknown"),
                "text": results["documents"][0][i][:500],  
                "distance": float(results["distances"][0][i]),
                "similarity": float(1 - results["distances"][0][i]),  
            })

        return docs

    def process_query(self, query_text: str) -> dict:
       
        embedding = self.embed_query(query_text)


        memberships = self.predict_cluster_memberships(embedding)
        dominant_cluster = int(np.argmax(memberships))


        retrieved_docs = self.retrieve_documents(embedding)


        result = {
            "retrieved_documents": retrieved_docs,
            "dominant_cluster": dominant_cluster,
            "cluster_memberships": {
                str(i): round(float(m), 4)
                for i, m in enumerate(memberships)
                if m > 0.01  
            },
            "top_newsgroups": list({
                doc["newsgroup"] for doc in retrieved_docs[:3]
            }),
        }

        return embedding, memberships, result