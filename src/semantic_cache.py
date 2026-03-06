import time
import json
import threading
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

import numpy as np
from loguru import logger

@dataclass
class CacheEntry:
    entry_id: int                         
    query_text: str                      
    embedding: np.ndarray                 
    result: Any                            
    cluster_memberships: np.ndarray        
    dominant_cluster: int                 
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        """Serialisable representation (for /cache/stats and persistence)."""
        return {
            "entry_id": self.entry_id,
            "query_text": self.query_text,
            "dominant_cluster": self.dominant_cluster,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            
        }
        
@dataclass
class CacheHit:
  
    matched_query: str
    similarity_score: float
    result: Any
    dominant_cluster: int
    entry_id: int
    
class SemanticCache:


    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_size: int = 10_000,
        cluster_membership_threshold: float = 0.15,
    ):
      
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.cluster_membership_threshold = cluster_membership_threshold


        self._store: dict[int, CacheEntry] = {}


        self._lru: OrderedDict[int, None] = OrderedDict()


        self._cluster_index: dict[int, set] = defaultdict(set)


        self._hit_count = 0
        self._miss_count = 0


        self._lock = threading.RLock()

        self._next_id = 0

        logger.info(
            f"SemanticCache initialized: threshold={similarity_threshold}, "
            f"max_size={max_size}, cluster_membership_threshold={cluster_membership_threshold}"
        )

    def get(
        self,
        query_embedding: np.ndarray,
        cluster_memberships: np.ndarray,
    ) -> Optional[CacheHit]:
       
        with self._lock:

            candidate_clusters = self._get_relevant_clusters(cluster_memberships)

            if not candidate_clusters:

                candidate_ids = set(self._store.keys())
                logger.debug("No candidate clusters found, falling back to full scan")
            else:

                candidate_ids = set()
                for cluster_id in candidate_clusters:
                    candidate_ids.update(self._cluster_index[cluster_id])

            if not candidate_ids:
                self._miss_count += 1
                return None

            best_score = -1.0
            best_entry = None

            for entry_id in candidate_ids:
                entry = self._store.get(entry_id)
                if entry is None:
                    continue  

                score = float(np.dot(query_embedding, entry.embedding))

                if score > best_score:
                    best_score = score
                    best_entry = entry


            if best_score >= self.similarity_threshold and best_entry is not None:

                best_entry.access_count += 1
                best_entry.last_accessed = time.time()
                self._lru.move_to_end(best_entry.entry_id)
                self._hit_count += 1

                logger.debug(
                    f"Cache HIT: '{best_entry.query_text[:50]}' "
                    f"(sim={best_score:.4f}, threshold={self.similarity_threshold})"
                )

                return CacheHit(
                    matched_query=best_entry.query_text,
                    similarity_score=best_score,
                    result=best_entry.result,
                    dominant_cluster=best_entry.dominant_cluster,
                    entry_id=best_entry.entry_id,
                )
            else:
                self._miss_count += 1
                logger.debug(
                    f"Cache MISS: best score={best_score:.4f} < threshold={self.similarity_threshold}"
                    f" (searched {len(candidate_ids)} candidates out of {len(self._store)} total)"
                )
                return None

    def put(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        cluster_memberships: np.ndarray,
        result: Any,
    ) -> CacheEntry:
        
        with self._lock:

            if len(self._store) >= self.max_size:
                self._evict_lru()

            entry_id = self._next_id
            self._next_id += 1

            dominant_cluster = int(np.argmax(cluster_memberships))

            entry = CacheEntry(
                entry_id=entry_id,
                query_text=query_text,
                embedding=query_embedding.copy(),  
                result=result,
                cluster_memberships=cluster_memberships.copy(),
                dominant_cluster=dominant_cluster,
            )


            self._store[entry_id] = entry


            self._lru[entry_id] = None


            relevant_clusters = self._get_relevant_clusters(cluster_memberships)
            for cluster_id in relevant_clusters:
                self._cluster_index[cluster_id].add(entry_id)

            logger.debug(
                f"Cache PUT: '{query_text[:50]}' → entry_id={entry_id}, "
                f"dominant_cluster={dominant_cluster}, "
                f"indexed_in_clusters={relevant_clusters}"
            )

            return entry

    def flush(self) -> None:

        with self._lock:
            self._store.clear()
            self._lru.clear()
            self._cluster_index.clear()
            self._hit_count = 0
            self._miss_count = 0
            self._next_id = 0
            logger.info("Cache flushed. All entries and statistics cleared.")


    def stats(self) -> dict:

        with self._lock:
            total = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total if total > 0 else 0.0

            return {
                "total_entries": len(self._store),
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": round(hit_rate, 4),
                "max_size": self.max_size,
                "similarity_threshold": self.similarity_threshold,
                "cluster_index_summary": {
                    str(cluster_id): len(entry_ids)
                    for cluster_id, entry_ids in self._cluster_index.items()
                    if len(entry_ids) > 0
                },
            }

    def get_all_entries_summary(self) -> list[dict]:

        with self._lock:
            return [entry.to_dict() for entry in self._store.values()]

    

    def _get_relevant_clusters(self, cluster_memberships: np.ndarray) -> list[int]:
      
        return [
            int(i)
            for i, membership in enumerate(cluster_memberships)
            if membership >= self.cluster_membership_threshold
        ]

    def _evict_lru(self) -> None:
       
        if not self._lru:
            return

        evicted_id, _ = self._lru.popitem(last=False)

        evicted_entry = self._store.pop(evicted_id, None)

        if evicted_entry is not None:
            relevant_clusters = self._get_relevant_clusters(evicted_entry.cluster_memberships)
            for cluster_id in relevant_clusters:
                self._cluster_index[cluster_id].discard(evicted_id)

            logger.debug(
                f"Evicted LRU entry: id={evicted_id}, "
                f"query='{evicted_entry.query_text[:40]}', "
                f"access_count={evicted_entry.access_count}"
            )

    

    def explore_threshold_behavior(
        self,
        query_embedding: np.ndarray,
        cluster_memberships: np.ndarray,
        thresholds: list[float] = None,
    ) -> dict:
        
        if thresholds is None:
            thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]

        candidate_clusters = self._get_relevant_clusters(cluster_memberships)
        candidate_ids = set()
        for cluster_id in candidate_clusters:
            candidate_ids.update(self._cluster_index[cluster_id])


        similarities = []
        for entry_id in candidate_ids:
            entry = self._store.get(entry_id)
            if entry is not None:
                score = float(np.dot(query_embedding, entry.embedding))
                similarities.append((score, entry.query_text, entry_id))

        similarities.sort(reverse=True)

        result = {}
        for threshold in thresholds:
            hits = [(s, q, i) for s, q, i in similarities if s >= threshold]
            result[threshold] = {
                "would_hit": len(hits) > 0,
                "num_candidates": len(hits),
                "best_match": similarities[0][1][:80] if similarities else None,
                "best_score": similarities[0][0] if similarities else None,
            }

        return result
