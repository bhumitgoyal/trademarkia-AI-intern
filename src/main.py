import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

from .query_engine import QueryEngine
from .semantic_cache import SemanticCache

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
CLUSTERING_ARTIFACTS_DIR = os.getenv("CLUSTERING_ARTIFACTS_DIR", "data")
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.85"))
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "10000"))

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural language query to search the newsgroups corpus",
        examples=["What are the arguments about gun control?"],
    )

    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override the cache similarity threshold for this request",
    )


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    result: dict
    dominant_cluster: int


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    max_size: int
    similarity_threshold: float


class CacheFlushResponse(BaseModel):
    message: str
    entries_cleared: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    logger.info("Starting up NewsGroups Semantic Search API...")


    required_paths = [
        CHROMA_PERSIST_DIR,
        os.path.join(CLUSTERING_ARTIFACTS_DIR, "pca_model.joblib"),
        os.path.join(CLUSTERING_ARTIFACTS_DIR, "fcm_centers.npy"),
    ]
    for path in required_paths:
        if not os.path.exists(path):
            logger.error(
                f"Required artifact not found: {path}\n"
                "Run the setup pipeline first:\n"
                "  python scripts/run_pipeline.py --dataset-path <path_to_20newsgroups>"
            )
            raise RuntimeError(f"Missing required artifact: {path}")


    logger.info("Initializing QueryEngine...")
    app.state.query_engine = QueryEngine(
        chroma_persist_dir=CHROMA_PERSIST_DIR,
        clustering_artifacts_dir=CLUSTERING_ARTIFACTS_DIR,
    )


    logger.info(f"Initializing SemanticCache (threshold={CACHE_SIMILARITY_THRESHOLD})...")
    app.state.cache = SemanticCache(
        similarity_threshold=CACHE_SIMILARITY_THRESHOLD,
        max_size=CACHE_MAX_SIZE,
    )

    logger.info("API startup complete. Ready to serve requests.")

    yield 


    logger.info("Shutting down API.")


app = FastAPI(
    title="20 Newsgroups Semantic Search",
    description=(
        "Semantic search over the 20 Newsgroups corpus with fuzzy clustering "
        "and a custom semantic cache that recognises paraphrased queries."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Semantic search with cache",
    description=(
        "Embeds the query, checks the semantic cache for a similar cached query, "
        "and returns either a cached result (with cache_hit=true) or freshly "
        "computed results from the vector database."
    ),
)
async def post_query(request: Request, body: QueryRequest) -> QueryResponse:
    
    engine: QueryEngine = request.app.state.query_engine
    cache: SemanticCache = request.app.state.cache


    original_threshold = cache.similarity_threshold
    if body.similarity_threshold is not None:
        cache.similarity_threshold = body.similarity_threshold

    try:

        embedding, memberships, result_data = engine.process_query(body.query)


        cache_hit = cache.get(
            query_embedding=embedding,
            cluster_memberships=memberships,
        )

        if cache_hit is not None:

            logger.info(
                f"Cache HIT for query: '{body.query[:60]}' "
                f"(matched: '{cache_hit.matched_query[:60]}', "
                f"sim={cache_hit.similarity_score:.4f})"
            )
            return QueryResponse(
                query=body.query,
                cache_hit=True,
                matched_query=cache_hit.matched_query,
                similarity_score=round(cache_hit.similarity_score, 4),
                result=cache_hit.result,
                dominant_cluster=cache_hit.dominant_cluster,
            )

        else:

            logger.info(f"Cache MISS for query: '{body.query[:60]}' — querying vector DB")

            cache.put(
                query_text=body.query,
                query_embedding=embedding,
                cluster_memberships=memberships,
                result=result_data,
            )

            return QueryResponse(
                query=body.query,
                cache_hit=False,
                matched_query=None,
                similarity_score=None,
                result=result_data,
                dominant_cluster=result_data["dominant_cluster"],
            )

    except Exception as e:
        logger.exception(f"Error processing query '{body.query[:60]}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}",
        )
    finally:

        cache.similarity_threshold = original_threshold


@app.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    summary="Cache statistics",
    description="Returns the current state of the semantic cache including hit/miss counts.",
)
async def get_cache_stats(request: Request) -> CacheStatsResponse:

    cache: SemanticCache = request.app.state.cache
    stats = cache.stats()

    return CacheStatsResponse(
        total_entries=stats["total_entries"],
        hit_count=stats["hit_count"],
        miss_count=stats["miss_count"],
        hit_rate=stats["hit_rate"],
        max_size=stats["max_size"],
        similarity_threshold=stats["similarity_threshold"],
    )


@app.delete(
    "/cache",
    response_model=CacheFlushResponse,
    summary="Flush cache",
    description="Clears all cache entries and resets hit/miss statistics.",
)
async def delete_cache(request: Request) -> CacheFlushResponse:

    cache: SemanticCache = request.app.state.cache
    entries_before = cache.stats()["total_entries"]

    cache.flush()

    logger.info(f"Cache flushed via API. Cleared {entries_before} entries.")

    return CacheFlushResponse(
        message=f"Cache cleared. {entries_before} entries removed.",
        entries_cleared=entries_before,
    )


@app.get(
    "/cache/entries",
    summary="List cache entries",
    description="Returns a summary of all currently cached queries (without embeddings).",
)
async def get_cache_entries(request: Request) -> dict:
    """List all cached entries for debugging/inspection."""
    cache: SemanticCache = request.app.state.cache
    entries = cache.get_all_entries_summary()
    return {
        "total": len(entries),
        "entries": entries,
    }


@app.get(
    "/health",
    summary="Health check",
    description="Simple liveness probe — returns 200 if the service is running.",
)
async def health_check(request: Request) -> dict:

    try:
        engine: QueryEngine = request.app.state.query_engine
        cache: SemanticCache = request.app.state.cache
        return {
            "status": "healthy",
            "corpus_size": engine.collection.count(),
            "cache_entries": cache.stats()["total_entries"],
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)},
        )
