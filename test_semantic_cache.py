import numpy as np
from src.semantic_cache import SemanticCache


def random_embedding(dim=384):
    """Generate normalized random embedding."""
    v = np.random.randn(dim)
    v = v / np.linalg.norm(v)
    return v.astype(np.float32)


def random_memberships(n_clusters=10):
    """Generate valid fuzzy cluster membership distribution."""
    v = np.random.rand(n_clusters)
    v = v / v.sum()
    return v


def test_cache_miss_then_put_then_hit():
    cache = SemanticCache(similarity_threshold=0.85, max_size=100)

    query = "What is artificial intelligence?"
    embedding = random_embedding()
    memberships = random_memberships()

    # First lookup → MISS
    hit = cache.get(embedding, memberships)
    assert hit is None

    # Store result
    result = {"answer": "AI is the simulation of human intelligence."}
    cache.put(query, embedding, memberships, result)

    # Second lookup → HIT
    hit = cache.get(embedding, memberships)
    assert hit is not None
    assert hit.result == result


def test_paraphrase_hit():
    cache = SemanticCache(similarity_threshold=0.80)

    query1 = "What is machine learning?"
    emb1 = random_embedding()
    memberships = random_memberships()

    result = {"answer": "Machine learning is a subset of AI."}
    cache.put(query1, emb1, memberships, result)

    # simulate paraphrase (very similar embedding)
    emb2 = emb1 + np.random.normal(0, 0.01, emb1.shape)
    emb2 = emb2 / np.linalg.norm(emb2)

    hit = cache.get(emb2, memberships)

    assert hit is not None
    assert hit.result == result


def test_cluster_gating():
    cache = SemanticCache()

    emb = random_embedding()

    memberships1 = np.array([0.9, 0.1, 0, 0, 0])
    memberships2 = np.array([0.0, 0.9, 0.1, 0, 0])

    cache.put("query1", emb, memberships1, {"a": 1})

    # query in different cluster
    hit = cache.get(emb, memberships2)

    # likely miss because clusters differ
    assert hit is None or hit.similarity_score < cache.similarity_threshold


def test_lru_eviction():
    cache = SemanticCache(max_size=3)

    memberships = random_memberships()

    for i in range(4):
        emb = random_embedding()
        cache.put(f"query{i}", emb, memberships, {"id": i})

    # cache should not exceed max size
    stats = cache.stats()
    assert stats["total_entries"] <= 3


def test_cache_stats():
    cache = SemanticCache()

    emb = random_embedding()
    memberships = random_memberships()

    cache.get(emb, memberships)  # miss

    cache.put("query", emb, memberships, {"x": 1})

    cache.get(emb, memberships)  # hit

    stats = cache.stats()

    assert stats["hit_count"] == 1
    assert stats["miss_count"] == 1
    assert stats["total_entries"] == 1