from src.query_engine import QueryEngine

engine = QueryEngine(
    chroma_persist_dir="data/chroma_db",
    clustering_artifacts_dir="data",
)

query = "How do graphics cards work?"

embedding, memberships, result = engine.process_query(query)

print("Embedding shape:", embedding.shape)
print("Cluster memberships:", memberships)
print("Result:", result)