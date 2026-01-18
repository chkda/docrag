from functools import lru_cache

from stores.qdrant.qdrant_store import QdrantStore


class VectorStore:
    def __init__(self, db_type: str, collection_name: str, **connection_config):
        self.db_type = db_type
        self.collection_name = collection_name

        if db_type == "qdrant":
            self.store = QdrantStore(
                collection_name=collection_name,
                **connection_config
            )
        else:
            raise ValueError(f"Unsupported db_type: {db_type}")

    def insert(self, points):
        self.store.add(points)

    def retrieve(self, query_vector, limit: int = 10, query_filter=None, score_threshold: float = None):
        return self.store.search(
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )

@lru_cache
def get_vector_store(db_type: str, collection_name: str, **connection_config)->VectorStore:
    return VectorStore(db_type, collection_name, **connection_config)