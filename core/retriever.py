from core.embedding_generator import EmbeddingGenerator
from stores.vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore, embedding_generator: EmbeddingGenerator):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator

    def search(self, query: str, car: str) :
        query_embedding = self.embedding_generator.embed(query)
        results = self.vector_store.retrieve(query_vector=query_embedding, limit=3, car=car)
        return results