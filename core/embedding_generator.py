from functools import lru_cache

from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    def __init__(self, model_name: str, dim: int):
        self.model_name = model_name
        self.dim = dim
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str):
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

@lru_cache
def get_embedding_generator(model_name: str, dim: int) -> EmbeddingGenerator:
    return EmbeddingGenerator(model_name, dim)