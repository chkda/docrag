from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    def __init__(self, model_name: str, dim: int):
        self.model_name = model_name
        self.dim = dim
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str):
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()