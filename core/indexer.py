import uuid
from qdrant_client.models import PointStruct

from core import EmbeddingGenerator
from stores.vector_store import VectorStore


class Indexer:
    def __init__(self, vector_store:VectorStore, embedding_generator:EmbeddingGenerator):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator

    def index(self, text: str, metadata: dict):
        embedding = self.embedding_generator.embed(text)

        point_id = metadata.get("id", str(uuid.uuid4()))

        payload = {
            "text": text,
            **metadata
        }

        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )

        self.vector_store.insert([point])

    def index_batch(self, items: list[dict]):
        points = []

        for item in items:
            text = item["text"]
            metadata = item.get("metadata", {})

            embedding = self.embedding_generator.embed(text)

            point_id = metadata.get("id", str(uuid.uuid4()))

            payload = {
                "text": text,
                **metadata
            }

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )

            points.append(point)

        self.vector_store.insert(points)