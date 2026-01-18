from qdrant_client import QdrantClient
from qdrant_client.http.models import QueryResponse
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    PointStruct,
    Filter,
)


class QdrantStore:
    def __init__(
        self,
        collection_name: str,
        url: str = "localhost",
        port: int = 6333,
        vector_size: int = 256,
        distance: Distance = Distance.COSINE
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self.client = QdrantClient(url=url, port=port)

        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                ),
            )

    def add(self, points: list[PointStruct]):
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def update(self, points: list[PointStruct]):
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def delete(self, point_ids: list):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=point_ids,
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        query_filter: Filter = None,
        score_threshold: float = None,
    ):
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )
