from core.embedding_generator import get_embedding_generator
from core import Ingestor
from stores.vector_store import get_vector_store


DOCUMENT_PATHS = [
    ("tiago","documents/APP-TIAGO-FINAL-OMSB.pdf"),
    ("astor","documents/Astor-Manual.pdf"),
]

QDRANT_URL = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def ingest_documents():
    vector_store = get_vector_store(
        db_type="qdrant",
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
        port=QDRANT_PORT,
        vector_size=EMBEDDING_DIM
    )

    embedding_generator = get_embedding_generator(
        model_name=EMBEDDING_MODEL,
        dim=EMBEDDING_DIM
    )

    ingestor = Ingestor(
        vector_store=vector_store,
        embedding_generator=embedding_generator
    )

    for document in DOCUMENT_PATHS:
        car = document[0]
        document_path = document[1]
        ingestor.index(car=car,document_path=document_path, mode="section")


if __name__ == "__main__":
    ingest_documents()