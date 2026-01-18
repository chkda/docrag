from core.embedding_generator import EmbeddingGenerator
from core.extractor import Extractor
from core.indexer import Indexer
from stores.vector_store import VectorStore


class Ingestor:
    def __init__(
        self,
        vector_store:VectorStore,
        embedding_generator:EmbeddingGenerator,
    ):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator

        self.extractor = Extractor()
        self.indexer = Indexer(
            vector_store=vector_store,
            embedding_generator=embedding_generator
        )

    def index(self, document_path: str, mode: str = "page"):
        self.extractor.set_doc(document_path=document_path, mode=mode)
        document_name = document_path.split("/")[-1]

        for chunk in self.extractor.extract():
            text = chunk["text"]

            metadata = {
                "document_name": document_name,
                **chunk
            }
            del metadata["text"]

            self.indexer.index(text=text, metadata=metadata)

        self.extractor.close()