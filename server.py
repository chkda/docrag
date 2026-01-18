from typing import List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from core import Retriever
from core.embedding_generator import get_embedding_generator
from stores.vector_store import get_vector_store


QDRANT_URL = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

app = FastAPI()

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

retriever = Retriever(
    vector_store=vector_store,
    embedding_generator=embedding_generator
)


class SearchRequest(BaseModel):
    query: str

class Citation(BaseModel):
    document_name: str
    page_number: int
    text: str

class SearchResponse(BaseModel):
    answer: str
    citations: List[Citation]


@app.post("/search")
def search(request: SearchRequest):
    results = retriever.search(request.query)
    citations = []
    for result in results.points:
        result = result.payload
        print(result)
        citations.append(Citation(text=result["text"], document_name=result["document_name"], page_number=result["page_number"] if "page_number" in result else result["start_page"]))

    return SearchResponse(answer="This is the answer", citations=citations)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)