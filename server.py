from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from core import Retriever
from core.generator import get_generator
from core.embedding_generator import get_embedding_generator
from stores.vector_store import get_vector_store


QDRANT_URL = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

cars = {
    "astor": {"mg astor", "astor", "mgastor"},
    "tiago": {"tata tiago", "tiago", "tatatiago"},
}

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

generator = get_generator("cpu")


class SearchRequest(BaseModel):
    query: str

class Citation(BaseModel):
    document_name: str
    page_number: int
    text: str

class SearchResponse(BaseModel):
    answer: str
    citations: Optional[List[Citation]] = None


@app.post("/search")
def search(request: SearchRequest):
    request.query = request.query.lower()
    car = ""
    for car_name, possible_names in cars.items():
        for name in possible_names:
            if name in request.query:
                car = car_name
                break

    if car == "":
        return SearchResponse(answer="Manual is not available for this car/model.", citations=[])
    results = retriever.search(request.query, car)
    citations = []
    chunks = []

    for result in results.points:
        result = result.payload
        citation = Citation(text=result["text"], document_name=result["document_name"], page_number=result["page_number"] if "page_number" in result else result["start_page"])
        citations.append(citation)
        chunks.append(citation.model_dump())

    answer = generator.generate_answer(query=request.query, chunks=chunks, car_model=car)

    return SearchResponse(answer=answer, citations=citations)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)