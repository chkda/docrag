# DocRAG

A document retrieval-augmented generation (RAG) system for indexing and searching PDF documents.

## Architecture

```
docrag/
├── core/                    # Core processing modules
├── stores/                  # Storage backends
├── server.py               # FastAPI server
├── app.py                  # Streamlit frontend
└── document_ingestion.py   # Document indexing script
```

## Modules

### Core (`core/`)

#### Extractor (`extractor.py`)
Extracts text from PDF documents using PyMuPDF. Supports two modes:
- `page`: Extract text page by page
- `section`: Extract text section by section using PDF table of contents

#### EmbeddingGenerator (`embedding_generator.py`)
Generates vector embeddings from text using sentence-transformers models.

#### Indexer (`indexer.py`)
Takes text and metadata, generates embeddings, and stores them in the vector database.

#### Ingestor (`ingestor.py`)
Orchestrates the document ingestion pipeline. Combines Extractor and Indexer to process documents end-to-end.

#### Retriever (`retriever.py`)
Handles semantic search. Takes a query, generates its embedding, and retrieves relevant documents from the vector store.

#### LLM (`llm.py`)
Wrapper for language model inference using HuggingFace transformers.

### Stores (`stores/`)

#### VectorStore (`vector_store.py`)
Abstract interface for vector database operations. Currently supports Qdrant.

#### QdrantStore (`qdrant/qdrant_store.py`)
Qdrant-specific implementation for vector storage and retrieval. Handles:
- Collection creation with HNSW index
- CRUD operations (add, update, delete, search)
- Filtered search

## Setup

1. Start Qdrant:
```bash
docker compose up -d
```

2. Index documents:
```bash
python document_ingestion.py
```

3. Run the API server:
```bash
python server.py
```

4. Run the frontend:
```bash
streamlit run app.py
```

## API

### POST /search
Search for relevant documents.

**Request:**
```json
{
  "query": "your search query"
}
```

**Response:**
```json
{
  "answer": "...",
  "citations": [
    {
      "document_name": "doc.pdf",
      "page_number": 1,
      "text": "..."
    }
  ]
}
```