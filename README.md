# AI Embeddings Service

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi)
![AI](https://img.shields.io/badge/AI-Embeddings-purple?style=flat-square)

A lightweight FastAPI service for generating, storing, and searching text embeddings. Uses OpenAI's embedding models with an in-memory vector store and cosine similarity search.

## Features

- Generate embeddings via OpenAI API
- In-memory vector store with cosine similarity search
- Top-K search with score thresholds
- Document metadata support
- REST API with automatic OpenAPI docs

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/embed` | Generate and store an embedding |
| POST | `/search` | Search for similar documents |
| DELETE | `/documents/{id}` | Remove a document |
| GET | `/stats` | Store statistics |
| GET | `/health` | Health check |

## Quick Start

```bash
export OPENAI_API_KEY=sk-...
pip install -r requirements.txt
uvicorn src.main:app --reload
```

## Usage

```bash
# Embed a document
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "FastAPI is a modern web framework", "id": "doc-1"}'

# Search similar
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "web frameworks for Python", "top_k": 3}'
```

## License

MIT
