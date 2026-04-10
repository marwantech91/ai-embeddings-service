from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from typing import Optional
import os

from .embeddings import EmbeddingStore, get_openai_embedding

app = FastAPI(title="AI Embeddings Service", version="0.1.0")
store = EmbeddingStore()


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    id: Optional[str] = None
    metadata: Optional[dict] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    top_k: int = Field(default=5, ge=1, le=100)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class EmbedResponse(BaseModel):
    id: str
    dimensions: int


class SearchResult(BaseModel):
    id: str
    score: float
    metadata: Optional[dict] = None


@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """Generate and store an embedding for the given text."""
    try:
        embedding = await get_openai_embedding(request.text)
        doc_id = store.add(embedding, doc_id=request.id, metadata=request.metadata)
        return EmbedResponse(id=doc_id, dimensions=len(embedding))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=list[SearchResult])
async def search_similar(request: SearchRequest):
    """Search for similar documents by query text."""
    try:
        query_embedding = await get_openai_embedding(request.query)
        results = store.search(query_embedding, top_k=request.top_k, threshold=request.threshold)
        return [
            SearchResult(id=r["id"], score=r["score"], metadata=r.get("metadata"))
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document from the store."""
    removed = store.remove(doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": doc_id}


@app.get("/stats")
async def get_stats():
    """Get store statistics."""
    return store.stats()


@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Retrieve a document's metadata and dimensions."""
    doc = store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.post("/batch-embed")
async def batch_embed(requests: list[EmbedRequest]):
    """Embed multiple texts in a single request."""
    results = []
    for req in requests:
        try:
            embedding = await get_openai_embedding(req.text)
            doc_id = store.add(embedding, doc_id=req.id, metadata=req.metadata)
            results.append({"id": doc_id, "dimensions": len(embedding), "error": None})
        except Exception as e:
            results.append({"id": req.id, "dimensions": 0, "error": str(e)})
    return {"results": results, "total": len(results)}


@app.get("/health")
async def health_check():
    return {"status": "ok", "documents": store.count()}
