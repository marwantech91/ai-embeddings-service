import numpy as np
from typing import Optional
from openai import AsyncOpenAI
import uuid
import os


client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


async def get_openai_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Get embedding from OpenAI API."""
    response = await client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class EmbeddingStore:
    """In-memory vector store for embeddings."""

    def __init__(self):
        self._embeddings: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict] = {}

    def add(
        self,
        embedding: list[float],
        doc_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Add an embedding to the store."""
        doc_id = doc_id or str(uuid.uuid4())
        self._embeddings[doc_id] = np.array(embedding, dtype=np.float32)
        if metadata:
            self._metadata[doc_id] = metadata
        return doc_id

    def remove(self, doc_id: str) -> bool:
        """Remove a document from the store."""
        if doc_id not in self._embeddings:
            return False
        del self._embeddings[doc_id]
        self._metadata.pop(doc_id, None)
        return True

    def search(
        self,
        query: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[dict]:
        """Search for similar embeddings."""
        if not self._embeddings:
            return []

        query_vec = np.array(query, dtype=np.float32)
        scores: list[tuple[str, float]] = []

        for doc_id, embedding in self._embeddings.items():
            score = cosine_similarity(query_vec, embedding)
            if score >= threshold:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in scores[:top_k]:
            result = {"id": doc_id, "score": round(score, 4)}
            if doc_id in self._metadata:
                result["metadata"] = self._metadata[doc_id]
            results.append(result)

        return results

    def get(self, doc_id: str) -> Optional[dict]:
        """Get a document by ID."""
        if doc_id not in self._embeddings:
            return None
        return {
            "id": doc_id,
            "dimensions": len(self._embeddings[doc_id]),
            "metadata": self._metadata.get(doc_id),
        }

    def count(self) -> int:
        return len(self._embeddings)

    def stats(self) -> dict:
        dimensions = set()
        for emb in self._embeddings.values():
            dimensions.add(len(emb))
        return {
            "total_documents": self.count(),
            "dimensions": list(dimensions),
        }
