"""Tests for the FastAPI endpoints, with OpenAI embedding calls mocked."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app, store


FAKE_EMBEDDING = [0.1] * 128  # deterministic fake embedding


@pytest.fixture(autouse=True)
def _clear_store():
    """Reset the in-memory store between tests."""
    store._embeddings.clear()
    store._metadata.clear()
    yield
    store._embeddings.clear()
    store._metadata.clear()


@pytest.fixture
def client():
    return TestClient(app)


def _mock_openai_embedding():
    """Return a patch that replaces get_openai_embedding with a coroutine
    that returns FAKE_EMBEDDING."""
    return patch(
        "src.main.get_openai_embedding",
        new_callable=AsyncMock,
        return_value=FAKE_EMBEDDING,
    )


# ---------------------------------------------------------------------------
# POST /embed
# ---------------------------------------------------------------------------

class TestEmbedEndpoint:
    def test_embed_success(self, client):
        with _mock_openai_embedding():
            resp = client.post("/embed", json={"text": "hello world"})
        assert resp.status_code == 200
        body = resp.json()
        assert "id" in body
        assert body["dimensions"] == len(FAKE_EMBEDDING)

    def test_embed_with_custom_id(self, client):
        with _mock_openai_embedding():
            resp = client.post(
                "/embed",
                json={"text": "test", "id": "custom-1"},
            )
        assert resp.status_code == 200
        assert resp.json()["id"] == "custom-1"

    def test_embed_with_metadata(self, client):
        with _mock_openai_embedding():
            resp = client.post(
                "/embed",
                json={"text": "test", "id": "meta-1", "metadata": {"k": "v"}},
            )
        assert resp.status_code == 200
        assert resp.json()["id"] == "meta-1"
        # Verify metadata was actually stored
        doc = store.get("meta-1")
        assert doc["metadata"] == {"k": "v"}

    def test_embed_empty_text_rejected(self, client):
        resp = client.post("/embed", json={"text": ""})
        assert resp.status_code == 422

    def test_embed_missing_text_rejected(self, client):
        resp = client.post("/embed", json={})
        assert resp.status_code == 422

    def test_embed_openai_failure(self, client):
        with patch(
            "src.main.get_openai_embedding",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API down"),
        ):
            resp = client.post("/embed", json={"text": "fail"})
        assert resp.status_code == 500
        assert "API down" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# POST /search
# ---------------------------------------------------------------------------

class TestSearchEndpoint:
    def _seed_documents(self):
        """Insert a few documents directly into the store."""
        store.add(FAKE_EMBEDDING, doc_id="doc-1", metadata={"label": "a"})
        # Slightly different embedding for variety
        other = [x + 0.01 for x in FAKE_EMBEDDING]
        store.add(other, doc_id="doc-2", metadata={"label": "b"})

    def test_search_success(self, client):
        self._seed_documents()
        with _mock_openai_embedding():
            resp = client.post("/search", json={"query": "hello"})
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) >= 1
        assert "id" in results[0]
        assert "score" in results[0]

    def test_search_top_k(self, client):
        self._seed_documents()
        with _mock_openai_embedding():
            resp = client.post(
                "/search", json={"query": "hello", "top_k": 1}
            )
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_search_with_threshold(self, client):
        self._seed_documents()
        with _mock_openai_embedding():
            resp = client.post(
                "/search",
                json={"query": "hello", "threshold": 0.9999},
            )
        assert resp.status_code == 200
        results = resp.json()
        # Only the exact-match embedding should survive the high threshold
        ids = [r["id"] for r in results]
        assert "doc-1" in ids

    def test_search_empty_store(self, client):
        with _mock_openai_embedding():
            resp = client.post("/search", json={"query": "anything"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_search_empty_query_rejected(self, client):
        resp = client.post("/search", json={"query": ""})
        assert resp.status_code == 422

    def test_search_openai_failure(self, client):
        with patch(
            "src.main.get_openai_embedding",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API down"),
        ):
            resp = client.post("/search", json={"query": "fail"})
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# DELETE /documents/{doc_id}
# ---------------------------------------------------------------------------

class TestDeleteEndpoint:
    def test_delete_existing(self, client):
        store.add(FAKE_EMBEDDING, doc_id="del-1")
        resp = client.delete("/documents/del-1")
        assert resp.status_code == 200
        assert resp.json() == {"deleted": "del-1"}
        assert store.count() == 0

    def test_delete_nonexistent(self, client):
        resp = client.delete("/documents/no-such-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /stats
# ---------------------------------------------------------------------------

class TestStatsEndpoint:
    def test_stats_empty(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_documents"] == 0

    def test_stats_after_inserts(self, client):
        store.add(FAKE_EMBEDDING, doc_id="s1")
        store.add(FAKE_EMBEDDING, doc_id="s2")
        resp = client.get("/stats")
        assert resp.status_code == 200
        assert resp.json()["total_documents"] == 2


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["documents"] == 0

    def test_health_reflects_count(self, client):
        store.add(FAKE_EMBEDDING, doc_id="h1")
        resp = client.get("/health")
        assert resp.json()["documents"] == 1
