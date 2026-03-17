"""Tests for the EmbeddingStore class and cosine_similarity function."""

import numpy as np
import pytest

from src.embeddings import EmbeddingStore, cosine_similarity


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        b = -a
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, b) == 0.0
        assert cosine_similarity(b, a) == 0.0

    def test_both_zero_vectors(self):
        z = np.array([0.0, 0.0])
        assert cosine_similarity(z, z) == 0.0

    def test_known_value(self):
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 1.0])
        expected = 1.0 / np.sqrt(2)
        assert cosine_similarity(a, b) == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# EmbeddingStore — add
# ---------------------------------------------------------------------------

class TestEmbeddingStoreAdd:
    def test_add_returns_given_id(self):
        store = EmbeddingStore()
        doc_id = store.add([1.0, 2.0, 3.0], doc_id="abc")
        assert doc_id == "abc"

    def test_add_generates_id_when_none(self):
        store = EmbeddingStore()
        doc_id = store.add([1.0, 2.0])
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    def test_add_increments_count(self):
        store = EmbeddingStore()
        assert store.count() == 0
        store.add([1.0, 2.0], doc_id="a")
        assert store.count() == 1
        store.add([3.0, 4.0], doc_id="b")
        assert store.count() == 2

    def test_add_stores_metadata(self):
        store = EmbeddingStore()
        store.add([1.0], doc_id="m1", metadata={"source": "test"})
        doc = store.get("m1")
        assert doc is not None
        assert doc["metadata"] == {"source": "test"}

    def test_add_without_metadata(self):
        store = EmbeddingStore()
        store.add([1.0], doc_id="m2")
        doc = store.get("m2")
        assert doc is not None
        assert doc["metadata"] is None

    def test_add_overwrites_existing_id(self):
        store = EmbeddingStore()
        store.add([1.0, 2.0], doc_id="dup")
        store.add([9.0, 8.0], doc_id="dup")
        assert store.count() == 1
        doc = store.get("dup")
        assert doc["dimensions"] == 2


# ---------------------------------------------------------------------------
# EmbeddingStore — remove
# ---------------------------------------------------------------------------

class TestEmbeddingStoreRemove:
    def test_remove_existing(self):
        store = EmbeddingStore()
        store.add([1.0], doc_id="r1")
        assert store.remove("r1") is True
        assert store.count() == 0

    def test_remove_nonexistent(self):
        store = EmbeddingStore()
        assert store.remove("nope") is False

    def test_remove_cleans_metadata(self):
        store = EmbeddingStore()
        store.add([1.0], doc_id="r2", metadata={"k": "v"})
        store.remove("r2")
        assert store.get("r2") is None


# ---------------------------------------------------------------------------
# EmbeddingStore — search
# ---------------------------------------------------------------------------

class TestEmbeddingStoreSearch:
    def test_search_empty_store(self):
        store = EmbeddingStore()
        results = store.search([1.0, 2.0])
        assert results == []

    def test_search_returns_most_similar_first(self):
        store = EmbeddingStore()
        store.add([1.0, 0.0], doc_id="a")
        store.add([0.0, 1.0], doc_id="b")
        store.add([1.0, 1.0], doc_id="c")

        # Query close to [1, 0] — "a" should rank first, then "c", then "b"
        results = store.search([1.0, 0.1], top_k=3)
        ids = [r["id"] for r in results]
        assert ids[0] == "a"
        assert "score" in results[0]

    def test_search_respects_top_k(self):
        store = EmbeddingStore()
        for i in range(10):
            store.add([float(i), 1.0], doc_id=f"d{i}")
        results = store.search([5.0, 1.0], top_k=3)
        assert len(results) == 3

    def test_search_respects_threshold(self):
        store = EmbeddingStore()
        store.add([1.0, 0.0], doc_id="x")
        store.add([0.0, 1.0], doc_id="y")

        # High threshold should exclude the orthogonal vector
        results = store.search([1.0, 0.0], top_k=10, threshold=0.9)
        ids = [r["id"] for r in results]
        assert "x" in ids
        assert "y" not in ids

    def test_search_includes_metadata(self):
        store = EmbeddingStore()
        store.add([1.0, 0.0], doc_id="meta_doc", metadata={"tag": "hello"})
        results = store.search([1.0, 0.0], top_k=1)
        assert results[0]["metadata"] == {"tag": "hello"}

    def test_search_score_rounded(self):
        store = EmbeddingStore()
        store.add([1.0, 0.0], doc_id="s1")
        results = store.search([1.0, 0.0], top_k=1)
        score_str = str(results[0]["score"])
        # At most 4 decimal places
        if "." in score_str:
            assert len(score_str.split(".")[1]) <= 4


# ---------------------------------------------------------------------------
# EmbeddingStore — get / stats
# ---------------------------------------------------------------------------

class TestEmbeddingStoreGetAndStats:
    def test_get_existing(self):
        store = EmbeddingStore()
        store.add([1.0, 2.0, 3.0], doc_id="g1")
        doc = store.get("g1")
        assert doc is not None
        assert doc["id"] == "g1"
        assert doc["dimensions"] == 3

    def test_get_nonexistent(self):
        store = EmbeddingStore()
        assert store.get("missing") is None

    def test_stats_empty(self):
        store = EmbeddingStore()
        s = store.stats()
        assert s["total_documents"] == 0
        assert s["dimensions"] == []

    def test_stats_populated(self):
        store = EmbeddingStore()
        store.add([1.0, 2.0, 3.0], doc_id="s1")
        store.add([4.0, 5.0, 6.0], doc_id="s2")
        s = store.stats()
        assert s["total_documents"] == 2
        assert 3 in s["dimensions"]
