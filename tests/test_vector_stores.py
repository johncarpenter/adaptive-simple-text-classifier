"""Tests for pluggable vector store interface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from adaptive_classifier.vector_stores import FaissVectorStore, VectorStore
from adaptive_classifier.index import ClassificationIndex
from adaptive_classifier.embeddings import CallableEmbedder
from adaptive_classifier.types import ClassificationSource


# ---------------------------------------------------------------------------
# Mock in-memory vector store (simulates a DB-backed store)
# ---------------------------------------------------------------------------

class InMemoryVectorStore:
    """A simple in-memory vector store that simulates a DB-backed store.

    Vectors persist in memory (like a database connection would), so
    file_suffix is None — no file persistence needed.
    """

    def __init__(self, dimension: int):
        self._dimension = dimension
        self._vectors: np.ndarray | None = None

    @property
    def size(self) -> int:
        return 0 if self._vectors is None else self._vectors.shape[0]

    @property
    def file_suffix(self) -> str | None:
        return None

    def add(self, vectors: np.ndarray) -> None:
        if self._vectors is None:
            self._vectors = vectors.copy()
        else:
            self._vectors = np.vstack([self._vectors, vectors])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.size == 0:
            n = queries.shape[0]
            return np.zeros((n, 1), dtype=np.float32), -np.ones((n, 1), dtype=np.int64)

        # Brute-force cosine similarity (vectors assumed normalized)
        scores = queries @ self._vectors.T  # (N, M)
        actual_k = min(k, self.size)
        indices = np.argsort(-scores, axis=1)[:, :actual_k]
        distances = np.take_along_axis(scores, indices, axis=1)
        return distances.astype(np.float32), indices.astype(np.int64)

    def reset(self) -> None:
        self._vectors = None

    def save(self, path: Path) -> None:
        pass  # DB-backed — no-op

    def load(self, path: Path) -> None:
        pass  # DB-backed — no-op


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIMENSION = 8


def _random_embedder() -> CallableEmbedder:
    """Embedder that returns deterministic normalized vectors based on text hash."""
    def _embed(texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            rng = np.random.RandomState(hash(t) % 2**31)
            v = rng.randn(DIMENSION).astype(np.float32)
            v /= np.linalg.norm(v)
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)

    return CallableEmbedder(fn=_embed, dimension=DIMENSION)


# ---------------------------------------------------------------------------
# Tests: VectorStore protocol compliance
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    def test_faiss_is_vector_store(self):
        store = FaissVectorStore(DIMENSION)
        assert isinstance(store, VectorStore)

    def test_in_memory_is_vector_store(self):
        store = InMemoryVectorStore(DIMENSION)
        assert isinstance(store, VectorStore)


# ---------------------------------------------------------------------------
# Tests: FaissVectorStore
# ---------------------------------------------------------------------------

class TestFaissVectorStore:
    def test_add_and_size(self):
        store = FaissVectorStore(DIMENSION)
        assert store.size == 0
        vecs = np.random.randn(5, DIMENSION).astype(np.float32)
        store.add(vecs)
        assert store.size == 5

    def test_search(self):
        store = FaissVectorStore(DIMENSION)
        vecs = np.eye(DIMENSION, dtype=np.float32)
        store.add(vecs)

        query = vecs[:1]  # search for first vector
        distances, indices = store.search(query, k=1)
        assert indices[0][0] == 0
        assert distances[0][0] == pytest.approx(1.0, abs=1e-5)

    def test_reset(self):
        store = FaissVectorStore(DIMENSION)
        store.add(np.random.randn(3, DIMENSION).astype(np.float32))
        assert store.size == 3
        store.reset()
        assert store.size == 0

    def test_save_load(self, tmp_path):
        store = FaissVectorStore(DIMENSION)
        vecs = np.random.randn(4, DIMENSION).astype(np.float32)
        store.add(vecs)

        path = tmp_path / "test.faiss"
        store.save(path)

        store2 = FaissVectorStore(DIMENSION)
        store2.load(path)
        assert store2.size == 4

    def test_file_suffix(self):
        store = FaissVectorStore(DIMENSION)
        assert store.file_suffix == ".faiss"


# ---------------------------------------------------------------------------
# Tests: InMemoryVectorStore (DB-backed simulation)
# ---------------------------------------------------------------------------

class TestInMemoryVectorStore:
    def test_add_and_size(self):
        store = InMemoryVectorStore(DIMENSION)
        assert store.size == 0
        vecs = np.random.randn(5, DIMENSION).astype(np.float32)
        store.add(vecs)
        assert store.size == 5

    def test_search(self):
        store = InMemoryVectorStore(DIMENSION)
        vecs = np.eye(DIMENSION, dtype=np.float32)
        store.add(vecs)

        query = vecs[:1]
        distances, indices = store.search(query, k=1)
        assert indices[0][0] == 0
        assert distances[0][0] == pytest.approx(1.0, abs=1e-5)

    def test_reset(self):
        store = InMemoryVectorStore(DIMENSION)
        store.add(np.random.randn(3, DIMENSION).astype(np.float32))
        store.reset()
        assert store.size == 0

    def test_file_suffix_is_none(self):
        store = InMemoryVectorStore(DIMENSION)
        assert store.file_suffix is None


# ---------------------------------------------------------------------------
# Tests: ClassificationIndex with DB-backed store
# ---------------------------------------------------------------------------

class TestClassificationIndexWithDbStore:
    def test_add_and_search(self):
        embedder = _random_embedder()
        store = InMemoryVectorStore(DIMENSION)
        index = ClassificationIndex(embedder=embedder, vector_store=store)

        index.add(
            texts=["cheeseburger", "pepperoni pizza"],
            category_paths=["Food > Burgers", "Food > Pizza"],
        )
        assert index.size == 2

        results = index.search(["cheeseburger"], k=1)
        assert len(results) == 1
        assert len(results[0]) == 1
        assert results[0][0].category_path == "Food > Burgers"

    def test_save_metadata_only(self, tmp_path):
        """DB-backed store should save only .meta.json, no vector file."""
        embedder = _random_embedder()
        store = InMemoryVectorStore(DIMENSION)
        index_path = tmp_path / "test_index"
        index = ClassificationIndex(
            embedder=embedder,
            index_path=index_path,
            vector_store=store,
        )

        index.add(
            texts=["burger"],
            category_paths=["Food > Burgers"],
        )
        index.save()

        meta_path = index_path.with_suffix(".meta.json")
        assert meta_path.exists()
        # No vector file should exist
        assert not index_path.with_suffix(".faiss").exists()
        assert not index_path.with_suffix(".vectors").exists()

        # Metadata should be valid JSON
        with open(meta_path) as f:
            metadata = json.load(f)
        assert len(metadata) == 1
        assert metadata[0]["category_path"] == "Food > Burgers"

    def test_load_metadata_only(self, tmp_path):
        """DB-backed store should load metadata from .meta.json without needing vector file."""
        embedder = _random_embedder()
        store = InMemoryVectorStore(DIMENSION)
        index_path = tmp_path / "test_index"

        # Pre-populate the store (simulates data already in DB)
        vecs = embedder.embed(["burger"])
        store.add(vecs)

        # Write metadata file manually
        meta_path = index_path.with_suffix(".meta.json")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump([{
                "category_path": "Food > Burgers",
                "original_text": "burger",
                "source": "llm",
            }], f)

        # Creating index should auto-load metadata
        index = ClassificationIndex(
            embedder=embedder,
            index_path=index_path,
            vector_store=store,
        )
        assert index.size == 1
        assert len(index._metadata) == 1

    def test_index_files_exist_db_store(self, tmp_path):
        """DB-backed store: _index_files_exist returns True when .meta.json exists."""
        embedder = _random_embedder()
        store = InMemoryVectorStore(DIMENSION)
        index_path = tmp_path / "test_index"

        index = ClassificationIndex(
            embedder=embedder,
            index_path=index_path,
            vector_store=store,
        )

        # No files yet
        assert not index._index_files_exist()

        # Create metadata file
        meta_path = index_path.with_suffix(".meta.json")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump([], f)

        # Now should return True for DB-backed store
        assert index._index_files_exist()


# ---------------------------------------------------------------------------
# Tests: ClassificationIndex with FAISS (file-based) store
# ---------------------------------------------------------------------------

class TestClassificationIndexWithFaissStore:
    def test_save_creates_vector_and_meta_files(self, tmp_path):
        embedder = _random_embedder()
        index_path = tmp_path / "test_index"
        index = ClassificationIndex(
            embedder=embedder,
            index_path=index_path,
        )

        index.add(
            texts=["burger"],
            category_paths=["Food > Burgers"],
        )
        index.save()

        assert index_path.with_suffix(".faiss").exists()
        assert index_path.with_suffix(".meta.json").exists()

    def test_round_trip_save_load(self, tmp_path):
        embedder = _random_embedder()
        index_path = tmp_path / "test_index"

        # Create and save
        index1 = ClassificationIndex(
            embedder=embedder,
            index_path=index_path,
        )
        index1.add(
            texts=["burger", "pizza"],
            category_paths=["Food > Burgers", "Food > Pizza"],
        )
        index1.save()

        # Load into new instance
        index2 = ClassificationIndex(
            embedder=embedder,
            index_path=index_path,
        )
        assert index2.size == 2
        assert len(index2._metadata) == 2


# ---------------------------------------------------------------------------
# Tests: AdaptiveClassifier with custom vector store
# ---------------------------------------------------------------------------

class TestAdaptiveClassifierWithCustomStore:
    def test_classifier_accepts_custom_store(self):
        from adaptive_classifier import AdaptiveClassifier, Taxonomy

        store = InMemoryVectorStore(DIMENSION)
        embedder = _random_embedder()

        taxonomy = Taxonomy.from_dict({
            "Food": ["Burgers", "Pizza"],
        })

        classifier = AdaptiveClassifier(
            taxonomy=taxonomy,
            embedder=embedder,
            vector_store=store,
        )

        assert classifier.index.size > 0  # taxonomy seeded
        assert classifier.index._store is store
