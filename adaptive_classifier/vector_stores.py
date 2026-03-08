"""Pluggable vector store backends for nearest-neighbor search.

Provides a Protocol-based abstraction so any vector store can be used.
Includes a default FAISS implementation (IndexFlatIP for cosine similarity
on normalized vectors).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector store backends."""

    @property
    def size(self) -> int:
        """Number of vectors in the store."""
        ...

    @property
    def file_suffix(self) -> str | None:
        """File extension for persistence, or None if store manages its own persistence."""
        ...

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to the store. vectors is (N, dimension) float32."""
        ...

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors.

        Args:
            queries: (N, dimension) float32 query vectors.
            k: Number of neighbors to return.

        Returns:
            Tuple of (distances, indices) each shaped (N, k).
            Use index -1 for missing results.
        """
        ...

    def reset(self) -> None:
        """Clear all vectors from the store."""
        ...

    def save(self, path: Path) -> None:
        """Persist the vector store to disk."""
        ...

    def load(self, path: Path) -> None:
        """Load the vector store from disk."""
        ...


class FaissVectorStore:
    """FAISS-backed vector store using IndexFlatIP (inner product / cosine).

    This is the default vector store. Uses exact search which is accurate
    for small-to-medium indexes (up to ~1M vectors).
    """

    def __init__(self, dimension: int):
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss required: pip install faiss-cpu")

        self._faiss = faiss
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        logger.info(f"Created FAISS vector store (dim={dimension})")

    @property
    def size(self) -> int:
        return self._index.ntotal

    @property
    def file_suffix(self) -> str | None:
        return ".faiss"

    def add(self, vectors: np.ndarray) -> None:
        self._index.add(vectors)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        actual_k = min(k, self.size) if self.size > 0 else 1
        distances, indices = self._index.search(queries, actual_k)
        return distances, indices

    def reset(self) -> None:
        self._index = self._faiss.IndexFlatIP(self._dimension)

    def save(self, path: Path) -> None:
        self._faiss.write_index(self._index, str(path))

    def load(self, path: Path) -> None:
        self._index = self._faiss.read_index(str(path))
