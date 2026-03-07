"""Embedding providers.

Default uses sentence-transformers (local, fast, free).
Extensible via the EmbeddingProvider protocol.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns (N, dimension) float32 array."""
        ...


class SentenceTransformerEmbedder:
    """Local embeddings via sentence-transformers.

    Default model is all-MiniLM-L6-v2: 384 dims, fast, good quality.
    For higher quality at the cost of speed, use all-mpnet-base-v2 (768 dims).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required: pip install sentence-transformers"
            )
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Loaded embedding model: {model_name} (dim={self._dimension})")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)


class CallableEmbedder:
    """Wrap any function as an embedding provider.

    Usage:
        embedder = CallableEmbedder(
            fn=my_embed_function,  # (list[str]) -> np.ndarray
            dimension=768,
        )
    """

    def __init__(self, fn, dimension: int):
        self._fn = fn
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        result = self._fn(texts)
        if not isinstance(result, np.ndarray):
            result = np.array(result, dtype=np.float32)
        return result
