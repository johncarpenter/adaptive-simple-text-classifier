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


class OpenAIEmbedder:
    """Cloud embeddings via OpenAI API.

    Uses text-embedding-3-small (1536 dims) by default.
    For higher quality, use text-embedding-3-large (3072 dims).

    Both models support a `dimensions` parameter to reduce output size
    (e.g., 384 or 512) while preserving quality via Matryoshka representation.

    Usage:
        embedder = OpenAIEmbedder()  # uses OPENAI_API_KEY env var
        embedder = OpenAIEmbedder(api_key="sk-...", model="text-embedding-3-small", dimensions=384)
    """

    # Known output dimensions for OpenAI models (when dimensions param is not set)
    _MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        api_key: str | None = None,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai required: pip install adaptive-simple-text-classifier[openai]"
            )

        self._client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
        self._model = model
        self._dimensions = dimensions
        self._dimension_value = dimensions or self._MODEL_DIMENSIONS.get(model, 1536)
        logger.info(
            f"Loaded OpenAI embedding model: {model} (dim={self._dimension_value})"
        )

    @property
    def dimension(self) -> int:
        return self._dimension_value

    def embed(self, texts: list[str]) -> np.ndarray:
        kwargs: dict = {"input": texts, "model": self._model}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions

        response = self._client.embeddings.create(**kwargs)
        embeddings = np.array(
            [item.embedding for item in response.data], dtype=np.float32
        )
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        return embeddings


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
