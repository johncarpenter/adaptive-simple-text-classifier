"""adaptive_classifier - Self-building hybrid classifier.

FAISS embedding search + LLM fallback with automatic feedback loop.
Classifies messy text into structured taxonomies.
"""

from .classifier import AdaptiveClassifier
from .embeddings import CallableEmbedder, EmbeddingProvider, SentenceTransformerEmbedder
from .index import ClassificationIndex
from .normalizer import create_normalizer, DEFAULT_ABBREVIATIONS
from .providers import (
    AnthropicProvider,
    BedrockProvider,
    CallableLLMProvider,
    LLMProvider,
    VertexProvider,
)
from .taxonomy import Taxonomy
from .vector_stores import FaissVectorStore, VectorStore
from .types import (
    BatchStats,
    Classification,
    ClassificationBatch,
    ClassificationSource,
)

__version__ = "0.1.2"

__all__ = [
    "AdaptiveClassifier",
    # Taxonomy
    "Taxonomy",
    # Types
    "Classification",
    "ClassificationBatch",
    "ClassificationSource",
    "BatchStats",
    # Providers
    "LLMProvider",
    "AnthropicProvider",
    "VertexProvider",
    "BedrockProvider",
    "CallableLLMProvider",
    # Embeddings
    "EmbeddingProvider",
    "SentenceTransformerEmbedder",
    "CallableEmbedder",
    # Index
    "ClassificationIndex",
    # Vector stores
    "VectorStore",
    "FaissVectorStore",
    # Normalizer
    "create_normalizer",
    "DEFAULT_ABBREVIATIONS",
]
