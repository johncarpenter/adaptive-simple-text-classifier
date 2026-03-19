"""adaptive_classifier - Self-building hybrid classifier.

FAISS embedding search + LLM fallback with automatic feedback loop.
Classifies messy text into structured taxonomies.
"""

from .classifier import AdaptiveClassifier
from .embeddings import CallableEmbedder, EmbeddingProvider

# Lazy imports for optional embedding providers
def __getattr__(name):
    if name == "SentenceTransformerEmbedder":
        from .embeddings import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder
    if name == "OpenAIEmbedder":
        from .embeddings import OpenAIEmbedder
        return OpenAIEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
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

__version__ = "0.2.0"

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
    "OpenAIEmbedder",
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
