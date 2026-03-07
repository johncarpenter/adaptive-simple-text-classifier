"""Core data types for adaptive_classifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ClassificationSource(str, Enum):
    EMBEDDING = "embedding"
    LLM = "llm"
    CACHE = "cache"
    MANUAL = "manual"


@dataclass
class Classification:
    """Result of classifying a single item."""

    input_text: str
    category_path: str  # Full path: "Food > Burgers > Cheeseburger"
    confidence: float  # 0.0 - 1.0
    source: ClassificationSource
    leaf_label: str = ""  # Just the leaf: "Cheeseburger"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.leaf_label and self.category_path:
            self.leaf_label = self.category_path.split(" > ")[-1]

    @property
    def path_parts(self) -> list[str]:
        return [p.strip() for p in self.category_path.split(" > ")]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input_text,
            "category": self.category_path,
            "leaf": self.leaf_label,
            "confidence": round(self.confidence, 4),
            "source": self.source.value,
            "metadata": self.metadata,
        }


@dataclass
class ClassificationBatch:
    """Result of classifying a batch of items."""

    results: list[Classification]
    stats: BatchStats = field(default_factory=lambda: BatchStats())

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, idx) -> Classification:
        return self.results[idx]

    @property
    def embedding_hits(self) -> list[Classification]:
        return [r for r in self.results if r.source == ClassificationSource.EMBEDDING]

    @property
    def llm_hits(self) -> list[Classification]:
        return [r for r in self.results if r.source == ClassificationSource.LLM]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.results]


@dataclass
class BatchStats:
    total: int = 0
    embedding_hits: int = 0
    llm_calls: int = 0
    llm_items: int = 0
    fed_back: int = 0
    avg_confidence: float = 0.0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "embedding_hits": self.embedding_hits,
            "llm_calls": self.llm_calls,
            "llm_items": self.llm_items,
            "fed_back": self.fed_back,
            "avg_confidence": round(self.avg_confidence, 4),
            "hit_rate": round(self.embedding_hits / self.total, 4) if self.total else 0,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }
