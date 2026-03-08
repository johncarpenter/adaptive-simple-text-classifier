"""Vector index management with persistence and feedback loop.

The index stores (embedding, category_path, original_text) triples.
It grows over time as LLM results are fed back in.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from .embeddings import EmbeddingProvider
from .types import ClassificationSource
from .vector_stores import FaissVectorStore, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    category_path: str
    distance: float
    confidence: float
    original_text: str
    source: ClassificationSource = ClassificationSource.EMBEDDING


class ClassificationIndex:
    """Vector index for fast nearest-neighbor classification.

    Stores embeddings alongside metadata (category path, original text, source).
    Persists to disk as a pair of files: {name}.vectors + {name}.meta.json.
    Uses a pluggable VectorStore backend (FAISS by default).
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        index_path: str | Path | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.embedder = embedder
        self.index_path = Path(index_path) if index_path else None
        self._dimension = embedder.dimension

        # Metadata parallel to vector store entries
        self._metadata: list[dict[str, Any]] = []

        # Vector store backend
        self._store = vector_store or FaissVectorStore(self._dimension)

        # Try to load existing index
        if self.index_path and self._index_files_exist():
            self._load()

    @property
    def size(self) -> int:
        return self._store.size

    def _index_files_exist(self) -> bool:
        if not self.index_path:
            return False
        meta_path = self.index_path.with_suffix(".meta.json")
        if not meta_path.exists():
            return False
        # DB-backed stores manage their own persistence — only need metadata file
        if self._store.file_suffix is None:
            return True
        vector_path = self.index_path.with_suffix(self._store.file_suffix)
        return vector_path.exists()

    def _vector_file_path(self) -> Path | None:
        """Resolve the vector file path, or None if the store manages its own persistence."""
        if self._store.file_suffix is None:
            return None
        return self.index_path.with_suffix(self._store.file_suffix)

    def seed_taxonomy(self, leaf_paths: list[str]) -> int:
        """Seed the index with taxonomy leaf labels.

        Embeds each leaf label and adds it to the index. This gives the
        embedding search something to match against from the start.
        Returns the number of entries added.
        """
        if not leaf_paths:
            return 0

        # Use the leaf label (last segment) as the text to embed,
        # but also embed the full path for better matching
        texts_to_embed = []
        metadata_entries = []

        for path in leaf_paths:
            leaf = path.split(" > ")[-1]
            # Embed both the leaf label and the full path
            for text in [leaf, path]:
                texts_to_embed.append(text)
                metadata_entries.append({
                    "category_path": path,
                    "original_text": text,
                    "source": ClassificationSource.MANUAL.value,
                })

        embeddings = self.embedder.embed(texts_to_embed)
        self._store.add(embeddings)
        self._metadata.extend(metadata_entries)

        logger.info(f"Seeded index with {len(texts_to_embed)} entries from {len(leaf_paths)} taxonomy paths")
        return len(texts_to_embed)

    def add(
        self,
        texts: list[str],
        category_paths: list[str],
        source: ClassificationSource = ClassificationSource.LLM,
    ) -> int:
        """Add classified items to the index (feedback loop).

        Returns number of items added.
        """
        if not texts:
            return 0

        assert len(texts) == len(category_paths), "texts and category_paths must be same length"

        embeddings = self.embedder.embed(texts)
        self._store.add(embeddings)

        for text, path in zip(texts, category_paths):
            self._metadata.append({
                "category_path": path,
                "original_text": text,
                "source": source.value,
            })

        logger.info(f"Added {len(texts)} items to index (source={source.value})")
        return len(texts)

    def search(
        self,
        texts: list[str],
        k: int = 3,
    ) -> list[list[SearchResult]]:
        """Search for nearest neighbors.

        Returns list of list of SearchResults (one list per query text).
        Each inner list has up to k results sorted by confidence.
        """
        if self.size == 0:
            return [[] for _ in texts]

        embeddings = self.embedder.embed(texts)
        distances, indices = self._store.search(embeddings, k)

        results = []
        for i in range(len(texts)):
            hits = []
            for j in range(indices.shape[1]):
                idx = indices[i][j]
                if idx == -1:
                    continue
                dist = float(distances[i][j])
                # Cosine similarity on normalized vectors is already in [0, 1]
                confidence = max(0.0, min(1.0, dist))
                meta = self._metadata[idx]
                hits.append(SearchResult(
                    category_path=meta["category_path"],
                    distance=dist,
                    confidence=confidence,
                    original_text=meta["original_text"],
                    source=ClassificationSource(meta.get("source", "embedding")),
                ))
            results.append(hits)

        return results

    def search_best(
        self,
        texts: list[str],
        k: int = 3,
        min_confidence: float = 0.0,
    ) -> list[SearchResult | None]:
        """Return the single best match per text, with optional confidence threshold.

        Uses majority voting among top-k neighbors: if the most common
        category among the k nearest neighbors appears in a majority,
        use the average confidence of those matching neighbors.
        """
        all_results = self.search(texts, k=k)
        best = []

        for hits in all_results:
            if not hits:
                best.append(None)
                continue

            # Majority vote among top-k
            category_votes: dict[str, list[float]] = {}
            for hit in hits:
                cat = hit.category_path
                if cat not in category_votes:
                    category_votes[cat] = []
                category_votes[cat].append(hit.confidence)

            # Pick category with most votes, break ties by avg confidence
            winner = max(
                category_votes.items(),
                key=lambda x: (len(x[1]), sum(x[1]) / len(x[1])),
            )
            category_path = winner[0]
            avg_confidence = sum(winner[1]) / len(winner[1])

            if avg_confidence < min_confidence:
                best.append(None)
            else:
                best.append(SearchResult(
                    category_path=category_path,
                    distance=avg_confidence,
                    confidence=avg_confidence,
                    original_text=hits[0].original_text,
                ))

        return best

    def save(self) -> None:
        """Persist index and metadata to disk."""
        if not self.index_path:
            logger.warning("No index_path set, skipping save")
            return

        vector_path = self._vector_file_path()
        meta_path = self.index_path.with_suffix(".meta.json")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        if vector_path is not None:
            self._store.save(vector_path)

        with open(meta_path, "w") as f:
            json.dump(self._metadata, f)

        logger.info(f"Saved index ({self.size} vectors) to {vector_path or 'db-backed store'}")

    def _load(self) -> None:
        """Load index and metadata from disk."""
        vector_path = self._vector_file_path()
        meta_path = self.index_path.with_suffix(".meta.json")

        if vector_path is not None:
            self._store.load(vector_path)

        with open(meta_path) as f:
            self._metadata = json.load(f)

        logger.info(f"Loaded index ({self.size} vectors) from {vector_path or 'db-backed store'}")

    def clear(self) -> None:
        """Reset the index."""
        self._store.reset()
        self._metadata = []
        logger.info("Index cleared")

    def stats(self) -> dict[str, Any]:
        """Return index statistics."""
        source_counts: dict[str, int] = {}
        category_counts: dict[str, int] = {}
        for m in self._metadata:
            src = m.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
            cat = m["category_path"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "total_vectors": self.size,
            "dimension": self._dimension,
            "by_source": source_counts,
            "unique_categories": len(category_counts),
            "category_distribution": category_counts,
        }
