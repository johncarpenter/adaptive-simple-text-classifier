"""Adaptive Classifier - the main orchestrator.

Hybrid classification: FAISS embedding search first, LLM fallback for
low-confidence items, results fed back into the index for continuous
improvement.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

from .embeddings import EmbeddingProvider, SentenceTransformerEmbedder
from .index import ClassificationIndex
from .providers import LLMProvider, resolve_provider
from .taxonomy import Taxonomy
from .vector_stores import VectorStore
from .types import (
    BatchStats,
    Classification,
    ClassificationBatch,
    ClassificationSource,
)

logger = logging.getLogger(__name__)


class AdaptiveClassifier:
    """Self-building hybrid classifier.

    Combines fast FAISS nearest-neighbor search with LLM fallback.
    LLM results feed back into the index, so accuracy improves over time
    and LLM costs decrease.

    Usage:
        classifier = AdaptiveClassifier(
            taxonomy={"Food": {"Burgers": ["Hamburger", "Cheeseburger"]}},
            provider="anthropic",
            index_path="./my_index",
        )

        results = classifier.classify(["chz brgr", "lg pepperoni pza"])

        for r in results:
            print(f"{r.input_text} -> {r.category_path} ({r.confidence:.2f}, {r.source.value})")
    """

    def __init__(
        self,
        taxonomy: dict[str, Any] | list[str] | Taxonomy | str | Path,
        provider: str | LLMProvider | Callable | None = "anthropic",
        embedder: EmbeddingProvider | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        vector_store: VectorStore | None = None,
        index_path: str | Path | None = None,
        confidence_threshold: float = 0.65,
        k_neighbors: int = 5,
        llm_batch_size: int = 50,
        auto_feedback: bool = True,
        auto_save: bool = True,
        normalizer: Callable[[str], str] | None = None,
        provider_kwargs: dict[str, Any] | None = None,
    ):
        """
        Args:
            taxonomy: Classification taxonomy as dict, path list, Taxonomy, or file path.
            provider: LLM provider - "anthropic", "vertex", "bedrock", callable, or LLMProvider.
            embedder: Custom embedding provider. If None, uses SentenceTransformer.
            embedding_model: Model name for default SentenceTransformer embedder.
            vector_store: Custom vector store backend. If None, uses FAISS.
            index_path: Path to persist the index. None = in-memory only.
            confidence_threshold: Minimum confidence for embedding match (0.0-1.0).
            k_neighbors: Number of neighbors for majority voting.
            llm_batch_size: Items per LLM API call.
            auto_feedback: Automatically feed LLM results back into the index.
            auto_save: Automatically save index after classification.
            normalizer: Optional text normalization function applied before embedding.
            provider_kwargs: Extra kwargs passed to the LLM provider constructor.
        """
        # Resolve taxonomy
        if isinstance(taxonomy, Taxonomy):
            self.taxonomy = taxonomy
        elif isinstance(taxonomy, (str, Path)):
            self.taxonomy = Taxonomy.from_file(taxonomy)
        elif isinstance(taxonomy, list):
            self.taxonomy = Taxonomy.from_flat(taxonomy)
        elif isinstance(taxonomy, dict):
            self.taxonomy = Taxonomy.from_dict(taxonomy)
        else:
            raise TypeError(f"Unsupported taxonomy type: {type(taxonomy)}")

        # Resolve embedder
        self.embedder = embedder or SentenceTransformerEmbedder(embedding_model)

        # Build index
        self.index = ClassificationIndex(
            embedder=self.embedder,
            index_path=index_path,
            vector_store=vector_store,
        )

        # Seed index with taxonomy if it's empty
        if self.index.size == 0:
            self.index.seed_taxonomy(self.taxonomy.leaf_paths)

        # Resolve LLM provider
        self.provider = resolve_provider(provider, **(provider_kwargs or {}))

        # Config
        self.confidence_threshold = confidence_threshold
        self.k_neighbors = k_neighbors
        self.llm_batch_size = llm_batch_size
        self.auto_feedback = auto_feedback
        self.auto_save = auto_save
        self.normalizer = normalizer
        self.index_path = index_path

    def classify(
        self,
        items: list[str] | str,
        confidence_threshold: float | None = None,
    ) -> ClassificationBatch:
        """Classify items using hybrid embedding + LLM approach.

        1. Normalize inputs
        2. Search FAISS index for confident matches
        3. Route low-confidence items to LLM
        4. Feed LLM results back into index
        5. Return all results

        Args:
            items: Single string or list of strings to classify.
            confidence_threshold: Override instance threshold for this call.

        Returns:
            ClassificationBatch with results and stats.
        """
        if isinstance(items, str):
            items = [items]

        threshold = confidence_threshold or self.confidence_threshold
        start = time.time()

        # Step 1: Normalize
        normalized = [self._normalize(item) for item in items]

        # Step 2: Embedding search
        best_matches = self.index.search_best(
            normalized,
            k=self.k_neighbors,
            min_confidence=threshold,
        )

        # Split into confident vs uncertain
        results: dict[int, Classification] = {}
        llm_indices: list[int] = []

        for i, match in enumerate(best_matches):
            if match is not None:
                results[i] = Classification(
                    input_text=items[i],
                    category_path=match.category_path,
                    confidence=match.confidence,
                    source=ClassificationSource.EMBEDDING,
                )
            else:
                llm_indices.append(i)

        logger.info(
            f"Embedding pass: {len(results)} hits, {len(llm_indices)} to LLM "
            f"(threshold={threshold})"
        )

        # Step 3: LLM fallback for uncertain items
        llm_call_count = 0
        if llm_indices and self.provider:
            llm_items = [normalized[i] for i in llm_indices]
            taxonomy_prompt = self.taxonomy.render_for_prompt()

            try:
                llm_results = self.provider.classify_batch(
                    items=llm_items,
                    taxonomy_prompt=taxonomy_prompt,
                    batch_size=self.llm_batch_size,
                )
                llm_call_count = (len(llm_items) + self.llm_batch_size - 1) // self.llm_batch_size

                # Map LLM results back
                llm_result_map = {r["input"]: r["category"] for r in llm_results}

                feedback_texts = []
                feedback_paths = []

                for idx in llm_indices:
                    norm_text = normalized[idx]
                    category = llm_result_map.get(norm_text)

                    if not category:
                        # Try fuzzy match on input
                        for r in llm_results:
                            if r["input"].lower().strip() == norm_text.lower().strip():
                                category = r["category"]
                                break

                    if category:
                        results[idx] = Classification(
                            input_text=items[idx],
                            category_path=category,
                            confidence=0.9,  # LLM classifications get high confidence
                            source=ClassificationSource.LLM,
                        )
                        feedback_texts.append(norm_text)
                        feedback_paths.append(category)

                        # Also feed back the original (pre-normalization) text
                        if items[idx] != norm_text:
                            feedback_texts.append(items[idx])
                            feedback_paths.append(category)
                    else:
                        # LLM didn't return a result for this item
                        results[idx] = Classification(
                            input_text=items[idx],
                            category_path="UNCLASSIFIED",
                            confidence=0.0,
                            source=ClassificationSource.LLM,
                        )

                # Step 4: Feedback loop
                fed_back = 0
                if self.auto_feedback and feedback_texts:
                    fed_back = self.index.add(
                        feedback_texts,
                        feedback_paths,
                        source=ClassificationSource.LLM,
                    )
                    logger.info(f"Fed {fed_back} items back into index")

            except Exception as e:
                logger.error(f"LLM classification failed: {e}")
                for idx in llm_indices:
                    if idx not in results:
                        results[idx] = Classification(
                            input_text=items[idx],
                            category_path="UNCLASSIFIED",
                            confidence=0.0,
                            source=ClassificationSource.LLM,
                            metadata={"error": str(e)},
                        )
                fed_back = 0

        else:
            fed_back = 0

        # Step 5: Assemble ordered results
        ordered = [results[i] for i in range(len(items))]

        elapsed = time.time() - start
        confidences = [r.confidence for r in ordered if r.confidence > 0]
        stats = BatchStats(
            total=len(items),
            embedding_hits=len(items) - len(llm_indices),
            llm_calls=llm_call_count,
            llm_items=len(llm_indices),
            fed_back=fed_back,
            avg_confidence=sum(confidences) / len(confidences) if confidences else 0,
            elapsed_seconds=elapsed,
        )

        # Auto-save
        if self.auto_save and self.index_path:
            self.index.save()

        batch = ClassificationBatch(results=ordered, stats=stats)
        logger.info(f"Classification complete: {stats.to_dict()}")
        return batch

    def classify_one(
        self,
        item: str,
        confidence_threshold: float | None = None,
    ) -> Classification:
        """Classify a single item. Convenience wrapper."""
        batch = self.classify([item], confidence_threshold=confidence_threshold)
        return batch[0]

    def add_examples(
        self,
        examples: dict[str, str] | list[tuple[str, str]],
    ) -> int:
        """Manually add labeled examples to the index.

        Args:
            examples: Dict of {text: category_path} or list of (text, category_path) tuples.

        Returns:
            Number of examples added.
        """
        if isinstance(examples, dict):
            examples = list(examples.items())

        texts = [t for t, _ in examples]
        paths = [p for _, p in examples]

        return self.index.add(texts, paths, source=ClassificationSource.MANUAL)

    def save(self) -> None:
        """Explicitly save the index to disk."""
        self.index.save()

    def stats(self) -> dict[str, Any]:
        """Return index statistics."""
        return self.index.stats()

    def _normalize(self, text: str) -> str:
        """Apply text normalization."""
        if self.normalizer:
            return self.normalizer(text)
        return text.strip()
