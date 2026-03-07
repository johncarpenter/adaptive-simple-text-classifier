"""Benchmark: Adaptive Classifier on financial transaction data.

Loads the transaction parquet, trains on 50 stratified examples,
then tests against 100 held-out records. Reports accuracy, per-category
precision/recall, timing, memory, and index stats.

Runs three passes against the same test set:
  1. Embedding-only (no LLM) — fast baseline, no index persistence
  2. Hybrid (embedding + Haiku LLM fallback) — LLM results fed back & saved
  3. Post-feedback embedding-only — re-test using the enriched index (no LLM)

The index is persisted to example/benchmark_index so subsequent runs
start with an already-enriched index and hit the LLM less over time.

Usage:
    uv run python example/benchmark.py
    uv run python example/benchmark.py --runs 5        # repeat 5 times to watch learning
    uv run python example/benchmark.py --reset          # clear saved index first
    uv run python example/benchmark.py --embedding-only  # skip LLM passes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc

# Prevent OMP threading conflict between FAISS and PyTorch
os.environ.setdefault("OMP_NUM_THREADS", "1")
from collections import Counter
from pathlib import Path

import pandas as pd

# Ensure the package is importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive_classifier import AdaptiveClassifier


EXAMPLE_DIR = Path(__file__).resolve().parent
PARQUET_PATH = EXAMPLE_DIR / "0000.parquet"
CATEGORIES_PATH = EXAMPLE_DIR / "categories.json"
INDEX_PATH = EXAMPLE_DIR / "benchmark_index"

TRAIN_SIZE = 50
TEST_SIZE = 100
RANDOM_SEED = 42
CONFIDENCE_THRESHOLD = 0.65


def load_taxonomy() -> dict:
    """Build taxonomy dict from categories.json."""
    with open(CATEGORIES_PATH) as f:
        data = json.load(f)
    return {cat["name"]: cat["name"] for cat in data["categories"]}


def sample_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified sample: TRAIN_SIZE for training, TEST_SIZE for testing.

    Ensures no overlap and balanced category representation.
    """
    categories = df["category"].unique()
    train_per_cat = max(1, TRAIN_SIZE // len(categories))
    test_per_cat = max(1, TEST_SIZE // len(categories))

    train_frames = []
    test_frames = []

    for cat in categories:
        cat_df = df[df["category"] == cat].sample(
            n=train_per_cat + test_per_cat,
            random_state=RANDOM_SEED,
        )
        train_frames.append(cat_df.iloc[:train_per_cat])
        test_frames.append(cat_df.iloc[train_per_cat : train_per_cat + test_per_cat])

    train_df = pd.concat(train_frames).sample(frac=1, random_state=RANDOM_SEED)
    test_df = pd.concat(test_frames).sample(frac=1, random_state=RANDOM_SEED)

    return train_df, test_df


def build_classifier(
    taxonomy_dict: dict,
    train_df: pd.DataFrame,
    *,
    use_llm: bool = False,
    confidence_threshold: float = 0.0,
    index_path: Path | None = None,
    auto_feedback: bool = False,
) -> tuple[AdaptiveClassifier, float]:
    """Create classifier, seed with taxonomy, add training examples.

    Args:
        use_llm: If True, use Anthropic Haiku as LLM fallback.
        confidence_threshold: Embedding confidence below this routes to LLM.
        index_path: Path to persist/load index. None = in-memory only.
        auto_feedback: If True, LLM results are fed back into the index.

    Returns (classifier, build_time_s).
    """
    t0 = time.perf_counter()

    classifier = AdaptiveClassifier(
        taxonomy=taxonomy_dict,
        provider="anthropic" if use_llm else lambda items, taxonomy_prompt, batch_size: [],
        confidence_threshold=confidence_threshold,
        index_path=index_path,
        auto_save=index_path is not None,
        auto_feedback=auto_feedback,
    )

    # Only add training examples if the index is freshly seeded (taxonomy-only).
    # If we loaded a persisted index that already has training data, skip.
    if classifier.index.size <= len(classifier.taxonomy.leaf_paths) * 2:
        examples = {
            row["transaction_description"]: row["category"]
            for _, row in train_df.iterrows()
        }
        classifier.add_examples(examples)
        if index_path:
            classifier.save()

    build_time = time.perf_counter() - t0
    return classifier, build_time


def evaluate(
    classifier: AdaptiveClassifier,
    test_df: pd.DataFrame,
) -> dict:
    """Run classification and compute metrics."""
    descriptions = test_df["transaction_description"].tolist()
    expected = test_df["category"].tolist()

    tracemalloc.start()
    t0 = time.perf_counter()
    batch = classifier.classify(descriptions)
    classify_time = time.perf_counter() - t0
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    predicted = [r.category_path for r in batch]
    confidences = [r.confidence for r in batch]
    sources = [r.source.value for r in batch]

    # Overall accuracy
    correct = sum(1 for p, e in zip(predicted, expected) if p == e)
    accuracy = correct / len(expected)

    # Per-category precision / recall
    categories = sorted(set(expected))
    per_category = {}
    for cat in categories:
        tp = sum(1 for p, e in zip(predicted, expected) if p == cat and e == cat)
        fp = sum(1 for p, e in zip(predicted, expected) if p == cat and e != cat)
        fn = sum(1 for p, e in zip(predicted, expected) if p != cat and e == cat)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_category[cat] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for e in expected if e == cat),
        }

    # Aggregate F1
    macro_f1 = sum(m["f1"] for m in per_category.values()) / len(per_category)

    # Misclassifications detail
    misclassified = [
        {
            "input": descriptions[i],
            "expected": expected[i],
            "predicted": predicted[i],
            "confidence": round(confidences[i], 4),
            "source": sources[i],
        }
        for i in range(len(expected))
        if predicted[i] != expected[i]
    ]

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "correct": correct,
        "total": len(expected),
        "classify_time_s": round(classify_time, 3),
        "items_per_second": round(len(expected) / classify_time, 1),
        "peak_memory_mb": round(peak_memory / 1024 / 1024, 2),
        "avg_confidence": round(sum(confidences) / len(confidences), 4),
        "source_distribution": dict(Counter(sources)),
        "batch_stats": batch.stats.to_dict(),
        "per_category": per_category,
        "misclassified": misclassified,
    }


def print_report(
    title: str,
    results: dict,
    build_time: float,
    index_stats: dict,
    train_size: int,
    test_size: int,
) -> None:
    """Pretty-print the benchmark report."""
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)

    print(f"\n{'Dataset':.<30} {PARQUET_PATH.name}")
    print(f"{'Training examples':.<30} {train_size}")
    print(f"{'Test examples':.<30} {test_size}")
    print(f"{'Index vectors':.<30} {index_stats['total_vectors']}")
    print(f"{'Embedding dimension':.<30} {index_stats['dimension']}")

    print("\n--- Performance ---")
    print(f"{'Index build time':.<30} {build_time:.3f}s")
    print(f"{'Classification time':.<30} {results['classify_time_s']:.3f}s")
    print(f"{'Throughput':.<30} {results['items_per_second']:.1f} items/s")
    print(f"{'Peak memory (classify)':.<30} {results['peak_memory_mb']:.2f} MB")

    stats = results["batch_stats"]
    print("\n--- Routing ---")
    print(f"{'Embedding hits':.<30} {stats['embedding_hits']}/{stats['total']}")
    print(f"{'LLM fallback items':.<30} {stats['llm_items']}")
    print(f"{'LLM API calls':.<30} {stats['llm_calls']}")
    print(f"{'Fed back to index':.<30} {stats['fed_back']}")

    print("\n--- Accuracy ---")
    print(f"{'Overall accuracy':.<30} {results['accuracy']:.1%}  ({results['correct']}/{results['total']})")
    print(f"{'Macro F1':.<30} {results['macro_f1']:.4f}")
    print(f"{'Avg confidence':.<30} {results['avg_confidence']:.4f}")
    print(f"{'Source distribution':.<30} {results['source_distribution']}")

    print("\n--- Per-Category Breakdown ---")
    header = f"{'Category':<30} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}"
    print(header)
    print("-" * len(header))
    for cat, m in sorted(results["per_category"].items()):
        print(f"{cat:<30} {m['precision']:>6.2%} {m['recall']:>6.2%} {m['f1']:>6.4f} {m['support']:>8}")

    if results["misclassified"]:
        print(f"\n--- Misclassified ({len(results['misclassified'])}) ---")
        for m in results["misclassified"]:
            print(f"  [{m['confidence']:.2f}] [{m['source']:>9s}] \"{m['input']}\"")
            print(f"         expected: {m['expected']}")
            print(f"        predicted: {m['predicted']}")
    else:
        print("\n  No misclassifications!")

    print("\n" + "=" * 70)


def print_comparison(all_results: list[tuple[str, dict]]) -> None:
    """Print side-by-side comparison of all runs."""
    if len(all_results) < 2:
        return

    print("\n" + "=" * 70)
    print("  COMPARISON ACROSS ALL PASSES")
    print("=" * 70)

    metrics = [
        ("Accuracy", "accuracy", ".1%"),
        ("Macro F1", "macro_f1", ".4f"),
        ("Avg confidence", "avg_confidence", ".4f"),
        ("Classification time", "classify_time_s", ".3f"),
        ("Throughput (items/s)", "items_per_second", ".1f"),
    ]

    # Build column headers
    col_width = 16
    header = f"{'Metric':<25}"
    for label, _ in all_results:
        header += f" {label:>{col_width}}"
    print(f"\n{header}")
    print("-" * len(header))

    for metric_label, key, fmt in metrics:
        row = f"{metric_label:<25}"
        for _, results in all_results:
            val = results[key]
            row += f" {val:>{col_width}{fmt}}"
        print(row)

    # Routing row
    row = f"{'Embedding hits':<25}"
    for _, results in all_results:
        val = results["batch_stats"]["embedding_hits"]
        total = results["batch_stats"]["total"]
        row += f" {f'{val}/{total}':>{col_width}}"
    print(row)

    row = f"{'LLM fallback items':<25}"
    for _, results in all_results:
        val = results["batch_stats"]["llm_items"]
        row += f" {val:>{col_width}}"
    print(row)

    row = f"{'LLM API calls':<25}"
    for _, results in all_results:
        val = results["batch_stats"]["llm_calls"]
        row += f" {val:>{col_width}}"
    print(row)

    row = f"{'Fed back to index':<25}"
    for _, results in all_results:
        val = results["batch_stats"]["fed_back"]
        row += f" {val:>{col_width}}"
    print(row)

    print("\n" + "=" * 70)


def print_learning_summary(run_results: list[list[tuple[str, dict]]]) -> None:
    """Print accuracy progression across multiple runs."""
    print("\n" + "=" * 70)
    print("  LEARNING CURVE ACROSS RUNS")
    print("=" * 70)

    header = f"{'Run':<6}"
    # Use pass labels from the first run
    for label, _ in run_results[0]:
        header += f" {label:>16}"
    print(f"\n{header}")
    print("-" * len(header))

    for i, passes in enumerate(run_results):
        row = f"{'#' + str(i + 1):<6}"
        for _, results in passes:
            row += f" {results['accuracy']:>15.1%}"
        print(row)

    # Show LLM calls decreasing
    print()
    header2 = f"{'Run':<6} {'LLM items':>16} {'LLM calls':>16} {'Index size':>16}"
    print(header2)
    print("-" * len(header2))
    for i, passes in enumerate(run_results):
        # Find the hybrid pass (has LLM items)
        hybrid = next((r for label, r in passes if r["batch_stats"]["llm_items"] > 0), None)
        if hybrid:
            row = f"{'#' + str(i + 1):<6}"
            row += f" {hybrid['batch_stats']['llm_items']:>16}"
            row += f" {hybrid['batch_stats']['llm_calls']:>16}"
            row += f" {hybrid.get('index_size', 'n/a'):>16}"
            print(row)

    print("\n" + "=" * 70)


def index_exists() -> bool:
    """Check if the persisted index files exist."""
    return (
        INDEX_PATH.with_suffix(".faiss").exists()
        and INDEX_PATH.with_suffix(".meta.json").exists()
    )


def clear_index() -> None:
    """Remove persisted index files."""
    for ext in (".faiss", ".meta.json"):
        p = INDEX_PATH.with_suffix(ext)
        if p.exists():
            p.unlink()
    print(f"  Cleared index at {INDEX_PATH}")


def run_single(
    taxonomy_dict: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float,
    run_embedding_only: bool,
    run_llm: bool,
) -> list[tuple[str, dict]]:
    """Execute one full benchmark cycle. Returns list of (label, results) pairs."""
    all_pass_results: list[tuple[str, dict]] = []

    # --- Pass 1: Embedding-only (cold, no persisted index) ---
    if run_embedding_only:
        print("\n" + "#" * 70)
        print("  PASS 1: Embedding-Only (no LLM)")
        print("#" * 70)

        classifier, build_time = build_classifier(
            taxonomy_dict, train_df,
            use_llm=False,
            confidence_threshold=0.0,
            index_path=None,  # in-memory, no persistence
        )
        index_stats = classifier.stats()
        print(f"  Built in {build_time:.3f}s ({index_stats['total_vectors']} vectors)")

        results = evaluate(classifier, test_df)
        print_report(
            "EMBEDDING-ONLY RESULTS",
            results, build_time, index_stats, len(train_df), len(test_df),
        )
        all_pass_results.append(("Embed-only", results))

    # --- Pass 2: Hybrid (Embedding + Haiku, feedback ON, persisted) ---
    if run_llm:
        print("\n" + "#" * 70)
        print(f"  PASS 2: Hybrid (Embedding + Haiku, threshold={threshold})")
        print("#" * 70)

        if index_exists():
            print(f"  Loading existing index from {INDEX_PATH}")
        else:
            print("  No existing index — starting fresh")

        classifier, build_time = build_classifier(
            taxonomy_dict, train_df,
            use_llm=True,
            confidence_threshold=threshold,
            index_path=INDEX_PATH,
            auto_feedback=True,
        )
        index_stats = classifier.stats()
        print(f"  Built in {build_time:.3f}s ({index_stats['total_vectors']} vectors)")

        print("\n  Classifying (low-confidence -> Haiku, results fed back to index)...")
        results = evaluate(classifier, test_df)
        results["index_size"] = classifier.index.size
        index_stats = classifier.stats()  # refresh after feedback
        print_report(
            f"HYBRID RESULTS (threshold={threshold})",
            results, build_time, index_stats, len(train_df), len(test_df),
        )
        all_pass_results.append(("Hybrid+LLM", results))

        # --- Pass 3: Post-feedback embedding-only (re-test enriched index) ---
        print("\n" + "#" * 70)
        print("  PASS 3: Post-Feedback Embedding-Only (enriched index, no LLM)")
        print("#" * 70)

        classifier_post, build_time_post = build_classifier(
            taxonomy_dict, train_df,
            use_llm=False,
            confidence_threshold=0.0,
            index_path=INDEX_PATH,
        )
        index_stats_post = classifier_post.stats()
        print(f"  Loaded index: {index_stats_post['total_vectors']} vectors")

        results_post = evaluate(classifier_post, test_df)
        results_post["index_size"] = classifier_post.index.size
        print_report(
            "POST-FEEDBACK EMBEDDING-ONLY RESULTS",
            results_post, build_time_post, index_stats_post, len(train_df), len(test_df),
        )
        all_pass_results.append(("Post-feedback", results_post))

    return all_pass_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark adaptive classifier")
    parser.add_argument("--embedding-only", action="store_true", help="Skip LLM passes")
    parser.add_argument("--llm-only", action="store_true", help="Skip initial embedding-only pass")
    parser.add_argument(
        "--threshold", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"Embedding confidence threshold for LLM fallback (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument("--reset", action="store_true", help="Clear saved index before running")
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of runs to show learning curve (default: 1)",
    )
    args = parser.parse_args()

    if args.reset:
        clear_index()

    print("Loading parquet data...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  {len(df):,} total records, {df['category'].nunique()} categories")

    taxonomy_dict = load_taxonomy()
    train_df, test_df = sample_data(df)

    print(f"\nSampled {len(train_df)} train / {len(test_df)} test (stratified, no overlap)")

    run_embedding = not args.llm_only
    run_llm = not args.embedding_only

    all_run_results: list[list[tuple[str, dict]]] = []

    for run_num in range(args.runs):
        if args.runs > 1:
            print("\n" + "*" * 70)
            print(f"  RUN {run_num + 1} / {args.runs}")
            print("*" * 70)

        pass_results = run_single(
            taxonomy_dict, train_df, test_df, args.threshold,
            run_embedding_only=run_embedding,
            run_llm=run_llm,
        )
        all_run_results.append(pass_results)

        # After first run, skip the cold embedding-only pass
        if args.runs > 1 and run_num == 0:
            run_embedding = False

    # Print comparisons
    if len(all_run_results[-1]) > 1:
        print_comparison(all_run_results[-1])

    if args.runs > 1:
        print_learning_summary(all_run_results)

    if index_exists():
        print(f"\nIndex saved to: {INDEX_PATH}.faiss + {INDEX_PATH}.meta.json")

    final = all_run_results[-1][-1][1]
    return final


if __name__ == "__main__":
    results = main()
    sys.exit(0 if results["accuracy"] > 0 else 1)
