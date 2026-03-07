# Benchmark: Adaptive Classifier on Financial Transactions

This benchmark evaluates the adaptive classifier against a real-world financial transaction categorization dataset. It demonstrates the hybrid embedding + LLM approach and the feedback loop that reduces LLM costs over time.

## Dataset

**[Financial Transaction Categorization Dataset](https://huggingface.co/datasets/mitulshah/transaction-categorization)** by Mitul Shah

- 4.5M+ transaction records across 10 categories
- 5 countries (USA, UK, Canada, Australia, India)
- 5 currencies (USD, GBP, CAD, AUD, INR)

### Categories

1. Food & Dining
2. Transportation
3. Shopping & Retail
4. Entertainment & Recreation
5. Healthcare & Medical
6. Utilities & Services
7. Financial Services
8. Income
9. Government & Legal
10. Charity & Donations

### Download

Download the parquet file from Hugging Face and place it in this directory:

```bash
# Using the Hugging Face CLI
pip install huggingface_hub
huggingface-cli download mitulshah/transaction-categorization \
  --repo-type dataset \
  --local-dir example/
```

Or download manually from https://huggingface.co/datasets/mitulshah/transaction-categorization and place `0000.parquet` in the `example/` directory.

You should end up with:

```
example/
  0000.parquet        # 71 MB, 4.5M records
  categories.json     # category definitions
  benchmark.py        # this benchmark script
```

## Setup

```bash
# Install dependencies (includes pandas, pyarrow, anthropic)
uv sync

# Set your Anthropic API key (required for hybrid mode)
export ANTHROPIC_API_KEY=sk-ant-...
```

## Running the Benchmark

### Full benchmark (3 passes)

```bash
uv run python example/benchmark.py
```

This runs three passes against 100 stratified test records:

1. **Embedding-only** — FAISS nearest-neighbor search with 50 training examples, no LLM
2. **Hybrid** — Embedding search + Haiku LLM fallback for low-confidence items. LLM results are fed back into the index and persisted to disk
3. **Post-feedback embedding-only** — Re-tests using the enriched index with no LLM calls

### Watch the learning curve

```bash
uv run python example/benchmark.py --runs 3
```

Repeats the hybrid + post-feedback passes multiple times. Each run benefits from the previous run's feedback, so LLM calls decrease as the index learns.

### Other options

```bash
# Clear saved index and start fresh
uv run python example/benchmark.py --reset

# Embedding-only (no API key needed)
uv run python example/benchmark.py --embedding-only

# LLM passes only (skip initial embedding baseline)
uv run python example/benchmark.py --llm-only

# Adjust the confidence threshold for LLM fallback (default: 0.65)
uv run python example/benchmark.py --threshold 0.5
```

## Results

Benchmark run with 50 training examples, 100 test records, and `--runs 3`:

### Comparison: Hybrid vs Post-Feedback (Run 3)

```
Metric                          Hybrid+LLM    Post-feedback
-----------------------------------------------------------
Accuracy                             90.0%            84.0%
Macro F1                            0.8972           0.8366
Avg confidence                      0.8158           0.7859
Classification time                  2.005            0.212
Throughput (items/s)                  49.9            471.7
Embedding hits                      85/100          100/100
LLM fallback items                      15                0
LLM API calls                            1                0
Fed back to index                       15                0
```

### Learning Curve

Accuracy across three consecutive runs:

```
Run          Embed-only       Hybrid+LLM    Post-feedback
---------------------------------------------------------
#1               42.0%           90.0%           72.0%
#2                               90.0%           82.0%
#3                               90.0%           84.0%
```

LLM usage decreasing as the index learns:

```
Run           LLM items        LLM calls       Index size
---------------------------------------------------------
#1                   98                2              168
#2                   48                1              216
#3                   15                1              231
```

### Key Takeaways

- **Embedding-only baseline**: 42% accuracy with just 50 training examples — the index only knows taxonomy labels and a few examples per category
- **Hybrid with Haiku**: 90% accuracy — the LLM correctly classifies items the embeddings can't
- **Feedback loop**: Post-feedback accuracy climbs from 72% to 84% across 3 runs as LLM results enrich the index
- **LLM cost reduction**: LLM fallback items drop from 98 to 15 (85% reduction) across 3 runs. By run 3, only 15% of items need LLM help
- **Throughput**: Post-feedback embedding-only runs at 472 items/s vs 50 items/s for hybrid — 9x faster once the index is trained
- **Index growth**: The index grows from 70 vectors (taxonomy + 50 examples) to 231 vectors after 3 runs of feedback

## How It Works

```
Input text ──> Normalize ──> Embedding search (FAISS)
                                    │
                         ┌──────────┴──────────┐
                    confidence              confidence
                    >= threshold            < threshold
                         │                      │
                    Return result          LLM fallback
                                           (Haiku)
                                                │
                                        ┌───────┴───────┐
                                   Return result   Feed back into
                                                   FAISS index
                                                        │
                                                   Save to disk
```

The feedback loop is the key mechanism: every LLM classification enriches the FAISS index, so the same pattern gets an embedding hit next time instead of an LLM call. Over time, LLM costs approach zero as the index learns your data distribution.

## Index Persistence

The benchmark persists its index to:

```
example/benchmark_index.faiss       # FAISS vector data
example/benchmark_index.meta.json   # category paths + metadata
```

These files are `.gitignore`d. Use `--reset` to clear and start fresh.
