# adaptive-simple-text-classifier

[![CI](https://github.com/johncarpenter/adaptive-simple-text-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/johncarpenter/adaptive-simple-text-classifier/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/adaptive-simple-text-classifier)](https://pypi.org/project/adaptive-simple-text-classifier/)
[![Python](https://img.shields.io/pypi/pyversions/adaptive-simple-text-classifier)](https://pypi.org/project/adaptive-simple-text-classifier/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Self-building hybrid text classifier. FAISS embedding search with LLM fallback and automatic feedback loop.

Classifies messy, abbreviated text into structured taxonomies. The index grows as LLM results feed back, so accuracy improves and LLM costs decrease over time.

## Install

```bash
pip install adaptive-simple-text-classifier

# With LLM provider:
pip install adaptive-simple-text-classifier[anthropic]   # Direct Anthropic API
pip install adaptive-simple-text-classifier[vertex]      # Google Cloud Vertex AI
pip install adaptive-simple-text-classifier[bedrock]     # AWS Bedrock
pip install adaptive-simple-text-classifier[all]         # All providers
```

## How It Works

```
Input Text ──> Normalize ──> FAISS Search ──┬──> Confident? ──> Return result
                                            │
                                            └──> Uncertain? ──> LLM Classify ──> Return result
                                                                     │
                                                                     └──> Feed back into FAISS index
```

1. **First run**: Most items go to the LLM (cold start, only taxonomy labels in the index)
2. **LLM results** get embedded and stored back in the FAISS index
3. **Subsequent runs**: FAISS handles most items, LLM handles only novel patterns
4. **Over time**: Hit rate climbs toward 100%, LLM costs drop to near zero

## Quick Start

```python
from adaptive_classifier import AdaptiveClassifier, create_normalizer

# Define your taxonomy (nested dict, flat list, or YAML/JSON file)
taxonomy = {
    "Food": {
        "Burgers": ["Hamburger", "Cheeseburger", "Veggie Burger"],
        "Pizza": ["Pepperoni", "Margherita", "Hawaiian"],
        "Drinks": ["Coffee", "Juice", "Soda"],
    },
    "Retail": {
        "Electronics": ["Phone", "Laptop", "Tablet"],
        "Furniture": ["Chair", "Table", "Bookshelf"],
    },
}

classifier = AdaptiveClassifier(
    taxonomy=taxonomy,
    provider="anthropic",              # or "vertex", "bedrock", callable
    index_path="./my_classifier",      # persists to disk
    confidence_threshold=0.65,         # below this -> LLM fallback
    normalizer=create_normalizer(),    # expands abbreviations, strips noise
)

results = classifier.classify([
    "chz brgr",
    "lg pep pizza",
    "bkshf oak - $249.99",
    "iced coffee lg",
])

for r in results:
    print(f"{r.input_text:30s} -> {r.category_path:40s} ({r.confidence:.2f}, {r.source.value})")

# Check stats
print(results.stats.to_dict())
# {'total': 4, 'embedding_hits': 1, 'llm_calls': 1, 'llm_items': 3, 'fed_back': 6, ...}

# Run again - more hits from the index, fewer LLM calls
results2 = classifier.classify(["double chz burger", "pepperoni pza sm"])
print(f"Hit rate: {results2.stats.embedding_hits}/{results2.stats.total}")
```

## Benchmark Results

Tested against the [Financial Transaction Categorization Dataset](https://huggingface.co/datasets/mitulshah/transaction-categorization) (4.5M records, 10 categories) with 50 training examples and 100 test records. See [`example/`](example/) for the full benchmark.

### Accuracy across 3 runs

```
Run          Embed-only       Hybrid+LLM    Post-feedback
---------------------------------------------------------
#1               42.0%           90.0%           72.0%
#2                               90.0%           82.0%
#3                               90.0%           84.0%
```

### LLM usage decreasing as the index learns

```
Run           LLM items        LLM calls       Index size
---------------------------------------------------------
#1                   98                2              168
#2                   48                1              216
#3                   15                1              231
```

### Run 3 performance

| Metric | Hybrid+LLM | Post-feedback (no LLM) |
|--------|-----------|----------------------|
| Accuracy | 90.0% | 84.0% |
| Macro F1 | 0.8972 | 0.8366 |
| Throughput | 49.9 items/s | 471.7 items/s |
| Embedding hits | 85/100 | 100/100 |
| LLM fallback items | 15 | 0 |

By run 3, LLM usage dropped 85% (98 -> 15 items) and embedding-only throughput is 9x faster than hybrid. The index grows from 70 vectors to 231 as LLM results feed back.

## Use Case Examples

### Banking Transaction Classification

```python
from adaptive_classifier import AdaptiveClassifier, create_normalizer

envelopes = {
    "Housing": ["Rent", "Mortgage", "Property Tax", "Home Insurance", "Maintenance"],
    "Transportation": ["Gas", "Car Payment", "Insurance", "Parking", "Transit"],
    "Food": ["Groceries", "Restaurants", "Coffee Shops", "Fast Food"],
    "Utilities": ["Electric", "Gas Utility", "Water", "Internet", "Phone"],
    "Health": ["Doctor", "Dentist", "Pharmacy", "Gym"],
    "Entertainment": ["Streaming", "Movies", "Games", "Books"],
    "Savings": ["Emergency Fund", "Retirement", "Investment"],
}

classifier = AdaptiveClassifier(
    taxonomy=envelopes,
    provider="anthropic",
    index_path="./budget_classifier",
    normalizer=create_normalizer(
        abbreviations={"wal-mart": "walmart grocery", "amzn": "amazon"},
        strip_codes=True,
    ),
)

transactions = [
    "WALMART SUPERCENTER #4532",
    "SHELL OIL 57442",
    "NETFLIX.COM",
    "CITY OF CALGARY UTILITIES",
    "TIM HORTONS #0891",
    "PHARMACHOICE #112",
]

results = classifier.classify(transactions)
for r in results:
    print(f"{r.input_text:35s} -> {r.leaf_label}")
```

### Property Valuation CRN Lookup

```python
from adaptive_classifier import AdaptiveClassifier, Taxonomy

crn_taxonomy = Taxonomy.from_flat([
    "Furniture > Seating > Office Chair",
    "Furniture > Seating > Dining Chair",
    "Furniture > Storage > Bookshelf",
    "Furniture > Storage > Filing Cabinet",
    "Furniture > Tables > Desk",
    "Furniture > Tables > Dining Table",
    "Electronics > Computing > Desktop Computer",
    "Electronics > Computing > Laptop",
    "Electronics > Audio Visual > Television",
    "Electronics > Audio Visual > Projector",
    "Appliances > Kitchen > Refrigerator",
    "Appliances > Kitchen > Dishwasher",
    "Appliances > Laundry > Washing Machine",
])

classifier = AdaptiveClassifier(
    taxonomy=crn_taxonomy,
    provider="vertex",
    index_path="./crn_classifier",
    provider_kwargs={"project_id": "my-gcp-project", "region": "us-east5"},
)

items = [
    "oak bkshf 5-shelf",
    "Herman Miller Aeron",
    "Samsung 65in QLED",
    "ikea kallax",
    "dell latitude 5540",
]

results = classifier.classify(items)
```

### POS Product Hierarchy

```python
from adaptive_classifier import AdaptiveClassifier, create_normalizer

menu = {
    "Burgers": {
        "Beef": ["Hamburger", "Cheeseburger", "Bacon Burger", "Double Burger"],
        "Chicken": ["Chicken Burger", "Spicy Chicken", "Grilled Chicken"],
        "Plant": ["Veggie Burger", "Beyond Burger"],
    },
    "Sides": {
        "Fries": ["Regular Fries", "Sweet Potato Fries", "Poutine"],
        "Salads": ["Garden Salad", "Caesar Salad", "Coleslaw"],
    },
    "Drinks": {
        "Hot": ["Coffee", "Tea", "Hot Chocolate"],
        "Cold": ["Soda", "Iced Tea", "Milkshake", "Water"],
    },
}

classifier = AdaptiveClassifier(
    taxonomy=menu,
    provider="bedrock",
    index_path="./pos_classifier",
    normalizer=create_normalizer(
        abbreviations={
            "chz": "cheese", "brgr": "burger", "dbl": "double",
            "reg": "regular", "sw pot": "sweet potato",
        }
    ),
)

pos_entries = [
    "dbl chz brgr",
    "reg fry",
    "lg coff blk",
    "spcy chkn sndwch",
    "grdn salad",
    "sw pot fry",
]

results = classifier.classify(pos_entries)
```

## Pluggable LLM Backend

```python
# Any callable works
def my_custom_llm(items, system_prompt, user_prompt):
    # Call OpenAI, local model, whatever
    response = my_api.complete(system=system_prompt, user=user_prompt)
    return response.text  # Must return JSON string

classifier = AdaptiveClassifier(
    taxonomy=my_taxonomy,
    provider=my_custom_llm,
)

# Or implement the LLMProvider protocol directly
from adaptive_classifier import LLMProvider

class MyProvider:
    def classify_batch(self, items, taxonomy_prompt, batch_size=50):
        # Your implementation
        return [{"input": item, "category": "..."} for item in items]
```

## Pluggable Vector Store

FAISS is the default, but you can swap in any vector backend by implementing the `VectorStore` protocol:

```python
from adaptive_classifier import AdaptiveClassifier, VectorStore
import numpy as np
from pathlib import Path

class MyVectorStore:
    """Drop-in replacement - e.g. Pinecone, Qdrant, Annoy, etc."""

    @property
    def size(self) -> int: ...
    def add(self, vectors: np.ndarray) -> None: ...
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...
    def reset(self) -> None: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...

classifier = AdaptiveClassifier(
    taxonomy=my_taxonomy,
    vector_store=MyVectorStore(),
)
```

## Pre-seeding with Known Mappings

```python
# If you already have labeled data, seed the index directly
classifier.add_examples({
    "WALMART SUPERCENTER": "Food > Groceries",
    "COSTCO WHOLESALE": "Food > Groceries",
    "NETFLIX.COM": "Entertainment > Streaming",
    "SPOTIFY": "Entertainment > Streaming",
})
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence_threshold` | `0.65` | Below this -> LLM fallback |
| `k_neighbors` | `5` | Neighbors for majority voting |
| `llm_batch_size` | `50` | Items per LLM API call |
| `auto_feedback` | `True` | Feed LLM results back to index |
| `auto_save` | `True` | Save index after each classify() |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `vector_store` | `None` (FAISS) | Custom `VectorStore` backend |

## Taxonomy Formats

```python
# Nested dict
taxonomy = {"Category": {"Subcategory": ["Leaf1", "Leaf2"]}}

# Flat path list
taxonomy = ["Category > Subcategory > Leaf1", "Category > Subcategory > Leaf2"]

# From file
classifier = AdaptiveClassifier(taxonomy="./taxonomy.json")
classifier = AdaptiveClassifier(taxonomy="./taxonomy.yaml")
```

## Architecture

```
adaptive_classifier/
├── classifier.py      # AdaptiveClassifier orchestrator
├── taxonomy.py        # Taxonomy tree management
├── index.py           # Vector index + persistence + feedback
├── vector_stores.py   # Pluggable vector store backends (FAISS default)
├── embeddings.py      # Embedding provider abstraction
├── providers.py       # Pluggable LLM backends
├── normalizer.py      # Text normalization / abbreviation expansion
└── types.py           # Classification, BatchStats, etc.
```

## Development

### Setup

```bash
# Clone the repo
git clone https://github.com/johncarpenter/adaptive-simple-text-classifier.git
cd adaptive-simple-text-classifier

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

### Testing

```bash
# Run unit tests
uv run pytest -v

# Run the benchmark (requires ANTHROPIC_API_KEY for hybrid mode)
uv run python example/benchmark.py --embedding-only  # no API key needed
uv run python example/benchmark.py                   # full hybrid benchmark
uv run python example/benchmark.py --runs 3           # watch the learning curve
```

See [`example/README.md`](example/README.md) for benchmark details and dataset setup.

### Linting

```bash
uv run ruff check .
uv run ruff format .
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Releasing

Releases are published to [PyPI](https://pypi.org/project/adaptive-simple-text-classifier/) automatically when a GitHub release is created.

To create a new release:

1. Update the version in `pyproject.toml` and `adaptive_classifier/__init__.py`
2. Commit: `git commit -am "Bump version to X.Y.Z"`
3. Tag: `git tag vX.Y.Z`
4. Push: `git push origin main --tags`
5. Create a [GitHub release](https://github.com/johncarpenter/adaptive-simple-text-classifier/releases/new) from the tag

The [publish workflow](.github/workflows/publish.yml) will build and upload to PyPI using trusted publishing.

## License

MIT - see [LICENSE](LICENSE) for details.
