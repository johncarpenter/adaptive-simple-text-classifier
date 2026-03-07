# Contributing to adaptive-classifier

Thanks for your interest in contributing! This guide will help you get started.

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Git

### Setup

```bash
git clone https://github.com/2lines-software/adaptive-classifier.git
cd adaptive-classifier
uv sync
```

This installs all dependencies including dev tools (pytest, ruff, pandas, pyarrow, anthropic).

### Verify your setup

```bash
uv run pytest -v
uv run ruff check .
```

## Development Workflow

### 1. Create a branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make your changes

- Follow existing code patterns and conventions
- Keep changes focused and minimal
- Add tests for new functionality

### 3. Run checks

```bash
# Tests
uv run pytest -v

# Linting
uv run ruff check .

# Formatting
uv run ruff format .
```

All checks must pass before submitting a PR.

### 4. Commit and push

Write clear commit messages that explain *why*, not just *what*.

```bash
git commit -m "Add support for custom distance metrics in vector search"
git push origin feature/your-feature-name
```

### 5. Open a pull request

- Target the `main` branch
- Include a clear description of what changed and why
- Reference any related issues

## What to Contribute

### Good first contributions

- Bug fixes with test cases
- Documentation improvements
- Additional test coverage
- New normalizer abbreviation sets for specific industries

### Feature ideas

- New vector store backends (Qdrant, Pinecone, Annoy, etc.)
- Additional LLM provider implementations
- Async classification support
- Batch processing optimizations

### Before starting large features

Open an issue first to discuss the approach. This avoids wasted effort if the design needs adjustment.

## Code Guidelines

### Style

- Follow [ruff](https://docs.astral.sh/ruff/) defaults (configured in `pyproject.toml`)
- Line length: 100 characters
- Target: Python 3.10+
- Use type hints for public APIs

### Testing

- Write tests for all new functionality
- Test behavior, not implementation details
- Use descriptive test names: `test_normalizer_strips_currency_symbols`
- Keep tests fast and deterministic (no network calls in unit tests)
- Tests go in `tests/` and follow existing patterns

### Architecture

- **Composition over inheritance** — use protocols and dependency injection
- **Keep it simple** — avoid premature abstractions
- **Explicit over implicit** — clear data flow, no magic

### Commit messages

- Use imperative mood: "Add feature" not "Added feature"
- First line: concise summary (under 72 characters)
- Body: explain why the change was made, not what (the diff shows what)

## Project Structure

```
adaptive_classifier/
├── classifier.py      # Main orchestrator (AdaptiveClassifier)
├── taxonomy.py        # Taxonomy tree parsing and rendering
├── index.py           # FAISS vector index + persistence
├── vector_stores.py   # Pluggable vector store protocol + FAISS impl
├── embeddings.py      # Embedding provider protocol + SentenceTransformer impl
├── providers.py       # LLM provider protocol + Anthropic/Vertex/Bedrock impls
├── normalizer.py      # Text normalization and abbreviation expansion
└── types.py           # Data classes (Classification, BatchStats, etc.)

tests/
└── test_core.py       # Unit tests

example/
├── benchmark.py       # Benchmark script
├── 0000.parquet       # Transaction dataset (not in repo, see example/README.md)
├── categories.json    # Category definitions
└── README.md          # Benchmark documentation
```

## Reporting Issues

- Use the [GitHub issue tracker](https://github.com/2lines-software/adaptive-classifier/issues)
- Include Python version, OS, and package version
- Provide a minimal reproduction if possible
- For bugs, include the full traceback

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
