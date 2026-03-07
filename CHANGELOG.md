# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-14

### Added

- Initial release
- `AdaptiveClassifier` with hybrid embedding + LLM classification
- FAISS vector store with persistence (`.faiss` + `.meta.json`)
- Automatic feedback loop: LLM results feed back into the index
- Pluggable LLM providers: Anthropic, Vertex AI, Bedrock, callable
- Pluggable vector store protocol (FAISS default)
- Pluggable embedding providers (SentenceTransformer default)
- Taxonomy support: nested dict, flat path list, JSON, YAML
- Text normalizer with abbreviation expansion, code/amount stripping
- Majority voting with k-nearest neighbors
- Batch classification with stats tracking
- Index persistence and auto-save
