# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **CLI** (`mx`): Token-efficient command-line interface
  - Search, browse, and modify KB entries
  - Designed for automation and scripting
  - Minimal output format reduces context usage

- **Hybrid Search**: Best-of-both-worlds search approach
  - BM25 keyword search via Whoosh
  - Semantic search via ChromaDB + sentence-transformers
  - Reciprocal Rank Fusion (RRF) for result merging
  - Context-aware boosting for project relevance

- **Bidirectional Links**: Wiki-style `[[link]]` syntax
  - Auto-resolved to existing entries
  - Backlink tracking across the KB
  - Stale link detection in health checks

### Technical

- Python 3.11+ with type hints throughout
- Pydantic models for all data structures
- Async-first architecture
- Configurable via environment variables
