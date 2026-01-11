# Contributing to Memex

Thanks for your interest in contributing to Memex! This document provides guidelines and information for contributors.

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package manager)

### Getting Started

```bash
# Clone the repository
git clone https://github.com/chriskd/memex.git
cd memex

# Install dependencies (including dev tools)
uv sync --dev

# Run tests
uv run pytest

# Run the MCP server
uv run memex

# Run the CLI
uv run mx --help

# Run the web explorer
uv run memex-web
```

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check linting
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

Key style points:
- Line length: 100 characters
- Python 3.11+ features encouraged
- Type hints for function signatures
- Docstrings for public APIs

## Testing

Tests use pytest with pytest-asyncio for async support.

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_search_content.py

# Run tests matching a pattern
uv run pytest -k "test_search"
```

## Pull Request Process

1. **Fork and branch**: Create a feature branch from `main`
2. **Make changes**: Keep commits focused and atomic
3. **Test**: Ensure all tests pass
4. **Lint**: Run `uv run ruff check .` and fix issues
5. **Document**: Update docstrings and README if needed
6. **Submit**: Open a PR with a clear description

### Commit Messages

Use conventional commit format:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `refactor:` Code changes that don't add features or fix bugs
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Example: `feat: add tag filtering to search`

## Project Structure

```
memex/
├── src/memex/
│   ├── server.py       # FastMCP server (MCP protocol)
│   ├── cli.py          # Command-line interface
│   ├── core.py         # Core business logic
│   ├── config.py       # Configuration constants
│   ├── models.py       # Pydantic data models
│   ├── indexer/        # Search indexing (Whoosh + ChromaDB)
│   ├── parser/         # Markdown parsing and link extraction
│   └── webapp/         # Web explorer (FastAPI + static)
├── tests/              # Test suite
└── kb/                 # Sample knowledge base entries
```

## Architecture Notes

- **MCP Server** (`server.py`): Thin wrapper around core functions
- **CLI** (`cli.py`): Token-efficient alternative to MCP for automation
- **Core** (`core.py`): All business logic, no protocol dependencies
- **Hybrid Search**: Combines BM25 (keyword) and semantic search via RRF

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
