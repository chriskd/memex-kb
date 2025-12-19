"""Configuration management for voidlabs-kb."""

import os
from pathlib import Path


def get_kb_root() -> Path:
    """Get the knowledge base root directory."""
    root = os.environ.get("KB_ROOT")
    if root:
        return Path(root)
    # Default to kb/ relative to package
    return Path(__file__).parent.parent.parent / "kb"


def get_index_root() -> Path:
    """Get the search index root directory."""
    root = os.environ.get("INDEX_ROOT")
    if root:
        return Path(root)
    # Default to .indices/ relative to package
    return Path(__file__).parent.parent.parent / ".indices"


# Embedding model for semantic search
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Search configuration
DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 50

# RRF constant for hybrid search
RRF_K = 60
