"""Configuration management for memex.

This module contains all configurable constants for the knowledge base.
Magic numbers are documented here rather than scattered throughout the codebase.
"""

import os
from pathlib import Path


def get_kb_root() -> Path:
    """Get the knowledge base root directory."""
    root = os.environ.get("MEMEX_KB_ROOT")
    if root:
        return Path(root)
    # Default to kb/ relative to package
    return Path(__file__).parent.parent.parent / "kb"


def get_index_root() -> Path:
    """Get the search index root directory."""
    root = os.environ.get("MEMEX_INDEX_ROOT")
    if root:
        return Path(root)
    # Default to .indices/ relative to package
    return Path(__file__).parent.parent.parent / ".indices"


# =============================================================================
# Embedding Model
# =============================================================================

# Sentence-transformers model for semantic embeddings.
# MiniLM is a good balance of speed and quality for knowledge base search.
# Produces 384-dimensional embeddings, trained on 1B+ sentence pairs.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =============================================================================
# Search Limits
# =============================================================================

# Default number of results returned by search
DEFAULT_SEARCH_LIMIT = 10

# Maximum number of results allowed (prevents expensive queries)
MAX_SEARCH_LIMIT = 50

# Maximum directory traversal depth when searching for .kbcontext files
# Prevents infinite loops on circular symlinks or unusual filesystems.
# 50 levels is far beyond typical project depths (5-10 levels).
MAX_CONTEXT_SEARCH_DEPTH = 50

# Maximum results to hydrate with full document content
# Higher values increase response size; keep small for API performance
MAX_CONTENT_RESULTS = 20


# =============================================================================
# Hybrid Search (Reciprocal Rank Fusion)
# =============================================================================

# RRF constant for combining keyword and semantic search rankings.
# Formula: score(d) = sum(1 / (k + rank)) across ranking lists.
# Higher k values reduce the impact of rank differences.
# k=60 is the standard value from the RRF paper (Cormack et al., 2009).
RRF_K = 60


# =============================================================================
# Search Ranking Boosts
# =============================================================================

# Boost per matching tag in query (e.g., searching "python" boosts entries tagged "python")
# Applied additively: 2 matching tags = +0.10 boost
TAG_MATCH_BOOST = 0.05

# Boost for entries created from the current project (source_project matches)
# Helps surface project-specific documentation when working within that project
PROJECT_CONTEXT_BOOST = 0.15

# Boost for entries matching .kbcontext path patterns
# Slightly lower than project boost to prioritize exact project matches
KB_PATH_CONTEXT_BOOST = 0.12


# =============================================================================
# Link and Tag Suggestions
# =============================================================================

# Minimum semantic similarity score for suggesting links between entries
# 0.5 = moderate similarity, filters out weak connections
LINK_SUGGESTION_MIN_SCORE = 0.5

# Minimum similarity for including entries in tag frequency analysis
# Lower than link threshold to capture broader context
TAG_SUGGESTION_MIN_SCORE = 0.3

# Score weight for tags from semantically similar entries
# Contributes to tag frequency ranking when suggesting tags for new entries
SIMILAR_ENTRY_TAG_WEIGHT = 0.5
