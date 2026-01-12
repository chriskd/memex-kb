"""Configuration management for memex.

This module contains all configurable constants for the knowledge base.
Magic numbers are documented here rather than scattered throughout the codebase.
"""

import os
from pathlib import Path


class ConfigurationError(Exception):
    """Raised when required configuration is missing."""

    pass


def get_kb_root() -> Path:
    """Get the knowledge base root directory.

    Discovery order:
    1. MEMEX_KB_ROOT environment variable (explicit override)
    2. Walk up from cwd looking for kb/.kbconfig (project scope)
    3. ~/.memex/kb/ if it exists with .kbconfig (user scope)
    4. Error with helpful message

    Raises:
        ConfigurationError: If no KB can be found.
    """
    # 1. Explicit env var takes precedence
    root = os.environ.get("MEMEX_KB_ROOT")
    if root:
        return Path(root)

    # 2. Look for project-scope KB (walk up from cwd)
    project_kb = _discover_project_kb()
    if project_kb:
        return project_kb

    # 3. Check for user-scope KB
    user_kb = Path.home() / ".memex" / "kb"
    if (user_kb / ".kbconfig").exists():
        return user_kb

    # 4. No KB found
    raise ConfigurationError(
        "No knowledge base found. Options:\n"
        "  1. Run 'mx init' to create a project KB at ./kb/\n"
        "  2. Run 'mx init --user' to create a personal KB at ~/.memex/kb/\n"
        "  3. Set MEMEX_KB_ROOT to an existing KB directory"
    )


def _discover_project_kb(start_dir: Path | None = None, max_depth: int = 10) -> Path | None:
    """Walk up from start_dir looking for kb/.kbconfig.

    Args:
        start_dir: Directory to start from (defaults to cwd)
        max_depth: Maximum directories to traverse up

    Returns:
        Path to KB root if found, None otherwise.
    """
    current = Path(start_dir or os.getcwd()).resolve()

    for _ in range(max_depth):
        kb_dir = current / "kb"
        if (kb_dir / ".kbconfig").exists():
            return kb_dir

        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent

    return None


def get_index_root() -> Path:
    """Get the search index root directory.

    Discovery order:
    1. MEMEX_INDEX_ROOT environment variable (explicit override)
    2. {kb_root}/.indices/ for discovered KBs

    Raises:
        ConfigurationError: If no index root can be determined.
    """
    # 1. Explicit env var takes precedence
    root = os.environ.get("MEMEX_INDEX_ROOT")
    if root:
        return Path(root)

    # 2. Use .indices/ inside KB root
    try:
        kb_root = get_kb_root()
        return kb_root / ".indices"
    except ConfigurationError:
        raise ConfigurationError(
            "MEMEX_INDEX_ROOT environment variable is not set and no KB found. "
            "Set it to the path where search indices should be stored."
        )


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
# Semantic Search Thresholds
# =============================================================================

# Minimum raw semantic similarity score for search results.
# Results below this threshold are filtered out to prevent misleading high scores
# for gibberish or unrelated queries. Cosine similarity ranges from -1 to 1,
# but for normalized embeddings, practical range is 0 to 1.
# 0.3 filters very weak matches while allowing exploratory searches.
SEMANTIC_MIN_SIMILARITY = 0.3

# Higher threshold for --strict mode.
# 0.5 requires moderate similarity, filtering out most speculative matches.
# Use this when precision matters more than recall.
SEMANTIC_STRICT_SIMILARITY = 0.5


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


# =============================================================================
# Duplicate Detection
# =============================================================================

# Minimum semantic similarity score to flag as potential duplicate
# 0.75 = high similarity, requires near-identical content
DUPLICATE_DETECTION_MIN_SCORE = 0.75

# Maximum number of potential duplicates to return
DUPLICATE_DETECTION_LIMIT = 3
