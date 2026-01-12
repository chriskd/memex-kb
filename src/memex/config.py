"""Configuration management for memex.

This module contains all configurable constants for the knowledge base.
Magic numbers are documented here rather than scattered throughout the codebase.
"""

import os
from pathlib import Path


class ConfigurationError(Exception):
    """Raised when required configuration is missing."""

    pass


def get_project_kb_root() -> Path | None:
    """Get the project-scope KB root directory.

    Returns:
        Path to project KB if found, None otherwise.
    """
    project_config = _discover_project_config()
    if project_config:
        config_path, kb_path = project_config
        return kb_path
    return None


def get_user_kb_root() -> Path | None:
    """Get the user-scope KB root directory.

    Returns:
        Path to user KB if it exists, None otherwise.
    """
    user_kb = Path.home() / ".memex" / "kb"
    if (user_kb / ".kbconfig").exists():
        return user_kb
    return None


def get_kb_roots(project_only: bool = False) -> list[Path]:
    """Get all active KB root directories.

    By default, returns both project and user KBs (additive scope).
    Use project_only=True to restrict to project KB only.

    Args:
        project_only: If True, only return project KB (not user KB).

    Returns:
        List of KB paths. Project KB comes first if present.
        Empty list if no KBs found.
    """
    # Check for explicit env var override (single KB mode)
    root = os.environ.get("MEMEX_KB_ROOT")
    if root:
        return [Path(root)]

    roots = []

    # Project KB (if in a project with .kbconfig)
    project_kb = get_project_kb_root()
    if project_kb:
        roots.append(project_kb)

    # User KB (unless project_only is set)
    if not project_only:
        user_kb = get_user_kb_root()
        if user_kb and user_kb not in roots:
            roots.append(user_kb)

    return roots


def get_kb_root() -> Path:
    """Get the primary knowledge base root directory.

    This returns the first/primary KB for write operations.
    For read operations that should span multiple KBs, use get_kb_roots().

    Discovery order:
    1. MEMEX_KB_ROOT environment variable (explicit override)
    2. Walk up from cwd looking for .kbconfig with kb_path field
    3. ~/.memex/kb/ if it exists with .kbconfig (user scope)
    4. Error with helpful message

    Raises:
        ConfigurationError: If no KB can be found.
    """
    # 1. Explicit env var takes precedence
    root = os.environ.get("MEMEX_KB_ROOT")
    if root:
        return Path(root)

    # 2. Look for .kbconfig at project root with kb_path
    project_kb = get_project_kb_root()
    if project_kb:
        return project_kb

    # 3. Check for user-scope KB
    user_kb = get_user_kb_root()
    if user_kb:
        return user_kb

    # 4. No KB found
    raise ConfigurationError(
        "No knowledge base found. Options:\n"
        "  1. Run 'mx init' to create a project KB at ./kb/\n"
        "  2. Run 'mx init --user' to create a personal KB at ~/.memex/kb/\n"
        "  3. Set MEMEX_KB_ROOT to an existing KB directory"
    )


def parse_scoped_path(path: str) -> tuple[str | None, str]:
    """Parse a scoped path into (scope, relative_path).

    Paths can have scope prefixes like @project/ or @user/.
    Paths without prefixes return (None, path).

    Args:
        path: Path string, possibly with scope prefix.

    Returns:
        Tuple of (scope, relative_path). Scope is None if no prefix.

    Examples:
        "@project/guides/setup.md" -> ("project", "guides/setup.md")
        "@user/personal/notes.md" -> ("user", "personal/notes.md")
        "guides/setup.md" -> (None, "guides/setup.md")
    """
    if path.startswith("@"):
        # Find the first slash after @
        slash_idx = path.find("/")
        if slash_idx > 1:
            scope = path[1:slash_idx]
            relative = path[slash_idx + 1:]
            return (scope, relative)
    return (None, path)


def resolve_scoped_path(path: str) -> Path:
    """Resolve a scoped path to an absolute filesystem path.

    Args:
        path: Path string, possibly with scope prefix.

    Returns:
        Absolute Path to the entry.

    Raises:
        ConfigurationError: If scope KB not found.
    """
    scope, relative = parse_scoped_path(path)

    if scope == "project":
        kb_root = get_project_kb_root()
        if not kb_root:
            raise ConfigurationError("No project KB found for @project/ path")
    elif scope == "user":
        kb_root = get_user_kb_root()
        if not kb_root:
            raise ConfigurationError("No user KB found for @user/ path")
    else:
        # No scope prefix - use primary KB
        kb_root = get_kb_root()

    return kb_root / relative


def get_kb_roots_for_indexing(project_only: bool = False) -> list[tuple[str, Path]]:
    """Get KB roots formatted for indexing (with scope labels).

    Args:
        project_only: If True, only return project KB.

    Returns:
        List of (scope, path) tuples for use with HybridSearcher.reindex().
    """
    # Check for explicit env var override first (single KB mode, used in tests)
    root = os.environ.get("MEMEX_KB_ROOT")
    if root:
        return [(None, Path(root))]

    result = []

    project_kb = get_project_kb_root()
    user_kb = get_user_kb_root()

    # If only one KB exists, use single-KB mode (no scope prefix)
    if project_kb and not user_kb:
        return [(None, project_kb)]
    if user_kb and not project_kb:
        return [(None, user_kb)]

    # Multi-KB mode: add scope prefixes
    if project_kb:
        result.append(("project", project_kb))
    if not project_only and user_kb:
        result.append(("user", user_kb))

    return result


def _discover_project_config(start_dir: Path | None = None, max_depth: int = 10) -> tuple[Path, Path] | None:
    """Walk up from start_dir looking for .kbconfig with kb_path.

    Args:
        start_dir: Directory to start from (defaults to cwd)
        max_depth: Maximum directories to traverse up

    Returns:
        Tuple of (config_path, kb_path) if found, None otherwise.
    """
    import yaml

    current = Path(start_dir or os.getcwd()).resolve()

    for _ in range(max_depth):
        config_file = current / ".kbconfig"
        if config_file.exists():
            try:
                content = config_file.read_text(encoding="utf-8")
                data = yaml.safe_load(content) or {}
                if "kb_path" in data:
                    kb_path = (current / data["kb_path"]).resolve()
                    if kb_path.exists() and kb_path.is_dir():
                        return (config_file, kb_path)
            except (OSError, yaml.YAMLError):
                pass

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
