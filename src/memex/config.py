"""Configuration management for memex.

This module contains all configurable constants for the knowledge base.
Magic numbers are documented here rather than scattered throughout the codebase.
"""

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(Exception):
    """Raised when required configuration is missing."""

    pass


def get_project_kb_root() -> Path | None:
    """Get the project-scope KB root directory.

    Returns:
        Path to project KB if found, None otherwise.

    Note:
        Set MEMEX_SKIP_PROJECT_KB=1 to disable project KB discovery.
        This is useful for testing to ensure isolation.
    """
    # Allow tests to disable project KB discovery
    if os.environ.get("MEMEX_SKIP_PROJECT_KB"):
        return None

    project_config = _discover_project_config()
    if project_config:
        config_path, kb_path = project_config
        return kb_path
    return None


def get_user_kb_root() -> Path | None:
    """Get the user-scope KB root directory.

    Checks MEMEX_USER_KB_ROOT env var first, falls back to ~/.memex/kb.

    Returns:
        Path to user KB if it exists, None otherwise.
    """
    # Check for explicit env var override
    env_root = os.environ.get("MEMEX_USER_KB_ROOT")
    if env_root:
        user_kb = Path(env_root)
        if user_kb.exists():
            return user_kb
        return None

    # Default location
    user_kb = Path.home() / ".memex" / "kb"
    if (user_kb / ".kbconfig").exists():
        return user_kb
    return None


def get_kb_roots(scope: str | None = None) -> list[Path]:
    """Get all active KB root directories.

    By default, returns both project and user KBs (additive scope).

    Args:
        scope: Filter to specific scope - "project", "user", or None for all.

    Returns:
        List of KB paths. Project KB comes first if present.
        Empty list if no KBs found.
    """
    project_kb = get_project_kb_root()
    user_kb = get_user_kb_root()

    if scope == "project":
        return [project_kb] if project_kb else []
    elif scope == "user":
        return [user_kb] if user_kb else []
    else:
        # Return both KBs
        roots = []
        if project_kb:
            roots.append(project_kb)
        if user_kb and user_kb not in roots:
            roots.append(user_kb)
        return roots


def get_kb_root_by_scope(scope: str) -> Path:
    """Get a specific KB root by scope name.

    Args:
        scope: Either "project" or "user".

    Returns:
        Path to the requested KB.

    Raises:
        ConfigurationError: If the requested scope KB doesn't exist.
        ValueError: If scope is not "project" or "user".
    """
    if scope == "project":
        kb_root = get_project_kb_root()
        if not kb_root:
            raise ConfigurationError(
                "No project KB found. Run 'mx init' in your project to create one."
            )
        return kb_root
    elif scope == "user":
        kb_root = get_user_kb_root()
        if not kb_root:
            raise ConfigurationError(
                "No user KB found. Run 'mx init --user' to create one at ~/.memex/kb/"
            )
        return kb_root
    else:
        raise ValueError(f"Invalid scope '{scope}'. Must be 'project' or 'user'.")


def get_kb_root() -> Path:
    """Get the primary knowledge base root directory.

    This returns the first/primary KB for write operations.
    For read operations that should span multiple KBs, use get_kb_roots().

    Discovery order:
    1. Project KB: Walk up from cwd looking for .kbconfig with kb_path field
    2. User KB: ~/.memex/kb/ or MEMEX_USER_KB_ROOT if set
    3. Error with helpful message

    Raises:
        ConfigurationError: If no KB can be found.
    """
    # 1. Look for .kbconfig at project root with kb_path
    project_kb = get_project_kb_root()
    if project_kb:
        return project_kb

    # 2. Check for user-scope KB
    user_kb = get_user_kb_root()
    if user_kb:
        return user_kb

    # 3. No KB found
    raise ConfigurationError(
        "No knowledge base found. Options:\n"
        "  1. Run 'mx init' to create a project KB at ./kb/\n"
        "  2. Run 'mx init --user' to create a personal KB at ~/.memex/kb/\n"
        "  3. Set MEMEX_USER_KB_ROOT to an existing KB directory"
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


def get_kb_roots_for_indexing(scope: str | None = None) -> list[tuple[str | None, Path]]:
    """Get KB roots formatted for indexing (with scope labels).

    Args:
        scope: Filter to specific scope - "project", "user", or None for all.

    Returns:
        List of (scope, path) tuples for use with HybridSearcher.reindex().
    """
    project_kb = get_project_kb_root()
    user_kb = get_user_kb_root()

    # If only one KB exists, use single-KB mode (no scope prefix)
    if project_kb and not user_kb:
        return [(None, project_kb)]
    if user_kb and not project_kb:
        return [(None, user_kb)]

    # Multi-KB mode: filter by scope or return both
    if scope == "project":
        return [("project", project_kb)] if project_kb else []
    elif scope == "user":
        return [("user", user_kb)] if user_kb else []
    else:
        # Return both KBs
        result = []
        if project_kb:
            result.append(("project", project_kb))
        if user_kb:
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


# =============================================================================
# Auto Semantic Links
# =============================================================================

# Enable automatic bidirectional semantic link creation on entry add/update.
# When enabled, new entries will be linked to similar existing entries.
SEMANTIC_LINK_ENABLED = True

# Minimum similarity score for creating semantic links (0.0-1.0).
# 0.6 = moderate similarity, filters out weak connections.
SEMANTIC_LINK_MIN_SCORE = 0.6

# Maximum number of semantic links to create per entry.
# Higher values create denser link graphs but may add noise.
SEMANTIC_LINK_K = 5

# Minimum similarity score for triggering memory evolution on neighbors.
# Higher than SEMANTIC_LINK_MIN_SCORE to focus evolution on strong connections.
MEMORY_EVOLUTION_MIN_SCORE = 0.7


# =============================================================================
# LLM Provider Configuration
# =============================================================================


@dataclass
class LLMConfig:
    """Configuration for LLM provider and models.

    Applies to all LLM-powered features (evolution, etc.).
    Provider is auto-detected from available API keys unless explicitly set.

    Example .kbconfig:
        llm:
          provider: anthropic  # or "openrouter"
          model: claude-3.5-haiku
          models:
            memory_evolution: claude-3.5-haiku
    """

    provider: str | None = None
    """LLM provider: 'anthropic' or 'openrouter'. Auto-detected if None."""

    model: str = "claude-3.5-haiku"
    """Default model for all LLM features."""

    models: dict[str, str] | None = None
    """Per-feature model overrides. Keys: 'memory_evolution'"""

    def get_model(self, feature: str) -> str:
        """Get the model for a specific feature.

        Args:
            feature: Feature name like 'memory_evolution'.

        Returns:
            Model name, using feature-specific override if set.
        """
        if self.models and feature in self.models:
            return self.models[feature]
        return self.model


def get_llm_config() -> LLMConfig:
    """Load LLM config from .kbconfig.

    Config is loaded from the project .kbconfig file's llm section.
    Returns default config if not configured or KB not found.

    Example .kbconfig:
        llm:
          provider: anthropic
          model: claude-3.5-haiku
          models:
            memory_evolution: claude-3.5-haiku

    Returns:
        LLMConfig with loaded or default values.
    """
    import yaml

    try:
        project_config = _discover_project_config()
        if not project_config:
            return LLMConfig()

        config_path, _ = project_config
        content = config_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content) or {}

        llm_data = data.get("llm", {})
        if not llm_data:
            return LLMConfig()

        return LLMConfig(
            provider=llm_data.get("provider"),
            model=llm_data.get("model", "claude-3.5-haiku"),
            models=llm_data.get("models"),
        )
    except (OSError, yaml.YAMLError):
        return LLMConfig()


# =============================================================================
# Memory Evolution (A-Mem style)
# =============================================================================

@dataclass
class MemoryEvolutionConfig:
    """Configuration for LLM-driven memory evolution.

    When a new entry is added and links to neighbors, the LLM analyzes
    the relationship and suggests keyword/context updates for neighbors.
    This makes the knowledge graph "learn" from new connections.

    Evolution is queued (non-blocking) and processed by `mx evolve`.
    Auto-triggers can spawn background evolution processing.

    Note: Model configuration has moved to the top-level llm: section.
    The model field here is kept for backwards compatibility but
    llm.models.memory_evolution takes precedence if set.
    """

    enabled: bool = False
    """Whether memory evolution is active. Requires ANTHROPIC_API_KEY or OPENROUTER_API_KEY."""

    model: str = "anthropic/claude-3-5-haiku"
    """Model ID for evolution analysis. Deprecated: prefer llm.models.memory_evolution."""

    min_score: float = MEMORY_EVOLUTION_MIN_SCORE
    """Minimum similarity score to trigger evolution (0.0-1.0)."""

    max_keywords_per_neighbor: int = 3
    """Maximum keywords to add to a neighbor per evolution."""

    batch_neighbors: bool = True
    """Batch multiple neighbors into single LLM call when possible."""

    auto_probability: float = 0.0
    """Probability (0.0-1.0) to spawn background `mx evolve` after add."""

    auto_queue_threshold: int = 0
    """Spawn background `mx evolve` when queue exceeds this size (0=disabled)."""

    strengthen_on_add: bool = False
    """Run strengthen analysis synchronously during add_entry() when neighbors found.

    Conservative default (False) - user must opt-in via .kbconfig.
    When enabled, adds an LLM call to add_entry() which may slow down the operation."""


def get_memory_evolution_config() -> MemoryEvolutionConfig:
    """Load memory evolution config from .kbconfig.

    Config is loaded from the project .kbconfig file's memory_evolution section.
    Returns default (disabled) config if not configured or KB not found.

    Example .kbconfig:
        memory_evolution:
          enabled: true
          model: anthropic/claude-3-5-haiku
          min_score: 0.7

    Returns:
        MemoryEvolutionConfig with loaded or default values.
    """
    import yaml

    try:
        project_config = _discover_project_config()
        if not project_config:
            return MemoryEvolutionConfig()

        config_path, _ = project_config
        content = config_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content) or {}

        evolution_data = data.get("memory_evolution", {})
        if not evolution_data:
            return MemoryEvolutionConfig()

        return MemoryEvolutionConfig(
            enabled=evolution_data.get("enabled", False),
            model=evolution_data.get("model", "anthropic/claude-3-5-haiku"),
            min_score=evolution_data.get("min_score", MEMORY_EVOLUTION_MIN_SCORE),
            max_keywords_per_neighbor=evolution_data.get("max_keywords_per_neighbor", 3),
            batch_neighbors=evolution_data.get("batch_neighbors", True),
            auto_probability=evolution_data.get("auto_probability", 0.0),
            auto_queue_threshold=evolution_data.get("auto_queue_threshold", 0),
            strengthen_on_add=evolution_data.get("strengthen_on_add", True),
        )
    except (OSError, yaml.YAMLError):
        return MemoryEvolutionConfig()


# =============================================================================
# A-Mem Strict Mode
# =============================================================================


class AMEMStrictError(ValueError):
    """Raised when A-Mem strict mode is enabled but keywords are missing."""

    pass


AMEM_STRICT_ERROR_MESSAGE = """\
A-Mem strict mode is enabled but --keywords was not provided.

To fix this, add keywords that capture key concepts in your entry:

    mx add --title="Your Title" \\
           --tags="tag1,tag2" \\
           --keywords="concept1,concept2,concept3" \\
           --content="..."

Keywords should be:
  • Key concepts mentioned in the content (e.g., "REST", "caching")
  • Related terms not explicitly mentioned (e.g., "API design")
  • Domain-specific terminology (e.g., "microservices")

Typically 3-7 keywords work well. They improve search relevance
and semantic linking accuracy.

To disable this check: set amem_strict: false in .kbconfig"""


def is_amem_strict_enabled() -> bool:
    """Check if A-Mem strict mode is enabled in .kbconfig.

    When enabled, add_entry() and update_entry() will fail if keywords
    are not provided, helping LLMs remember to populate this field.

    Example .kbconfig:
        amem_strict: true

    Returns:
        True if amem_strict is enabled, False otherwise.
    """
    import yaml

    try:
        project_config = _discover_project_config()
        if not project_config:
            return False

        config_path, _ = project_config
        content = config_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content) or {}

        return bool(data.get("amem_strict", False))
    except (OSError, yaml.YAMLError):
        return False
