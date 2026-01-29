"""Project context discovery and loading for memex.

This module provides context-aware behavior when working within a project directory.
A .kbconfig file tells the KB which paths are most relevant for that project.

Example .kbconfig file:
    primary: projects/memex    # Default write directory
    paths:                           # Boost these in search (supports globs)
      - projects/memex
      - guides/*
      - infrastructure/*
    default_tags:                    # Suggested for new entries
      - memex
"""

import os
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import yaml

from .config import MAX_CONTEXT_SEARCH_DEPTH

# KB config filename (marks a directory as a KB root)
LOCAL_KB_CONFIG_FILENAME = ".kbconfig"

# Default project KB directory name (scope=project)
LOCAL_KB_DIR = "kb"

# User KB directory (scope=user)
USER_KB_DIR = Path.home() / ".memex" / "kb"

# Cache for context discovery (per-session)
_context_cache: dict[str, "KBContext | None"] = {}

# Cache for .kbconfig loading (per-session)
_kbconfig_cache: dict[str, "KBConfig | None"] = {}


@dataclass
class KBConfig:
    """KB configuration from .kbconfig file.

    This config is stored inside the KB directory itself (e.g., kb/.kbconfig)
    and configures behavior for that specific KB.
    """

    default_tags: list[str] = field(default_factory=list)
    """Suggested tags for entries created in this KB."""

    exclude: list[str] = field(default_factory=list)
    """Glob patterns for files to exclude from indexing."""

    source_file: Path | None = None
    """Path to the .kbconfig file that was loaded."""

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_file: Path | None = None) -> "KBConfig":
        """Create KBConfig from parsed YAML dict."""
        return cls(
            default_tags=data.get("default_tags", []),
            exclude=data.get("exclude", []),
            source_file=source_file,
        )


def load_kbconfig(kb_path: Path) -> KBConfig | None:
    """Load .kbconfig from a KB directory.

    Args:
        kb_path: Path to the KB directory (e.g., kb/ or ~/.memex/kb/).

    Returns:
        KBConfig if found and valid, None otherwise.
    """
    config_file = kb_path / LOCAL_KB_CONFIG_FILENAME

    if not config_file.exists():
        return None

    try:
        content = config_file.read_text(encoding="utf-8")
        data = yaml.safe_load(content)

        # Handle empty file or all-comments file
        if data is None:
            return KBConfig(source_file=config_file)

        if not isinstance(data, dict):
            return None

        return KBConfig.from_dict(data, source_file=config_file)

    except (OSError, yaml.YAMLError):
        return None


def get_kbconfig(kb_path: Path) -> KBConfig | None:
    """Get KB config (cached).

    Args:
        kb_path: Path to the KB directory.

    Returns:
        KBConfig if found and valid, None otherwise.
    """
    cache_key = str(kb_path.resolve())

    if cache_key not in _kbconfig_cache:
        _kbconfig_cache[cache_key] = load_kbconfig(kb_path)

    return _kbconfig_cache[cache_key]


def clear_kbconfig_cache() -> None:
    """Clear the kbconfig cache. Useful for testing or after .kbconfig changes."""
    _kbconfig_cache.clear()


@dataclass
class KBContext:
    """Project context configuration from .kbconfig file."""

    primary: str | None = None
    """Default directory for new entries (e.g., 'projects/memex')."""

    paths: list[str] = field(default_factory=list)
    """Paths to boost in search results. Supports glob patterns (* and **)."""

    default_tags: list[str] = field(default_factory=list)
    """Suggested tags for new entries created in this context."""

    project: str | None = None
    """Override for project name (auto-detected from directory if not set)."""

    project_kb: str | None = None
    """Relative path to KB directory for this project (e.g., './kb')."""

    publish_base_url: str | None = None
    """Base URL for published site (e.g., '/repo-name' for GitHub Pages subdirectory)."""

    publish_index_entry: str | None = None
    """Entry path to use as landing page (e.g., 'guides/welcome')."""

    source_file: Path | None = None
    """Path to the .kbconfig file that was loaded."""

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_file: Path | None = None) -> "KBContext":
        """Create KBContext from parsed YAML dict."""
        return cls(
            primary=data.get("primary"),
            paths=data.get("paths", []),
            default_tags=data.get("default_tags", []),
            project=data.get("project"),
            project_kb=data.get("project_kb"),
            publish_base_url=data.get("publish_base_url"),
            publish_index_entry=data.get("publish_index_entry"),
            source_file=source_file,
        )

    def get_project_name(self) -> str | None:
        """Get the project name, either from config or source directory."""
        if self.project:
            return self.project
        if self.source_file:
            return self.source_file.parent.name
        return None

    def get_all_boost_paths(self) -> list[str]:
        """Get all paths to boost, including primary if set."""
        paths = list(self.paths)
        # Auto-include primary in boost paths if not already present
        if self.primary and self.primary not in paths:
            # Check if primary matches any existing pattern
            already_matched = any(matches_glob(self.primary, p) for p in paths)
            if not already_matched:
                paths.append(self.primary)
        return paths


def matches_glob(path: str, pattern: str) -> bool:
    """Check if a KB entry path matches a glob pattern.

    Supports:
    - Exact prefix matching: 'projects/foo' matches 'projects/foo/bar.md'
    - Single-level wildcard: 'projects/*' matches 'projects/foo/bar.md'
    - Recursive wildcard: 'projects/**' matches any depth under projects/

    Args:
        path: KB entry path (e.g., 'projects/memex/docs.md')
        pattern: Glob pattern from .kbconfig

    Returns:
        True if path matches the pattern.
    """
    # Normalize paths (remove trailing slashes, .md extension for matching)
    path = path.rstrip("/")
    pattern = pattern.rstrip("/")

    # Remove .md extension for path and pattern matching
    path_normalized = path[:-3] if path.endswith(".md") else path
    pattern_normalized = pattern[:-3] if pattern.endswith(".md") else pattern

    # Handle ** recursive wildcard
    if "**" in pattern_normalized:
        # Convert ** to match any number of path segments
        # projects/** -> matches projects/foo, projects/foo/bar, etc.
        base = pattern_normalized.split("**")[0].rstrip("/")
        return path_normalized.startswith(base) or path_normalized == base.rstrip("/")

    # Handle * single-level wildcard
    if "*" in pattern_normalized:
        # Use fnmatch for pattern matching
        # projects/* matches projects/foo but not projects/foo/bar
        # For KB paths, we want to match if the pattern matches the start
        if fnmatch(path_normalized, pattern_normalized):
            return True
        # Also check if the pattern matches a prefix
        # e.g., 'infrastructure/*' should match 'infrastructure/docker/guide.md'
        pattern_prefix = pattern_normalized.rstrip("/*")
        if path_normalized.startswith(pattern_prefix + "/"):
            return True
        return False

    # Exact prefix match (no wildcards)
    # 'projects/foo' matches 'projects/foo/bar.md' and 'projects/foo.md'
    return path_normalized.startswith(pattern_normalized) or path_normalized == pattern_normalized


def discover_kb_context(start_dir: Path | None = None) -> KBContext | None:
    """Walk up from start_dir to find and parse .kbconfig file.

    Discovery order:
    1. Walk up from start_dir (or cwd) looking for .kbconfig
    2. Stop at first config file found
    Skips discovery entirely when MEMEX_SKIP_PROJECT_KB=1.

    Args:
        start_dir: Directory to start searching from. Defaults to cwd.

    Returns:
        KBContext if found and valid, None otherwise.
    """
    if os.environ.get("MEMEX_SKIP_PROJECT_KB"):
        return None

    # Walk up from start_dir looking for .kbconfig
    current = (start_dir or Path.cwd()).resolve()
    depth = 0

    while depth < MAX_CONTEXT_SEARCH_DEPTH:
        kbconfig_file = current / ".kbconfig"
        if kbconfig_file.exists():
            context = _load_kbconfig_as_context(kbconfig_file)
            if context:
                return context

        # Move up one directory
        parent = current.parent
        if parent == current:
            # Reached filesystem root
            break
        current = parent
        depth += 1

    return None


def _load_kbconfig_as_context(config_path: Path) -> KBContext | None:
    """Load .kbconfig and convert to KBContext.

    The new .kbconfig format supports all context fields plus kb_path.

    Args:
        config_path: Path to the .kbconfig file.

    Returns:
        KBContext if valid, None if parsing fails.
    """
    try:
        content = config_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)

        if not isinstance(data, dict):
            return None

        # Map .kbconfig fields to KBContext fields
        # boost_paths in new format -> paths in KBContext
        project_kb = data.get("project_kb")
        if not project_kb:
            project_kb = data.get("kb_path")

        context_data = {
            "primary": data.get("primary"),
            "paths": data.get("boost_paths", []),
            "default_tags": data.get("default_tags", []),
            "project": data.get("project"),
            "project_kb": project_kb,
            "publish_base_url": data.get("publish_base_url"),
            "publish_index_entry": data.get("publish_index_entry"),
        }

        return KBContext.from_dict(context_data, source_file=config_path)

    except (OSError, yaml.YAMLError):
        return None


def get_kb_context(start_dir: Path | None = None) -> KBContext | None:
    """Get the KB context for the current session (cached).

    Caches results per starting directory to avoid repeated filesystem walks.

    Args:
        start_dir: Directory to start searching from. Defaults to cwd.

    Returns:
        KBContext if found and valid, None otherwise.
    """
    # Use cwd as the cache key if start_dir not specified
    cache_key = str((start_dir or Path.cwd()).resolve())

    if cache_key not in _context_cache:
        _context_cache[cache_key] = discover_kb_context(start_dir)

    return _context_cache[cache_key]


def clear_context_cache() -> None:
    """Clear the context cache. Useful for testing or after .kbconfig changes."""
    _context_cache.clear()


def validate_context(context: KBContext, kb_root: Path) -> list[str]:
    """Validate a KBContext against the knowledge base.

    Checks that:
    - primary directory exists (or can be created)
    - paths reference valid locations (warning only)

    Args:
        context: The context to validate.
        kb_root: Root directory of the knowledge base.

    Returns:
        List of warning messages (empty if valid).
    """
    warnings: list[str] = []

    # Check primary directory
    if context.primary:
        primary_path = kb_root / context.primary
        if not primary_path.exists():
            warnings.append(
                f"Primary directory '{context.primary}' does not exist. "
                "It will be created when adding entries."
            )

    # Check paths (non-glob patterns only - globs can match future entries)
    for pattern in context.paths:
        if "*" in pattern:
            # Skip glob patterns - they may match future entries
            continue

        path = kb_root / pattern
        if not path.exists():
            # Check if it might be a file without .md extension
            md_path = kb_root / f"{pattern}.md"
            if not md_path.exists():
                warnings.append(f"Path '{pattern}' does not exist in the knowledge base.")

    return warnings
