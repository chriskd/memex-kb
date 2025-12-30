"""Project context discovery and loading for voidlabs-kb.

This module provides context-aware behavior when working within a project directory.
A .kbcontext file tells the KB which paths are most relevant for that project.

Example .kbcontext file:
    primary: projects/voidlabs-kb    # Default write directory
    paths:                           # Boost these in search (supports globs)
      - projects/voidlabs-kb
      - tooling/beads
      - infrastructure/*
    default_tags:                    # Suggested for new entries
      - voidlabs-kb
"""

import os
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import yaml

# Context filename
CONTEXT_FILENAME = ".kbcontext"

# Cache for context discovery (per-session)
_context_cache: dict[str, "KBContext | None"] = {}


@dataclass
class KBContext:
    """Project context configuration from .kbcontext file."""

    primary: str | None = None
    """Default directory for new entries (e.g., 'projects/voidlabs-kb')."""

    paths: list[str] = field(default_factory=list)
    """Paths to boost in search results. Supports glob patterns (* and **)."""

    default_tags: list[str] = field(default_factory=list)
    """Suggested tags for new entries created in this context."""

    project: str | None = None
    """Override for project name (auto-detected from directory if not set)."""

    source_file: Path | None = None
    """Path to the .kbcontext file that was loaded."""

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_file: Path | None = None) -> "KBContext":
        """Create KBContext from parsed YAML dict."""
        return cls(
            primary=data.get("primary"),
            paths=data.get("paths", []),
            default_tags=data.get("default_tags", []),
            project=data.get("project"),
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
        path: KB entry path (e.g., 'projects/voidlabs-kb/docs.md')
        pattern: Glob pattern from .kbcontext

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
    """Walk up from start_dir to find and parse .kbcontext file.

    Discovery order:
    1. Check VL_KB_CONTEXT environment variable for explicit path
    2. Walk up from start_dir (or cwd) looking for .kbcontext
    3. Stop at first .kbcontext file found

    Args:
        start_dir: Directory to start searching from. Defaults to cwd.

    Returns:
        KBContext if found and valid, None otherwise.
    """
    # Check environment variable first (explicit override)
    env_context = os.environ.get("VL_KB_CONTEXT")
    if env_context:
        context_path = Path(env_context)
        if context_path.exists():
            return _load_context_file(context_path)
        # Env var set but file doesn't exist - treat as no context
        return None

    # Walk up from start_dir looking for .kbcontext
    current = (start_dir or Path.cwd()).resolve()

    # Prevent infinite loop - stop at filesystem root
    max_depth = 50
    depth = 0

    while depth < max_depth:
        context_file = current / CONTEXT_FILENAME
        if context_file.exists():
            return _load_context_file(context_file)

        # Move up one directory
        parent = current.parent
        if parent == current:
            # Reached filesystem root
            break
        current = parent
        depth += 1

    return None


def _load_context_file(context_path: Path) -> KBContext | None:
    """Load and parse a .kbcontext file.

    Args:
        context_path: Path to the .kbcontext file.

    Returns:
        KBContext if valid, None if parsing fails.
    """
    try:
        content = context_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)

        if not isinstance(data, dict):
            return None

        return KBContext.from_dict(data, source_file=context_path)

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
    """Clear the context cache. Useful for testing or after .kbcontext changes."""
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
                warnings.append(
                    f"Path '{pattern}' does not exist in the knowledge base."
                )

    return warnings


def create_default_context(project_name: str, kb_directory: str | None = None) -> str:
    """Generate default .kbcontext content for a new project.

    Args:
        project_name: Name of the project.
        kb_directory: Optional KB directory path (e.g., 'projects/myapp').

    Returns:
        YAML content for .kbcontext file.
    """
    directory = kb_directory or f"projects/{project_name}"

    return f"""# .kbcontext - Project knowledge base context
# This file tells voidlabs-kb which KB entries are relevant to this project.

# Default directory for new entries created from this project
primary: {directory}

# Boost these paths in search results (supports * and ** globs)
paths:
  - {directory}
  # - tooling/*
  # - infrastructure/**

# Suggested tags for new entries
default_tags:
  - {project_name}

# Override auto-detected project name (optional)
# project: {project_name}
"""
