"""Title-to-path index for resolving wiki-style links.

Enables resolution of [[Title]] and [[Alias]] style links in addition
to path-style [[path/to/entry]] links.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import frontmatter

log = logging.getLogger(__name__)


class TitleEntry(NamedTuple):
    """A title/alias mapping to a path."""

    title: str
    path: str
    is_alias: bool = False


def build_title_index(kb_root: Path) -> dict[str, str]:
    """Build an index mapping titles and aliases to paths.

    Scans all markdown files in the KB and creates a case-insensitive
    lookup from title/alias to the entry's path.

    Args:
        kb_root: Root directory of the knowledge base.

    Returns:
        Dict mapping lowercase title/alias to path (relative, without .md).
    """
    if not kb_root.exists() or not kb_root.is_dir():
        return {}

    title_index: dict[str, str] = {}

    for md_file in kb_root.rglob("*.md"):
        # Skip special files
        if md_file.name.startswith("_"):
            continue

        try:
            post = frontmatter.load(md_file)
        except Exception as e:
            log.debug("Skipping %s during title index build: %s", md_file, e)
            continue

        if not post.metadata:
            continue

        # Get path relative to kb_root, without .md
        rel_path = md_file.relative_to(kb_root)
        path_str = str(rel_path.with_suffix(""))

        # Index the title
        title = post.metadata.get("title")
        if title:
            # Use lowercase for case-insensitive matching
            title_key = title.lower().strip()
            if title_key not in title_index:
                title_index[title_key] = path_str

        # Index any aliases
        aliases = post.metadata.get("aliases", [])
        if isinstance(aliases, list):
            for alias in aliases:
                if alias:
                    alias_key = alias.lower().strip()
                    if alias_key not in title_index:
                        title_index[alias_key] = path_str

    return title_index


def resolve_link_target(
    target: str,
    title_index: dict[str, str],
    source_path: str | None = None,
) -> str | None:
    """Resolve a link target to a path.

    Attempts resolution in order:
    1. Exact path match (if target looks like a path)
    2. Title/alias lookup (case-insensitive)
    3. Filename match (for [[filename]] without path)

    Args:
        target: The link target from [[target]].
        title_index: Title/alias to path mapping.
        source_path: Optional source file path for relative resolution.

    Returns:
        Resolved path (without .md) or None if not resolvable.
    """
    normalized = target.strip()

    # Remove .md extension if present
    if normalized.endswith(".md"):
        normalized = normalized[:-3]

    # Normalize path separators
    normalized = normalized.replace("\\", "/").strip("/")

    # If it contains a path separator, it's likely a path reference
    if "/" in normalized:
        return normalized

    # Try title/alias lookup (case-insensitive)
    lookup_key = normalized.lower()
    if lookup_key in title_index:
        return title_index[lookup_key]

    # Try matching just the filename part of paths in the index
    # This handles [[entry-name]] matching "category/entry-name"
    for _title, path in title_index.items():
        if path.endswith(f"/{normalized}") or path == normalized:
            return path

    # Also check if normalized matches the end of any indexed path
    normalized_lower = normalized.lower()
    for _title, path in title_index.items():
        path_lower = path.lower()
        if path_lower.endswith(f"/{normalized_lower}") or path_lower == normalized_lower:
            return path

    return None
