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


class TitleIndex(NamedTuple):
    """Title index with secondary indices for O(1) lookups."""

    title_to_path: dict[str, str]
    filename_to_paths: dict[str, list[str]]


def build_title_index(
    kb_root: Path, *, include_filename_index: bool = True
) -> dict[str, str] | TitleIndex:
    """Build an index mapping titles and aliases to paths.

    Scans all markdown files in the KB and creates a case-insensitive
    lookup from title/alias to the entry's path.

    Args:
        kb_root: Root directory of the knowledge base.
        include_filename_index: If True, returns a TitleIndex with secondary
            indices for O(1) filename lookups. If False, returns just the
            dict for backward compatibility.

    Returns:
        If include_filename_index is True: TitleIndex with title_to_path and
            filename_to_paths mappings.
        If include_filename_index is False: Dict mapping lowercase title/alias
            to path (relative, without .md).
    """
    if not kb_root.exists() or not kb_root.is_dir():
        if include_filename_index:
            return TitleIndex(title_to_path={}, filename_to_paths={})
        return {}

    title_index: dict[str, str] = {}
    filename_to_paths: dict[str, list[str]] = {}

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

        # Build filename index for O(1) lookups
        # Extract the basename (last component) of the path
        if "/" in path_str:
            filename = path_str.rsplit("/", 1)[1]
        else:
            filename = path_str

        # Index by both original case and lowercase for case-insensitive matching
        filename_lower = filename.lower()
        if filename_lower not in filename_to_paths:
            filename_to_paths[filename_lower] = []
        if path_str not in filename_to_paths[filename_lower]:
            filename_to_paths[filename_lower].append(path_str)

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

    if include_filename_index:
        return TitleIndex(title_to_path=title_index, filename_to_paths=filename_to_paths)
    return title_index


def resolve_link_target(
    target: str,
    title_index: dict[str, str] | TitleIndex,
    source_path: str | None = None,
    *,
    filename_index: dict[str, list[str]] | None = None,
) -> str | None:
    """Resolve a link target to a path.

    Attempts resolution in order:
    1. Exact path match (if target looks like a path)
    2. Title/alias lookup (case-insensitive)
    3. Filename match (for [[filename]] without path) - O(1) with filename_index

    Args:
        target: The link target from [[target]].
        title_index: Title/alias to path mapping, or a TitleIndex with both
            title_to_path and filename_to_paths for O(1) lookups.
        source_path: Optional source file path for relative resolution.
        filename_index: Optional secondary index mapping filename to paths.
            If title_index is a TitleIndex, this is ignored and the embedded
            filename_to_paths is used instead.

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

    # Extract the actual title_to_path dict and filename_to_paths if available
    if isinstance(title_index, TitleIndex):
        title_to_path = title_index.title_to_path
        fn_index = title_index.filename_to_paths
    else:
        title_to_path = title_index
        fn_index = filename_index

    # Try title/alias lookup (case-insensitive)
    lookup_key = normalized.lower()
    if lookup_key in title_to_path:
        return title_to_path[lookup_key]

    # Try O(1) filename lookup if we have the filename index
    if fn_index is not None:
        paths = fn_index.get(lookup_key)
        if paths:
            # Return the first matching path (consistent with original behavior)
            return paths[0]
        return None

    # Fallback: O(n) linear search for backward compatibility when no filename_index
    # This handles [[entry-name]] matching "category/entry-name"
    for _title, path in title_to_path.items():
        if path.endswith(f"/{normalized}") or path == normalized:
            return path

    # Also check if normalized matches the end of any indexed path (case-insensitive)
    for _title, path in title_to_path.items():
        path_lower = path.lower()
        if path_lower.endswith(f"/{lookup_key}") or path_lower == lookup_key:
            return path

    return None
