"""Bidirectional link extraction and resolution."""

import re
from pathlib import Path

from .md_renderer import extract_links_only, normalize_link
from .title_index import TitleIndex, build_title_index, resolve_link_target

def extract_links(content: str) -> list[str]:
    """Extract bidirectional links from markdown content.

    Uses AST-based parsing via markdown-it-py for robust extraction
    that handles edge cases like links in code blocks correctly.

    Args:
        content: Markdown content to extract links from.

    Returns:
        List of unique link targets (normalized, without .md extension).
    """
    return extract_links_only(content)


# Pattern for [[link]] syntax - used for link replacement operations
# Handles [[path/to/entry]] and [[entry]] formats
LINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")


# Backward compatibility alias - the canonical implementation is in md_renderer.py
_normalize_link = normalize_link


def resolve_backlinks(
    kb_root: Path,
    title_index: dict[str, str] | TitleIndex | None = None,
) -> dict[str, list[str]]:
    """Build a backlink index for all markdown files.

    Scans all .md files in the KB and builds an index mapping
    each entry to the list of entries that link to it.

    Supports both path-style links ([[path/to/entry]]) and
    title-style links ([[Entry Title]]).

    Args:
        kb_root: Root directory of the knowledge base.
        title_index: Optional pre-built title index. If None, builds one
            internally with O(1) filename lookups. Can be a dict (legacy)
            or TitleIndex (with filename index for O(1) lookups).

    Returns:
        Dict mapping entry paths (relative to kb_root, without .md) to
        list of paths that link to them.
    """
    if not kb_root.exists() or not kb_root.is_dir():
        return {}

    # Use provided title index or build one with filename index for O(1) lookups
    if title_index is None:
        title_index = build_title_index(kb_root, include_filename_index=True)

    # Collect all markdown files and their links
    # Maps: normalized_path -> set of files that contain [[normalized_path]]
    # Use sets for O(1) deduplication instead of O(n) list membership checks
    backlinks: dict[str, set[str]] = {}

    # First pass: collect all links from each file
    # forward_links: source_path -> list of raw link targets
    forward_links: dict[str, list[str]] = {}

    for md_file in kb_root.rglob("*.md"):
        # Skip special files
        if md_file.name.startswith("_"):
            continue

        # Get path relative to kb_root, without .md extension
        rel_path = md_file.relative_to(kb_root)
        source_path = str(rel_path.with_suffix(""))

        try:
            content = md_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        links = extract_links(content)
        forward_links[source_path] = links

    # Second pass: resolve and invert to get backlinks
    for source, targets in forward_links.items():
        for target in targets:
            # Try title resolution first, then fall back to path resolution
            resolved = resolve_link_target(target, title_index, source)

            if resolved is None:
                # Fall back to relative path resolution for path-style links
                resolved = _resolve_relative_link(source, target)

            if resolved not in backlinks:
                backlinks[resolved] = set()
            backlinks[resolved].add(source)  # O(1) set add

    # Convert sets to lists for API compatibility
    return {k: list(v) for k, v in backlinks.items()}


def _resolve_relative_link(source: str, target: str) -> str:
    """Resolve a potentially relative link target.

    Args:
        source: Source file path (e.g., "path/to/source").
        target: Link target (e.g., "../other" or "absolute/path").

    Returns:
        Resolved absolute path within the KB.
    """
    # If target contains no path separators or starts with /, treat as absolute
    if "/" not in target or target.startswith("/"):
        return target.lstrip("/")

    # If target contains .. or ., resolve relative to source directory
    if target.startswith("..") or target.startswith("./"):
        source_parts = source.split("/")[:-1]  # Get parent directory
        target_parts = target.split("/")

        result_parts = list(source_parts)
        for part in target_parts:
            if part == "..":
                if result_parts:
                    result_parts.pop()
            elif part == ".":
                continue
            else:
                result_parts.append(part)

        return "/".join(result_parts)

    # Otherwise, treat as relative to KB root (absolute path without leading /)
    return target


def update_links_in_files(
    kb_root: Path, old_path: str, new_path: str
) -> int:
    """Update [[links]] in all files when an entry moves.

    Replaces all occurrences of [[old_path]] with [[new_path]] across
    all markdown files in the knowledge base.

    Args:
        kb_root: Knowledge base root directory.
        old_path: Original path without .md (e.g., "development/old-entry").
        new_path: New path without .md (e.g., "architecture/old-entry").

    Returns:
        Number of files that were updated.
    """
    # Normalize paths (remove .md extension if present)
    old_normalized = old_path[:-3] if old_path.endswith(".md") else old_path
    new_normalized = new_path[:-3] if new_path.endswith(".md") else new_path

    # Build patterns to match both with and without .md extension
    # Matches [[old/path]], [[old/path.md]]
    pattern = re.compile(
        rf"\[\[{re.escape(old_normalized)}(\.md)?\]\]"
    )
    replacement = f"[[{new_normalized}]]"

    updated_count = 0

    for md_file in kb_root.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            original = content

            # Replace all occurrences
            content = pattern.sub(replacement, content)

            if content != original:
                md_file.write_text(content, encoding="utf-8")
                updated_count += 1
        except (OSError, UnicodeDecodeError):
            continue

    return updated_count


def update_links_batch(
    kb_root: Path, path_mapping: dict[str, str]
) -> int:
    """Update multiple link mappings in a single pass through all files.

    More efficient than calling update_links_in_files() multiple times
    when moving a directory with multiple entries.

    Args:
        kb_root: Knowledge base root directory.
        path_mapping: Dict mapping old_path -> new_path (without .md).

    Returns:
        Number of files that were updated.
    """
    if not path_mapping:
        return 0

    # Build a combined pattern for all old paths
    patterns = []
    replacements = {}

    for old_path, new_path in path_mapping.items():
        old_normalized = old_path[:-3] if old_path.endswith(".md") else old_path
        new_normalized = new_path[:-3] if new_path.endswith(".md") else new_path

        # Key is the escaped pattern for lookup
        pattern_key = re.escape(old_normalized)
        patterns.append(rf"\[\[{pattern_key}(\.md)?\]\]")
        replacements[old_normalized] = new_normalized

    combined_pattern = re.compile("|".join(patterns))

    def replace_func(match: re.Match) -> str:
        """Replace matched link with new path."""
        matched_text = match.group(0)
        # Extract the path from [[path]] or [[path.md]]
        inner = matched_text[2:-2]  # Remove [[ and ]]
        if inner.endswith(".md"):
            inner = inner[:-3]

        if inner in replacements:
            return f"[[{replacements[inner]}]]"
        return matched_text

    updated_count = 0

    for md_file in kb_root.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            original = content

            content = combined_pattern.sub(replace_func, content)

            if content != original:
                md_file.write_text(content, encoding="utf-8")
                updated_count += 1
        except (OSError, UnicodeDecodeError):
            continue

    return updated_count
