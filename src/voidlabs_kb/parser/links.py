"""Bidirectional link extraction and resolution."""

import re
from pathlib import Path

# Pattern for [[link]] syntax - captures content between double brackets
# Handles [[path/to/entry]] and [[entry]] formats
LINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")


def extract_links(content: str) -> list[str]:
    """Extract bidirectional links from markdown content.

    Args:
        content: Markdown content to extract links from.

    Returns:
        List of unique link targets (normalized, without .md extension).
    """
    matches = LINK_PATTERN.findall(content)

    # Normalize and deduplicate
    seen: set[str] = set()
    links: list[str] = []

    for link in matches:
        normalized = _normalize_link(link)
        if normalized and normalized not in seen:
            seen.add(normalized)
            links.append(normalized)

    return links


def _normalize_link(link: str) -> str:
    """Normalize a link target.

    - Strips whitespace
    - Removes .md extension
    - Normalizes path separators

    Args:
        link: Raw link target.

    Returns:
        Normalized link target.
    """
    link = link.strip()

    # Remove .md extension if present
    if link.endswith(".md"):
        link = link[:-3]

    # Normalize path separators (use forward slashes)
    link = link.replace("\\", "/")

    # Remove leading/trailing slashes
    link = link.strip("/")

    return link


def resolve_backlinks(kb_root: Path) -> dict[str, list[str]]:
    """Build a backlink index for all markdown files.

    Scans all .md files in the KB and builds an index mapping
    each entry to the list of entries that link to it.

    Args:
        kb_root: Root directory of the knowledge base.

    Returns:
        Dict mapping entry paths (relative to kb_root, without .md) to
        list of paths that link to them.
    """
    if not kb_root.exists() or not kb_root.is_dir():
        return {}

    # Collect all markdown files and their links
    # Maps: normalized_path -> list of files that contain [[normalized_path]]
    backlinks: dict[str, list[str]] = {}

    # First pass: collect all links from each file
    # forward_links: source_path -> list of target_paths
    forward_links: dict[str, list[str]] = {}

    for md_file in kb_root.rglob("*.md"):
        # Get path relative to kb_root, without .md extension
        rel_path = md_file.relative_to(kb_root)
        source_path = str(rel_path.with_suffix(""))

        try:
            content = md_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        links = extract_links(content)
        forward_links[source_path] = links

    # Second pass: invert to get backlinks
    for source, targets in forward_links.items():
        for target in targets:
            # Resolve relative paths
            resolved = _resolve_relative_link(source, target)

            if resolved not in backlinks:
                backlinks[resolved] = []
            backlinks[resolved].append(source)

    return backlinks


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
