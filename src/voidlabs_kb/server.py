"""FastMCP server for voidlabs-kb."""

import os
import re
import shutil
import subprocess
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from fastmcp import FastMCP

from .backlinks_cache import ensure_backlink_cache, rebuild_backlink_cache
from .config import MAX_CONTENT_RESULTS, get_kb_root
from .evaluation import run_quality_checks
from .indexer import HybridSearcher
from .models import DocumentChunk, IndexStatus, KBEntry, QualityReport, SearchResponse, SearchResult
from .parser import ParseError, extract_links, parse_entry, update_links_batch


def _get_current_contributor() -> str | None:
    """Get the current contributor identity from git config or environment.

    Returns:
        Contributor string like "Name <email>" or "Name", or None if unavailable.
    """
    # Try git config first
    try:
        name_result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        email_result = subprocess.run(
            ["git", "config", "user.email"],
            capture_output=True,
            text=True,
            timeout=2,
        )

        name = name_result.stdout.strip() if name_result.returncode == 0 else None
        email = email_result.stdout.strip() if email_result.returncode == 0 else None

        if name and email:
            return f"{name} <{email}>"
        elif name:
            return name
    except (subprocess.TimeoutExpired, OSError):
        pass

    # Fall back to environment variables
    name = os.environ.get("GIT_AUTHOR_NAME") or os.environ.get("USER")
    email = os.environ.get("GIT_AUTHOR_EMAIL")

    if name and email:
        return f"{name} <{email}>"
    elif name:
        return name

    return None


mcp = FastMCP(
    name="voidlabs-kb",
    instructions="""
Voidlabs Knowledge Base - Organization-wide knowledge with semantic search.

Use `search` to find documentation, patterns, and operational guides.
Use `add` to contribute new knowledge entries.
Use `update` to modify existing entries.

Best practices:
- Search before creating to avoid duplicates
- Use consistent tags from the existing taxonomy
- Add [[bidirectional links]] to related entries
""",
)

# Lazy-initialized searcher
_searcher: HybridSearcher | None = None
_searcher_ready = False

def _maybe_initialize_searcher(searcher: HybridSearcher) -> None:
    """Ensure search indices are populated before first query."""
    global _searcher_ready
    if _searcher_ready:
        return

    status = searcher.status()
    if status.kb_files > 0 and (status.whoosh_docs == 0 or status.chroma_docs == 0):
        kb_root = get_kb_root()
        if kb_root.exists():
            searcher.reindex(kb_root)

    _searcher_ready = True


def _get_searcher() -> HybridSearcher:
    """Get the HybridSearcher, initializing lazily."""
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher()
    _maybe_initialize_searcher(_searcher)
    return _searcher


def _slugify(title: str) -> str:
    """Convert title to URL-friendly slug."""
    # Lowercase
    slug = title.lower()
    # Replace spaces and underscores with hyphens
    slug = re.sub(r"[\s_]+", "-", slug)
    # Remove non-alphanumeric characters (except hyphens)
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    # Remove consecutive hyphens
    slug = re.sub(r"-+", "-", slug)
    # Strip leading/trailing hyphens
    slug = slug.strip("-")
    return slug


def _get_valid_categories() -> list[str]:
    """Get list of valid category directories."""
    kb_root = get_kb_root()
    if not kb_root.exists():
        return []
    return [
        d.name
        for d in kb_root.iterdir()
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
    ]


def _relative_kb_path(kb_root: Path, file_path: Path) -> str:
    """Return file path relative to KB root."""
    return str(file_path.relative_to(kb_root))


def _normalize_chunks(chunks: list[DocumentChunk], relative_path: str) -> list[DocumentChunk]:
    """Ensure all chunk paths share the relative file path."""
    return [chunk.model_copy(update={"path": relative_path}) for chunk in chunks]


def _apply_section_updates(content: str, updates: dict[str, str]) -> str:
    """Apply section-level updates to markdown content."""

    updated = content
    for section, replacement in updates.items():
        clean_section = section.strip()
        clean_replacement = replacement.strip()
        if not clean_section or not clean_replacement:
            continue

        pattern = re.compile(
            rf"(^##\s+{re.escape(clean_section)}\s*$)(.*?)(?=^##\s+|\Z)",
            re.MULTILINE | re.DOTALL,
        )

        if pattern.search(updated):
            updated = pattern.sub(
                f"## {clean_section}\n\n{clean_replacement}\n\n", updated, count=1
            )
        else:
            updated = updated.rstrip() + f"\n\n## {clean_section}\n\n{clean_replacement}\n"

    return updated


def _get_backlink_index() -> dict[str, list[str]]:
    """Return cached backlink index, refreshing if files changed."""
    kb_root = get_kb_root()
    return ensure_backlink_cache(kb_root)


def _validate_nested_path(path: str) -> tuple[Path, str]:
    """Validate a nested path within the KB.

    Args:
        path: Relative path like "development/python/file.md" or "development/python/"

    Returns:
        Tuple of (absolute_path, normalized_relative_path)

    Raises:
        ValueError: If path is invalid or outside KB.
    """
    kb_root = get_kb_root()

    # Security: prevent path traversal
    normalized = Path(path).as_posix()
    if ".." in normalized or normalized.startswith("/"):
        raise ValueError(f"Invalid path: {path}")

    # Check for hidden/special path components
    for part in Path(path).parts:
        if part.startswith(".") or part.startswith("_"):
            raise ValueError(f"Invalid path component: {part}")

    abs_path = (kb_root / path).resolve()

    # Ensure path is within KB root
    try:
        abs_path.relative_to(kb_root.resolve())
    except ValueError:
        raise ValueError(f"Path escapes knowledge base: {path}")

    return abs_path, normalized


def _directory_exists(directory: str) -> bool:
    """Check if a directory path exists within the KB."""
    kb_root = get_kb_root()
    dir_path = kb_root / directory
    return dir_path.exists() and dir_path.is_dir()


def _get_parent_category(path: str) -> str:
    """Extract top-level category from a nested path.

    Args:
        path: "development/python/frameworks/file.md"

    Returns:
        "development"
    """
    parts = Path(path).parts
    if not parts:
        raise ValueError("Empty path")
    return parts[0]


def _hydrate_content(results: list[SearchResult]) -> list[SearchResult]:
    """Add full document content to search results.

    Reads content from disk using parse_entry() for each result.

    Args:
        results: Search results to hydrate.

    Returns:
        New list of SearchResult with content populated.
    """
    if not results:
        return results

    kb_root = get_kb_root()
    hydrated = []

    for result in results:
        file_path = kb_root / result.path
        content = None

        if file_path.exists():
            try:
                _, content, _ = parse_entry(file_path)
            except ParseError:
                # Fall back to None if parsing fails
                pass

        # Create new result with content
        hydrated.append(
            SearchResult(
                path=result.path,
                title=result.title,
                snippet=result.snippet,
                score=result.score,
                tags=result.tags,
                section=result.section,
                created=result.created,
                updated=result.updated,
                token_count=result.token_count,
                content=content,
            )
        )

    return hydrated


@mcp.tool(
    name="search",
    description=(
        "Search the knowledge base using hybrid keyword + semantic search. "
        "Returns relevant entries ranked by relevance. "
        "Use include_content=true to get full document content instead of snippets."
    ),
)
async def search_tool(
    query: str,
    limit: int = 10,
    mode: Literal["hybrid", "keyword", "semantic"] = "hybrid",
    tags: list[str] | None = None,
    include_content: bool = False,
) -> SearchResponse:
    """Search the knowledge base.

    Args:
        query: Search query string.
        limit: Maximum number of results (default 10).
        mode: Search mode - "hybrid" (default), "keyword", or "semantic".
        tags: Optional list of tags to filter results.
        include_content: If True, include full document content in results.
                         Default False (snippet only). Limited to MAX_CONTENT_RESULTS.

    Returns:
        SearchResponse with results and optional warnings.
    """
    searcher = _get_searcher()
    results = searcher.search(query, limit=limit, mode=mode)
    warnings: list[str] = []

    # Filter by tags if specified
    if tags:
        tag_set = set(tags)
        results = [r for r in results if tag_set.intersection(r.tags)]

    # Hydrate with full content if requested
    if include_content:
        if len(results) > MAX_CONTENT_RESULTS:
            warnings.append(
                f"Results limited to {MAX_CONTENT_RESULTS} for content hydration "
                f"(requested {len(results)}). Reduce limit or use get tool for remaining."
            )
            results = results[:MAX_CONTENT_RESULTS]
        results = _hydrate_content(results)

    return SearchResponse(results=results, warnings=warnings)


@mcp.tool(
    name="quality",
    description="Run the built-in search accuracy evaluation suite and return metrics.",
)
async def quality_tool(limit: int = 5, cutoff: int = 3) -> QualityReport:
    """Evaluate MCP search accuracy."""

    searcher = _get_searcher()
    return run_quality_checks(searcher, limit=limit, cutoff=cutoff)


@mcp.tool(
    name="add",
    description=(
        "Create a new knowledge base entry. Generates a slug from the title "
        "and places the entry in the specified category or directory."
    ),
)
async def add_tool(
    title: str,
    content: str,
    tags: list[str],
    category: str = "",
    directory: str | None = None,
    links: list[str] | None = None,
) -> str:
    """Create a new KB entry.

    Args:
        title: Entry title.
        content: Markdown content (without frontmatter).
        tags: List of tags for the entry.
        category: Top-level category directory (e.g., "development", "devops").
                  Deprecated - use 'directory' for nested paths.
        directory: Full directory path (e.g., "development/python/frameworks").
                   Takes precedence over 'category' if both are provided.
        links: Optional list of paths to link to using [[link]] syntax.

    Returns:
        Path of the created file.
    """
    kb_root = get_kb_root()

    # Determine target directory: prefer 'directory' over 'category'
    if directory:
        # Validate the directory is within KB, auto-create if needed
        abs_dir, normalized_dir = _validate_nested_path(directory)
        if abs_dir.exists() and not abs_dir.is_dir():
            raise ValueError(f"Path exists but is not a directory: {directory}")
        # Auto-create directory if it doesn't exist
        abs_dir.mkdir(parents=True, exist_ok=True)
        target_dir = abs_dir
        rel_dir = normalized_dir
    elif category:
        # Auto-create category directory if it doesn't exist
        target_dir = kb_root / category
        target_dir.mkdir(parents=True, exist_ok=True)
        rel_dir = category
    else:
        valid_categories = _get_valid_categories()
        raise ValueError(
            "Either 'category' or 'directory' must be provided. "
            f"Existing categories: {', '.join(valid_categories)}"
        )

    if not tags:
        raise ValueError("At least one tag is required")

    # Generate slug and path
    slug = _slugify(title)
    if not slug:
        raise ValueError("Title must contain at least one alphanumeric character")

    file_path = target_dir / f"{slug}.md"
    rel_path = f"{rel_dir}/{slug}.md"

    if file_path.exists():
        raise ValueError(f"Entry already exists at {rel_path}")

    # Add links to content if specified
    final_content = content
    if links:
        link_section = "\n\n## Related\n\n"
        for link in links:
            link_section += f"- [[{link}]]\n"
        final_content += link_section

    # Build frontmatter
    today = date.today().isoformat()
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)

    # Get contributor identity
    contributor = _get_current_contributor()
    contributors_yaml = f"\ncontributors:\n  - {contributor}" if contributor else ""

    frontmatter = f"""---
title: {title}
tags:
{tags_yaml}
created: {today}{contributors_yaml}
---

"""
    # Write the file
    file_path.write_text(frontmatter + final_content, encoding="utf-8")
    rebuild_backlink_cache(kb_root)

    # Reindex the new file
    try:
        _, _, chunks = parse_entry(file_path)
        searcher = _get_searcher()
        if chunks:
            relative_path = _relative_kb_path(kb_root, file_path)
            normalized_chunks = _normalize_chunks(chunks, relative_path)
            searcher.index_chunks(normalized_chunks)
    except ParseError:
        # File was written but indexing failed - still return success
        pass

    return rel_path


@mcp.tool(
    name="update",
    description=(
        "Update an existing knowledge base entry. "
        "Preserves frontmatter and updates the 'updated' date."
    ),
)
async def update_tool(
    path: str,
    content: str | None = None,
    tags: list[str] | None = None,
    section_updates: dict[str, str] | None = None,
) -> str:
    """Update an existing KB entry.

    Args:
        path: Path to the entry relative to KB root (e.g., "development/python-tooling.md").
        content: New markdown content (without frontmatter).
        tags: Optional new list of tags. If provided, replaces existing tags.

    Returns:
        Updated path.
    """
    kb_root = get_kb_root()
    file_path = kb_root / path

    if not file_path.exists():
        raise ValueError(f"Entry not found: {path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if content is None and not section_updates:
        raise ValueError("Provide new content or section_updates")

    # Parse existing entry to get metadata
    try:
        metadata, existing_content, _ = parse_entry(file_path)
    except ParseError as e:
        raise ValueError(f"Failed to parse existing entry: {e}") from e

    # Update metadata
    today = date.today()
    new_tags = tags if tags is not None else list(metadata.tags)

    if not new_tags:
        raise ValueError("At least one tag is required")

    # Add current contributor if not already present
    contributors = list(metadata.contributors)
    current_contributor = _get_current_contributor()
    if current_contributor and current_contributor not in contributors:
        contributors.append(current_contributor)

    # Build updated frontmatter
    tags_yaml = "\n".join(f"  - {tag}" for tag in new_tags)
    contributors_yaml = "\n".join(f"  - {c}" for c in contributors)
    aliases_yaml = "\n".join(f"  - {a}" for a in metadata.aliases)

    frontmatter_parts = [
        "---",
        f"title: {metadata.title}",
        "tags:",
        tags_yaml,
        f"created: {metadata.created.isoformat()}",
        f"updated: {today.isoformat()}",
    ]

    if contributors:
        frontmatter_parts.append("contributors:")
        frontmatter_parts.append(contributors_yaml)

    if metadata.aliases:
        frontmatter_parts.append("aliases:")
        frontmatter_parts.append(aliases_yaml)

    if metadata.status != "published":
        frontmatter_parts.append(f"status: {metadata.status}")

    frontmatter_parts.append("---\n\n")
    frontmatter = "\n".join(frontmatter_parts)

    new_content = content if content is not None else existing_content
    if section_updates:
        new_content = _apply_section_updates(new_content, section_updates)

    # Write updated file
    file_path.write_text(frontmatter + new_content, encoding="utf-8")
    rebuild_backlink_cache(kb_root)

    relative_path = _relative_kb_path(kb_root, file_path)

    # Reindex
    searcher = _get_searcher()
    try:
        # Remove old index entries
        searcher.delete_document(relative_path)
        # Parse and index new content
        _, _, chunks = parse_entry(file_path)
        if chunks:
            normalized_chunks = _normalize_chunks(chunks, relative_path)
            searcher.index_chunks(normalized_chunks)
    except (ParseError, Exception):
        # Indexing failed but file was updated
        pass

    return relative_path


@mcp.tool(
    name="get",
    description=(
        "Get a full knowledge base entry including metadata, content, links, and backlinks."
    ),
)
async def get_tool(path: str) -> KBEntry:
    """Read a KB entry.

    Args:
        path: Path to the entry relative to KB root (e.g., "development/python-tooling.md").

    Returns:
        KBEntry with metadata, content, links, and backlinks.
    """
    kb_root = get_kb_root()
    file_path = kb_root / path

    if not file_path.exists():
        raise ValueError(f"Entry not found: {path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    try:
        metadata, content, _ = parse_entry(file_path)
    except ParseError as e:
        raise ValueError(f"Failed to parse entry: {e}") from e

    # Extract links from content
    links = extract_links(content)

    # Get backlinks
    all_backlinks = _get_backlink_index()
    # Normalize path for lookup (remove .md extension)
    path_key = path[:-3] if path.endswith(".md") else path
    entry_backlinks = all_backlinks.get(path_key, [])

    # Record the view (non-blocking, fire-and-forget)
    try:
        from .views_tracker import record_view

        record_view(path)
    except Exception:
        pass  # Don't let view tracking failures affect get

    return KBEntry(
        path=path,
        metadata=metadata,
        content=content,
        links=links,
        backlinks=entry_backlinks,
    )


@mcp.tool(
    name="list",
    description="List knowledge base entries, optionally filtered by category, directory, or tag.",
)
async def list_tool(
    category: str | None = None,
    directory: str | None = None,
    tag: str | None = None,
    recursive: bool = True,
    limit: int = 20,
) -> list[dict]:
    """List KB entries.

    Args:
        category: Optional top-level category to filter by (e.g., "development").
        directory: Optional directory path to filter by (e.g., "development/python").
                   Takes precedence over 'category' if both are provided.
        tag: Optional tag to filter by.
        recursive: If True (default), include entries in subdirectories.
                   If False, only list direct children of the directory.
        limit: Maximum number of entries to return (default 20).

    Returns:
        List of {path, title, tags, created, updated} dictionaries.
    """
    kb_root = get_kb_root()

    if not kb_root.exists():
        return []

    # Determine search path: prefer 'directory' over 'category'
    if directory:
        abs_path, _ = _validate_nested_path(directory)
        if not abs_path.exists() or not abs_path.is_dir():
            raise ValueError(f"Directory not found: {directory}")
        search_path = abs_path
    elif category:
        search_path = kb_root / category
        if not search_path.exists() or not search_path.is_dir():
            raise ValueError(f"Category not found: {category}")
    else:
        search_path = kb_root

    results = []

    # Choose glob pattern based on recursive flag
    md_files = search_path.rglob("*.md") if recursive else search_path.glob("*.md")

    for md_file in md_files:
        # Skip index files
        if md_file.name.startswith("_"):
            continue

        rel_path = str(md_file.relative_to(kb_root))

        try:
            metadata, _, _ = parse_entry(md_file)
        except ParseError:
            continue

        # Filter by tag if specified
        if tag and tag not in metadata.tags:
            continue

        results.append(
            {
                "path": rel_path,
                "title": metadata.title,
                "tags": metadata.tags,
                "created": metadata.created.isoformat() if metadata.created else None,
                "updated": metadata.updated.isoformat() if metadata.updated else None,
            }
        )

        if len(results) >= limit:
            break

    return results


@mcp.tool(
    name="whats_new",
    description=(
        "List recently created or updated knowledge base entries, "
        "sorted by most recent activity first."
    ),
)
async def whats_new_tool(
    days: int = 30,
    limit: int = 10,
    include_created: bool = True,
    include_updated: bool = True,
    category: str | None = None,
    tag: str | None = None,
) -> list[dict]:
    """List recent KB entries.

    Args:
        days: Look back period in days (default 30).
        limit: Maximum entries to return (default 10).
        include_created: Include newly created entries (default True).
        include_updated: Include recently updated entries (default True).
        category: Optional category filter.
        tag: Optional tag filter.

    Returns:
        List of {path, title, tags, created, updated, activity_type, activity_date}.
    """
    kb_root = get_kb_root()

    if not kb_root.exists():
        return []

    # Determine search path
    if category:
        search_path = kb_root / category
        if not search_path.exists() or not search_path.is_dir():
            raise ValueError(f"Category not found: {category}")
    else:
        search_path = kb_root

    cutoff_date = date.today() - timedelta(days=days)
    candidates: list[dict] = []

    for md_file in search_path.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue

        rel_path = str(md_file.relative_to(kb_root))

        try:
            metadata, _, _ = parse_entry(md_file)
        except ParseError:
            continue

        # Filter by tag if specified
        if tag and tag not in metadata.tags:
            continue

        # Determine activity type and date
        activity_type: str | None = None
        activity_date: date | None = None

        # Check updated first (takes precedence if both qualify)
        if include_updated and metadata.updated and metadata.updated >= cutoff_date:
            activity_type = "updated"
            activity_date = metadata.updated
        elif include_created and metadata.created >= cutoff_date:
            activity_type = "created"
            activity_date = metadata.created

        if activity_type is None:
            continue

        candidates.append(
            {
                "path": rel_path,
                "title": metadata.title,
                "tags": metadata.tags,
                "created": metadata.created.isoformat() if metadata.created else None,
                "updated": metadata.updated.isoformat() if metadata.updated else None,
                "activity_type": activity_type,
                "activity_date": activity_date.isoformat(),
            }
        )

    # Sort by activity_date descending
    candidates.sort(key=lambda x: x["activity_date"], reverse=True)

    return candidates[:limit]


@mcp.tool(
    name="popular",
    description=(
        "List most frequently accessed knowledge base entries. "
        "Tracks views from 'get' tool calls."
    ),
)
async def popular_tool(
    limit: int = 10,
    days: int | None = None,
    category: str | None = None,
    tag: str | None = None,
) -> list[dict]:
    """List popular KB entries by view count.

    Args:
        limit: Maximum entries to return (default 10).
        days: Optional time window (None = all time).
        category: Optional category filter.
        tag: Optional tag filter.

    Returns:
        List of {path, title, tags, view_count, last_viewed}.
    """
    from .views_tracker import get_popular

    kb_root = get_kb_root()

    if not kb_root.exists():
        return []

    # Get popular entries (fetch extra to allow for filtering)
    popular_entries = get_popular(limit=limit * 3, days=days)

    results: list[dict] = []

    for entry_path, stats in popular_entries:
        file_path = kb_root / entry_path

        if not file_path.exists():
            continue

        # Apply category filter
        if category:
            parts = entry_path.split("/")
            if len(parts) < 2 or parts[0] != category:
                continue

        try:
            metadata, _, _ = parse_entry(file_path)
        except ParseError:
            continue

        # Filter by tag if specified
        if tag and tag not in metadata.tags:
            continue

        results.append(
            {
                "path": entry_path,
                "title": metadata.title,
                "tags": metadata.tags,
                "view_count": stats.total_views,
                "last_viewed": stats.last_viewed.isoformat()
                if stats.last_viewed
                else None,
            }
        )

        if len(results) >= limit:
            break

    return results


@mcp.tool(
    name="backlinks",
    description="Find all entries that link to a specific entry.",
)
async def backlinks_tool(path: str) -> list[str]:
    """Find entries that link to this path.

    Args:
        path: Path to the entry (e.g., "development/python-tooling.md"
            or "development/python-tooling").

    Returns:
        List of paths that link to this entry.
    """
    # Normalize path (remove .md extension for lookup)
    path_key = path[:-3] if path.endswith(".md") else path

    all_backlinks = _get_backlink_index()
    return all_backlinks.get(path_key, [])


@mcp.tool(
    name="reindex",
    description="Rebuild the search indices from all markdown files in the knowledge base.",
)
async def reindex_tool() -> IndexStatus:
    """Rebuild search indices.

    Returns:
        IndexStatus with document counts.
    """
    kb_root = get_kb_root()
    searcher = _get_searcher()

    searcher.reindex(kb_root)
    rebuild_backlink_cache(kb_root)

    # Clean up views for deleted entries
    try:
        from .views_tracker import cleanup_stale_entries

        valid_paths = {
            str(p.relative_to(kb_root))
            for p in kb_root.rglob("*.md")
            if not p.name.startswith("_")
        }
        cleanup_stale_entries(valid_paths)
    except Exception:
        pass  # Don't fail reindex if cleanup fails

    status = searcher.status()
    return status


@mcp.tool(
    name="tree",
    description="Display the directory structure of the knowledge base or a subdirectory.",
)
async def tree_tool(
    path: str = "",
    depth: int = 3,
    include_files: bool = True,
) -> dict:
    """Show directory tree.

    Args:
        path: Starting path relative to KB root (empty for root).
        depth: Maximum depth to display (default 3).
        include_files: Whether to include .md files (default True).

    Returns:
        Dict with tree structure and counts.
    """
    kb_root = get_kb_root()

    if path:
        abs_path, _ = _validate_nested_path(path)
        if not abs_path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        start_path = abs_path
    else:
        start_path = kb_root

    def _build_tree(current: Path, current_depth: int) -> dict:
        """Recursively build tree structure."""
        result = {}

        if current_depth >= depth:
            return result

        try:
            items = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return result

        for item in items:
            # Skip hidden and special files
            if item.name.startswith(".") or item.name.startswith("_"):
                continue

            if item.is_dir():
                children = _build_tree(item, current_depth + 1)
                result[item.name] = {"_type": "directory", **children}
            elif item.is_file() and item.suffix == ".md" and include_files:
                # Try to get title from frontmatter
                title = None
                try:
                    metadata, _, _ = parse_entry(item)
                    title = metadata.title
                except (ParseError, Exception):
                    pass
                result[item.name] = {"_type": "file", "title": title}

        return result

    tree = _build_tree(start_path, 0)

    # Count directories and files
    def _count(node: dict) -> tuple[int, int]:
        dirs, files = 0, 0
        for key, value in node.items():
            if key == "_type":
                continue
            if isinstance(value, dict):
                if value.get("_type") == "directory":
                    dirs += 1
                    sub_dirs, sub_files = _count(value)
                    dirs += sub_dirs
                    files += sub_files
                elif value.get("_type") == "file":
                    files += 1
        return dirs, files

    total_dirs, total_files = _count(tree)

    return {
        "tree": tree,
        "directories": total_dirs,
        "files": total_files,
    }


@mcp.tool(
    name="mkdir",
    description="Create a new directory within the knowledge base hierarchy.",
)
async def mkdir_tool(path: str) -> str:
    """Create a directory.

    Args:
        path: Directory path relative to KB root (e.g., "development/python/frameworks").
              Must start with a valid top-level category.

    Returns:
        Created directory path.

    Raises:
        ValueError: If path is invalid or directory already exists.
    """
    kb_root = get_kb_root()

    # Validate path format
    abs_path, normalized = _validate_nested_path(path)

    # Ensure it starts with a valid category
    parent_category = _get_parent_category(normalized)
    valid_categories = _get_valid_categories()

    if parent_category not in valid_categories:
        raise ValueError(
            f"Path must start with a valid category. Valid categories: {', '.join(valid_categories)}"
        )

    # Check if already exists
    if abs_path.exists():
        raise ValueError(f"Directory already exists: {path}")

    # Create the directory (and any missing parents within the category)
    abs_path.mkdir(parents=True, exist_ok=False)

    return normalized


@mcp.tool(
    name="move",
    description=(
        "Move an entry or directory to a new location. "
        "Updates all bidirectional links and search indices."
    ),
)
async def move_tool(
    source: str,
    destination: str,
    update_links: bool = True,
) -> dict:
    """Move entry or directory.

    Args:
        source: Current path (e.g., "development/old-entry.md" or "development/python/").
        destination: New path (e.g., "architecture/old-entry.md" or "patterns/python/").
        update_links: Whether to update [[links]] in other files (default True).

    Returns:
        Dict with moved paths and updated link count.

    Raises:
        ValueError: If source doesn't exist or destination conflicts.
    """
    kb_root = get_kb_root()

    # Validate source path
    source_abs, source_normalized = _validate_nested_path(source)
    if not source_abs.exists():
        raise ValueError(f"Source not found: {source}")

    # Validate destination path
    dest_abs, dest_normalized = _validate_nested_path(destination)

    # Check destination doesn't already exist
    if dest_abs.exists():
        raise ValueError(f"Destination already exists: {destination}")

    # Ensure destination parent directory exists
    dest_parent = dest_abs.parent
    if not dest_parent.exists():
        raise ValueError(
            f"Destination parent directory does not exist: {dest_parent.relative_to(kb_root)}. "
            "Use mkdir to create it first."
        )

    # Ensure destination is within a valid category
    dest_category = _get_parent_category(dest_normalized)
    valid_categories = _get_valid_categories()
    if dest_category not in valid_categories:
        raise ValueError(
            f"Destination must be within a valid category. Valid categories: {', '.join(valid_categories)}"
        )

    # Protect category root directories from being moved
    if source_normalized in valid_categories:
        raise ValueError(f"Cannot move category root directory: {source}")

    # Collect files to move and build path mapping for link updates
    path_mapping: dict[str, str] = {}  # old_path -> new_path (without .md)
    moved_files: list[str] = []

    if source_abs.is_file():
        # Moving a single file
        old_rel = source_normalized[:-3] if source_normalized.endswith(".md") else source_normalized
        new_rel = dest_normalized[:-3] if dest_normalized.endswith(".md") else dest_normalized
        path_mapping[old_rel] = new_rel
        moved_files.append(f"{source_normalized} -> {dest_normalized}")
    else:
        # Moving a directory - collect all .md files
        for md_file in source_abs.rglob("*.md"):
            old_rel_path = str(md_file.relative_to(kb_root))
            old_rel = old_rel_path[:-3] if old_rel_path.endswith(".md") else old_rel_path

            # Compute new path
            relative_to_source = md_file.relative_to(source_abs)
            new_rel_path = str(Path(dest_normalized) / relative_to_source)
            new_rel = new_rel_path[:-3] if new_rel_path.endswith(".md") else new_rel_path

            path_mapping[old_rel] = new_rel
            moved_files.append(f"{old_rel_path} -> {new_rel_path}")

    # Perform the move
    shutil.move(str(source_abs), str(dest_abs))

    # Update links in other files
    links_updated = 0
    if update_links and path_mapping:
        links_updated = update_links_batch(kb_root, path_mapping)

    # Reindex moved files
    searcher = _get_searcher()
    entries_reindexed = 0

    for old_path, new_path in path_mapping.items():
        old_path_md = f"{old_path}.md"
        new_path_md = f"{new_path}.md"

        # Delete old index entries
        searcher.delete_document(old_path_md)

        # Index under new path
        new_file_path = kb_root / new_path_md
        if new_file_path.exists():
            try:
                _, _, chunks = parse_entry(new_file_path)
                if chunks:
                    normalized_chunks = _normalize_chunks(chunks, new_path_md)
                    searcher.index_chunks(normalized_chunks)
                    entries_reindexed += 1
            except ParseError:
                pass

    # Rebuild backlink cache
    rebuild_backlink_cache(kb_root)

    return {
        "moved": moved_files,
        "links_updated": links_updated,
        "entries_reindexed": entries_reindexed,
    }


@mcp.tool(
    name="rmdir",
    description="Remove an empty directory from the knowledge base.",
)
async def rmdir_tool(path: str, force: bool = False) -> str:
    """Remove directory.

    Args:
        path: Directory path relative to KB root.
        force: If True, also remove empty subdirectories recursively.

    Returns:
        Removed directory path.

    Raises:
        ValueError: If directory not empty (without force), doesn't exist, or is a category root.
    """
    kb_root = get_kb_root()

    # Validate path
    abs_path, normalized = _validate_nested_path(path)

    if not abs_path.exists():
        raise ValueError(f"Directory not found: {path}")

    if not abs_path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    # Protect category root directories
    valid_categories = _get_valid_categories()
    if normalized in valid_categories:
        raise ValueError(f"Cannot remove category root directory: {path}")

    # Check if directory is empty (or force is True)
    has_files = any(abs_path.rglob("*"))
    has_md_files = any(abs_path.rglob("*.md"))

    if has_md_files:
        raise ValueError(
            f"Directory contains entries: {path}. Move or delete entries first."
        )

    if has_files and not force:
        raise ValueError(
            f"Directory not empty: {path}. Use force=True to remove empty subdirectories."
        )

    # Remove the directory
    if force:
        shutil.rmtree(str(abs_path))
    else:
        abs_path.rmdir()

    return normalized


@mcp.tool(
    name="delete",
    description=(
        "Delete a knowledge base entry. "
        "Warns if other entries link to it (has backlinks). "
        "Use force=True to delete anyway."
    ),
)
async def delete_tool(path: str, force: bool = False) -> dict:
    """Delete an entry.

    Args:
        path: Path to the entry (e.g., "development/old-entry.md").
        force: If True, delete even if other entries link to this one.

    Returns:
        Dict with deleted path and any backlinks that were pointing to it.

    Raises:
        ValueError: If entry doesn't exist or has backlinks (without force).
    """
    kb_root = get_kb_root()

    # Validate path
    abs_path, normalized = _validate_nested_path(path)

    if not abs_path.exists():
        raise ValueError(f"Entry not found: {path}")

    if not abs_path.is_file():
        raise ValueError(f"Path is not a file: {path}. Use rmdir for directories.")

    if not normalized.endswith(".md"):
        raise ValueError(f"Path must be a markdown file: {path}")

    # Check for backlinks
    path_key = normalized[:-3] if normalized.endswith(".md") else normalized
    all_backlinks = _get_backlink_index()
    entry_backlinks = all_backlinks.get(path_key, [])

    if entry_backlinks and not force:
        raise ValueError(
            f"Entry has {len(entry_backlinks)} backlink(s): {', '.join(entry_backlinks)}. "
            "Use force=True to delete anyway, or update linking entries first."
        )

    # Remove from search index
    searcher = _get_searcher()
    searcher.delete_document(normalized)

    # Remove from view tracking
    try:
        from .views_tracker import delete_entry_views

        delete_entry_views(normalized)
    except Exception:
        pass  # Don't fail delete if view cleanup fails

    # Delete the file
    abs_path.unlink()

    # Rebuild backlink cache
    rebuild_backlink_cache(kb_root)

    return {
        "deleted": normalized,
        "had_backlinks": entry_backlinks,
    }


@mcp.tool(
    name="tags",
    description=(
        "List all tags used in the knowledge base with usage counts. "
        "Helps discover available tags and find inconsistencies."
    ),
)
async def tags_tool(
    min_count: int = 1,
    include_entries: bool = False,
) -> list[dict]:
    """List all tags with usage counts.

    Args:
        min_count: Only include tags used at least this many times (default 1).
        include_entries: If True, include list of entry paths for each tag.

    Returns:
        List of {tag, count, entries?} sorted by count descending.
    """
    kb_root = get_kb_root()

    if not kb_root.exists():
        return []

    # Collect all tags
    tag_entries: dict[str, list[str]] = {}

    for md_file in kb_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue

        rel_path = str(md_file.relative_to(kb_root))

        try:
            metadata, _, _ = parse_entry(md_file)
        except ParseError:
            continue

        for tag in metadata.tags:
            if tag not in tag_entries:
                tag_entries[tag] = []
            tag_entries[tag].append(rel_path)

    # Build results
    results = []
    for tag, entries in sorted(tag_entries.items(), key=lambda x: -len(x[1])):
        if len(entries) >= min_count:
            result: dict = {"tag": tag, "count": len(entries)}
            if include_entries:
                result["entries"] = entries
            results.append(result)

    return results


@mcp.tool(
    name="health",
    description=(
        "Audit the knowledge base for problems: orphaned entries (no incoming links), "
        "broken links, stale content, and empty directories."
    ),
)
async def health_tool(
    stale_days: int = 90,
    check_orphans: bool = True,
    check_broken_links: bool = True,
    check_stale: bool = True,
    check_empty_dirs: bool = True,
) -> dict:
    """Audit KB health.

    Args:
        stale_days: Entries not updated in this many days are considered stale.
        check_orphans: Check for entries with no incoming backlinks.
        check_broken_links: Check for [[links]] pointing to non-existent entries.
        check_stale: Check for entries not updated recently.
        check_empty_dirs: Check for directories with no entries.

    Returns:
        Dict with lists of issues found in each category.
    """
    kb_root = get_kb_root()

    if not kb_root.exists():
        return {
            "orphans": [],
            "broken_links": [],
            "stale": [],
            "empty_dirs": [],
            "summary": {"total_issues": 0},
        }

    results: dict = {
        "orphans": [],
        "broken_links": [],
        "stale": [],
        "empty_dirs": [],
    }

    # Collect all entries and their metadata
    all_entries: dict[str, dict] = {}  # path -> {title, tags, created, updated, links}
    cutoff_date = date.today() - timedelta(days=stale_days)

    for md_file in kb_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue

        rel_path = str(md_file.relative_to(kb_root))
        path_key = rel_path[:-3] if rel_path.endswith(".md") else rel_path

        try:
            metadata, content, _ = parse_entry(md_file)
            links = extract_links(content)
        except ParseError:
            continue

        all_entries[path_key] = {
            "path": rel_path,
            "title": metadata.title,
            "tags": metadata.tags,
            "created": metadata.created,
            "updated": metadata.updated,
            "links": links,
        }

    # Get backlink index
    all_backlinks = _get_backlink_index()

    # Check for orphans (entries with no incoming backlinks)
    if check_orphans:
        for path_key, entry in all_entries.items():
            incoming = all_backlinks.get(path_key, [])
            if not incoming:
                results["orphans"].append({
                    "path": entry["path"],
                    "title": entry["title"],
                })

    # Check for broken links
    if check_broken_links:
        valid_targets = set(all_entries.keys())
        # Also accept paths with .md extension
        valid_targets.update(f"{k}.md" for k in all_entries.keys())
        # Also accept titles as link targets
        title_to_path = {e["title"]: k for k, e in all_entries.items()}

        for path_key, entry in all_entries.items():
            for link in entry["links"]:
                # Normalize link target
                link_normalized = link[:-3] if link.endswith(".md") else link

                # Check if link is valid (by path or by title)
                if (
                    link not in valid_targets
                    and link_normalized not in valid_targets
                    and link not in title_to_path
                ):
                    results["broken_links"].append({
                        "source": entry["path"],
                        "broken_link": link,
                    })

    # Check for stale entries
    if check_stale:
        for path_key, entry in all_entries.items():
            last_activity = entry["updated"] or entry["created"]
            if last_activity and last_activity < cutoff_date:
                days_old = (date.today() - last_activity).days
                results["stale"].append({
                    "path": entry["path"],
                    "title": entry["title"],
                    "last_activity": last_activity.isoformat(),
                    "days_old": days_old,
                })

    # Check for empty directories
    if check_empty_dirs:
        for dir_path in kb_root.rglob("*"):
            if not dir_path.is_dir():
                continue
            if dir_path.name.startswith(".") or dir_path.name.startswith("_"):
                continue

            # Check if directory has any .md files
            has_entries = any(
                f.suffix == ".md" and not f.name.startswith("_")
                for f in dir_path.rglob("*")
                if f.is_file()
            )

            if not has_entries:
                rel_dir = str(dir_path.relative_to(kb_root))
                results["empty_dirs"].append(rel_dir)

    # Add summary
    total_issues = sum(len(v) for v in results.values() if isinstance(v, list))
    results["summary"] = {
        "total_issues": total_issues,
        "orphans_count": len(results["orphans"]),
        "broken_links_count": len(results["broken_links"]),
        "stale_count": len(results["stale"]),
        "empty_dirs_count": len(results["empty_dirs"]),
        "total_entries": len(all_entries),
    }

    return results


def main():
    """Run the MCP server."""
    # Check for preload request via environment variable
    if os.environ.get("KB_PRELOAD", "").lower() in ("1", "true", "yes"):
        import sys
        print("Preloading embedding model...", file=sys.stderr)
        searcher = _get_searcher()
        searcher.preload()
        print("Embedding model ready.", file=sys.stderr)

    mcp.run()


if __name__ == "__main__":
    main()
