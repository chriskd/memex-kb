"""FastMCP server for voidlabs-kb."""

import re
import subprocess
from datetime import date
from typing import Literal

from fastmcp import FastMCP

from .config import get_kb_root
from .indexer import HybridSearcher
from .models import IndexStatus, KBEntry, SearchResult
from .parser import ParseError, extract_links, parse_entry, resolve_backlinks

mcp = FastMCP(
    name="voidlabs-kb",
    instructions="""
Voidlabs Knowledge Base - Organization-wide knowledge with semantic search.

Use `search` to find documentation, patterns, and operational guides.
Use `add` to contribute new knowledge entries.
Use `sync` to commit and push your KB changes.

Best practices:
- Search before creating to avoid duplicates
- Use consistent tags from the existing taxonomy
- Add [[bidirectional links]] to related entries
- Run sync after making significant contributions
""",
)

# Lazy-initialized searcher
_searcher: HybridSearcher | None = None


def _get_searcher() -> HybridSearcher:
    """Get the HybridSearcher, initializing lazily."""
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher()
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


@mcp.tool(
    name="search",
    description=(
        "Search the knowledge base using hybrid keyword + semantic search. "
        "Returns relevant entries ranked by relevance."
    ),
)
async def search_tool(
    query: str,
    limit: int = 10,
    mode: Literal["hybrid", "keyword", "semantic"] = "hybrid",
    tags: list[str] | None = None,
) -> list[SearchResult]:
    """Search the knowledge base.

    Args:
        query: Search query string.
        limit: Maximum number of results (default 10).
        mode: Search mode - "hybrid" (default), "keyword", or "semantic".
        tags: Optional list of tags to filter results.

    Returns:
        List of SearchResult objects.
    """
    searcher = _get_searcher()
    results = searcher.search(query, limit=limit, mode=mode)

    # Filter by tags if specified
    if tags:
        tag_set = set(tags)
        results = [r for r in results if tag_set.intersection(r.tags)]

    return results


@mcp.tool(
    name="add",
    description=(
        "Create a new knowledge base entry. Generates a slug from the title "
        "and places the entry in the specified category."
    ),
)
async def add_tool(
    title: str,
    content: str,
    tags: list[str],
    category: str,
    links: list[str] | None = None,
) -> str:
    """Create a new KB entry.

    Args:
        title: Entry title.
        content: Markdown content (without frontmatter).
        tags: List of tags for the entry.
        category: Category directory (e.g., "development", "devops").
        links: Optional list of paths to link to using [[link]] syntax.

    Returns:
        Path of the created file.
    """
    kb_root = get_kb_root()
    valid_categories = _get_valid_categories()

    if category not in valid_categories:
        raise ValueError(
            f"Invalid category '{category}'. Valid categories: {', '.join(valid_categories)}"
        )

    if not tags:
        raise ValueError("At least one tag is required")

    # Generate slug and path
    slug = _slugify(title)
    if not slug:
        raise ValueError("Title must contain at least one alphanumeric character")

    file_path = kb_root / category / f"{slug}.md"
    rel_path = f"{category}/{slug}.md"

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
    frontmatter = f"""---
title: {title}
tags:
{tags_yaml}
created: {today}
---

"""
    # Write the file
    file_path.write_text(frontmatter + final_content, encoding="utf-8")

    # Reindex the new file
    try:
        metadata, raw_content, chunks = parse_entry(file_path)
        searcher = _get_searcher()
        if chunks:
            searcher.index_chunks(chunks)
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
    content: str,
    tags: list[str] | None = None,
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

    # Parse existing entry to get metadata
    try:
        metadata, old_content, _ = parse_entry(file_path)
    except ParseError as e:
        raise ValueError(f"Failed to parse existing entry: {e}") from e

    # Update metadata
    today = date.today()
    new_tags = tags if tags is not None else list(metadata.tags)

    if not new_tags:
        raise ValueError("At least one tag is required")

    # Build updated frontmatter
    tags_yaml = "\n".join(f"  - {tag}" for tag in new_tags)
    contributors_yaml = "\n".join(f"  - {c}" for c in metadata.contributors)
    aliases_yaml = "\n".join(f"  - {a}" for a in metadata.aliases)

    frontmatter_parts = [
        "---",
        f"title: {metadata.title}",
        "tags:",
        tags_yaml,
        f"created: {metadata.created.isoformat()}",
        f"updated: {today.isoformat()}",
    ]

    if metadata.contributors:
        frontmatter_parts.append("contributors:")
        frontmatter_parts.append(contributors_yaml)

    if metadata.aliases:
        frontmatter_parts.append("aliases:")
        frontmatter_parts.append(aliases_yaml)

    if metadata.status != "published":
        frontmatter_parts.append(f"status: {metadata.status}")

    frontmatter_parts.append("---\n\n")
    frontmatter = "\n".join(frontmatter_parts)

    # Write updated file
    file_path.write_text(frontmatter + content, encoding="utf-8")

    # Reindex
    searcher = _get_searcher()
    try:
        # Remove old index entries
        searcher.delete_document(path)
        # Parse and index new content
        _, _, chunks = parse_entry(file_path)
        if chunks:
            searcher.index_chunks(chunks)
    except (ParseError, Exception):
        # Indexing failed but file was updated
        pass

    return path


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
    all_backlinks = resolve_backlinks(kb_root)
    # Normalize path for lookup (remove .md extension)
    path_key = path[:-3] if path.endswith(".md") else path
    entry_backlinks = all_backlinks.get(path_key, [])

    return KBEntry(
        path=path,
        metadata=metadata,
        content=content,
        links=links,
        backlinks=entry_backlinks,
    )


@mcp.tool(
    name="list",
    description="List knowledge base entries, optionally filtered by category or tag.",
)
async def list_tool(
    category: str | None = None,
    tag: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """List KB entries.

    Args:
        category: Optional category to filter by.
        tag: Optional tag to filter by.
        limit: Maximum number of entries to return (default 20).

    Returns:
        List of {path, title, tags} dictionaries.
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

    results = []

    for md_file in search_path.rglob("*.md"):
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
    kb_root = get_kb_root()

    # Normalize path (remove .md extension for lookup)
    path_key = path[:-3] if path.endswith(".md") else path

    all_backlinks = resolve_backlinks(kb_root)
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

    status = searcher.status()
    return status


@mcp.tool(
    name="sync",
    description="Commit and push knowledge base changes to git.",
)
async def sync_tool(message: str | None = None) -> str:
    """Commit and push KB changes.

    Args:
        message: Optional commit message. Defaults to "Update knowledge base".

    Returns:
        Success or failure message.
    """
    kb_root = get_kb_root()
    commit_message = message or "Update knowledge base"

    try:
        # Check for changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=kb_root,
            capture_output=True,
            text=True,
            check=True,
        )

        if not result.stdout.strip():
            return "No changes to commit"

        # Stage all changes in kb directory
        subprocess.run(
            ["git", "add", "."],
            cwd=kb_root,
            capture_output=True,
            text=True,
            check=True,
        )

        # Commit
        subprocess.run(
            ["git", "commit", "-m", commit_message],
            cwd=kb_root,
            capture_output=True,
            text=True,
            check=True,
        )

        # Push
        push_result = subprocess.run(
            ["git", "push"],
            cwd=kb_root,
            capture_output=True,
            text=True,
        )

        if push_result.returncode != 0:
            return f"Committed but push failed: {push_result.stderr}"

        return f"Successfully committed and pushed: {commit_message}"

    except subprocess.CalledProcessError as e:
        return f"Git operation failed: {e.stderr or str(e)}"
    except FileNotFoundError:
        return "Git not found. Please ensure git is installed."
    except Exception as e:
        return f"Sync failed: {e}"


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
