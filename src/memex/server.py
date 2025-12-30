"""FastMCP server for memex.

This module provides MCP protocol wrappers around the core business logic.
All actual logic lives in core.py - this file just handles MCP serialization.
"""

import os
from typing import Literal

from fastmcp import FastMCP

from . import core
from .config import DEFAULT_SEARCH_LIMIT, LINK_SUGGESTION_MIN_SCORE
from .models import KBEntry, QualityReport, SearchResponse


mcp = FastMCP(
    name="memex",
    instructions="Knowledge base with semantic search. Use search/get/add/update for entries. Use [[links]] for connections.",
)


# ─────────────────────────────────────────────────────────────────────────────
# Re-export commonly used helpers for backwards compatibility
# ─────────────────────────────────────────────────────────────────────────────

# These are used by cli.py for the search command workaround
_get_searcher = core.get_searcher
_get_current_project = core.get_current_project


# ─────────────────────────────────────────────────────────────────────────────
# MCP Tool Wrappers
# ─────────────────────────────────────────────────────────────────────────────


@mcp.tool(
    name="search",
    description="Search the knowledge base using hybrid keyword + semantic search. Returns ranked entries.",
)
async def search_tool(
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
    mode: Literal["hybrid", "keyword", "semantic"] = "hybrid",
    tags: list[str] | None = None,
    include_content: bool = False,
) -> SearchResponse:
    """Search the knowledge base."""
    return await core.search(
        query=query,
        limit=limit,
        mode=mode,
        tags=tags,
        include_content=include_content,
    )


@mcp.tool(
    name="add",
    description=(
        "Create a new knowledge base entry. "
        "Checks for potential duplicates first. "
        "If duplicates detected, returns created=False with warning. "
        "Use force=True to bypass duplicate check."
    ),
)
async def add_tool(
    title: str,
    content: str,
    tags: list[str],
    category: str = "",
    directory: str | None = None,
    links: list[str] | None = None,
    force: bool = False,
) -> core.AddEntryResponse:
    """Create a new KB entry with duplicate detection."""
    return await core.add_entry(
        title=title,
        content=content,
        tags=tags,
        category=category,
        directory=directory,
        links=links,
        force=force,
    )


@mcp.tool(
    name="update",
    description="Update an existing knowledge base entry. Preserves frontmatter, updates date.",
)
async def update_tool(
    path: str,
    content: str | None = None,
    tags: list[str] | None = None,
    section_updates: dict[str, str] | None = None,
) -> dict:
    """Update an existing KB entry."""
    return await core.update_entry(
        path=path,
        content=content,
        tags=tags,
        section_updates=section_updates,
    )


@mcp.tool(
    name="get",
    description="Get a knowledge base entry with metadata, content, and links.",
)
async def get_tool(path: str) -> KBEntry:
    """Read a KB entry."""
    return await core.get_entry(path=path)


@mcp.tool(
    name="list",
    description="List knowledge base entries, optionally filtered by category, directory, or tag.",
)
async def list_tool(
    category: str | None = None,
    directory: str | None = None,
    tag: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """List KB entries."""
    return await core.list_entries(
        category=category, directory=directory, tag=tag, limit=limit
    )


@mcp.tool(
    name="whats_new",
    description="List recently created or updated entries, sorted by recency.",
)
async def whats_new_tool(
    days: int = 30,
    limit: int = DEFAULT_SEARCH_LIMIT,
    project: str | None = None,
) -> list[dict]:
    """List recent KB entries."""
    return await core.whats_new(days=days, limit=limit, project=project)


@mcp.tool(
    name="backlinks",
    description="Find all entries that link to a specific entry.",
)
async def backlinks_tool(path: str) -> list[str]:
    """Find entries that link to this path."""
    return await core.backlinks(path=path)


@mcp.tool(
    name="tree",
    description="Display the directory structure of the knowledge base or a subdirectory.",
)
async def tree_tool(path: str = "", depth: int = 3) -> dict:
    """Show directory tree."""
    return await core.tree(path=path, depth=depth)


@mcp.tool(
    name="mkdir",
    description="Create a new directory within the knowledge base hierarchy.",
)
async def mkdir_tool(path: str) -> str:
    """Create a directory."""
    return await core.mkdir(path=path)


@mcp.tool(
    name="move",
    description="Move an entry or directory. Updates links automatically.",
)
async def move_tool(source: str, destination: str) -> dict:
    """Move entry or directory."""
    return await core.move(source=source, destination=destination)


@mcp.tool(
    name="rmdir",
    description="Remove an empty directory from the knowledge base.",
)
async def rmdir_tool(path: str, force: bool = False) -> str:
    """Remove directory."""
    return await core.rmdir(path=path, force=force)


@mcp.tool(
    name="delete",
    description="Delete an entry. Warns if other entries link to it.",
)
async def delete_tool(path: str, force: bool = False) -> dict:
    """Delete an entry."""
    return await core.delete_entry(path=path, force=force)


@mcp.tool(
    name="tags",
    description="List all tags with usage counts.",
)
async def tags_tool() -> list[dict]:
    """List all tags with usage counts."""
    return await core.tags()


@mcp.tool(
    name="health",
    description="Audit KB for orphans, broken links, stale content.",
)
async def health_tool(stale_days: int = 90) -> dict:
    """Audit KB health."""
    return await core.health(stale_days=stale_days)


@mcp.tool(
    name="quality",
    description="Evaluate KB search accuracy against test queries.",
)
async def quality_tool(limit: int = 5, cutoff: int = 3) -> QualityReport:
    """Run KB quality checks."""
    return await core.quality(limit=limit, cutoff=cutoff)


@mcp.tool(
    name="suggest_links",
    description="Suggest related entries to link to based on semantic similarity.",
)
async def suggest_links_tool(
    path: str,
    limit: int = 5,
    min_score: float = LINK_SUGGESTION_MIN_SCORE,
) -> list[dict]:
    """Suggest links to add to an entry based on content similarity."""
    return await core.suggest_links(path=path, limit=limit, min_score=min_score)


def main():
    """Run the MCP server."""
    import logging
    from ._logging import configure_logging

    configure_logging()
    log = logging.getLogger(__name__)

    # Check for preload request via environment variable
    if os.environ.get("MEMEX_PRELOAD", "").lower() in ("1", "true", "yes"):
        log.info("Preloading embedding model...")
        searcher = core.get_searcher()
        searcher.preload()
        log.info("Embedding model ready")

    mcp.run()


if __name__ == "__main__":
    main()
