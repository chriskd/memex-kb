"""FastMCP server for voidlabs-kb.

This module provides MCP protocol wrappers around the core business logic.
All actual logic lives in core.py - this file just handles MCP serialization.
"""

import os
from typing import Literal

from fastmcp import FastMCP

from . import core
from .models import IndexStatus, KBEntry, QualityReport, SearchResponse


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

Beads integration:
- Link KB entries to beads issues using frontmatter fields:
  - `beads_issues: [issue-id1, issue-id2]` - link to specific issues
  - `beads_project: project-name` - link to all issues in a project
- Use this to connect documentation with related work items
""",
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
    """Search the knowledge base."""
    return await core.search(
        query=query,
        limit=limit,
        mode=mode,
        tags=tags,
        include_content=include_content,
    )


@mcp.tool(
    name="quality",
    description="Run the built-in search accuracy evaluation suite and return metrics.",
)
async def quality_tool(limit: int = 5, cutoff: int = 3) -> QualityReport:
    """Evaluate MCP search accuracy."""
    return await core.quality(limit=limit, cutoff=cutoff)


@mcp.tool(
    name="add",
    description=(
        "Create a new knowledge base entry. Generates a slug from the title "
        "and places the entry in the specified category or directory. "
        "Returns the path, suggested links to related entries, and suggested "
        "additional tags based on content similarity and existing taxonomy."
    ),
)
async def add_tool(
    title: str,
    content: str,
    tags: list[str],
    category: str = "",
    directory: str | None = None,
    links: list[str] | None = None,
) -> dict:
    """Create a new KB entry."""
    return await core.add_entry(
        title=title,
        content=content,
        tags=tags,
        category=category,
        directory=directory,
        links=links,
    )


@mcp.tool(
    name="update",
    description=(
        "Update an existing knowledge base entry. "
        "Preserves frontmatter and updates the 'updated' date. "
        "Returns the path, suggested links to related entries, and suggested "
        "additional tags based on content similarity and existing taxonomy."
    ),
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
    name="link_beads",
    description=(
        "Link a KB entry to beads issues. "
        "Use this to connect documentation with related work items."
    ),
)
async def link_beads_tool(
    path: str,
    issues: list[str] | None = None,
    project: str | None = None,
) -> dict:
    """Link a KB entry to beads issues."""
    return await core.link_beads(path=path, issues=issues, project=project)


@mcp.tool(
    name="register_beads_project",
    description=(
        "Register a beads project for cross-project issue resolution. "
        "Use this when starting work on a project so its issues can be "
        "resolved when linked from KB entries."
    ),
)
async def register_beads_project_tool(
    path: str,
    prefix: str | None = None,
) -> dict:
    """Register a beads project in the KB registry."""
    return await core.register_beads_project(path=path, prefix=prefix)


@mcp.tool(
    name="get",
    description=(
        "Get a full knowledge base entry including metadata, content, links, and backlinks."
    ),
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
    recursive: bool = True,
    limit: int = 20,
) -> list[dict]:
    """List KB entries."""
    return await core.list_entries(
        category=category,
        directory=directory,
        tag=tag,
        recursive=recursive,
        limit=limit,
    )


@mcp.tool(
    name="whats_new",
    description=(
        "List recently created or updated knowledge base entries, "
        "sorted by most recent activity first. "
        "Use 'project' to filter by project name (matches path, source_project metadata, or tags)."
    ),
)
async def whats_new_tool(
    days: int = 30,
    limit: int = 10,
    include_created: bool = True,
    include_updated: bool = True,
    category: str | None = None,
    tag: str | None = None,
    project: str | None = None,
) -> list[dict]:
    """List recent KB entries."""
    return await core.whats_new(
        days=days,
        limit=limit,
        include_created=include_created,
        include_updated=include_updated,
        category=category,
        tag=tag,
        project=project,
    )


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
    """List popular KB entries by view count."""
    return await core.popular(
        limit=limit,
        days=days,
        category=category,
        tag=tag,
    )


@mcp.tool(
    name="backlinks",
    description="Find all entries that link to a specific entry.",
)
async def backlinks_tool(path: str) -> list[str]:
    """Find entries that link to this path."""
    return await core.backlinks(path=path)


@mcp.tool(
    name="reindex",
    description="Rebuild the search indices from all markdown files in the knowledge base.",
)
async def reindex_tool() -> IndexStatus:
    """Rebuild search indices."""
    return await core.reindex()


@mcp.tool(
    name="tree",
    description="Display the directory structure of the knowledge base or a subdirectory.",
)
async def tree_tool(
    path: str = "",
    depth: int = 3,
    include_files: bool = True,
) -> dict:
    """Show directory tree."""
    return await core.tree(path=path, depth=depth, include_files=include_files)


@mcp.tool(
    name="mkdir",
    description="Create a new directory within the knowledge base hierarchy.",
)
async def mkdir_tool(path: str) -> str:
    """Create a directory."""
    return await core.mkdir(path=path)


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
    """Move entry or directory."""
    return await core.move(
        source=source,
        destination=destination,
        update_links=update_links,
    )


@mcp.tool(
    name="rmdir",
    description="Remove an empty directory from the knowledge base.",
)
async def rmdir_tool(path: str, force: bool = False) -> str:
    """Remove directory."""
    return await core.rmdir(path=path, force=force)


@mcp.tool(
    name="delete",
    description=(
        "Delete a knowledge base entry. "
        "Warns if other entries link to it (has backlinks). "
        "Use force=True to delete anyway."
    ),
)
async def delete_tool(path: str, force: bool = False) -> dict:
    """Delete an entry."""
    return await core.delete_entry(path=path, force=force)


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
    """List all tags with usage counts."""
    return await core.tags(min_count=min_count, include_entries=include_entries)


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
    """Audit KB health."""
    return await core.health(
        stale_days=stale_days,
        check_orphans=check_orphans,
        check_broken_links=check_broken_links,
        check_stale=check_stale,
        check_empty_dirs=check_empty_dirs,
    )


@mcp.tool(
    name="hubs",
    description=(
        "Find the most connected entries in the knowledge base (hub notes). "
        "These are key concepts that many other entries link to."
    ),
)
async def hubs_tool(limit: int = 10) -> list[dict]:
    """Find hub entries with most connections."""
    return await core.hubs(limit=limit)


@mcp.tool(
    name="dead_ends",
    description=(
        "Find dead-end entries: notes that receive links but don't link out. "
        "These may need more context or related links added."
    ),
)
async def dead_ends_tool(limit: int = 10) -> list[dict]:
    """Find entries with incoming links but no outgoing links."""
    return await core.dead_ends(limit=limit)


@mcp.tool(
    name="suggest_links",
    description=(
        "Suggest entries to link to based on semantic similarity. "
        "Uses embeddings to find conceptually related entries that aren't already linked. "
        "Only returns suggestions above the min_score threshold (default 0.5)."
    ),
)
async def suggest_links_tool(
    path: str,
    limit: int = 5,
    min_score: float = 0.5,
) -> list[dict]:
    """Suggest links to add to an entry based on content similarity."""
    return await core.suggest_links(path=path, limit=limit, min_score=min_score)


def main():
    """Run the MCP server."""
    # Check for preload request via environment variable
    if os.environ.get("KB_PRELOAD", "").lower() in ("1", "true", "yes"):
        import sys
        print("Preloading embedding model...", file=sys.stderr)
        searcher = core.get_searcher()
        searcher.preload()
        print("Embedding model ready.", file=sys.stderr)

    mcp.run()


if __name__ == "__main__":
    main()
