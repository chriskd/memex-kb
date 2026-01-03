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

# ─────────────────────────────────────────────────────────────────────────────
# Input Validation
# ─────────────────────────────────────────────────────────────────────────────


class ValidationError(ValueError):
    """Raised when MCP tool input validation fails."""

    pass


def _validate_non_empty_string(value: str, field_name: str) -> str:
    """Validate that a string is not empty or whitespace-only.

    Args:
        value: The string to validate.
        field_name: Name of the field for error messages.

    Returns:
        The stripped string value.

    Raises:
        ValidationError: If the value is empty or whitespace-only.
    """
    if not value or not value.strip():
        raise ValidationError(f"{field_name} cannot be empty or whitespace-only")
    return value.strip()


def _validate_tags(tags: list[str]) -> list[str]:
    """Validate tags list.

    Args:
        tags: List of tags to validate.

    Returns:
        List of validated, stripped tags.

    Raises:
        ValidationError: If tags list is empty or contains invalid tags.
    """
    if not tags:
        raise ValidationError("tags cannot be empty - at least one tag is required")

    validated_tags = []
    for i, tag in enumerate(tags):
        if not isinstance(tag, str):
            raise ValidationError(f"tags[{i}] must be a string, got {type(tag).__name__}")
        stripped = tag.strip()
        if not stripped:
            raise ValidationError(f"tags[{i}] cannot be empty or whitespace-only")
        validated_tags.append(stripped)

    return validated_tags


def _validate_path(path: str) -> str:
    """Validate a path for path traversal attempts.

    Args:
        path: The path to validate.

    Returns:
        The validated path.

    Raises:
        ValidationError: If the path contains path traversal attempts.
    """
    if not path or not path.strip():
        raise ValidationError("path cannot be empty")

    path = path.strip()

    # Check for path traversal
    if ".." in path:
        raise ValidationError("path cannot contain '..' (path traversal not allowed)")

    if path.startswith("/"):
        raise ValidationError("path must be relative (cannot start with '/')")

    return path


def _validate_section_updates(section_updates: dict[str, str] | None) -> dict[str, str] | None:
    """Validate section_updates dictionary.

    Args:
        section_updates: Dictionary of section name -> new content.

    Returns:
        Validated section_updates or None.

    Raises:
        ValidationError: If section_updates contains invalid data.
    """
    if section_updates is None:
        return None

    if not isinstance(section_updates, dict):
        raise ValidationError(
            f"section_updates must be a dictionary, got {type(section_updates).__name__}"
        )

    validated = {}
    for section, content in section_updates.items():
        if not isinstance(section, str):
            raise ValidationError(
                f"section_updates key must be a string, got {type(section).__name__}"
            )
        if not isinstance(content, str):
            raise ValidationError(
                f"section_updates['{section}'] must be a string, got {type(content).__name__}"
            )

        section_stripped = section.strip()
        if not section_stripped:
            raise ValidationError("section_updates key cannot be empty or whitespace-only")

        validated[section_stripped] = content

    return validated


mcp = FastMCP(
    name="memex",
    instructions=(
        "Knowledge base with semantic search. "
        "Use search/get/add/update for entries. Use [[links]] for connections."
    ),
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
        "Returns ranked entries."
    ),
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
    # Input validation - fail fast at MCP layer
    validated_title = _validate_non_empty_string(title, "title")
    validated_content = _validate_non_empty_string(content, "content")
    validated_tags = _validate_tags(tags)

    return await core.add_entry(
        title=validated_title,
        content=validated_content,
        tags=validated_tags,
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
    # Input validation - fail fast at MCP layer
    validated_path = _validate_path(path)
    validated_section_updates = _validate_section_updates(section_updates)

    # Validate tags if provided
    validated_tags = _validate_tags(tags) if tags is not None else None

    # Validate content if provided (not None, but can be empty string intentionally)
    validated_content = content
    if content is not None:
        # Allow empty string for content replacement (user might want to clear content)
        # but reject whitespace-only as likely unintended
        if content and not content.strip():
            raise ValidationError("content cannot be whitespace-only (use empty string to clear)")

    return await core.update_entry(
        path=validated_path,
        content=validated_content,
        tags=validated_tags,
        section_updates=validated_section_updates,
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
