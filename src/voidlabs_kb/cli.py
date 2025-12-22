#!/usr/bin/env python3
"""
vl-kb: CLI for voidlabs knowledge base

Token-efficient alternative to MCP tools. Wraps existing voidlabs_kb functionality.

Usage:
    vl-kb search "query"              # Search entries
    vl-kb get path/to/entry.md        # Read an entry
    vl-kb add --title="..." --tags=.. # Create entry
    vl-kb tree                        # Browse structure
    vl-kb health                      # Audit KB health
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click

# Lazy imports to speed up CLI startup
# The heavy imports (chromadb, sentence-transformers) only load when needed


def run_async(coro):
    """Run async function synchronously."""
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────────────
# Output Formatting
# ─────────────────────────────────────────────────────────────────────────────


def format_table(rows: list[dict], columns: list[str], max_widths: Optional[dict] = None) -> str:
    """Format rows as a simple table."""
    if not rows:
        return ""

    max_widths = max_widths or {}
    widths = {col: len(col) for col in columns}

    for row in rows:
        for col in columns:
            val = str(row.get(col, ""))
            limit = max_widths.get(col, 50)
            if len(val) > limit:
                val = val[: limit - 3] + "..."
            widths[col] = max(widths[col], len(val))

    # Header
    header = "  ".join(col.upper().ljust(widths[col]) for col in columns)
    separator = "  ".join("-" * widths[col] for col in columns)

    # Rows
    lines = [header, separator]
    for row in rows:
        vals = []
        for col in columns:
            val = str(row.get(col, ""))
            limit = max_widths.get(col, 50)
            if len(val) > limit:
                val = val[: limit - 3] + "..."
            vals.append(val.ljust(widths[col]))
        lines.append("  ".join(vals))

    return "\n".join(lines)


def output(data, as_json: bool = False):
    """Output data as JSON or formatted text."""
    if as_json:
        click.echo(json.dumps(data, indent=2, default=str))
    else:
        click.echo(data)


# ─────────────────────────────────────────────────────────────────────────────
# Main CLI Group
# ─────────────────────────────────────────────────────────────────────────────


@click.group()
@click.version_option(version="0.1.0", prog_name="vl-kb")
def cli():
    """vl-kb: Token-efficient CLI for voidlabs knowledge base.

    Search, browse, and manage KB entries without MCP context overhead.

    \b
    Quick start:
      vl-kb search "deployment"     # Find entries
      vl-kb get tooling/beads.md    # Read an entry
      vl-kb tree                    # Browse structure
      vl-kb health                  # Check KB health
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Search Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("query")
@click.option("--tags", "-t", help="Filter by tags (comma-separated)")
@click.option("--mode", type=click.Choice(["hybrid", "keyword", "semantic"]), default="hybrid")
@click.option("--limit", "-n", default=10, help="Max results")
@click.option("--content", "-c", is_flag=True, help="Include full content in results")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def search(query: str, tags: Optional[str], mode: str, limit: int, content: bool, as_json: bool):
    """Search the knowledge base.

    \b
    Examples:
      vl-kb search "deployment"
      vl-kb search "docker" --tags=infrastructure
      vl-kb search "api" --mode=semantic --limit=5
    """
    from .server import _get_searcher, _get_current_project

    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    searcher = _get_searcher()
    project_context = _get_current_project()
    results = searcher.search(query, limit=limit, mode=mode, project_context=project_context)

    # Filter by tags if specified
    if tag_list:
        tag_set = set(tag_list)
        results = [r for r in results if tag_set.intersection(r.tags)]

    if as_json:
        output([{"path": r.path, "title": r.title, "score": r.score, "snippet": r.snippet} for r in results], as_json=True)
    else:
        if not results:
            click.echo("No results found.")
            return

        rows = [
            {"path": r.path, "title": r.title, "score": f"{r.score:.2f}"}
            for r in results
        ]
        click.echo(format_table(rows, ["path", "title", "score"], {"path": 40, "title": 35}))


# ─────────────────────────────────────────────────────────────────────────────
# Get Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("path")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON with metadata")
@click.option("--metadata", "-m", is_flag=True, help="Show only metadata")
def get(path: str, as_json: bool, metadata: bool):
    """Read a knowledge base entry.

    \b
    Examples:
      vl-kb get tooling/beads-issue-tracker.md
      vl-kb get tooling/beads-issue-tracker.md --json
      vl-kb get tooling/beads-issue-tracker.md --metadata
    """
    from .server import get_tool

    try:
        entry = run_async(get_tool(path=path))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        output(entry.model_dump(), as_json=True)
    elif metadata:
        click.echo(f"Title:    {entry.metadata.title}")
        click.echo(f"Tags:     {', '.join(entry.metadata.tags)}")
        click.echo(f"Created:  {entry.metadata.created}")
        click.echo(f"Updated:  {entry.metadata.updated or 'never'}")
        click.echo(f"Links:    {len(entry.links)}")
        click.echo(f"Backlinks: {len(entry.backlinks)}")
    else:
        # Human-readable: show header + content
        click.echo(f"# {entry.metadata.title}")
        click.echo(f"Tags: {', '.join(entry.metadata.tags)}")
        click.echo("-" * 60)
        click.echo(entry.content)


# ─────────────────────────────────────────────────────────────────────────────
# Add Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--title", "-t", required=True, help="Entry title")
@click.option("--tags", required=True, help="Tags (comma-separated)")
@click.option("--category", "-c", default="", help="Category/directory")
@click.option("--content", help="Content (or use --file/--stdin)")
@click.option("--file", "-f", "file_path", type=click.Path(exists=True), help="Read content from file")
@click.option("--stdin", is_flag=True, help="Read content from stdin")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def add(
    title: str,
    tags: str,
    category: str,
    content: Optional[str],
    file_path: Optional[str],
    stdin: bool,
    as_json: bool,
):
    """Create a new knowledge base entry.

    \b
    Examples:
      vl-kb add --title="My Entry" --tags="foo,bar" --content="# Content here"
      vl-kb add --title="My Entry" --tags="foo,bar" --file=content.md
      cat content.md | vl-kb add --title="My Entry" --tags="foo,bar" --stdin
    """
    from .server import add_tool

    # Resolve content source
    if stdin:
        content = sys.stdin.read()
    elif file_path:
        content = Path(file_path).read_text()
    elif not content:
        click.echo("Error: Must provide --content, --file, or --stdin", err=True)
        sys.exit(1)

    tag_list = [t.strip() for t in tags.split(",")]

    try:
        result = run_async(add_tool(title=title, content=content, tags=tag_list, category=category))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        output(result.model_dump(), as_json=True)
    else:
        click.echo(f"Created: {result.path}")
        if result.suggested_links:
            click.echo("\nSuggested links:")
            for link in result.suggested_links[:5]:
                click.echo(f"  - {link.path} ({link.score:.2f})")
        if result.suggested_tags:
            click.echo("\nSuggested tags:")
            for tag in result.suggested_tags[:5]:
                click.echo(f"  - {tag.tag} ({tag.reason})")


# ─────────────────────────────────────────────────────────────────────────────
# Update Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("path")
@click.option("--tags", help="New tags (comma-separated)")
@click.option("--content", help="New content")
@click.option("--file", "-f", "file_path", type=click.Path(exists=True), help="Read content from file")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def update(path: str, tags: Optional[str], content: Optional[str], file_path: Optional[str], as_json: bool):
    """Update an existing knowledge base entry.

    \b
    Examples:
      vl-kb update path/entry.md --tags="new,tags"
      vl-kb update path/entry.md --file=updated-content.md
    """
    from .server import update_tool

    if file_path:
        content = Path(file_path).read_text()

    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    try:
        result = run_async(update_tool(path=path, content=content, tags=tag_list))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        output(result.model_dump(), as_json=True)
    else:
        click.echo(f"Updated: {result.path}")


# ─────────────────────────────────────────────────────────────────────────────
# Tree Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("path", default="")
@click.option("--depth", "-d", default=3, help="Max depth")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def tree(path: str, depth: int, as_json: bool):
    """Display knowledge base directory structure.

    \b
    Examples:
      vl-kb tree
      vl-kb tree tooling --depth=2
    """
    from .server import tree_tool

    result = run_async(tree_tool(path=path, depth=depth))

    if as_json:
        output(result.model_dump(), as_json=True)
    else:
        click.echo(result.tree)


# ─────────────────────────────────────────────────────────────────────────────
# List Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command("list")
@click.option("--tag", "-t", help="Filter by tag")
@click.option("--category", "-c", help="Filter by category")
@click.option("--limit", "-n", default=20, help="Max results")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def list_entries(tag: Optional[str], category: Optional[str], limit: int, as_json: bool):
    """List knowledge base entries.

    \b
    Examples:
      vl-kb list
      vl-kb list --tag=tooling
      vl-kb list --category=infrastructure --limit=10
    """
    from .server import list_tool

    result = run_async(list_tool(tag=tag, category=category, limit=limit))

    if as_json:
        output([e.model_dump() for e in result.entries], as_json=True)
    else:
        if not result.entries:
            click.echo("No entries found.")
            return

        rows = [{"path": e.path, "title": e.title} for e in result.entries]
        click.echo(format_table(rows, ["path", "title"], {"path": 45, "title": 40}))


# ─────────────────────────────────────────────────────────────────────────────
# What's New Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command("whats-new")
@click.option("--days", "-d", default=30, help="Look back N days")
@click.option("--limit", "-n", default=10, help="Max results")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def whats_new(days: int, limit: int, as_json: bool):
    """Show recently created or updated entries.

    \b
    Examples:
      vl-kb whats-new
      vl-kb whats-new --days=7 --limit=5
    """
    from .server import whats_new_tool

    result = run_async(whats_new_tool(days=days, limit=limit))

    if as_json:
        output([e.model_dump() for e in result.entries], as_json=True)
    else:
        if not result.entries:
            click.echo(f"No entries created or updated in the last {days} days.")
            return

        rows = [
            {"path": e.path, "title": e.title, "date": str(e.activity_date)[:10]}
            for e in result.entries
        ]
        click.echo(format_table(rows, ["path", "title", "date"], {"path": 40, "title": 30}))


# ─────────────────────────────────────────────────────────────────────────────
# Health Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def health(as_json: bool):
    """Audit knowledge base for problems.

    Checks for orphaned entries, broken links, stale content, empty directories.

    \b
    Examples:
      vl-kb health
      vl-kb health --json
    """
    from .server import health_tool

    result = run_async(health_tool())

    if as_json:
        output(result.model_dump(), as_json=True)
    else:
        click.echo("Knowledge Base Health Report")
        click.echo("=" * 40)

        # Orphans
        if result.orphans:
            click.echo(f"\n⚠ Orphaned entries ({len(result.orphans)}):")
            for o in result.orphans[:10]:
                click.echo(f"  - {o}")
        else:
            click.echo("\n✓ No orphaned entries")

        # Broken links
        if result.broken_links:
            click.echo(f"\n⚠ Broken links ({len(result.broken_links)}):")
            for bl in result.broken_links[:10]:
                click.echo(f"  - {bl.source} -> {bl.target}")
        else:
            click.echo("\n✓ No broken links")

        # Stale
        if result.stale:
            click.echo(f"\n⚠ Stale entries ({len(result.stale)}):")
            for s in result.stale[:10]:
                click.echo(f"  - {s}")
        else:
            click.echo("\n✓ No stale entries")

        # Empty dirs
        if result.empty_dirs:
            click.echo(f"\n⚠ Empty directories ({len(result.empty_dirs)}):")
            for d in result.empty_dirs[:10]:
                click.echo(f"  - {d}")
        else:
            click.echo("\n✓ No empty directories")


# ─────────────────────────────────────────────────────────────────────────────
# Tags Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--min-count", default=1, help="Minimum usage count")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def tags(min_count: int, as_json: bool):
    """List all tags with usage counts.

    \b
    Examples:
      vl-kb tags
      vl-kb tags --min-count=3
    """
    from .server import tags_tool

    result = run_async(tags_tool(min_count=min_count))

    if as_json:
        output(result.model_dump(), as_json=True)
    else:
        if not result.tags:
            click.echo("No tags found.")
            return

        for tag in result.tags:
            click.echo(f"  {tag.tag}: {tag.count}")


# ─────────────────────────────────────────────────────────────────────────────
# Hubs Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--limit", "-n", default=10, help="Max results")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def hubs(limit: int, as_json: bool):
    """Show most connected entries (hub notes).

    These are key concepts that many other entries link to.

    \b
    Examples:
      vl-kb hubs
      vl-kb hubs --limit=5
    """
    from .server import hubs_tool

    result = run_async(hubs_tool(limit=limit))

    if as_json:
        output(result, as_json=True)
    else:
        if not result:
            click.echo("No hub entries found.")
            return

        rows = [{"path": h["path"], "backlinks": h["backlink_count"]} for h in result]
        click.echo(format_table(rows, ["path", "backlinks"], {"path": 50}))


# ─────────────────────────────────────────────────────────────────────────────
# Suggest Links Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command("suggest-links")
@click.argument("path")
@click.option("--limit", "-n", default=5, help="Max suggestions")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def suggest_links(path: str, limit: int, as_json: bool):
    """Suggest entries to link to based on semantic similarity.

    \b
    Examples:
      vl-kb suggest-links tooling/my-entry.md
    """
    from .server import suggest_links_tool

    result = run_async(suggest_links_tool(path=path, limit=limit))

    if as_json:
        output(result.model_dump(), as_json=True)
    else:
        if not result.suggestions:
            click.echo("No link suggestions found.")
            return

        click.echo(f"Suggested links for {path}:\n")
        for s in result.suggestions:
            click.echo(f"  {s.path} ({s.score:.2f})")
            click.echo(f"    {s.reason}")


# ─────────────────────────────────────────────────────────────────────────────
# Reindex Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
def reindex():
    """Rebuild search indices from all markdown files.

    Use this after bulk imports or if search results seem stale.

    \b
    Examples:
      vl-kb reindex
    """
    from .server import reindex_tool

    click.echo("Reindexing knowledge base...")
    result = run_async(reindex_tool())
    click.echo(f"✓ Indexed {result.total_entries} entries, {result.total_chunks} chunks")


# ─────────────────────────────────────────────────────────────────────────────
# Delete Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("path")
@click.option("--force", "-f", is_flag=True, help="Delete even if has backlinks")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def delete(path: str, force: bool, as_json: bool):
    """Delete a knowledge base entry.

    \b
    Examples:
      vl-kb delete path/to/entry.md
      vl-kb delete path/to/entry.md --force
    """
    from .server import delete_tool

    try:
        result = run_async(delete_tool(path=path, force=force))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        output(result, as_json=True)
    else:
        if result.get("warning"):
            click.echo(f"Warning: {result['warning']}", err=True)
        click.echo(f"Deleted: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """Entry point for vl-kb CLI."""
    cli()


if __name__ == "__main__":
    main()
