#!/usr/bin/env python3
"""
mx: CLI for memex knowledge base

Token-efficient alternative to MCP tools. Wraps existing memex functionality.

Usage:
    mx search "query"              # Search entries
    mx get path/to/entry.md        # Read an entry
    mx add --title="..." --tags=.. # Create entry
    mx tree                        # Browse structure
    mx health                      # Audit KB health
"""

import asyncio
import difflib
import json
import sys
from pathlib import Path
from typing import Optional

import click
from click.exceptions import ClickException, UsageError

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


def format_json_error(code: str, message: str, details: Optional[dict] = None) -> str:
    """Format an error as JSON for --json-errors output."""
    error = {"error": {"code": code, "message": message}}
    if details:
        error["error"]["details"] = details
    return json.dumps(error)


def get_error_code_for_exception(exc: Exception) -> str:
    """Map Click exceptions to error codes."""
    if isinstance(exc, click.BadParameter):
        return "INVALID_ARGUMENT"
    elif isinstance(exc, click.MissingParameter):
        return "MISSING_ARGUMENT"
    elif isinstance(exc, click.NoSuchOption):
        return "UNKNOWN_OPTION"
    elif isinstance(exc, UsageError):
        return "USAGE_ERROR"
    elif isinstance(exc, ClickException):
        return "CLI_ERROR"
    return "UNKNOWN_ERROR"


# ─────────────────────────────────────────────────────────────────────────────
# JSON Error Handling
# ─────────────────────────────────────────────────────────────────────────────


class JsonErrorGroup(click.Group):
    """Custom Click group that formats errors as JSON when --json-errors is set.

    This handles Click validation errors (bad option values, missing args, etc.)
    that occur before the command callback is invoked. Also provides typo
    suggestions for unknown commands.
    """

    def resolve_command(self, ctx, args):
        """Override to suggest similar commands for typos."""
        try:
            return super().resolve_command(ctx, args)
        except UsageError as e:
            # Check if this is a "No such command" error
            cmd_name = args[0] if args else ""
            if cmd_name and "No such command" in str(e):
                matches = difflib.get_close_matches(
                    cmd_name, self.list_commands(ctx), n=1, cutoff=0.6
                )
                if matches:
                    raise UsageError(
                        f"No such command '{cmd_name}'. Did you mean '{matches[0]}'?"
                    )
            raise

    def invoke(self, ctx):
        """Override invoke to catch and format errors."""
        try:
            return super().invoke(ctx)
        except ClickException as e:
            if ctx.params.get("json_errors"):
                code = get_error_code_for_exception(e)
                click.echo(format_json_error(code, e.format_message()), err=True)
                ctx.exit(1)
            raise

    def main(self, *args, standalone_mode=True, **kwargs):
        """Override main to catch errors during argument parsing.

        This catches errors that happen before invoke() is called,
        such as invalid option types or missing required arguments.
        """
        # We need to check if --json-errors was passed before Click parses it
        # because validation errors can occur during parsing itself.
        # Check sys.argv directly for the flag.
        json_errors_requested = "--json-errors" in sys.argv

        if not json_errors_requested:
            return super().main(*args, standalone_mode=standalone_mode, **kwargs)

        try:
            return super().main(*args, standalone_mode=standalone_mode, **kwargs)
        except SystemExit as e:
            # Click calls sys.exit() on errors; re-raise to preserve exit code
            raise
        except ClickException as e:
            # Format as JSON and exit
            code = get_error_code_for_exception(e)
            click.echo(format_json_error(code, e.format_message()), err=True)
            sys.exit(1)
        except Exception as e:
            # Unexpected errors
            click.echo(format_json_error("INTERNAL_ERROR", str(e)), err=True)
            sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Main CLI Group
# ─────────────────────────────────────────────────────────────────────────────


@click.group(cls=JsonErrorGroup)
@click.version_option(version="0.1.0", prog_name="mx")
@click.option("--json-errors", "json_errors", is_flag=True,
              help="Output errors as JSON (for programmatic use)")
@click.pass_context
def cli(ctx, json_errors: bool):
    """mx: Token-efficient CLI for memex knowledge base.

    Search, browse, and manage KB entries without MCP context overhead.

    \b
    Quick start:
      mx search "deployment"     # Find entries
      mx get tooling/beads.md    # Read an entry
      mx tree                    # Browse structure
      mx health                  # Check KB health

    \b
    Create content:
      mx add --title="Title" --tags="a,b" --content="..."
      mx append "Existing Title" --content="append this"

    \b
    Modify content:
      mx patch path.md --find "old text" --replace "new text"
      mx update path.md --tags="new,tags"
    """
    # Store json_errors in context for subcommands to access
    ctx.ensure_object(dict)
    ctx.obj["json_errors"] = json_errors


# ─────────────────────────────────────────────────────────────────────────────
# Prime Command (Agent Context Injection)
# ─────────────────────────────────────────────────────────────────────────────

PRIME_OUTPUT = """# Memex Knowledge Base

> Search organizational knowledge before reinventing. Add discoveries for future agents.

**⚡ Use `mx` CLI instead of MCP tools** - CLI uses ~0 tokens vs MCP schema overhead.

## CLI Quick Reference

```bash
# Search (hybrid keyword + semantic)
mx search "deployment"              # Find entries
mx search "docker" --tags=infra     # Filter by tag
mx search "api" --mode=semantic     # Semantic only

# Read entries
mx get tooling/beads.md             # Full entry
mx get tooling/beads.md --metadata  # Just metadata

# Browse
mx tree                             # Directory structure
mx list --tag=infrastructure        # Filter by tag
mx whats-new --days=7               # Recent changes
mx whats-new --project=myapp        # Recent changes for a project

# Contribute
mx add --title="My Entry" --tags="foo,bar" --content="..."
mx add --title="..." --tags="..." --file=content.md
cat notes.md | mx add --title="..." --tags="..." --stdin

# Maintenance
mx health                           # Audit for problems
mx suggest-links path/entry.md      # Find related entries
```

## When to Search KB

- ✅ Looking for org patterns, guides, troubleshooting
- ✅ Before implementing something that might exist
- ✅ Understanding infrastructure or deployment

## When to Contribute

- ✅ Discovered reusable pattern or solution
- ✅ Troubleshooting steps worth preserving
- ✅ Infrastructure or deployment knowledge

## Entry Format

Entries are Markdown with YAML frontmatter:

```markdown
---
title: Entry Title
tags: [tag1, tag2]
created: 2024-01-15
---

# Entry Title

Content with [[bidirectional links]] to other entries.
```

Use `[[path/to/entry.md|Display Text]]` for links.
"""

PRIME_MCP_OUTPUT = """# KB Quick Reference
Search: `mx search "query"` | Read: `mx get path.md` | Add: `mx add --title="..." --tags="..."`
"""


def _detect_mcp_mode() -> bool:
    """Detect if running in MCP context (minimal output preferred)."""
    import os
    # If MCP server is active, we're likely in a context where minimal output is better
    # Check for common MCP environment indicators
    return os.environ.get("MCP_SERVER_ACTIVE") == "1"


@cli.command()
@click.option("--full", is_flag=True, help="Force full CLI output (ignore MCP detection)")
@click.option("--mcp", is_flag=True, help="Force MCP mode (minimal output)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def prime(full: bool, mcp: bool, as_json: bool):
    """Output agent workflow context for session start.

    Automatically detects MCP vs CLI mode and adapts output:
    - CLI mode: Full command reference (~1-2k tokens)
    - MCP mode: Brief workflow reminders (~50 tokens)

    Designed for Claude Code hooks (SessionStart, PreCompact) to prevent
    agents from forgetting KB workflow after context compaction.

    \b
    Examples:
      mx prime              # Auto-detect mode
      mx prime --full       # Force full output
      mx prime --mcp        # Force minimal output
    """
    # Determine output mode
    if full:
        use_full = True
    elif mcp:
        use_full = False
    else:
        use_full = not _detect_mcp_mode()

    content = PRIME_OUTPUT if use_full else PRIME_MCP_OUTPUT

    if as_json:
        output({"mode": "full" if use_full else "mcp", "content": content}, as_json=True)
    else:
        click.echo(content)


# ─────────────────────────────────────────────────────────────────────────────
# Score Confidence Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _score_confidence(score: float) -> str:
    """Return confidence level for a score (for JSON output)."""
    if score >= 0.7:
        return "high"
    elif score >= 0.4:
        return "moderate"
    else:
        return "weak"


def _score_confidence_short(score: float) -> str:
    """Return short confidence label for table output."""
    if score >= 0.7:
        return "high"
    elif score >= 0.4:
        return "mod"
    else:
        return "weak"


# ─────────────────────────────────────────────────────────────────────────────
# Search Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("query")
@click.option("--tags", "-t", help="Filter by tags (comma-separated)")
@click.option("--mode", type=click.Choice(["hybrid", "keyword", "semantic"]), default="hybrid")
@click.option("--limit", "-n", default=10, type=click.IntRange(min=1), help="Max results")
@click.option("--min-score", type=click.FloatRange(min=0.0, max=1.0), default=None,
              help="Minimum score threshold (0.0-1.0). Scores: >=0.7 high, 0.4-0.7 moderate, <0.4 weak")
@click.option("--content", "-c", is_flag=True, help="Include full content in results")
@click.option("--strict", is_flag=True, help="Disable semantic fallback for keyword mode")
@click.option("--terse", is_flag=True, help="Output paths only (one per line)")
@click.option("--full-titles", is_flag=True, help="Show full titles without truncation")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def search(query: str, tags: Optional[str], mode: str, limit: int, min_score: Optional[float], content: bool, strict: bool, terse: bool, full_titles: bool, as_json: bool):
    """Search the knowledge base.

    Scores are normalized to 0.0-1.0 (higher = better match):

    \b
      >= 0.7  High confidence - strong keyword/semantic match
      0.4-0.7 Moderate - partial match, worth reviewing
      < 0.4   Weak - tangential relevance only

    \b
    Score composition varies by mode:
      hybrid:   Reciprocal Rank Fusion of keyword + semantic
      keyword:  BM25 text matching (exact terms matter)
      semantic: Cosine similarity of embeddings (meaning matters)

    Context boosts (+0.05-0.15) are applied for tag matches and project context.

    The --strict flag prevents semantic search from returning low-confidence results
    for unrelated queries (e.g., gibberish). Useful when you need precise matches.

    \b
    Examples:
      mx search "deployment"
      mx search "docker" --tags=infrastructure
      mx search "api" --mode=semantic --limit=5
      mx search "config" --min-score=0.5          # Only confident results
      mx search "query" --strict                  # No semantic fallback
      mx search "query" --terse                   # Paths only
    """
    from .core import search as core_search

    # Validate query is not empty
    if not query or not query.strip():
        raise UsageError("Query cannot be empty.")

    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    result = run_async(core_search(
        query=query,
        limit=limit,
        mode=mode,
        tags=tag_list,
        include_content=content,
        strict=strict,
    ))

    # Apply min_score filter if specified
    filtered_results = result.results
    if min_score is not None:
        filtered_results = [r for r in result.results if r.score >= min_score]

    if as_json:
        results_data = []
        for r in filtered_results:
            item = {
                "path": r.path,
                "title": r.title,
                "score": r.score,
                "confidence": _score_confidence(r.score),
            }
            if content and r.content:
                item["content"] = r.content
            else:
                item["snippet"] = r.snippet
            results_data.append(item)
        output(results_data, as_json=True)
    elif terse:
        for r in filtered_results:
            click.echo(r.path)
    else:
        if not filtered_results:
            if min_score is not None and result.results:
                click.echo(f"No results above score threshold {min_score:.2f}. ({len(result.results)} results filtered out)")
            else:
                click.echo("No results found.")
            return

        rows = [
            {"path": r.path, "title": r.title, "score": f"{r.score:.2f}", "conf": _score_confidence_short(r.score)}
            for r in filtered_results
        ]
        title_width = 10000 if full_titles else 30
        click.echo(format_table(rows, ["path", "title", "score", "conf"], {"path": 40, "title": title_width}))

        # Show full content below table when --content flag is used
        if content:
            click.echo("\n" + "=" * 60)
            for r in filtered_results:
                click.echo(f"\n## {r.path}")
                click.echo("-" * 40)
                if r.content:
                    click.echo(r.content)
                else:
                    click.echo(r.snippet)
                click.echo()


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
      mx get tooling/beads-issue-tracker.md
      mx get tooling/beads-issue-tracker.md --json
      mx get tooling/beads-issue-tracker.md --metadata
    """
    from .core import get_entry

    try:
        entry = run_async(get_entry(path=path))
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
      mx add --title="My Entry" --tags="foo,bar" --content="# Content here"
      mx add --title="My Entry" --tags="foo,bar" --file=content.md
      cat content.md | mx add --title="My Entry" --tags="foo,bar" --stdin
    """
    from .core import add_entry

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
        result = run_async(add_entry(title=title, content=content, tags=tag_list, category=category))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        output(result, as_json=True)
    else:
        click.echo(f"Created: {result['path']}")
        if result.get('suggested_links'):
            click.echo("\nSuggested links:")
            for link in result['suggested_links'][:5]:
                click.echo(f"  - {link['path']} ({link['score']:.2f})")
        if result.get('suggested_tags'):
            click.echo("\nSuggested tags:")
            for tag in result['suggested_tags'][:5]:
                click.echo(f"  - {tag['tag']} ({tag['reason']})")


# ─────────────────────────────────────────────────────────────────────────────
# Append Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("title")
@click.option("--content", "-c", help="Content to append (or use --file/--stdin)")
@click.option("--file", "-f", "file_path", type=click.Path(exists=True), help="Read content from file")
@click.option("--stdin", is_flag=True, help="Read content from stdin")
@click.option("--tags", "-t", help="Tags (comma-separated, required for new entries)")
@click.option("--category", help="Category for new entries")
@click.option("--directory", "-d", help="Directory for new entries")
@click.option("--no-create", is_flag=True, help="Error if entry not found (don't create)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def append(
    title: str,
    content: Optional[str],
    file_path: Optional[str],
    stdin: bool,
    tags: Optional[str],
    category: Optional[str],
    directory: Optional[str],
    no_create: bool,
    as_json: bool,
):
    """Append content to existing entry by title, or create new if not found.

    Finds an entry by title (case-insensitive) and appends the provided content.
    If no matching entry exists, creates a new entry with the given content.

    \b
    Examples:
      mx append "Daily Log" --content="Session summary"
      mx append "API Docs" --file=api.md --tags="api,docs"
      mx append "Debug Log" --content="..." --no-create  # Error if not found
      cat notes.md | mx append "Meeting Notes" --stdin --tags="meetings"
    """
    from .core import append_entry

    # Resolve content source
    if stdin:
        content = sys.stdin.read()
    elif file_path:
        content = Path(file_path).read_text()
    elif not content:
        click.echo("Error: Must provide --content, --file, or --stdin", err=True)
        sys.exit(1)

    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    try:
        result = run_async(append_entry(
            title=title,
            content=content,
            tags=tag_list,
            category=category or "",
            directory=directory,
            no_create=no_create,
        ))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        output(result, as_json=True)
    else:
        action = result.get("action", "updated")
        if action == "created":
            click.echo(f"Created: {result['path']}")
        else:
            click.echo(f"Appended to: {result['path']}")

        if result.get('suggested_links'):
            click.echo("\nSuggested links:")
            for link in result['suggested_links'][:5]:
                click.echo(f"  - {link['path']} ({link['score']:.2f})")
        if result.get('suggested_tags'):
            click.echo("\nSuggested tags:")
            for tag in result['suggested_tags'][:5]:
                click.echo(f"  - {tag['tag']} ({tag['reason']})")


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
      mx update path/entry.md --tags="new,tags"
      mx update path/entry.md --file=updated-content.md
    """
    from .core import update_entry

    if file_path:
        content = Path(file_path).read_text()

    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    try:
        result = run_async(update_entry(path=path, content=content, tags=tag_list))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        output(result, as_json=True)
    else:
        click.echo(f"Updated: {result['path']}")


# ─────────────────────────────────────────────────────────────────────────────
# Tree Command
# ─────────────────────────────────────────────────────────────────────────────


def format_tree(tree_data: dict, prefix: str = "") -> str:
    """Format tree dict as ASCII tree."""
    lines = []
    items = [(k, v) for k, v in tree_data.items() if k != "_type"]
    for i, (name, value) in enumerate(items):
        is_last = i == len(items) - 1
        connector = "└── " if is_last else "├── "

        if isinstance(value, dict) and value.get("_type") == "directory":
            lines.append(f"{prefix}{connector}{name}/")
            extension = "    " if is_last else "│   "
            lines.append(format_tree(value, prefix + extension))
        elif isinstance(value, dict) and value.get("_type") == "file":
            title = value.get("title", "")
            if title:
                lines.append(f"{prefix}{connector}{name} ({title})")
            else:
                lines.append(f"{prefix}{connector}{name}")

    return "\n".join(line for line in lines if line)


@cli.command()
@click.argument("path", default="")
@click.option("--depth", "-d", default=3, help="Max depth")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def tree(path: str, depth: int, as_json: bool):
    """Display knowledge base directory structure.

    \b
    Examples:
      mx tree
      mx tree tooling --depth=2
    """
    from .core import tree as core_tree

    result = run_async(core_tree(path=path, depth=depth))

    if as_json:
        output(result, as_json=True)
    else:
        formatted = format_tree(result["tree"])
        if formatted:
            click.echo(formatted)
        click.echo(f"\n{result['directories']} directories, {result['files']} files")


# ─────────────────────────────────────────────────────────────────────────────
# List Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command("list")
@click.option("--tag", "-t", help="Filter by tag")
@click.option("--category", "-c", help="Filter by category")
@click.option("--limit", "-n", default=20, help="Max results")
@click.option("--full-titles", is_flag=True, help="Show full titles without truncation")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def list_entries(tag: Optional[str], category: Optional[str], limit: int, full_titles: bool, as_json: bool):
    """List knowledge base entries.

    \b
    Examples:
      mx list
      mx list --tag=tooling
      mx list --category=infrastructure --limit=10
    """
    from .core import get_valid_categories, list_entries as core_list_entries

    try:
        result = run_async(core_list_entries(tag=tag, category=category, limit=limit))
    except ValueError as e:
        # Handle invalid category with helpful error message
        error_msg = str(e)
        if "not found" in error_msg.lower() and category:
            valid_categories = get_valid_categories()
            if valid_categories:
                click.echo(
                    f"Error: Invalid category '{category}'. "
                    f"Valid categories: {', '.join(sorted(valid_categories))}",
                    err=True,
                )
            else:
                click.echo(f"Error: Invalid category '{category}'. No categories exist yet.", err=True)
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        output(result, as_json=True)
    else:
        if not result:
            click.echo("No entries found.")
            return

        rows = [{"path": e["path"], "title": e["title"]} for e in result]
        title_width = 10000 if full_titles else 40
        click.echo(format_table(rows, ["path", "title"], {"path": 45, "title": title_width}))


# ─────────────────────────────────────────────────────────────────────────────
# What's New Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command("whats-new")
@click.option("--days", "-d", default=30, help="Look back N days")
@click.option("--limit", "-n", default=10, help="Max results")
@click.option("--project", "-p", help="Filter by project name (matches path, source_project, or tags)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def whats_new(days: int, limit: int, project: Optional[str], as_json: bool):
    """Show recently created or updated entries.

    \b
    Examples:
      mx whats-new
      mx whats-new --days=7 --limit=5
      mx whats-new --project=docviewer  # Filter by project
    """
    from .core import whats_new as core_whats_new

    result = run_async(core_whats_new(days=days, limit=limit, project=project))

    if as_json:
        output(result, as_json=True)
    else:
        if not result:
            if project:
                click.echo(f"No entries for project '{project}' in the last {days} days.")
            else:
                click.echo(f"No entries created or updated in the last {days} days.")
            return

        rows = [
            {"path": e["path"], "title": e["title"], "date": str(e["activity_date"])[:10]}
            for e in result
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
      mx health
      mx health --json
    """
    from .core import health as core_health

    result = run_async(core_health())

    if as_json:
        output(result, as_json=True)
    else:
        summary = result.get("summary", {})
        click.echo("Knowledge Base Health Report")
        click.echo("=" * 40)
        click.echo(f"Health Score: {summary.get('health_score', 0)}/100")
        click.echo(f"Total Entries: {summary.get('total_entries', 0)}")

        # Orphans
        orphans = result.get("orphans", [])
        if orphans:
            click.echo(f"\n⚠ Orphaned entries ({len(orphans)}):")
            for o in orphans[:10]:
                click.echo(f"  - {o['path']}")
        else:
            click.echo("\n✓ No orphaned entries")

        # Broken links
        broken_links = result.get("broken_links", [])
        if broken_links:
            click.echo(f"\n⚠ Broken links ({len(broken_links)}):")
            for bl in broken_links[:10]:
                click.echo(f"  - {bl['source']} -> {bl['broken_link']}")
        else:
            click.echo("\n✓ No broken links")

        # Stale
        stale = result.get("stale", [])
        if stale:
            click.echo(f"\n⚠ Stale entries ({len(stale)}):")
            for s in stale[:10]:
                click.echo(f"  - {s['path']}")
        else:
            click.echo("\n✓ No stale entries")

        # Empty dirs
        empty_dirs = result.get("empty_dirs", [])
        if empty_dirs:
            click.echo(f"\n⚠ Empty directories ({len(empty_dirs)}):")
            for d in empty_dirs[:10]:
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
      mx tags
      mx tags --min-count=3
    """
    from .core import tags as core_tags

    result = run_async(core_tags(min_count=min_count))

    if as_json:
        output(result, as_json=True)
    else:
        if not result:
            click.echo("No tags found.")
            return

        for tag_info in result:
            click.echo(f"  {tag_info['tag']}: {tag_info['count']}")


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
      mx hubs
      mx hubs --limit=5
    """
    from .core import hubs as core_hubs

    result = run_async(core_hubs(limit=limit))

    if as_json:
        output(result, as_json=True)
    else:
        if not result:
            click.echo("No hub entries found.")
            return

        rows = [{"path": h["path"], "incoming": h["incoming"], "outgoing": h["outgoing"], "total": h["total"]} for h in result]
        click.echo(format_table(rows, ["path", "incoming", "outgoing", "total"], {"path": 50}))


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
      mx suggest-links tooling/my-entry.md
    """
    from .core import suggest_links as core_suggest_links

    try:
        result = run_async(core_suggest_links(path=path, limit=limit))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        output(result, as_json=True)
    else:
        if not result:
            click.echo("No link suggestions found.")
            return

        click.echo(f"Suggested links for {path}:\n")
        for s in result:
            click.echo(f"  {s['path']} ({s['score']:.2f})")
            click.echo(f"    {s['reason']}")


# ─────────────────────────────────────────────────────────────────────────────
# Reindex Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
def reindex():
    """Rebuild search indices from all markdown files.

    Use this after bulk imports or if search results seem stale.

    \b
    Examples:
      mx reindex
    """
    from .core import reindex as core_reindex

    click.echo("Reindexing knowledge base...")
    result = run_async(core_reindex())
    click.echo(f"✓ Indexed {result.kb_files} entries, {result.whoosh_docs} keyword docs, {result.chroma_docs} semantic docs")


# ─────────────────────────────────────────────────────────────────────────────
# Context Command Group
# ─────────────────────────────────────────────────────────────────────────────


@cli.group(invoke_without_command=True)
@click.pass_context
def context(ctx):
    """Manage project KB context (.kbcontext file).

    The .kbcontext file configures KB behavior for a project:
    - primary: Default directory for new entries
    - paths: Boost these paths in search results
    - default_tags: Suggested tags for new entries

    \b
    Examples:
      mx context            # Show current context
      mx context init       # Create a new .kbcontext file
      mx context validate   # Check context paths exist in KB
    """
    # If no subcommand provided, show context
    if ctx.invoked_subcommand is None:
        ctx.invoke(context_show)


@context.command("show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def context_show(as_json: bool):
    """Show the current project context.

    Searches for .kbcontext file starting from current directory.

    \b
    Examples:
      mx context show
      mx context show --json
    """
    from .context import get_kb_context

    ctx = get_kb_context()

    if ctx is None:
        if as_json:
            output({"found": False, "message": "No .kbcontext file found"}, as_json=True)
        else:
            click.echo("No .kbcontext file found.")
            click.echo("Run 'mx context init' to create one.")
        return

    if as_json:
        output({
            "found": True,
            "source_file": str(ctx.source_file) if ctx.source_file else None,
            "primary": ctx.primary,
            "paths": ctx.paths,
            "default_tags": ctx.default_tags,
            "project": ctx.project,
        }, as_json=True)
    else:
        click.echo(f"Context file: {ctx.source_file}")
        click.echo(f"Primary:      {ctx.primary or '(not set)'}")
        click.echo(f"Paths:        {', '.join(ctx.paths) if ctx.paths else '(none)'}")
        click.echo(f"Default tags: {', '.join(ctx.default_tags) if ctx.default_tags else '(none)'}")
        if ctx.project:
            click.echo(f"Project:      {ctx.project}")


# Make 'show' the default command when 'context' is called without subcommand
@context.command("status", hidden=True)
@click.pass_context
def context_status(ctx):
    """Alias for 'show' - used when 'context' is called without subcommand."""
    ctx.invoke(context_show)


@context.command("init")
@click.option("--project", "-p", help="Project name (auto-detected from directory if not provided)")
@click.option("--directory", "-d", help="KB directory (defaults to projects/<project>)")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing .kbcontext file")
def context_init(project: Optional[str], directory: Optional[str], force: bool):
    """Create a new .kbcontext file in the current directory.

    \b
    Examples:
      mx context init
      mx context init --project myapp
      mx context init --project myapp --directory projects/myapp/docs
    """
    from .context import CONTEXT_FILENAME, create_default_context

    context_path = Path.cwd() / CONTEXT_FILENAME

    if context_path.exists() and not force:
        click.echo(f"Error: {CONTEXT_FILENAME} already exists. Use --force to overwrite.", err=True)
        sys.exit(1)

    # Auto-detect project name from directory
    if not project:
        project = Path.cwd().name

    content = create_default_context(project, directory)
    context_path.write_text(content, encoding="utf-8")

    click.echo(f"Created {CONTEXT_FILENAME}")
    click.echo(f"  Primary directory: {directory or f'projects/{project}'}")
    click.echo(f"  Default tags: {project}")
    click.echo("\nEdit the file to customize paths and tags.")


@context.command("validate")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def context_validate(as_json: bool):
    """Validate the current .kbcontext file against the knowledge base.

    Checks that:
    - primary directory exists (or can be created)
    - paths reference valid locations (warning only)

    \b
    Examples:
      mx context validate
      mx context validate --json
    """
    from .config import get_kb_root
    from .context import get_kb_context, validate_context

    ctx = get_kb_context()

    if ctx is None:
        if as_json:
            output({"valid": False, "error": "No .kbcontext file found"}, as_json=True)
        else:
            click.echo("Error: No .kbcontext file found.", err=True)
        sys.exit(1)

    kb_root = get_kb_root()
    warnings = validate_context(ctx, kb_root)

    if as_json:
        output({
            "valid": True,
            "source_file": str(ctx.source_file),
            "warnings": warnings,
        }, as_json=True)
    else:
        click.echo(f"Validating: {ctx.source_file}")

        if warnings:
            click.echo("\nWarnings:")
            for warning in warnings:
                click.echo(f"  ⚠ {warning}")
        else:
            click.echo("✓ All paths are valid")


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
      mx delete path/to/entry.md
      mx delete path/to/entry.md --force
    """
    from .core import delete_entry

    try:
        result = run_async(delete_entry(path=path, force=force))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        output(result, as_json=True)
    else:
        if result.get("had_backlinks"):
            click.echo(f"Warning: Entry had {len(result['had_backlinks'])} backlinks", err=True)
        click.echo(f"Deleted: {result['deleted']}")


# ─────────────────────────────────────────────────────────────────────────────
# Patch Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("path", metavar="PATH")
@click.option("--find", "find_text", help="Exact text to find and replace")
@click.option("--replace", "replace_text", help="Replacement text")
@click.option(
    "--find-file",
    type=click.Path(exists=True),
    help="Read --find text from file (for multi-line)",
)
@click.option(
    "--replace-file",
    type=click.Path(exists=True),
    help="Read --replace text from file (for multi-line)",
)
@click.option("--replace-all", is_flag=True, help="Replace all occurrences")
@click.option("--dry-run", is_flag=True, help="Preview changes without writing")
@click.option("--backup", is_flag=True, help="Create .bak backup before patching")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def patch(
    path: str,
    find_text: str | None,
    replace_text: str | None,
    find_file: str | None,
    replace_file: str | None,
    replace_all: bool,
    dry_run: bool,
    backup: bool,
    as_json: bool,
):
    """Apply surgical find-replace edits to a KB entry.

    PATH is relative to KB root (e.g., "tooling/my-entry.md").

    Finds exact occurrences of --find and replaces with --replace.
    Fails if --find is not found or matches multiple times (use --replace-all).

    For multi-line text or special characters, use --find-file and --replace-file.

    \b
    Exit codes:
      0: Success
      1: Text not found
      2: Multiple matches (ambiguous, use --replace-all)
      3: File error (not found, permission, encoding)

    \b
    Examples:
      mx patch tooling/notes.md --find "old text" --replace "new text"
      mx patch tooling/notes.md --find "TODO" --replace "DONE" --replace-all
      mx patch tooling/notes.md --find-file old.txt --replace-file new.txt
      mx patch tooling/notes.md --find "..." --replace "..." --dry-run
    """
    from .core import patch_entry

    # Resolve --find input source
    if find_file and find_text:
        click.echo("Error: --find and --find-file are mutually exclusive", err=True)
        sys.exit(3)
    if find_file:
        find_string = Path(find_file).read_text(encoding="utf-8")
    elif find_text is not None:
        find_string = find_text
    else:
        click.echo("Error: Must provide --find or --find-file", err=True)
        sys.exit(3)

    # Resolve --replace input source
    if replace_file and replace_text:
        click.echo("Error: --replace and --replace-file are mutually exclusive", err=True)
        sys.exit(3)
    if replace_file:
        replace_string = Path(replace_file).read_text(encoding="utf-8")
    elif replace_text is not None:
        replace_string = replace_text
    else:
        click.echo("Error: Must provide --replace or --replace-file", err=True)
        sys.exit(3)

    try:
        result = run_async(
            patch_entry(
                path=path,
                find_string=find_string,
                replace_string=replace_string,
                replace_all=replace_all,
                dry_run=dry_run,
                backup=backup,
            )
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(3)

    exit_code = result.get("exit_code", 0)

    if as_json:
        output(result, as_json=True)
    else:
        if result.get("success"):
            if dry_run:
                click.echo("Dry run - no changes made:")
                click.echo(result.get("diff", ""))
            else:
                click.echo(f"Patched: {result['path']} ({result['replacements']} replacement(s))")
        else:
            click.echo(f"Error: {result['message']}", err=True)
            # Show match contexts for ambiguous case
            if result.get("match_contexts"):
                click.echo("\nMatches found:", err=True)
                for ctx in result["match_contexts"]:
                    click.echo(f"  {ctx['preview']}", err=True)

    sys.exit(exit_code)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """Entry point for mx CLI."""
    from ._logging import configure_logging
    configure_logging()
    cli()


if __name__ == "__main__":
    main()
