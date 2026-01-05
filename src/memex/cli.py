#!/usr/bin/env python3
"""
mx: CLI for memex knowledge base

Token-efficient alternative to MCP tools. Wraps existing memex functionality.

Usage:
    mx search "query"              # Search entries
    mx get path/to/entry.md        # Read an entry
    mx add --title="..." --tags=.. # Create entry
    mx info                        # Show KB config
    mx tree                        # Browse structure
    mx health                      # Audit KB health
"""

import asyncio
import json
import sys
from pathlib import Path

import click

# Lazy imports to speed up CLI startup
# The heavy imports (chromadb, sentence-transformers) only load when needed


def run_async(coro):
    """Run async function synchronously."""
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────────────
# Output Formatting
# ─────────────────────────────────────────────────────────────────────────────


def format_table(rows: list[dict], columns: list[str], max_widths: dict | None = None) -> str:
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


def _normalize_error_message(message: str) -> str:
    """Normalize core error messages to CLI-friendly guidance."""
    normalized = message.replace("force=True", "--force")
    normalized = normalized.replace(
        "Either 'category' or 'directory' must be provided",
        "Either --category must be provided",
    )
    normalized = normalized.replace(
        "Use rmdir for directories.",
        "Delete entries inside or remove the directory manually.",
    )
    return normalized


def _format_missing_category_error(tags: list[str], message: str) -> str:
    """Format a helpful error when category is required."""
    from . import core

    valid_categories = core.get_valid_categories()
    tag_set = {tag.strip().lower() for tag in tags if tag.strip()}
    matches = [category for category in valid_categories if category.lower() in tag_set]
    suggestion = matches[0] if len(matches) == 1 else None

    lines = ["Error: --category required."]
    if "no .kbcontext file found" in message.lower():
        lines.append("No .kbcontext primary found. Run 'mx context init' or pass --category.")
    if tags:
        lines.append(f"Your tags: {', '.join(tags)}")
    if suggestion:
        lines.append(f"Suggested: --category={suggestion}")
    elif matches:
        lines.append(f"Tags matched categories: {', '.join(matches)}")
    if valid_categories:
        lines.append(f"Available categories: {', '.join(valid_categories)}")
    lines.append(
        "Example: mx add --title=\"...\" --tags=\"...\" --category=... --content=\"...\""
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main CLI Group
# ─────────────────────────────────────────────────────────────────────────────


@click.group()
@click.version_option(version="0.1.0", prog_name="mx")
def cli():
    """mx: Token-efficient CLI for memex knowledge base.

    Search, browse, and manage KB entries without MCP context overhead.

    \b
    Quick start:
      mx search "deployment"     # Find entries
      mx get tooling/beads.md    # Read an entry
      mx info                    # Show KB configuration
      mx tree                    # Browse structure
      mx health                  # Check KB health
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Prime Command (Agent Context Injection)
# ─────────────────────────────────────────────────────────────────────────────

PRIME_OUTPUT = """# Memex Knowledge Base

> Search organizational knowledge before reinventing. Add discoveries for future agents.

**⚡ Use `mx` CLI instead of MCP tools** - CLI uses ~0 tokens vs MCP schema overhead.

## Session Protocol

**Before searching codebase**: Check if memex has project patterns first
  `mx search "deployment"` or `mx whats-new --project=<project>`

**After discovering patterns**: Consider adding to KB for future agents
  `mx add --title="..." --tags="..." --content="..."`

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
mx info                             # Show KB configuration
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

PRIME_MCP_OUTPUT = """# Memex KB Active

**Session Start**: Search KB before implementing: `mx search "query"`
**Session End**: Consider adding discoveries: `mx add --title="..." --tags="..."`

Quick: search | get | add | tree | whats-new | health
"""


def _detect_mcp_mode() -> bool:
    """Detect if MCP server is configured for memex.

    Checks multiple locations for memex MCP server configuration:
    1. ~/.claude/settings.json - Global Claude settings
    2. .mcp.json - Project-level MCP configuration
    3. .claude-plugin/plugin.json - Plugin MCP configuration

    Returns True if memex MCP is configured anywhere, indicating
    the agent has access to MCP tools and minimal priming is preferred.
    """
    # Check ~/.claude/settings.json
    settings_path = Path.home() / ".claude" / "settings.json"
    if settings_path.exists():
        try:
            data = json.loads(settings_path.read_text())
            mcp_servers = data.get("mcpServers", {})
            if any("memex" in key.lower() for key in mcp_servers):
                return True
        except (json.JSONDecodeError, OSError):
            pass

    # Check .mcp.json in current directory and parents
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents)[:3]:  # Check up to 3 levels up
        mcp_json = parent / ".mcp.json"
        if mcp_json.exists():
            try:
                data = json.loads(mcp_json.read_text())
                mcp_servers = data.get("mcpServers", {})
                if any("memex" in key.lower() for key in mcp_servers):
                    return True
            except (json.JSONDecodeError, OSError):
                pass

    return False


def _detect_current_project() -> str | None:
    """Detect current project from git remote, directory, or beads.

    Returns:
        Project name or None if unavailable.
    """
    import subprocess

    cwd = Path.cwd()

    # Try git remote first
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            import re
            remote_url = result.stdout.strip()
            # Handle SSH format: git@github.com:user/repo.git
            ssh_match = re.search(r":([^/]+/[^/]+?)(?:\.git)?$", remote_url)
            if ssh_match:
                return ssh_match.group(1).split("/")[-1]
            # Handle HTTPS format
            https_match = re.search(r"/([^/]+?)(?:\.git)?$", remote_url)
            if https_match:
                return https_match.group(1)
    except (subprocess.TimeoutExpired, OSError):
        pass

    # Fallback to directory name
    return cwd.name


def _get_recent_project_entries(project: str, days: int = 7, limit: int = 5) -> list:
    """Get recent KB entries for a project.

    Args:
        project: Project name to filter by.
        days: Look back period.
        limit: Max entries to return.

    Returns:
        List of recent entry dicts.
    """
    from .core import whats_new as core_whats_new

    try:
        return run_async(core_whats_new(days=days, limit=limit, project=project))
    except Exception:
        return []


def _format_recent_entries(entries: list, project: str) -> str:
    """Format recent entries for display."""
    if not entries:
        return ""

    lines = [f"\n## Recent KB Updates for {project}\n"]

    for entry in entries:
        activity = "NEW" if entry.get("activity_type") == "created" else "UPD"
        date_str = str(entry.get("activity_date", ""))[:10]
        title = entry.get("title", "Untitled")
        path = entry.get("path", "")

        lines.append(f"{activity}  {path}")
        lines.append(f"     {title} ({date_str})")

    return "\n".join(lines)


@cli.command()
@click.option("--full", is_flag=True, help="Force full CLI output (ignore MCP detection)")
@click.option("--mcp", is_flag=True, help="Force MCP mode (minimal output)")
@click.option(
    "--project", "-p",
    help="Include recent entries for project (auto-detected if not specified)",
)
@click.option("--days", "-d", default=7, help="Days to look back for project entries")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def prime(full: bool, mcp: bool, project: str | None, days: int, as_json: bool):
    """Output agent workflow context for session start.

    Automatically detects MCP vs CLI mode and adapts output:
    - CLI mode: Full command reference (~1-2k tokens)
    - MCP mode: Brief workflow reminders (~50 tokens)

    When --project is specified (or auto-detected), includes recent KB
    changes for that project to help with session context recovery.

    Designed for Claude Code hooks (SessionStart, PreCompact) to prevent
    agents from forgetting KB workflow after context compaction.

    \b
    Examples:
      mx prime                    # Auto-detect mode
      mx prime --full             # Force full output
      mx prime --mcp              # Force minimal output
      mx prime --project=myapp    # Include myapp recent entries
      mx prime -p myapp -d 14     # Last 14 days of myapp changes
    """
    # Determine output mode
    if full:
        use_full = True
    elif mcp:
        use_full = False
    else:
        use_full = not _detect_mcp_mode()

    content = PRIME_OUTPUT if use_full else PRIME_MCP_OUTPUT

    # Auto-detect project if not specified
    detected_project = project or _detect_current_project()
    recent_entries = []
    recent_content = ""

    if detected_project:
        recent_entries = _get_recent_project_entries(detected_project, days=days)
        if recent_entries:
            recent_content = _format_recent_entries(recent_entries, detected_project)

    if as_json:
        output({
            "mode": "full" if use_full else "mcp",
            "content": content,
            "project": detected_project,
            "recent_entries": recent_entries,
        }, as_json=True)
    else:
        click.echo(content)
        if recent_content:
            click.echo(recent_content)


# ─────────────────────────────────────────────────────────────────────────────
# Search Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("query")
@click.option("--tags", "-t", help="Filter by tags (comma-separated)")
@click.option("--mode", type=click.Choice(["hybrid", "keyword", "semantic"]), default="hybrid")
@click.option("--limit", "-n", default=10, help="Max results")
@click.option("--content", "-c", is_flag=True, help="Include full content in results")
@click.option("--no-history", is_flag=True, help="Don't record this search in history")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def search(
    query: str, tags: str | None, mode: str, limit: int,
    content: bool, no_history: bool, as_json: bool,
):
    """Search the knowledge base.

    \b
    Examples:
      mx search "deployment"
      mx search "docker" --tags=infrastructure
      mx search "api" --mode=semantic --limit=5
    """
    from .core import search as core_search
    from .indexer.chroma_index import semantic_deps_available

    # Fail early if semantic search requested but deps not installed
    if mode == "semantic" and not semantic_deps_available():
        raise click.ClickException(
            "Semantic search requires additional dependencies.\n"
            "Install with: uv pip install -e '.[semantic]'\n"
            "Or use --mode=keyword for keyword-only search."
        )

    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    result = run_async(core_search(
        query=query,
        limit=limit,
        mode=mode,
        tags=tag_list,
        include_content=content,
    ))

    # Record search in history (unless --no-history flag is set)
    if not no_history:
        from . import search_history
        search_history.record_search(
            query=query,
            result_count=len(result.results),
            mode=mode,
            tags=tag_list,
        )

    if as_json:
        output(
            [{"path": r.path, "title": r.title, "score": r.score, "snippet": r.snippet}
             for r in result.results],
            as_json=True,
        )
    else:
        if not result.results:
            click.echo("No results found.")
            return

        rows = [
            {"path": r.path, "title": r.title, "score": f"{r.score:.2f}"}
            for r in result.results
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
      mx get tooling/beads-issue-tracker.md
      mx get tooling/beads-issue-tracker.md --json
      mx get tooling/beads-issue-tracker.md --metadata
    """
    from .core import get_entry

    try:
        entry = run_async(get_entry(path=path))
    except Exception as e:
        click.echo(f"Error: {_normalize_error_message(str(e))}", err=True)
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
@click.option(
    "--category",
    "-c",
    default="",
    help="Category/directory (required unless .kbcontext sets a primary path)",
)
@click.option("--content", help="Content (or use --file/--stdin)")
@click.option(
    "--file", "-f", "file_path",
    type=click.Path(exists=True), help="Read content from file",
)
@click.option("--stdin", is_flag=True, help="Read content from stdin")
@click.option("--force", is_flag=True, help="Create even if duplicates detected")
@click.option("--dry-run", is_flag=True, help="Preview path/frontmatter/content without creating")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def add(
    title: str,
    tags: str,
    category: str,
    content: str | None,
    file_path: str | None,
    stdin: bool,
    force: bool,
    dry_run: bool,
    as_json: bool,
):
    """Create a new knowledge base entry.

    \b
    Examples:
      mx add --title="My Entry" --tags="foo,bar" --content="# Content here"
      mx add --title="My Entry" --tags="foo,bar" --file=content.md
      cat content.md | mx add --title="My Entry" --tags="foo,bar" --stdin
      mx add --title="My Entry" --tags="foo,bar" --content="..." --dry-run

    \b
    Required:
      --title TEXT
      --tags TEXT
      --category TEXT  (required unless .kbcontext sets a primary path)

    \b
    Common issues:
      - Duplicate detected? Use --force to override
      - Category omitted? If a tag matches an existing category, it will be inferred
      - Preview first? Use --dry-run to inspect the output
      - Missing category? Run 'mx context init' or pass --category
    """
    from .core import add_entry, preview_add_entry

    # Resolve content source
    if stdin:
        content = sys.stdin.read()
    elif file_path:
        content = Path(file_path).read_text()
    elif not content:
        click.echo("Error: Must provide --content, --file, or --stdin", err=True)
        sys.exit(1)

    tag_list = [t.strip() for t in tags.split(",")]

    if dry_run:
        try:
            preview = run_async(preview_add_entry(
                title=title,
                content=content,
                tags=tag_list,
                category=category,
                force=force,
            ))
        except Exception as e:
            message = str(e)
            if "Either 'category' or 'directory' must be provided" in message:
                click.echo(_format_missing_category_error(tag_list, message), err=True)
            else:
                click.echo(f"Error: {_normalize_error_message(message)}", err=True)
            sys.exit(1)

        if as_json:
            data = preview.model_dump() if hasattr(preview, 'model_dump') else preview
            output(data, as_json=True)
            return

        click.echo(f"Would create: {preview.absolute_path}")
        click.echo(preview.frontmatter + preview.content)

        if preview.warning:
            click.echo(f"\nWarning: {preview.warning}")
            click.echo("Potential duplicates:")
            for dup in preview.potential_duplicates[:3]:
                click.echo(f"  - {dup.path} ({dup.score:.0%} similar)")
        elif force:
            click.echo("\nDuplicate check skipped (--force).")
        else:
            click.echo("\nNo duplicates detected.")
        return

    def _print_created(add_result):
        path = add_result.path if hasattr(add_result, 'path') else add_result.get('path')
        click.echo(f"Created: {path}")

        suggested_links = (
            add_result.suggested_links if hasattr(add_result, 'suggested_links')
            else add_result.get('suggested_links', [])
        )
        if suggested_links:
            click.echo("\nSuggested links:")
            for link in suggested_links[:5]:
                score = link.get('score', 0) if isinstance(link, dict) else link.score
                path_str = link.get('path', '') if isinstance(link, dict) else link.path
                click.echo(f"  - {path_str} ({score:.2f})")

        suggested_tags = (
            add_result.suggested_tags if hasattr(add_result, 'suggested_tags')
            else add_result.get('suggested_tags', [])
        )
        if suggested_tags:
            click.echo("\nSuggested tags:")
            for tag in suggested_tags[:5]:
                tag_name = tag.get('tag', '') if isinstance(tag, dict) else tag.tag
                reason = tag.get('reason', '') if isinstance(tag, dict) else tag.reason
                click.echo(f"  - {tag_name} ({reason})")

    def _print_duplicates(add_result):
        warning = add_result.warning or "Potential duplicates detected."
        warning = _normalize_error_message(warning)
        click.echo(f"Warning: {warning}")
        click.echo("Potential duplicates:")
        for dup in add_result.potential_duplicates[:3]:
            click.echo(f"  - {dup.path} ({dup.score:.0%} similar)")

    try:
        result = run_async(add_entry(
            title=title,
            content=content,
            tags=tag_list,
            category=category,
            force=force,
        ))
    except Exception as e:
        message = str(e)
        if "Either 'category' or 'directory' must be provided" in message:
            click.echo(_format_missing_category_error(tag_list, message), err=True)
        else:
            click.echo(f"Error: {_normalize_error_message(message)}", err=True)
        sys.exit(1)

    if as_json:
        output(result.model_dump() if hasattr(result, 'model_dump') else result, as_json=True)
        return

    # Handle AddEntryResponse or dict
    if hasattr(result, 'created') and not result.created:
        _print_duplicates(result)
        if not force and sys.stdin.isatty():
            if click.confirm("\nCreate anyway?"):
                try:
                    result = run_async(add_entry(
                        title=title,
                        content=content,
                        tags=tag_list,
                        category=category,
                        force=True,
                    ))
                except Exception as e:
                    click.echo(f"Error: {_normalize_error_message(str(e))}", err=True)
                    sys.exit(1)
                if hasattr(result, 'created') and not result.created:
                    _print_duplicates(result)
                else:
                    _print_created(result)
        return

    _print_created(result)


# ─────────────────────────────────────────────────────────────────────────────
# Quick-Add Command
# ─────────────────────────────────────────────────────────────────────────────


def _extract_title_from_content(content: str) -> str:
    """Extract title from markdown content.

    Tries:
    1. First H1 heading (# Title)
    2. First H2 heading (## Title)
    3. First non-empty line
    4. First 50 chars of content
    """
    import re

    lines = content.strip().split("\n")

    # Try H1 heading
    for line in lines:
        if line.startswith("# "):
            return line[2:].strip()

    # Try H2 heading
    for line in lines:
        if line.startswith("## "):
            return line[3:].strip()

    # Try first non-empty line (strip markdown syntax)
    for line in lines:
        clean = re.sub(r"^[#*>\-\s]+", "", line).strip()
        if clean and len(clean) > 3:
            # Truncate if too long
            if len(clean) > 60:
                clean = clean[:57] + "..."
            return clean

    # Fallback to first 50 chars
    return content[:50].strip() + "..."


def _suggest_tags_from_content(content: str, existing_tags: set) -> list[str]:
    """Suggest tags based on content keywords.

    Args:
        content: The entry content.
        existing_tags: Set of existing KB tags.

    Returns:
        List of suggested tags.
    """
    import re

    # Extract words from content
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9-]+\b', content.lower())
    word_counts: dict[str, int] = {}
    for word in words:
        if len(word) >= 3:
            word_counts[word] = word_counts.get(word, 0) + 1

    # Find matches with existing tags
    matches = []
    for tag in existing_tags:
        tag_lower = tag.lower()
        if tag_lower in word_counts:
            matches.append((tag, word_counts[tag_lower]))

    # Sort by frequency and return top matches
    matches.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in matches[:5]]


def _suggest_category_from_content(content: str, categories: list[str]) -> str | None:
    """Suggest category based on content.

    Args:
        content: The entry content.
        categories: List of valid categories.

    Returns:
        Suggested category or None.
    """
    content_lower = content.lower()

    # Simple keyword matching
    for cat in categories:
        cat_lower = cat.lower()
        if cat_lower in content_lower:
            return cat

    # Default to first category if available
    return categories[0] if categories else None


@cli.command("quick-add")
@click.option(
    "--file", "-f", "file_path",
    type=click.Path(exists=True), help="Read content from file",
)
@click.option("--stdin", is_flag=True, help="Read content from stdin")
@click.option("--content", "-c", help="Raw content to add")
@click.option("--title", "-t", help="Override auto-detected title")
@click.option("--tags", help="Override auto-suggested tags (comma-separated)")
@click.option("--category", help="Override auto-suggested category")
@click.option("--confirm", "-y", is_flag=True, help="Auto-confirm without prompting")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def quick_add(
    file_path: str | None,
    stdin: bool,
    content: str | None,
    title: str | None,
    tags: str | None,
    category: str | None,
    confirm: bool,
    as_json: bool,
):
    """Quickly add content with auto-generated metadata.

    Analyzes raw content to suggest title, tags, and category.
    In interactive mode, prompts for confirmation before creating.

    \b
    Examples:
      mx quick-add --stdin              # Paste content, auto-generate all
      mx quick-add -f notes.md          # From file with auto metadata
      mx quick-add -c "..." -y          # Auto-confirm creation
      echo "..." | mx quick-add --stdin --json  # Machine-readable
    """
    from .core import add_entry, get_valid_categories

    # Resolve content source
    if stdin:
        content = sys.stdin.read()
    elif file_path:
        content = Path(file_path).read_text()
    elif not content:
        click.echo("Error: Must provide --content, --file, or --stdin", err=True)
        sys.exit(1)

    if not content.strip():
        click.echo("Error: Content is empty", err=True)
        sys.exit(1)

    # Get existing KB structure
    valid_categories = get_valid_categories()

    # Collect all existing tags from KB
    from .config import get_kb_root
    from .parser import parse_entry

    kb_root = get_kb_root()
    existing_tags: set[str] = set()
    try:
        for md_file in kb_root.rglob("*.md"):
            try:
                metadata, _, _ = parse_entry(md_file)
                existing_tags.update(metadata.tags)
            except Exception:
                continue
    except Exception:
        pass

    # Auto-generate metadata
    auto_title = title or _extract_title_from_content(content)
    auto_tags = tags.split(",") if tags else _suggest_tags_from_content(content, existing_tags)
    auto_category = category or _suggest_category_from_content(content, valid_categories)

    # Ensure we have at least one tag
    if not auto_tags:
        auto_tags = ["uncategorized"]

    if as_json:
        # In JSON mode, output suggestions and let caller decide
        output({
            "title": auto_title,
            "tags": auto_tags,
            "category": auto_category,
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "categories_available": valid_categories,
        }, as_json=True)
        return

    # Interactive mode - show suggestions and prompt
    click.echo("\n=== Quick Add Analysis ===\n")
    click.echo(f"Title:    {auto_title}")
    click.echo(f"Tags:     {', '.join(auto_tags)}")
    click.echo(f"Category: {auto_category or '(none - will need to specify)'}")
    click.echo(f"Content:  {len(content)} chars")

    if not auto_category:
        click.echo(f"\nAvailable categories: {', '.join(valid_categories)}")
        default_cat = valid_categories[0] if valid_categories else "notes"
        auto_category = click.prompt("Category", default=default_cat)

    if not confirm:
        if not click.confirm("\nCreate entry with these settings?"):
            click.echo("Aborted.")
            return

    # Create the entry
    try:
        result = run_async(add_entry(
            title=auto_title,
            content=content,
            tags=auto_tags,
            category=auto_category,
            force=True,  # Skip duplicate check for quick-add
        ))
    except Exception as e:
        click.echo(f"Error: {_normalize_error_message(str(e))}", err=True)
        sys.exit(1)

    if hasattr(result, 'created') and not result.created:
        click.echo(f"\nWarning: {result.warning}")
        click.echo(f"Path would be: {result.path}")
    else:
        path = result.path if hasattr(result, 'path') else result.get('path')
        click.echo(f"\n✓ Created: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Update Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("path", metavar="PATH")
@click.option("--tags", help="Replace tags (comma-separated). Preserves existing if omitted")
@click.option("--content", help="New content (replaces existing unless --append)")
@click.option(
    "--file", "-f", "file_path",
    type=click.Path(exists=True), help="Read content from file",
)
@click.option("--stdin", is_flag=True, help="Read content from stdin")
@click.option("--append", is_flag=True, help="Append to end instead of replacing")
@click.option("--timestamp", is_flag=True, help="Add '## YYYY-MM-DD HH:MM UTC' header")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def update(
    path: str, tags: str | None, content: str | None,
    file_path: str | None, stdin: bool, append: bool, timestamp: bool, as_json: bool,
):
    """Update an existing knowledge base entry.

    PATH is relative to KB root (e.g., "tooling/my-entry.md").
    Requires --content, --file, --stdin, or --tags.

    \b
    Examples:
      mx update tooling/notes.md --tags="python,tooling"
      mx update tooling/notes.md --content="New content" --append
      mx update tooling/notes.md --file=session.md --append
      echo "Done for today" | mx update tooling/notes.md --stdin --append --timestamp
    """
    from datetime import datetime, timezone
    from .core import update_entry

    # Validate flag combinations
    if timestamp and not append:
        click.echo("Error: --timestamp requires --append", err=True)
        sys.exit(1)

    if stdin and file_path:
        click.echo("Error: --stdin and --file are mutually exclusive", err=True)
        sys.exit(1)

    if stdin and content:
        click.echo("Error: --stdin and --content are mutually exclusive", err=True)
        sys.exit(1)

    # Resolve content source
    if stdin:
        content = sys.stdin.read()
    elif file_path:
        content = Path(file_path).read_text()

    # Add timestamp header if requested
    if timestamp and content:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        content = f"## {ts}\n\n{content}"

    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    try:
        result = run_async(update_entry(path=path, content=content, tags=tag_list, append=append))
    except Exception as e:
        click.echo(f"Error: {_normalize_error_message(str(e))}", err=True)
        sys.exit(1)

    if as_json:
        output(result, as_json=True)
    else:
        action = "Appended to" if append else "Updated"
        click.echo(f"{action}: {result['path']}")


# ─────────────────────────────────────────────────────────────────────────────
# Upsert Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("title")
@click.option("--content", "-c", help="Content to add")
@click.option(
    "--file",
    "-f",
    "file_path",
    type=click.Path(exists=True),
    help="Read content from file",
)
@click.option("--stdin", is_flag=True, help="Read content from stdin")
@click.option("--tags", help="Tags for new entry (comma-separated)")
@click.option("--directory", "-d", help="Target directory for new entry")
@click.option("--no-timestamp", is_flag=True, help="Don't add timestamp header")
@click.option("--replace", is_flag=True, help="Replace content instead of appending")
@click.option(
    "--create/--no-create",
    default=True,
    help="Create entry if not found (default: create)",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def upsert(
    title: str,
    content: str | None,
    file_path: str | None,
    stdin: bool,
    tags: str | None,
    directory: str | None,
    no_timestamp: bool,
    replace: bool,
    create: bool,
    as_json: bool,
):
    """Create or append to entry by title.

    Searches for an existing entry with matching title. If found,
    appends content (with timestamp by default). If not found,
    creates a new entry.

    \b
    Examples:
      mx upsert "Project Notes" --content="Session summary here"
      mx upsert "Sessions Log" --stdin < notes.md
      mx upsert "API Docs" --file=api.md --tags="api,docs"
      mx upsert "Debug Log" --content="..." --no-create  # Error if not found

    \b
    Title matching:
      - Exact title match (case-insensitive)
      - Alias match (from entry frontmatter)
      - Fuzzy match (with confidence threshold)
    """
    from .core import AmbiguousMatchError, upsert_entry

    # Validate content source (mutually exclusive)
    content_sources = sum([bool(content), bool(file_path), stdin])
    if content_sources == 0:
        click.echo("Error: Must provide --content, --file, or --stdin", err=True)
        sys.exit(1)
    if content_sources > 1:
        click.echo("Error: --content, --file, and --stdin are mutually exclusive", err=True)
        sys.exit(1)

    # Get content from source
    if stdin:
        content = sys.stdin.read()
    elif file_path:
        content = Path(file_path).read_text(encoding="utf-8")

    # Parse tags
    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    try:
        result = run_async(
            upsert_entry(
                title=title,
                content=content,
                tags=tag_list,
                directory=directory,
                append=not replace,
                timestamp=not no_timestamp,
                create_if_missing=create,
            )
        )
    except AmbiguousMatchError as e:
        if as_json:
            output({
                "error": "ambiguous_match",
                "message": str(e),
                "matches": [m.model_dump() for m in e.matches],
            }, as_json=True)
        else:
            click.echo(f"Error: {e}", err=True)
            click.echo("\nCandidates:", err=True)
            for m in e.matches[:5]:
                click.echo(f"  - {m.path} \"{m.title}\" ({m.score:.0%})", err=True)
            click.echo("\nUse --json for full match list or provide more specific title.", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"Error: {_normalize_error_message(str(e))}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {_normalize_error_message(str(e))}", err=True)
        sys.exit(1)

    if as_json:
        output(result.model_dump(), as_json=True)
    else:
        if result.action == "created":
            click.echo(f"Created: {result.path}")
        else:
            match_info = f" (matched by {result.matched_by})" if result.matched_by else ""
            click.echo(f"Appended to: {result.path}{match_info}")


# ─────────────────────────────────────────────────────────────────────────────
# Session Log Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command("session-log")
@click.option("--message", "-m", help="Session summary message")
@click.option(
    "--file",
    "-f",
    "file_path",
    type=click.Path(exists=True),
    help="Read message from file",
)
@click.option("--stdin", is_flag=True, help="Read message from stdin")
@click.option("--entry", "-e", help="Explicit entry path (overrides context)")
@click.option("--tags", help="Additional tags (comma-separated)")
@click.option("--links", help="Wiki-style links to include (comma-separated)")
@click.option("--no-timestamp", is_flag=True, help="Don't add timestamp header")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def session_log(
    message: str | None,
    file_path: str | None,
    stdin: bool,
    entry: str | None,
    tags: str | None,
    links: str | None,
    no_timestamp: bool,
    as_json: bool,
):
    """Log a session summary to the project's session entry.

    Auto-detects the correct entry from .kbcontext, or uses --entry
    to specify explicitly. Creates the entry if it doesn't exist.

    \b
    Examples:
      mx session-log --message="Fixed auth bug, added tests"
      mx session-log --stdin < session_notes.md
      mx session-log -m "Deployed v2.1" --tags="deployment,release"
      mx session-log -m "..." --entry=projects/myapp/devlog.md

    \b
    Entry resolution:
      1. --entry flag (explicit)
      2. .kbcontext session_entry field
      3. {.kbcontext primary}/sessions.md
      4. Error with guidance if no context
    """
    from .core import log_session as core_log_session

    # Validate message source (mutually exclusive)
    content_sources = sum([bool(message), bool(file_path), stdin])
    if content_sources == 0:
        click.echo("Error: Must provide --message, --file, or --stdin", err=True)
        sys.exit(1)
    if content_sources > 1:
        click.echo("Error: --message, --file, and --stdin are mutually exclusive", err=True)
        sys.exit(1)

    # Get message from source
    if stdin:
        message = sys.stdin.read()
    elif file_path:
        message = Path(file_path).read_text(encoding="utf-8")

    # Parse tags and links
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    link_list = [l.strip() for l in links.split(",")] if links else None

    try:
        result = run_async(
            core_log_session(
                message=message,
                entry_path=entry,
                tags=tag_list,
                links=link_list,
                timestamp=not no_timestamp,
            )
        )
    except ValueError as e:
        click.echo(f"Error: {_normalize_error_message(str(e))}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {_normalize_error_message(str(e))}", err=True)
        sys.exit(1)

    if as_json:
        output(result.model_dump(), as_json=True)
    else:
        click.echo(f"Logged to: {result.path}")
        if result.project:
            click.echo(f"Project: {result.project}")


# ─────────────────────────────────────────────────────────────────────────────
# Patch Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("path", metavar="PATH")
@click.option("--old", help="Exact text to find and replace")
@click.option("--new", help="Replacement text")
@click.option(
    "--old-file",
    type=click.Path(exists=True),
    help="Read --old text from file (for multi-line)",
)
@click.option(
    "--new-file",
    type=click.Path(exists=True),
    help="Read --new text from file (for multi-line)",
)
@click.option("--replace-all", is_flag=True, help="Replace all occurrences")
@click.option("--dry-run", is_flag=True, help="Preview changes without modifying the entry")
@click.option("--backup", is_flag=True, help="Create .bak backup before patching (recommended for large changes)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def patch(
    path: str,
    old: str | None,
    new: str | None,
    old_file: str | None,
    new_file: str | None,
    replace_all: bool,
    dry_run: bool,
    backup: bool,
    as_json: bool,
):
    """Apply surgical find-replace edits to a knowledge base entry.

    PATH is relative to KB root (e.g., "tooling/my-entry.md").

    Finds exact occurrences of --old and replaces with --new.
    Fails if --old is not found or matches multiple times (use --replace-all).

    For multi-line text or special characters (quotes, newlines, tabs), use
    --old-file and --new-file to avoid shell quoting issues.

    If multiple matches are found, the command shows match contexts to help
    you decide whether --replace-all is safe or if you need more specific text.

    \b
    Exit codes:
      0: Success
      1: Text not found
      2: Multiple matches (ambiguous, use --replace-all)
      3: Input error (file not found, permission, encoding, invalid options)

    \b
    Examples:
      mx patch tooling/notes.md --old "old text" --new "new text"
      mx patch tooling/notes.md --old "TODO" --new "DONE" --replace-all
      mx patch tooling/notes.md --old-file old.txt --new-file new.txt
      mx patch tooling/notes.md --old "# TODO" --new "# DONE" --dry-run
    """
    from .core import patch_entry

    # Resolve --old input source
    if old_file and old:
        click.echo("Error: --old and --old-file are mutually exclusive", err=True)
        sys.exit(3)
    if old_file:
        old_text = Path(old_file).read_text(encoding="utf-8")
    elif old is not None:
        old_text = old
    else:
        click.echo("Error: Must provide --old or --old-file", err=True)
        sys.exit(3)

    # Resolve --new input source
    if new_file and new:
        click.echo("Error: --new and --new-file are mutually exclusive", err=True)
        sys.exit(3)
    if new_file:
        new_text = Path(new_file).read_text(encoding="utf-8")
    elif new is not None:
        new_text = new
    else:
        click.echo("Error: Must provide --new or --new-file", err=True)
        sys.exit(3)

    try:
        result = run_async(
            patch_entry(
                path=path,
                old_string=old_text,
                new_string=new_text,
                replace_all=replace_all,
                dry_run=dry_run,
                backup=backup,
            )
        )
    except Exception as e:
        click.echo(f"Error: {_normalize_error_message(str(e))}", err=True)
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
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def list_entries(tag: str | None, category: str | None, limit: int, as_json: bool):
    """List knowledge base entries.

    \b
    Examples:
      mx list
      mx list --tag=tooling
      mx list --category=infrastructure --limit=10
    """
    from .core import list_entries as core_list_entries

    result = run_async(core_list_entries(tag=tag, category=category, limit=limit))

    if as_json:
        output(result, as_json=True)
    else:
        if not result:
            click.echo("No entries found.")
            return

        rows = [{"path": e["path"], "title": e["title"]} for e in result]
        click.echo(format_table(rows, ["path", "title"], {"path": 45, "title": 40}))


# ─────────────────────────────────────────────────────────────────────────────
# What's New Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command("whats-new")
@click.option("--days", "-d", default=30, help="Look back N days")
@click.option("--limit", "-n", default=10, help="Max results")
@click.option(
    "--project", "-p",
    help="Filter by project name (matches path, source_project, or tags)",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def whats_new(days: int, limit: int, project: str | None, as_json: bool):
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
# Info Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def info(as_json: bool):
    """Show knowledge base configuration and stats.

    \b
    Examples:
      mx info
      mx info --json
    """
    from .config import ConfigurationError, get_index_root, get_kb_root
    from .core import get_valid_categories

    try:
        kb_root = get_kb_root()
        index_root = get_index_root()
    except ConfigurationError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    categories = get_valid_categories(kb_root)
    entry_count = sum(1 for _ in kb_root.rglob("*.md")) if kb_root.exists() else 0

    payload = {
        "kb_root": str(kb_root),
        "index_root": str(index_root),
        "categories": categories,
        "entry_count": entry_count,
    }

    if as_json:
        output(payload, as_json=True)
        return

    click.echo("Memex Info")
    click.echo("=" * 40)
    click.echo(f"KB Root:    {kb_root}")
    click.echo(f"Index Root: {index_root}")
    click.echo(f"Entries:    {entry_count}")
    if categories:
        click.echo(f"Categories: {', '.join(categories)}")
    else:
        click.echo("Categories: (none)")


@cli.command("config")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def config_alias(as_json: bool):
    """Alias for mx info."""
    ctx = click.get_current_context()
    ctx.invoke(info, as_json=as_json)


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

        rows = [
            {"path": h["path"], "incoming": h["incoming"],
             "outgoing": h["outgoing"], "total": h["total"]}
            for h in result
        ]
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
        click.echo(f"Error: {_normalize_error_message(str(e))}", err=True)
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
# History Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--limit", "-n", default=10, help="Max entries to show")
@click.option("--rerun", "-r", type=int, help="Re-execute search at position N (1=most recent)")
@click.option("--clear", is_flag=True, help="Clear all search history")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def history(limit: int, rerun: int | None, clear: bool, as_json: bool):
    """Show recent search history and optionally re-run searches.

    \b
    Examples:
      mx history                  # Show last 10 searches
      mx history -n 20            # Show last 20 searches
      mx history --rerun 1        # Re-run most recent search
      mx history -r 3             # Re-run 3rd most recent search
      mx history --clear          # Clear all history
    """
    from . import search_history

    if clear:
        count = search_history.clear_history()
        click.echo(f"Cleared {count} search history entries.")
        return

    if rerun is not None:
        entry = search_history.get_by_index(rerun)
        if entry is None:
            click.echo(f"Error: No search at position {rerun}", err=True)
            sys.exit(1)

        # Re-run the search using the search command logic
        click.echo(f"Re-running: {entry.query}")
        if entry.tags:
            click.echo(f"  Tags: {', '.join(entry.tags)}")
        click.echo(f"  Mode: {entry.mode}")
        click.echo()

        # Import and run search
        from .core import search as core_search

        result = run_async(core_search(
            query=entry.query,
            limit=10,
            mode=entry.mode,
            tags=entry.tags if entry.tags else None,
            include_content=False,
        ))

        # Record this re-run in history
        search_history.record_search(
            query=entry.query,
            result_count=len(result.results),
            mode=entry.mode,
            tags=entry.tags if entry.tags else None,
        )

        if as_json:
            output(
                [{"path": r.path, "title": r.title,
                  "score": r.score, "snippet": r.snippet}
                 for r in result.results],
                as_json=True,
            )
        else:
            if not result.results:
                click.echo("No results found.")
                return

            rows = [
                {"path": r.path, "title": r.title, "score": f"{r.score:.2f}"}
                for r in result.results
            ]
            click.echo(format_table(rows, ["path", "title", "score"], {"path": 40, "title": 35}))
        return

    # Show history
    entries = search_history.get_recent(limit=limit)

    if as_json:
        output([
            {
                "position": i + 1,
                "query": e.query,
                "timestamp": e.timestamp.isoformat(),
                "result_count": e.result_count,
                "mode": e.mode,
                "tags": e.tags,
            }
            for i, e in enumerate(entries)
        ], as_json=True)
        return

    if not entries:
        click.echo("No search history.")
        return

    click.echo("Recent searches:\n")
    for i, entry in enumerate(entries, 1):
        time_str = entry.timestamp.strftime("%Y-%m-%d %H:%M")
        tag_str = f" [tags: {', '.join(entry.tags)}]" if entry.tags else ""
        result_str = f"{entry.result_count} results" if entry.result_count else "no results"
        click.echo(f"  {i:2d}. {entry.query}")
        click.echo(f"      {time_str} | {entry.mode} | {result_str}{tag_str}")

    click.echo("\nTip: Use 'mx history --rerun N' to re-execute a search")


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
    click.echo(
        f"✓ Indexed {result.kb_files} entries, "
        f"{result.whoosh_docs} keyword docs, {result.chroma_docs} semantic docs"
    )


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
@click.option("--suggest", is_flag=True, help="Show bootstrap suggestions if no context found")
def context_show(as_json: bool, suggest: bool):
    """Show the current project context.

    Searches for .kbcontext file starting from current directory.
    When no context is found, use --suggest to show auto-detected
    project info and suggested bootstrap command.

    \b
    Examples:
      mx context show
      mx context show --suggest
      mx context show --json
    """
    from .context import detect_project_context, get_kb_context, get_session_entry_path

    ctx = get_kb_context()

    if ctx is None:
        # No .kbcontext found - show detected context if --suggest
        detected = detect_project_context() if suggest else None

        if as_json:
            result = {"found": False, "message": "No .kbcontext file found"}
            if detected and detected.project_name:
                result["detected"] = {
                    "project_name": detected.project_name,
                    "git_root": str(detected.git_root) if detected.git_root else None,
                    "suggested_kb_directory": detected.suggested_kb_directory,
                    "detection_method": detected.detection_method,
                }
                result["suggestion"] = f"mx context init --project={detected.project_name}"
            output(result, as_json=True)
        else:
            click.echo("No .kbcontext file found.")
            if detected and detected.project_name:
                click.echo()
                click.echo(f"Detected project: {detected.project_name} (from {detected.detection_method})")
                if detected.git_root:
                    click.echo(f"Git root:         {detected.git_root}")
                click.echo(f"Suggested KB dir: {detected.suggested_kb_directory}")
                click.echo()
                click.echo("To set up context:")
                click.echo(f"  mx context init --project={detected.project_name}")
            else:
                click.echo("Run 'mx context init' to create one.")
        return

    # Context found - show it
    session_entry = get_session_entry_path(ctx)

    if as_json:
        output({
            "found": True,
            "source_file": str(ctx.source_file) if ctx.source_file else None,
            "primary": ctx.primary,
            "paths": ctx.paths,
            "default_tags": ctx.default_tags,
            "project": ctx.project,
            "session_entry": ctx.session_entry,
            "session_entry_resolved": session_entry,
        }, as_json=True)
    else:
        click.echo(f"Context file: {ctx.source_file}")
        click.echo(f"Primary:      {ctx.primary or '(not set)'}")
        click.echo(f"Paths:        {', '.join(ctx.paths) if ctx.paths else '(none)'}")
        click.echo(f"Default tags: {', '.join(ctx.default_tags) if ctx.default_tags else '(none)'}")
        if ctx.project:
            click.echo(f"Project:      {ctx.project}")
        if session_entry:
            click.echo(f"Session log:  {session_entry}")


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
def context_init(project: str | None, directory: str | None, force: bool):
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
        click.echo(f"Error: {_normalize_error_message(str(e))}", err=True)
        sys.exit(1)

    if as_json:
        output(result, as_json=True)
    else:
        if result.get("had_backlinks"):
            click.echo(f"Warning: Entry had {len(result['had_backlinks'])} backlinks", err=True)
        click.echo(f"Deleted: {result['deleted']}")


# ─────────────────────────────────────────────────────────────────────────────
# Beads Integration
# ─────────────────────────────────────────────────────────────────────────────

PRIORITY_LABELS = {0: "critical", 1: "high", 2: "medium", 3: "low", 4: "backlog"}
PRIORITY_ABBREV = {0: "crit", 1: "high", 2: "med", 3: "low", 4: "back"}


def _load_beads_registry() -> dict[str, Path]:
    """Load beads project registry from .beads-registry.yaml.

    Returns:
        Dict mapping project prefix to resolved path.
    """
    import yaml

    from .config import get_kb_root

    kb_root = get_kb_root()
    registry_path = kb_root / ".beads-registry.yaml"

    if not registry_path.exists():
        return {}

    try:
        with open(registry_path) as f:
            raw = yaml.safe_load(f) or {}

        # Resolve relative paths
        resolved = {}
        for prefix, path_str in raw.items():
            if not isinstance(path_str, str) or path_str.startswith("#"):
                continue
            if path_str == ".":
                resolved[prefix] = kb_root
            else:
                path = Path(path_str)
                if not path.is_absolute():
                    path = kb_root / path
                resolved[prefix] = path.resolve()

        return resolved
    except Exception:
        return {}


def _resolve_beads_project(project: str | None) -> tuple[str, Path]:
    """Resolve beads project from prefix, cwd, or default.

    Args:
        project: Optional project prefix from --project flag

    Returns:
        Tuple of (prefix, project_path)

    Raises:
        click.ClickException: If project cannot be resolved
    """
    from .beads_client import find_beads_db
    from .config import get_kb_root

    registry = _load_beads_registry()

    if project:
        # Explicit project specified
        if project in registry:
            return project, registry[project]
        available = ", ".join(sorted(registry.keys())) if registry else "(none)"
        raise click.ClickException(f"Unknown project '{project}'. Available: {available}")

    # Try to detect from cwd
    cwd = Path.cwd()
    for prefix, path in registry.items():
        try:
            if cwd == path or cwd.is_relative_to(path):
                return prefix, path
        except ValueError:
            continue

    # Try KB root as fallback
    kb_root = get_kb_root()
    beads = find_beads_db(kb_root)
    if beads:
        for prefix, path in registry.items():
            if path == kb_root:
                return prefix, path
        return "kb", kb_root

    available = ", ".join(sorted(registry.keys())) if registry else "(none)"
    raise click.ClickException(
        f"No beads project found. Use --project or run from a project directory.\n"
        f"Available projects: {available}"
    )


def _parse_issue_id(issue_id: str, project: str | None) -> tuple[str, str]:
    """Parse issue ID, extracting project prefix if present.

    Args:
        issue_id: Issue ID like 'memex-42', 'voidlabs-kb-abc', or '42'
        project: Explicit project prefix if provided

    Returns:
        Tuple of (project_prefix, full_issue_id)
    """
    registry = _load_beads_registry()

    # Try to match a known prefix
    for prefix in sorted(registry.keys(), key=len, reverse=True):
        if issue_id.startswith(f"{prefix}-"):
            return prefix, issue_id

    # If project explicitly provided, use it
    if project:
        full_id = f"{project}-{issue_id}" if not issue_id.startswith(project) else issue_id
        return project, full_id

    raise click.ClickException(
        f"Cannot determine project for issue '{issue_id}'. "
        "Use format 'project-123' or specify --project."
    )


def _get_beads_db_or_fail(project_path: Path, project_prefix: str):
    """Get beads database or raise ClickException.

    Args:
        project_path: Path to project root
        project_prefix: Project prefix for error messages

    Returns:
        BeadsProject with validated db_path

    Raises:
        click.ClickException: If beads database not found
    """
    from .beads_client import find_beads_db

    if not project_path.exists():
        raise click.ClickException(f"Beads project path does not exist: {project_path}")

    beads = find_beads_db(project_path)
    if not beads:
        raise click.ClickException(
            f"No beads database found for '{project_prefix}' at: {project_path}/.beads/beads.db"
        )

    return beads


def _format_priority(priority: int | None) -> str:
    """Format priority as label."""
    if priority is None:
        return "medium"
    return PRIORITY_LABELS.get(priority, "medium")


@cli.group()
def beads():
    """Browse beads issue tracking across registered projects.

    Beads projects are registered in .beads-registry.yaml at KB root.
    Use --project to specify a project, or commands auto-detect from cwd.

    \b
    Quick start:
      mx beads list                    # List issues
      mx beads show epstein-42         # Show issue details
      mx beads kanban                  # Kanban board view
      mx beads status                  # Project statistics
      mx beads projects                # List registered projects
    """
    pass


@beads.command("list")
@click.option("--project", "-p", help="Beads project prefix from registry")
@click.option(
    "--status", "-s",
    type=click.Choice(["open", "in_progress", "closed", "all"]),
    default="all",
    help="Filter by status"
)
@click.option("--type", "-t", "issue_type", help="Filter by type (task, bug, feature, epic)")
@click.option("--limit", "-n", default=50, help="Max results")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def beads_list(project: str | None, status: str, issue_type: str | None, limit: int, as_json: bool):
    """List issues from a beads project.

    \b
    Examples:
      mx beads list                         # All issues from detected project
      mx beads list -p epstein              # Issues from epstein project
      mx beads list --status=open           # Only open issues
      mx beads list --type=bug --limit=10   # 10 bugs
    """
    from .beads_client import list_issues

    prefix, project_path = _resolve_beads_project(project)
    beads = _get_beads_db_or_fail(project_path, prefix)

    issues = list_issues(beads.db_path)

    # Apply filters
    if status != "all":
        issues = [i for i in issues if i.get("status") == status]
    if issue_type:
        issues = [i for i in issues if i.get("issue_type") == issue_type]

    # Limit results
    issues = issues[:limit]

    # Add priority labels
    for issue in issues:
        issue["priority_label"] = _format_priority(issue.get("priority"))

    if as_json:
        output(issues, as_json=True)
    else:
        if not issues:
            click.echo(f"No issues found for {prefix}")
            return

        rows = [
            {
                "id": i.get("id", ""),
                "status": i.get("status", ""),
                "priority": i.get("priority_label", ""),
                "type": i.get("issue_type", ""),
                "title": i.get("title", ""),
            }
            for i in issues
        ]
        click.echo(format_table(rows, ["id", "status", "priority", "type", "title"], {"title": 50}))
        click.echo(f"\nShowing {len(issues)} issues from {prefix}")


@beads.command("show")
@click.argument("issue_id")
@click.option("--project", "-p", help="Beads project prefix (auto-detected from issue ID)")
@click.option("--no-comments", is_flag=True, help="Exclude comments")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def beads_show(issue_id: str, project: str | None, no_comments: bool, as_json: bool):
    """Show detailed information for a specific issue.

    Issue ID can include project prefix (e.g., 'epstein-42') or just
    the number if --project is specified.

    \b
    Examples:
      mx beads show epstein-42              # Full issue details with comments
      mx beads show 42 -p epstein           # Equivalent with explicit project
      mx beads show epstein-42 --no-comments # Without comments
    """
    from .beads_client import get_comments, show_issue

    prefix, full_id = _parse_issue_id(issue_id, project)
    registry = _load_beads_registry()

    if prefix not in registry:
        available = ", ".join(sorted(registry.keys())) if registry else "(none)"
        raise click.ClickException(f"Unknown project '{prefix}'. Available: {available}")

    project_path = registry[prefix]
    beads = _get_beads_db_or_fail(project_path, prefix)

    issue = show_issue(beads.db_path, full_id)
    if not issue:
        raise click.ClickException(f"Issue not found: {full_id}")

    comments = [] if no_comments else get_comments(beads.db_path, full_id)

    if as_json:
        output({"issue": issue, "comments": comments}, as_json=True)
    else:
        click.echo(f"Issue: {full_id}")
        click.echo("=" * 80)
        click.echo()
        click.echo(f"Title:       {issue.get('title', '')}")
        click.echo(f"Status:      {issue.get('status', '')}")
        priority = issue.get('priority', 2)
        click.echo(f"Priority:    {_format_priority(priority)} ({priority})")
        click.echo(f"Type:        {issue.get('issue_type', '')}")
        click.echo(f"Created:     {issue.get('created_at', '')} by {issue.get('created_by', '')}")
        if issue.get("updated_at"):
            click.echo(f"Updated:     {issue.get('updated_at', '')}")

        if issue.get("description"):
            click.echo()
            click.echo("Description:")
            for line in issue["description"].split("\n"):
                click.echo(f"  {line}")

        if comments:
            click.echo()
            click.echo("-" * 80)
            click.echo(f"Comments ({len(comments)}):")
            click.echo("-" * 80)
            for c in comments:
                click.echo()
                click.echo(f"[{c.get('created_at', '')}] {c.get('author', '')}:")
                content = c.get("content", "")
                for line in content.split("\n"):
                    click.echo(f"  {line}")


@beads.command("kanban")
@click.option("--project", "-p", help="Beads project prefix from registry")
@click.option("--compact", is_flag=True, help="Compact view (titles only)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def beads_kanban(project: str | None, compact: bool, as_json: bool):
    """Display issues grouped by status (kanban board view).

    Shows issues in columns: Open | In Progress | Closed

    \b
    Examples:
      mx beads kanban                       # Kanban for detected project
      mx beads kanban -p voidlabs-ansible   # Specific project
      mx beads kanban --compact             # Titles only
    """
    from .beads_client import list_issues

    prefix, project_path = _resolve_beads_project(project)
    beads = _get_beads_db_or_fail(project_path, prefix)

    issues = list_issues(beads.db_path)

    # Group by status
    columns = {
        "open": {"status": "open", "label": "Open", "issues": []},
        "in_progress": {"status": "in_progress", "label": "In Progress", "issues": []},
        "closed": {"status": "closed", "label": "Closed", "issues": []},
    }

    for issue in issues:
        status = issue.get("status", "open")
        if status in columns:
            columns[status]["issues"].append({
                "id": issue.get("id", ""),
                "title": issue.get("title", ""),
                "priority": issue.get("priority", 3),
                "priority_label": _format_priority(issue.get("priority")),
            })

    # Sort by priority within each column
    for col in columns.values():
        col["issues"].sort(key=lambda x: x.get("priority", 3))

    if as_json:
        output({
            "project": prefix,
            "total_issues": len(issues),
            "columns": list(columns.values()),
        }, as_json=True)
    else:
        total = len(issues)
        click.echo(f"Kanban: {prefix} ({total} issues)")
        click.echo()

        # Format as columns
        col_width = 28
        col_list = list(columns.values())

        # Header
        headers = [f"{c['label'].upper()} ({len(c['issues'])})" for c in col_list]
        click.echo("  ".join(h.ljust(col_width) for h in headers))
        click.echo("  ".join("-" * col_width for _ in col_list))

        # Rows
        max_rows = max(len(c["issues"]) for c in col_list) if col_list else 0
        for row_idx in range(min(max_rows, 20)):  # Limit to 20 rows
            row_parts = []
            for col in col_list:
                if row_idx < len(col["issues"]):
                    issue = col["issues"][row_idx]
                    short_id = issue["id"].split("-")[-1] if "-" in issue["id"] else issue["id"]
                    if compact:
                        text = f"#{short_id} {issue['title']}"
                    else:
                        prio = PRIORITY_ABBREV.get(issue.get("priority", 3), "med")
                        text = f"[{prio}] #{short_id} {issue['title']}"
                    if len(text) > col_width - 1:
                        text = text[:col_width - 4] + "..."
                else:
                    text = ""
                row_parts.append(text.ljust(col_width))
            click.echo("  ".join(row_parts))


@beads.command("status")
@click.option("--project", "-p", help="Beads project prefix from registry")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def beads_status(project: str | None, as_json: bool):
    """Show project statistics and health summary.

    Displays counts by status, priority distribution, and type breakdown.

    \b
    Examples:
      mx beads status                       # Stats for detected project
      mx beads status -p memex              # Stats for memex project
    """
    from .beads_client import list_issues

    prefix, project_path = _resolve_beads_project(project)
    beads = _get_beads_db_or_fail(project_path, prefix)

    issues = list_issues(beads.db_path)

    # Count by status
    by_status = {"open": 0, "in_progress": 0, "closed": 0}
    for issue in issues:
        status = issue.get("status", "open")
        if status in by_status:
            by_status[status] += 1

    # Count by priority
    by_priority = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for issue in issues:
        prio = issue.get("priority", 2)
        if prio in by_priority:
            by_priority[prio] += 1

    # Count by type
    by_type: dict[str, int] = {}
    for issue in issues:
        itype = issue.get("issue_type", "task")
        by_type[itype] = by_type.get(itype, 0) + 1

    if as_json:
        output({
            "project": prefix,
            "project_path": str(project_path),
            "db_path": str(beads.db_path),
            "total": len(issues),
            "by_status": by_status,
            "by_priority": {PRIORITY_LABELS[k]: v for k, v in by_priority.items()},
            "by_type": by_type,
        }, as_json=True)
    else:
        click.echo(f"Beads Status: {prefix}")
        click.echo("=" * 80)
        click.echo()
        click.echo("By Status:")
        click.echo(f"  Open:         {by_status['open']} issues")
        click.echo(f"  In Progress:  {by_status['in_progress']} issues")
        click.echo(f"  Closed:       {by_status['closed']} issues")
        click.echo(f"  Total:        {len(issues)} issues")
        click.echo()
        click.echo("By Priority:")
        for prio, label in PRIORITY_LABELS.items():
            click.echo(f"  {label.capitalize():12}  {by_priority[prio]}")
        click.echo()
        click.echo("By Type:")
        for itype, count in sorted(by_type.items()):
            click.echo(f"  {itype.capitalize():12}  {count}")
        click.echo()
        click.echo(f"Project Path: {project_path}")
        click.echo(f"DB Path:      {beads.db_path}")


@beads.command("projects")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def beads_projects(as_json: bool):
    """List all registered beads projects from .beads-registry.yaml.

    Shows project prefix, path, and availability status.

    \b
    Examples:
      mx beads projects                     # List all projects
      mx beads projects --json              # JSON output
    """
    from .beads_client import find_beads_db
    from .config import get_kb_root

    registry = _load_beads_registry()
    kb_root = get_kb_root()
    registry_path = kb_root / ".beads-registry.yaml"

    projects = []
    for prefix, path in sorted(registry.items()):
        beads = find_beads_db(path)
        projects.append({
            "prefix": prefix,
            "path": str(path),
            "available": beads is not None,
        })

    if as_json:
        output({
            "registry_path": str(registry_path),
            "projects": projects,
        }, as_json=True)
    else:
        click.echo("BEADS PROJECTS")
        click.echo("=" * 80)
        click.echo()

        if not projects:
            click.echo("No projects registered.")
            click.echo(f"\nCreate {registry_path} to register projects.")
            return

        rows = [
            {
                "prefix": p["prefix"],
                "path": p["path"],
                "status": "available" if p["available"] else "not found",
            }
            for p in projects
        ]
        click.echo(format_table(rows, ["prefix", "path", "status"], {"path": 40}))
        click.echo()
        click.echo(f"Registry: {registry_path}")


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
