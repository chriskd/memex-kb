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
@click.version_option(version="0.1.0", prog_name="mx")
def cli():
    """mx: Token-efficient CLI for memex knowledge base.

    Search, browse, and manage KB entries without MCP context overhead.

    \b
    Quick start:
      mx search "deployment"     # Find entries
      mx get tooling/beads.md    # Read an entry
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


def _detect_current_project() -> Optional[str]:
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
@click.option("--project", "-p", help="Include recent entries for project (auto-detected if not specified)")
@click.option("--days", "-d", default=7, help="Days to look back for project entries")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def prime(full: bool, mcp: bool, project: Optional[str], days: int, as_json: bool):
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
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def search(query: str, tags: Optional[str], mode: str, limit: int, content: bool, as_json: bool):
    """Search the knowledge base.

    \b
    Examples:
      mx search "deployment"
      mx search "docker" --tags=infrastructure
      mx search "api" --mode=semantic --limit=5
    """
    from .core import search as core_search

    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    result = run_async(core_search(
        query=query,
        limit=limit,
        mode=mode,
        tags=tag_list,
        include_content=content,
    ))

    if as_json:
        output([{"path": r.path, "title": r.title, "score": r.score, "snippet": r.snippet} for r in result.results], as_json=True)
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
@click.option("--force", is_flag=True, help="Create even if potential duplicates are detected")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def add(
    title: str,
    tags: str,
    category: str,
    content: Optional[str],
    file_path: Optional[str],
    stdin: bool,
    force: bool,
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

    def _print_created(add_result):
        path = add_result.path if hasattr(add_result, 'path') else add_result.get('path')
        click.echo(f"Created: {path}")

        suggested_links = add_result.suggested_links if hasattr(add_result, 'suggested_links') else add_result.get('suggested_links', [])
        if suggested_links:
            click.echo("\nSuggested links:")
            for link in suggested_links[:5]:
                score = link.get('score', 0) if isinstance(link, dict) else link.score
                path_str = link.get('path', '') if isinstance(link, dict) else link.path
                click.echo(f"  - {path_str} ({score:.2f})")

        suggested_tags = add_result.suggested_tags if hasattr(add_result, 'suggested_tags') else add_result.get('suggested_tags', [])
        if suggested_tags:
            click.echo("\nSuggested tags:")
            for tag in suggested_tags[:5]:
                tag_name = tag.get('tag', '') if isinstance(tag, dict) else tag.tag
                reason = tag.get('reason', '') if isinstance(tag, dict) else tag.reason
                click.echo(f"  - {tag_name} ({reason})")

    def _print_duplicates(add_result):
        warning = add_result.warning or "Potential duplicates detected."
        warning = warning.replace("force=True", "--force")
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
        click.echo(f"Error: {e}", err=True)
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
                    click.echo(f"Error: {e}", err=True)
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


def _suggest_category_from_content(content: str, categories: list[str]) -> Optional[str]:
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
@click.option("--file", "-f", "file_path", type=click.Path(exists=True), help="Read content from file")
@click.option("--stdin", is_flag=True, help="Read content from stdin")
@click.option("--content", "-c", help="Raw content to add")
@click.option("--title", "-t", help="Override auto-detected title")
@click.option("--tags", help="Override auto-suggested tags (comma-separated)")
@click.option("--category", help="Override auto-suggested category")
@click.option("--confirm", "-y", is_flag=True, help="Auto-confirm without prompting")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def quick_add(
    file_path: Optional[str],
    stdin: bool,
    content: Optional[str],
    title: Optional[str],
    tags: Optional[str],
    category: Optional[str],
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
    from .core import add_entry, get_valid_categories, compute_tag_suggestions

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
        auto_category = click.prompt("Category", default=valid_categories[0] if valid_categories else "notes")

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
        click.echo(f"Error: {e}", err=True)
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
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def list_entries(tag: Optional[str], category: Optional[str], limit: int, as_json: bool):
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
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """Entry point for mx CLI."""
    from ._logging import configure_logging
    configure_logging()
    cli()


if __name__ == "__main__":
    main()
