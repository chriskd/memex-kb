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


def _handle_error(
    ctx: click.Context,
    error: Exception,
    fallback_message: str | None = None,
) -> None:
    """Handle an error with optional JSON output.

    If --json-errors is enabled, outputs structured JSON error.
    Otherwise, outputs human-readable error message.

    Args:
        ctx: Click context (must have obj["json_errors"] set).
        error: The exception that occurred.
        fallback_message: Optional message to use for non-MemexError exceptions.
    """
    from .errors import MemexError, format_error_json

    json_errors = ctx.obj.get("json_errors", False) if ctx.obj else False

    if isinstance(error, MemexError):
        if json_errors:
            click.echo(error.to_json(), err=True)
        else:
            click.echo(f"Error: {_normalize_error_message(error.message)}", err=True)
    else:
        message = fallback_message or str(error)
        if json_errors:
            # Map common exceptions to error codes
            code = _infer_error_code(error, message)
            click.echo(format_error_json(code, _normalize_error_message(message)), err=True)
        else:
            click.echo(f"Error: {_normalize_error_message(message)}", err=True)

    sys.exit(1)


def _infer_error_code(error: Exception, message: str):
    """Infer an error code from exception type and message.

    Used for backwards compatibility when non-MemexError exceptions are raised.
    """
    from .errors import ErrorCode

    message_lower = message.lower()

    # Check exception types first
    if isinstance(error, FileNotFoundError):
        return ErrorCode.ENTRY_NOT_FOUND
    if isinstance(error, PermissionError):
        return ErrorCode.PERMISSION_DENIED

    # Check message patterns
    if "not found" in message_lower:
        return ErrorCode.ENTRY_NOT_FOUND
    if "already exists" in message_lower:
        return ErrorCode.ENTRY_EXISTS
    if "duplicate" in message_lower:
        return ErrorCode.DUPLICATE_DETECTED
    if "invalid path" in message_lower or "path escapes" in message_lower:
        return ErrorCode.INVALID_PATH
    if "ambiguous" in message_lower:
        return ErrorCode.AMBIGUOUS_MATCH
    if "category" in message_lower and ("required" in message_lower or "not found" in message_lower):
        return ErrorCode.INVALID_CATEGORY
    if "tag" in message_lower and "required" in message_lower:
        return ErrorCode.INVALID_TAGS
    if "index" in message_lower and "unavailable" in message_lower:
        return ErrorCode.INDEX_UNAVAILABLE
    if "semantic" in message_lower and ("unavailable" in message_lower or "not available" in message_lower):
        return ErrorCode.SEMANTIC_SEARCH_UNAVAILABLE
    if "parse" in message_lower or "frontmatter" in message_lower:
        return ErrorCode.PARSE_ERROR

    # Default to a generic file error
    return ErrorCode.FILE_READ_ERROR


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


def _handle_add_error(ctx: click.Context, error: Exception, tags: list[str]) -> None:
    """Handle errors from add/quick-add with special category error formatting.

    Supports --json-errors output while preserving the category error guidance
    for human-readable output.
    """
    from .errors import ErrorCode, MemexError

    message = str(error)
    json_errors = ctx.obj.get("json_errors", False) if ctx.obj else False

    # Special handling for category errors
    if "Either 'category' or 'directory' must be provided" in message:
        if json_errors:
            from . import core

            valid_categories = core.get_valid_categories()
            tag_set = {tag.strip().lower() for tag in tags if tag.strip()}
            matches = [cat for cat in valid_categories if cat.lower() in tag_set]

            error = MemexError(
                ErrorCode.INVALID_CATEGORY,
                "--category is required",
                {
                    "suggestion": "Provide --category or run 'mx context init'",
                    "available_categories": valid_categories,
                    "matching_tags": matches if matches else None,
                    "your_tags": tags,
                },
            )
            click.echo(error.to_json(), err=True)
        else:
            click.echo(_format_missing_category_error(tags, message), err=True)
        sys.exit(1)

    # Use standard error handler for other errors
    _handle_error(ctx, error)


# ─────────────────────────────────────────────────────────────────────────────
# Status Output (default when no subcommand)
# ─────────────────────────────────────────────────────────────────────────────


def _show_status() -> None:
    """Show KB status with context, recent entries, and suggested commands.

    Displayed when running `mx` with no arguments. Provides quick orientation
    for agents and humans about the current KB state.
    """
    from .config import ConfigurationError, get_kb_root
    from .context import get_kb_context, detect_project_context

    # Track what we successfully loaded
    kb_root = None
    context = None
    detected = None
    entries = []
    project_name = None

    # Try to get KB configuration
    try:
        kb_root = get_kb_root()
    except ConfigurationError:
        pass

    # Try to get context
    context = get_kb_context()
    if context:
        project_name = context.get_project_name()
    else:
        detected = detect_project_context()
        project_name = detected.project_name if detected else None

    # Get recent entries if KB is available
    if kb_root and kb_root.exists():
        entries = _get_recent_entries_for_status(kb_root, project_name, limit=5)

    # Build output
    _output_status(kb_root, context, detected, entries, project_name)


def _get_recent_entries_for_status(
    kb_root: Path, project: str | None, limit: int = 5
) -> list[dict]:
    """Get recent entries for status display.

    Tries to get project-specific entries first, falls back to all entries.

    Args:
        kb_root: Knowledge base root directory.
        project: Optional project name to filter by.
        limit: Maximum entries to return.

    Returns:
        List of entry dicts with path, title, date, activity_type.
    """
    from .core import whats_new as core_whats_new

    try:
        # Try project-specific first
        if project:
            entries = run_async(core_whats_new(days=14, limit=limit, project=project))
            if entries:
                return entries

        # Fall back to all recent entries
        return run_async(core_whats_new(days=14, limit=limit))
    except Exception:
        # Fail silently - status output should be resilient
        return []


def _output_status(
    kb_root: Path | None,
    context,  # KBContext | None
    detected,  # DetectedContext | None
    entries: list[dict],
    project_name: str | None,
) -> None:
    """Output the status display.

    Args:
        kb_root: KB root path (None if not configured).
        context: Loaded KBContext (None if no .kbcontext).
        detected: Auto-detected context (None if context exists).
        entries: Recent entries to display.
        project_name: Current project name.
    """
    lines = []

    # Header
    lines.append("Memex Knowledge Base")
    lines.append("=" * 40)

    # Context section
    if kb_root:
        lines.append(f"KB Root: {kb_root}")

        if context:
            lines.append(f"Context: {context.source_file}")
            if context.primary:
                lines.append(f"Primary: {context.primary}")
            if context.default_tags:
                lines.append(f"Tags:    {', '.join(context.default_tags)}")
        elif detected and detected.project_name:
            lines.append(f"Project: {detected.project_name} (auto-detected)")
            lines.append("         Run 'mx context init' to configure")
        else:
            lines.append("Context: (none)")
    else:
        lines.append("KB Root: NOT CONFIGURED")
        lines.append("")
        lines.append("Set MEMEX_KB_ROOT and MEMEX_INDEX_ROOT environment variables")
        lines.append("to point to your knowledge base directory.")

    # Recent entries section
    if entries:
        lines.append("")
        header = "Recent Entries"
        if project_name and any(
            e.get("path", "").startswith(f"projects/{project_name}")
            or project_name in e.get("tags", [])
            for e in entries
        ):
            header = f"Recent Entries ({project_name})"
        lines.append(header)
        lines.append("-" * 40)

        for e in entries[:5]:
            activity = "NEW" if e.get("activity_type") == "created" else "UPD"
            date_str = str(e.get("activity_date", ""))[:10]
            path = e.get("path", "")
            title = e.get("title", "Untitled")

            # Truncate path if too long
            if len(path) > 35:
                path = "..." + path[-32:]

            lines.append(f"  {activity} {date_str}  {path}")
            if title and title != path:
                lines.append(f"                      {title[:40]}")

    # Suggested commands section
    lines.append("")
    lines.append("Commands")
    lines.append("-" * 40)

    if not kb_root:
        lines.append("  mx --help           Show all commands")
    else:
        if entries:
            # KB has content
            lines.append("  mx search \"query\"   Search the knowledge base")
            lines.append("  mx whats-new        Recent changes")
            lines.append("  mx tree             Browse structure")
        else:
            # Empty KB
            lines.append("  mx add --title=\"...\" --tags=\"...\"  Add first entry")
            lines.append("  mx tree             Browse structure")

        if not context:
            lines.append("  mx context init     Set up project context")

        lines.append("  mx --help           Show all commands")

    # Output
    click.echo("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Main CLI Group
# ─────────────────────────────────────────────────────────────────────────────


@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0", prog_name="mx")
@click.option(
    "--json-errors",
    is_flag=True,
    envvar="MX_JSON_ERRORS",
    help="Output errors as JSON for programmatic handling",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    envvar="MEMEX_QUIET",
    help="Suppress warnings, show only errors and essential output",
)
@click.pass_context
def cli(ctx: click.Context, json_errors: bool, quiet: bool):
    """mx: Token-efficient CLI for memex knowledge base.

    Search, browse, and manage KB entries without MCP context overhead.

    \b
    Quick start:
      mx search "deployment"     # Find entries
      mx get tooling/beads.md    # Read an entry
      mx info                    # Show KB configuration
      mx tree                    # Browse structure
      mx health                  # Check KB health

    \b
    For programmatic error handling:
      mx --json-errors add ...   # Errors output as JSON with error codes

    \b
    For quieter output:
      mx --quiet search ...      # Suppress warnings
      MEMEX_QUIET=1 mx search    # Or use environment variable
    """
    from ._logging import set_quiet_mode

    ctx.ensure_object(dict)
    ctx.obj["json_errors"] = json_errors
    ctx.obj["quiet"] = quiet

    if quiet:
        set_quiet_mode(True)
    if ctx.invoked_subcommand is None:
        _show_status()


# ─────────────────────────────────────────────────────────────────────────────
# Prime Command (Agent Context Injection)
# ─────────────────────────────────────────────────────────────────────────────

PRIME_OUTPUT = """# Memex Knowledge Base

> Search organizational knowledge before reinventing. Add discoveries for future agents.

**Use `mx` CLI instead of MCP tools** - CLI uses ~0 tokens vs MCP schema overhead.

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

- Looking for org patterns, guides, troubleshooting
- Before implementing something that might exist
- Understanding infrastructure or deployment

## When to Contribute

- Discovered reusable pattern or solution
- Troubleshooting steps worth preserving
- Infrastructure or deployment knowledge

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
# Init Command (KB Bootstrap)
# ─────────────────────────────────────────────────────────────────────────────

# Default categories to create in a new KB
DEFAULT_KB_CATEGORIES = ["projects", "tooling", "infrastructure"]


# Project indicator files for auto-detection
PROJECT_INDICATORS = [
    ".git",
    "package.json",
    "pyproject.toml",
    "setup.py",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
]


@cli.command("init")
@click.option(
    "--kb-root",
    type=click.Path(),
    help="KB root path (defaults to MEMEX_KB_ROOT or ~/.memex/kb)",
)
@click.option(
    "--index-root",
    type=click.Path(),
    help="Index root path (defaults to MEMEX_INDEX_ROOT or <kb-root>/../.memex-indices)",
)
@click.option("--no-context", is_flag=True, help="Skip .kbcontext creation prompt")
@click.option("--force", "-f", is_flag=True, help="Recreate directory structure (preserves existing entries)")
def init_kb(
    kb_root: str | None,
    index_root: str | None,
    no_context: bool,
    force: bool,
):
    """Initialize a new memex knowledge base.

    Creates the KB directory structure with default categories and optionally
    sets up a .kbcontext file for the current project.

    \b
    Default structure created:
      <kb-root>/
        ├── projects/       # Project-specific documentation
        ├── tooling/        # Tools and utilities
        └── infrastructure/ # Infrastructure and DevOps

    \b
    .kbcontext files (created per-project):
      When you run 'mx init' from a project directory, it can create a
      .kbcontext file that provides:
      - Auto-routing: new entries go to projects/<name> by default
      - Search boost: project entries rank higher in search results
      - Default tags: entries are pre-tagged with the project name

    \b
    Examples:
      mx init                           # Use defaults
      mx init --kb-root ~/my-kb         # Custom location
      mx init --no-context              # Skip .kbcontext setup
      mx init --force                   # Recreate dirs (safe, keeps entries)
    """
    from .config import ConfigurationError, get_index_root, get_kb_root
    from .context import CONTEXT_FILENAME, create_default_context, detect_project_context

    lines = []
    created_dirs = []
    env_exports = []

    # Determine KB root
    resolved_kb_root = None
    kb_from_env = False

    if kb_root:
        resolved_kb_root = Path(kb_root).expanduser().resolve()
    else:
        try:
            resolved_kb_root = get_kb_root()
            kb_from_env = True
        except ConfigurationError:
            # Default to ~/.memex/kb
            resolved_kb_root = Path.home() / ".memex" / "kb"

    # Determine index root
    resolved_index_root = None
    index_from_env = False

    if index_root:
        resolved_index_root = Path(index_root).expanduser().resolve()
    else:
        try:
            resolved_index_root = get_index_root()
            index_from_env = True
        except ConfigurationError:
            # Default to sibling of kb root
            resolved_index_root = resolved_kb_root.parent / ".memex-indices"

    # Check if already initialized
    kb_exists = resolved_kb_root.exists() and any(resolved_kb_root.iterdir())
    if kb_exists and not force:
        click.echo(f"✓ KB already initialized at: {resolved_kb_root}")
        click.echo("")
        click.echo("Your KB is ready to use. To recreate directory structure:")
        click.echo("  mx init --force  (safe - preserves existing entries)")
        sys.exit(0)

    # Create KB root
    if not resolved_kb_root.exists():
        resolved_kb_root.mkdir(parents=True, exist_ok=True)
        created_dirs.append(str(resolved_kb_root))

    # Create default category directories
    for category in DEFAULT_KB_CATEGORIES:
        cat_path = resolved_kb_root / category
        if not cat_path.exists():
            cat_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(f"  {category}/")

    # Create index root
    if not resolved_index_root.exists():
        resolved_index_root.mkdir(parents=True, exist_ok=True)
        created_dirs.append(str(resolved_index_root))

    # Build environment variable guidance
    if not kb_from_env:
        env_exports.append(f'export MEMEX_KB_ROOT="{resolved_kb_root}"')
    if not index_from_env:
        env_exports.append(f'export MEMEX_INDEX_ROOT="{resolved_index_root}"')

    # Output what was created
    lines.append("Memex KB Initialized")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"KB Root:    {resolved_kb_root}")
    lines.append(f"Index Root: {resolved_index_root}")
    lines.append("")

    if created_dirs:
        lines.append("Created:")
        for d in created_dirs:
            lines.append(f"  {d}")
        lines.append("")

    # Environment variable setup
    if env_exports:
        lines.append("Add to your shell profile (~/.bashrc, ~/.zshrc):")
        lines.append("-" * 40)
        for exp in env_exports:
            lines.append(exp)
        lines.append("")

    click.echo("\n".join(lines))

    # Handle .kbcontext creation
    if not no_context:
        context_path = Path.cwd() / CONTEXT_FILENAME

        if context_path.exists():
            click.echo(f"Note: {CONTEXT_FILENAME} already exists in current directory.")
        else:
            # Auto-detect project
            detected = detect_project_context()
            project_name = detected.project_name if detected else Path.cwd().name

            # Check if we're in a project directory (any common project indicator)
            cwd = Path.cwd()
            is_project = any((cwd / indicator).exists() for indicator in PROJECT_INDICATORS)

            if is_project:
                click.echo(f"Detected project: {project_name}")
                if click.confirm("Create .kbcontext for this project?", default=True):
                    directory = f"projects/{project_name}"
                    content = create_default_context(project_name, directory)
                    context_path.write_text(content, encoding="utf-8")
                    click.echo(f"Created {CONTEXT_FILENAME}")
                    click.echo(f"  Primary directory: {directory}")
                    click.echo(f"  Default tags: {project_name}")

    # Next steps
    click.echo("")
    click.echo("Next steps:")
    click.echo("-" * 40)
    if env_exports:
        click.echo("  1. Add environment variables to your shell profile")
        click.echo("  2. Restart your shell or run: source ~/.bashrc")
        click.echo("  3. Run: mx add --title=\"First Entry\" --tags=\"...\"")
    else:
        click.echo("  mx add --title=\"First Entry\" --tags=\"...\"")
        click.echo("  mx search \"query\"")
        click.echo("  mx --help")


# ─────────────────────────────────────────────────────────────────────────────
# Search Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("query")
@click.option("--tags", "-t", help="Filter by tags (comma-separated)")
@click.option(
    "--mode",
    type=click.Choice(["hybrid", "keyword", "semantic"]),
    default="hybrid",
    help="Search mode: hybrid (keyword+semantic, default), keyword (fast, exact), semantic (meaning-based, requires [semantic] extras)",
)
@click.option("--limit", "-n", default=10, help="Max results")
@click.option("--content", "-c", is_flag=True, help="Include full content in results")
@click.option("--no-history", is_flag=True, help="Don't record this search in history")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--strict", is_flag=True, help="Return empty results instead of semantic fallbacks")
@click.option("--terse", is_flag=True, help="Output paths only (one per line)")
@click.option("--compact", is_flag=True, help="Output minimal JSON (short keys: p, t, s)")
@click.option("--no-session", is_flag=True, help="Ignore session context for this search")
@click.pass_context
def search(
    ctx: click.Context,
    query: str, tags: str | None, mode: str, limit: int,
    content: bool, no_history: bool, as_json: bool, strict: bool,
    terse: bool, compact: bool, no_session: bool,
):
    """Search the knowledge base.

    Session context (from 'mx session start') is applied automatically:
    - Session tags are merged with --tags (union)
    - Session project boosts matching entries

    Use --no-session to ignore session context for a single search.

    \b
    Examples:
      mx search "deployment"
      mx search "docker" --tags=infrastructure
      mx search "api" --mode=semantic --limit=5
      mx search "config" --strict          # No semantic fallbacks
      mx search "deploy" --terse           # Paths only
      mx search "api" --compact            # Minimal JSON
      mx search "api" --no-session         # Ignore session context
    """
    from .core import search as core_search
    from .errors import MemexError
    from .indexer.chroma_index import semantic_deps_available

    # Fail early if semantic search requested but deps not installed
    if mode == "semantic" and not semantic_deps_available():
        _handle_error(ctx, MemexError.semantic_search_unavailable())

    # Parse CLI tags
    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    # Apply session context (unless --no-session)
    session_project = None
    if not no_session:
        from .session import get_session

        session_ctx = get_session()
        if not session_ctx.is_empty():
            # Merge session tags with CLI tags (union)
            tag_list = session_ctx.merge_tags(tag_list)
            session_project = session_ctx.project

    result = run_async(core_search(
        query=query,
        limit=limit,
        mode=mode,
        tags=tag_list,
        include_content=content,
        strict=strict,
        session_project=session_project,
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

    # Show warnings (e.g., semantic fallback warning)
    for warning in result.warnings:
        click.echo(f"Warning: {warning}", err=True)

    # Output format selection (mutually exclusive, priority: terse > compact > json > table)
    if terse:
        # Terse mode: just paths, one per line
        for r in result.results:
            click.echo(r.path)
    elif compact:
        # Compact JSON: minimal keys (p=path, t=title, s=score)
        output(
            [{"p": r.path, "t": r.title, "s": round(r.score, 2)} for r in result.results],
            as_json=True,
        )
    elif as_json:
        # Full JSON with match_type
        output(
            [{"path": r.path, "title": r.title, "score": r.score, "snippet": r.snippet,
              "match_type": r.match_type}
             for r in result.results],
            as_json=True,
        )
    else:
        if not result.results:
            click.echo("No results found.")
            return

        rows = [
            {"path": r.path, "title": r.title, "score": f"{r.score:.2f}",
             "match": r.match_type or ""}
            for r in result.results
        ]
        click.echo(format_table(rows, ["path", "title", "score", "match"], {"path": 40, "title": 30}))


# ─────────────────────────────────────────────────────────────────────────────
# Get Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("path")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON with metadata")
@click.option("--metadata", "-m", is_flag=True, help="Show only metadata")
@click.pass_context
def get(ctx: click.Context, path: str, as_json: bool, metadata: bool):
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
        _handle_error(ctx, e)

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
@click.option("--template", "-T", "template_name", help="Use a template (see 'mx templates list')")
@click.option("--force", is_flag=True, help="Create even if duplicates detected")
@click.option("--dry-run", is_flag=True, help="Preview path/frontmatter/content without creating")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def add(
    ctx: click.Context,
    title: str,
    tags: str,
    category: str,
    content: str | None,
    file_path: str | None,
    stdin: bool,
    template_name: str | None,
    force: bool,
    dry_run: bool,
    as_json: bool,
):
    """Create a new knowledge base entry.

    \b
    Examples:
      mx add --title="My Entry" --tags="foo,bar" --content="# Content here"
      mx add --title="My Entry" --tags="foo,bar" --file=content.md
      mx add --title="Fix login bug" --tags="auth" --template=troubleshooting
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
    from .errors import ErrorCode, MemexError
    from .templates import apply_template, get_template

    # Handle template if specified
    template = None
    if template_name:
        template = get_template(template_name)
        if not template:
            from .templates import list_templates
            available = ", ".join(t.name for t in list_templates())
            _handle_error(
                ctx,
                MemexError(
                    ErrorCode.INVALID_CONTENT,
                    f"Unknown template: {template_name}",
                    {"suggestion": f"Available templates: {available}"},
                ),
            )

    # Resolve content source
    if stdin:
        content = sys.stdin.read()
    elif file_path:
        content = Path(file_path).read_text()
    elif template:
        # Use template content
        content = apply_template(template, title)
    elif not content:
        _handle_error(
            ctx,
            MemexError(
                ErrorCode.MISSING_REQUIRED_FIELD,
                "Must provide --content, --file, --template, or --stdin",
                {"suggestion": "Use --content='...' or --file=path or --template=name or --stdin"},
            ),
        )

    # Build tag list, including template suggested tags
    tag_list = [t.strip() for t in tags.split(",")]
    if template and template.suggested_tags:
        # Add template tags that aren't already present
        for tag in template.suggested_tags:
            if tag not in tag_list:
                tag_list.append(tag)

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
            _handle_add_error(ctx, e, tag_list)

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
        _handle_add_error(ctx, e, tag_list)

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
                    _handle_add_error(ctx, e, tag_list)
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
# Templates Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("action", default="list", type=click.Choice(["list", "show"]))
@click.argument("name", required=False)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def templates(action: str, name: str | None, as_json: bool):
    """List or show entry templates.

    \b
    Examples:
      mx templates              # List all available templates
      mx templates list         # Same as above
      mx templates show pattern # Show the 'pattern' template content

    \b
    Templates provide scaffolding for structured entries. Use with:
      mx add --title="..." --tags="..." --template=<name>

    \b
    Template sources (in priority order):
      1. Project: .kbcontext templates: section
      2. User: ~/.config/memex/templates/*.yaml
      3. Built-in: troubleshooting, project, pattern, decision, runbook, api, meeting
    """
    from .templates import get_template, list_templates

    if action == "show":
        if not name:
            click.echo("Usage: mx templates show <name>", err=True)
            sys.exit(1)

        template = get_template(name)
        if not template:
            available = ", ".join(t.name for t in list_templates())
            click.echo(f"Unknown template: {name}", err=True)
            click.echo(f"Available: {available}", err=True)
            sys.exit(1)

        if as_json:
            output({
                "name": template.name,
                "description": template.description,
                "content": template.content,
                "suggested_tags": template.suggested_tags,
                "source": template.source,
            }, as_json=True)
            return

        click.echo(f"Template: {template.name}")
        click.echo(f"Source: {template.source}")
        click.echo(f"Description: {template.description}")
        if template.suggested_tags:
            click.echo(f"Suggested tags: {', '.join(template.suggested_tags)}")
        click.echo()
        click.echo("Content:")
        click.echo("─" * 40)
        click.echo(template.content if template.content else "(empty)")
        return

    # List templates
    all_templates = list_templates()

    if as_json:
        output([{
            "name": t.name,
            "description": t.description,
            "source": t.source,
            "suggested_tags": t.suggested_tags,
        } for t in all_templates], as_json=True)
        return

    click.echo("Available templates:\n")
    for t in all_templates:
        source_badge = f"[{t.source}]" if t.source != "builtin" else ""
        click.echo(f"  {t.name:16} {t.description} {source_badge}")

    click.echo()
    click.echo("Use: mx add --title='...' --tags='...' --template=<name>")
    click.echo("Show: mx templates show <name>")


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
@click.pass_context
def update(
    ctx: click.Context,
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
    from .errors import ErrorCode, MemexError

    # Validate flag combinations
    if timestamp and not append:
        _handle_error(ctx, MemexError(
            ErrorCode.INVALID_CONTENT,
            "--timestamp requires --append",
            {"suggestion": "Use --append with --timestamp"},
        ))

    if stdin and file_path:
        _handle_error(ctx, MemexError(
            ErrorCode.INVALID_CONTENT,
            "--stdin and --file are mutually exclusive",
        ))

    if stdin and content:
        _handle_error(ctx, MemexError(
            ErrorCode.INVALID_CONTENT,
            "--stdin and --content are mutually exclusive",
        ))

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
        _handle_error(ctx, e)

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
            action_verb = "Replaced" if result.action == "replaced" else "Appended to"
            click.echo(f"{action_verb}: {result.path}{match_info}")


# ─────────────────────────────────────────────────────────────────────────────
# Batch Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--file",
    "-f",
    "file_path",
    type=click.Path(exists=True),
    help="Read commands from file instead of stdin",
)
@click.option(
    "--continue-on-error/--stop-on-error",
    default=True,
    help="Continue processing after errors (default: continue)",
)
@click.pass_context
def batch(ctx: click.Context, file_path: str | None, continue_on_error: bool):
    """Execute multiple KB operations in a single invocation.

    Reads commands from stdin (or --file) and executes them sequentially.
    Output is always JSON with per-operation results.

    \b
    Supported commands:
      add --title='...' --tags='...' [--category=...] [--content='...'] [--force]
      update <path> [--tags='...'] [--content='...'] [--append]
      upsert <title> [--content='...'] [--tags='...'] [--directory=...]
      search <query> [--tags='...'] [--mode=...] [--limit=N]
      get <path> [--metadata]
      delete <path> [--force]

    \b
    Example:
      mx batch << 'EOF'
      add --title='Note 1' --tags='tag1' --category=tooling --content='Content'
      search 'api'
      EOF

    \b
    Output format:
      {
        "total": 2,
        "succeeded": 2,
        "failed": 0,
        "results": [
          {"index": 0, "command": "add ...", "success": true, "result": {...}},
          {"index": 1, "command": "search ...", "success": true, "result": {...}}
        ]
      }

    Exit code is 1 if any operation fails, 0 if all succeed.
    """
    from .batch import run_batch

    # Read input
    if file_path:
        lines = Path(file_path).read_text(encoding="utf-8").strip().split("\n")
    else:
        lines = sys.stdin.read().strip().split("\n")

    # Filter empty lines and comments
    commands = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]

    if not commands:
        click.echo(
            json.dumps({"error": "No commands provided", "code": 1501}), err=True
        )
        sys.exit(1)

    result = run_async(run_batch(commands, continue_on_error=continue_on_error))
    click.echo(result.model_dump_json(indent=2))

    # Exit with error if any commands failed
    if result.failed > 0:
        sys.exit(1)


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
    link_list = [link.strip() for link in links.split(",")] if links else None

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

    Checks for orphaned entries, broken links, stale content, empty directories,
    and entries missing descriptions.

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

        # Parse errors (most critical - entries not indexed)
        parse_errors = result.get("parse_errors", [])
        if parse_errors:
            click.echo(f"\n✗ Parse errors ({len(parse_errors)}):")
            for pe in parse_errors[:10]:
                click.echo(f"  - {pe['path']}")
                # Show truncated error message
                error_msg = pe.get("error", "Unknown error")
                if len(error_msg) > 80:
                    error_msg = error_msg[:77] + "..."
                click.echo(f"    {error_msg}")
            if len(parse_errors) > 10:
                click.echo(f"  ... and {len(parse_errors) - 10} more")
        else:
            click.echo("\n✓ No parse errors")

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
                if "suggestion" in bl:
                    click.echo(f"    → Did you mean: [[{bl['suggestion']}]]?")
            if len(broken_links) > 10:
                click.echo(f"  ... and {len(broken_links) - 10} more")
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
            if len(empty_dirs) > 10:
                click.echo(f"  ... and {len(empty_dirs) - 10} more")
        else:
            click.echo("\n✓ No empty directories")

        # Missing descriptions
        missing_descs = result.get("missing_descriptions", [])
        if missing_descs:
            click.echo(f"\nℹ Missing descriptions ({len(missing_descs)}):")
            for m in missing_descs[:10]:
                click.echo(f"  - {m['path']}")
            if len(missing_descs) > 10:
                click.echo(f"  ... and {len(missing_descs) - 10} more")
        else:
            click.echo("\n✓ All entries have descriptions")

        # Show fix guidance if there are issues
        total_issues = summary.get("total_issues", 0)
        if total_issues > 0:
            click.echo("\n" + "-" * 40)
            click.echo("Fix Guidance:")
            if parse_errors:
                click.echo("  • Parse errors: Fix frontmatter syntax in listed files")
            if broken_links:
                click.echo("  • Broken links: Update [[link]] targets or remove dead links")
            if orphans:
                click.echo("  • Orphans: Add [[links]] to these entries from other files")
            if stale:
                click.echo("  • Stale: Review and update entries, or mark them as archived")
            if empty_dirs:
                click.echo("  • Empty dirs: Add entries or remove with 'rmdir'")
            if missing_descs:
                click.echo("  • Missing descriptions: Add 'description:' to frontmatter or run 'mx summarize'")


# ─────────────────────────────────────────────────────────────────────────────
# Summarize Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--dry-run", is_flag=True, help="Preview changes without writing")
@click.option("--limit", type=int, help="Maximum entries to process")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def summarize(dry_run: bool, limit: int | None, as_json: bool):
    """Generate descriptions for entries missing them.

    Extracts a one-line summary from entry content to use as the description
    field in frontmatter. This improves search results and entry discoverability.

    \b
    Examples:
      mx summarize --dry-run         # Preview what would be generated
      mx summarize                   # Generate and write descriptions
      mx summarize --limit 5         # Process only 5 entries
      mx summarize --json            # Output as JSON
    """
    from .core import generate_descriptions

    results = run_async(generate_descriptions(dry_run=dry_run, limit=limit))

    if as_json:
        output(results, as_json=True)
    else:
        if not results:
            click.echo("All entries already have descriptions.")
            return

        updated = [r for r in results if r["status"] == "updated"]
        previewed = [r for r in results if r["status"] == "preview"]
        skipped = [r for r in results if r["status"] == "skipped"]
        errors = [r for r in results if r["status"] == "error"]

        if dry_run:
            click.echo("Preview of descriptions to generate:")
            click.echo("=" * 50)
            for r in previewed:
                click.echo(f"\n{r['path']}")
                click.echo(f"  Title: {r['title']}")
                click.echo(f"  Description: {r['description']}")
            click.echo(f"\n{len(previewed)} entries would be updated.")
        else:
            if updated:
                click.echo(f"✓ Generated descriptions for {len(updated)} entries:")
                for r in updated[:10]:
                    click.echo(f"  - {r['path']}")
                if len(updated) > 10:
                    click.echo(f"  ... and {len(updated) - 10} more")

        if skipped:
            click.echo(f"\n⚠ Skipped {len(skipped)} entries (no content to summarize)")

        if errors:
            click.echo(f"\n✗ {len(errors)} errors:")
            for r in errors[:5]:
                click.echo(f"  - {r['path']}: {r.get('reason', 'Unknown error')}")


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
    # Exclude hidden (.) and special (_) files, consistent with tree command
    entry_count = (
        sum(1 for p in kb_root.rglob("*.md") if not p.name.startswith((".", "_")))
        if kb_root.exists()
        else 0
    )

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
# Session Command Group
# ─────────────────────────────────────────────────────────────────────────────


@cli.group(invoke_without_command=True)
@click.pass_context
def session(ctx):
    """Manage session search context.

    Session context persists until explicitly cleared:
    - tags: Filter all searches to entries with these tags
    - project: Boost entries from this project in results

    Unlike .kbcontext (per-directory), session is global and explicit.

    \b
    Examples:
      mx session                    # Show current session
      mx session start --tags=infra # Start filtering by tags
      mx session set --project=api  # Set project boost
      mx session clear              # Clear session context
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(session_show)


@session.command("show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def session_show(as_json: bool):
    """Show the current session context.

    \b
    Examples:
      mx session show
      mx session show --json
    """
    from .session import get_session

    session_ctx = get_session()

    if as_json:
        output(
            {
                "active": not session_ctx.is_empty(),
                "tags": session_ctx.tags,
                "project": session_ctx.project,
            },
            as_json=True,
        )
    else:
        if session_ctx.is_empty():
            click.echo("No active session context.")
            click.echo("Use 'mx session start' to set filters.")
        else:
            click.echo("Session context:")
            if session_ctx.tags:
                click.echo(f"  Tags:    {', '.join(session_ctx.tags)}")
            if session_ctx.project:
                click.echo(f"  Project: {session_ctx.project}")


@session.command("start")
@click.option("--tags", "-t", help="Tags to filter by (comma-separated)")
@click.option("--project", "-p", help="Project to boost in results")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def session_start(tags: str | None, project: str | None, as_json: bool):
    """Start a new session with the given context.

    Replaces any existing session context.

    \b
    Examples:
      mx session start --tags=infra,docker
      mx session start --project=api-service
      mx session start --tags=python --project=memex
    """
    from .session import SessionContext, save_session

    if not tags and not project:
        click.echo("Error: At least one of --tags or --project required", err=True)
        sys.exit(1)

    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    session_ctx = SessionContext(tags=tag_list, project=project)
    save_session(session_ctx)

    if as_json:
        output(
            {
                "action": "started",
                "tags": session_ctx.tags,
                "project": session_ctx.project,
            },
            as_json=True,
        )
    else:
        click.echo("Session started:")
        if session_ctx.tags:
            click.echo(f"  Tags:    {', '.join(session_ctx.tags)}")
        if session_ctx.project:
            click.echo(f"  Project: {session_ctx.project}")


@session.command("set")
@click.option(
    "--tags", "-t", help="Tags to filter by (comma-separated, replaces existing)"
)
@click.option("--add-tags", help="Tags to add (comma-separated)")
@click.option("--project", "-p", help="Project to boost")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def session_set(
    tags: str | None, add_tags: str | None, project: str | None, as_json: bool
):
    """Update the current session context.

    Use --tags to replace all tags, --add-tags to append.
    Use --project to set/change project.

    \b
    Examples:
      mx session set --tags=docker          # Replace tags
      mx session set --add-tags=kubernetes  # Add tag
      mx session set --project=new-project  # Change project
    """
    from .session import get_session, save_session

    session_ctx = get_session()

    if tags is not None:
        session_ctx.tags = [t.strip() for t in tags.split(",") if t.strip()]
    if add_tags:
        new_tags = [t.strip() for t in add_tags.split(",") if t.strip()]
        session_ctx.tags = list(set(session_ctx.tags) | set(new_tags))
    if project is not None:
        session_ctx.project = project if project else None

    save_session(session_ctx)

    if as_json:
        output(
            {
                "action": "updated",
                "tags": session_ctx.tags,
                "project": session_ctx.project,
            },
            as_json=True,
        )
    else:
        click.echo("Session updated:")
        if session_ctx.tags:
            click.echo(f"  Tags:    {', '.join(session_ctx.tags)}")
        if session_ctx.project:
            click.echo(f"  Project: {session_ctx.project}")
        if session_ctx.is_empty():
            click.echo("  (no active context)")


@session.command("clear")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def session_clear(as_json: bool):
    """Clear the session context.

    \b
    Examples:
      mx session clear
    """
    from .session import clear_session

    cleared = clear_session()

    if as_json:
        output({"action": "cleared", "had_session": cleared}, as_json=True)
    else:
        if cleared:
            click.echo("Session context cleared.")
        else:
            click.echo("No active session to clear.")


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
# Publishing
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--output", "-o", "output_dir",
    type=click.Path(),
    default="_site",
    help="Output directory (default: _site)",
)
@click.option(
    "--base-url", "-b",
    default="",
    help="Base URL for links (e.g., /my-kb for subdirectory hosting)",
)
@click.option(
    "--include-drafts",
    is_flag=True,
    help="Include draft entries in output",
)
@click.option(
    "--include-archived",
    is_flag=True,
    help="Include archived entries in output",
)
@click.option(
    "--no-clean",
    is_flag=True,
    help="Don't remove output directory before build",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def publish(
    output_dir: str,
    base_url: str,
    include_drafts: bool,
    include_archived: bool,
    no_clean: bool,
    as_json: bool,
):
    """Generate static HTML site for GitHub Pages.

    Converts the knowledge base to a static site with:
    - Resolved [[wikilinks]] as HTML links
    - Client-side search (Lunr.js)
    - Tag pages and index
    - Minimal responsive theme with dark mode

    \b
    Examples:
      mx publish                           # Build to _site/
      mx publish -o docs                   # Build to docs/ for GitHub Pages
      mx publish --base-url /my-kb         # Set base URL for subdirectory hosting
      mx publish --include-drafts          # Include draft entries
    """
    from .core import publish as core_publish

    try:
        result = run_async(core_publish(
            output_dir=output_dir,
            base_url=base_url,
            include_drafts=include_drafts,
            include_archived=include_archived,
            clean=not no_clean,
        ))
    except Exception as e:
        click.echo(f"Error: {_normalize_error_message(str(e))}", err=True)
        sys.exit(1)

    if as_json:
        output(result, as_json=True)
    else:
        click.echo(f"Published {result['entries_published']} entries to {result['output_dir']}")

        broken_links = result.get("broken_links", [])
        if broken_links:
            click.echo(f"\n⚠ Broken links ({len(broken_links)}):")
            for bl in broken_links[:10]:
                click.echo(f"  - {bl['source']} -> {bl['target']}")
            if len(broken_links) > 10:
                click.echo(f"  ... and {len(broken_links) - 10} more")

        click.echo(f"\nSearch index: {result['search_index_path']}")
        click.echo("\nTo preview locally:")
        click.echo(f"  cd {result['output_dir']} && python -m http.server")


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
