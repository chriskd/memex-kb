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


def _handle_error(
    ctx: click.Context,
    error: Exception,
    fallback_message: str | None = None,
    exit_code: int = 1,
) -> None:
    """Handle an error with optional JSON output.

    If --json-errors is enabled, outputs structured JSON error.
    Otherwise, outputs human-readable error message.

    Args:
        ctx: Click context (must have obj["json_errors"] set).
        error: The exception that occurred.
        fallback_message: Optional message to use for non-MemexError exceptions.
        exit_code: Exit code to use (default 1). Some commands use specific codes.
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

    sys.exit(exit_code)


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
# Status Output (default when no subcommand)
# ─────────────────────────────────────────────────────────────────────────────


def _show_status() -> None:
    """Show KB status with context, recent entries, and suggested commands.

    Displayed when running `mx` with no arguments. Provides quick orientation
    for agents and humans about the current KB state.
    """
    from .config import ConfigurationError, get_kb_root
    from .context import get_kb_context

    # Track what we successfully loaded
    kb_root = None
    context = None
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
        # Fallback to current directory name as project
        project_name = Path.cwd().name

    # Get recent entries if KB is available
    if kb_root and kb_root.exists():
        entries = _get_recent_entries_for_status(kb_root, project_name, limit=5)

    # Build output
    _output_status(kb_root, context, entries, project_name)


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
    entries: list[dict],
    project_name: str | None,
) -> None:
    """Output the status display.

    Args:
        kb_root: KB root path (None if not configured).
        context: Loaded KBContext (None if no .kbcontext).
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
        elif project_name:
            lines.append(f"Project: {project_name} (auto-detected)")
            lines.append("         Run 'mx init' to configure")
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


@click.group(cls=JsonErrorGroup, invoke_without_command=True)
@click.version_option(version="0.1.0", prog_name="mx")
@click.option("--json-errors", "json_errors", is_flag=True,
              help="Output errors as JSON (for programmatic use)")
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
      mx tree                    # Browse structure
      mx health                  # Check KB health

    \b
    Create content:
      mx add --title="Title" --tags="a,b" --content="..."
      mx append "Existing Title" --content="append this"

    \b
    Modify content:
      mx patch path.md --find "old text" --replace "new text"
      mx replace path.md --tags="new,tags"

    \b
    For programmatic error handling:
      mx --json-errors add ...   # Errors output as JSON with error codes

    \b
    For quieter output:
      mx --quiet search ...      # Suppress warnings
      MEMEX_QUIET=1 mx search    # Or use environment variable
    """
    from ._logging import set_quiet_mode

    # Store options in context for subcommands to access
    ctx.ensure_object(dict)
    ctx.obj["json_errors"] = json_errors
    ctx.obj["quiet"] = quiet

    if quiet:
        set_quiet_mode(True)

    # Show status when no subcommand is provided
    if ctx.invoked_subcommand is None:
        _show_status()


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
# Init Command - KB Setup (project or user scope)
# ─────────────────────────────────────────────────────────────────────────────

# Default local KB directory name
LOCAL_KB_DIR = "kb"


@cli.command()
@click.option("--path", "-p", type=click.Path(), help="Custom location for KB (default: kb/)")
@click.option("--user", "-u", is_flag=True, help="Create user-scope KB at ~/.memex/kb/")
@click.option("--force", "-f", is_flag=True, help="Reinitialize existing KB")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def init(path: Optional[str], user: bool, force: bool, as_json: bool):
    """Initialize a knowledge base.

    By default, creates a project-scope KB at kb/ in the current directory.
    Use --user to create a user-scope KB at ~/.memex/kb/ for personal knowledge.

    \b
    Scopes:
      project (default)  kb/ in repo - shared with collaborators via git
      user               ~/.memex/kb/ - personal, available everywhere

    \b
    Examples:
      mx init                    # Project scope: creates kb/
      mx init --user             # User scope: creates ~/.memex/kb/
      mx init --path docs/kb     # Custom project location
      mx init --force            # Reinitialize existing
    """
    from .context import LOCAL_KB_CONFIG_FILENAME, USER_KB_DIR

    # Validate mutually exclusive options
    if user and path:
        if as_json:
            output({"error": "--user and --path are mutually exclusive"}, as_json=True)
        else:
            click.echo("Error: --user and --path are mutually exclusive", err=True)
        sys.exit(1)

    # Determine target directory based on scope
    if user:
        kb_path = USER_KB_DIR
        scope = "user"
    else:
        kb_path = Path(path) if path else Path.cwd() / LOCAL_KB_DIR
        scope = "project"

    # Check if already exists
    if kb_path.exists():
        if not force:
            scope_label = "User" if user else "Project"
            if as_json:
                output({
                    "error": f"{scope_label} KB already exists at {kb_path}",
                    "hint": "Use --force to reinitialize"
                }, as_json=True)
            else:
                click.echo(f"Error: {scope_label} KB already exists at {kb_path}", err=True)
                click.echo("Use --force to reinitialize.", err=True)
            sys.exit(1)

    # Create directory structure
    kb_path.mkdir(parents=True, exist_ok=True)

    # Create README with scope-appropriate content
    readme_path = kb_path / "README.md"
    if user:
        readme_content = """# User Knowledge Base

This directory contains your personal knowledge base entries managed by `mx`.
This KB is available everywhere and is not shared with collaborators.

## Usage

```bash
mx add --title="Entry" --tags="tag1,tag2" --content="..."
mx search "query"
mx list
```

## Structure

Entries are Markdown files with YAML frontmatter:

```markdown
---
title: Entry Title
tags: [tag1, tag2]
created: 2024-01-15
---

# Entry Title

Your content here.
```

## Scope

User KB entries are personal and available in all projects.
They are stored at ~/.memex/kb/ and are not committed to git.
"""
    else:
        readme_content = """# Project Knowledge Base

This directory contains project-specific knowledge base entries managed by `mx`.
Commit this directory to share knowledge with collaborators.

## Usage

```bash
mx add --title="Entry" --tags="tag1,tag2" --content="..." --local
mx search "query"   # Searches local KB first
mx list --local     # List only local entries
```

## Structure

Entries are Markdown files with YAML frontmatter:

```markdown
---
title: Entry Title
tags: [tag1, tag2]
created: 2024-01-15
---

# Entry Title

Your content here.
```

## Integration

Project KB entries take precedence over global KB entries in search results.
This keeps project-specific knowledge close to the code.
"""
    readme_path.write_text(readme_content, encoding="utf-8")

    # Create config file with scope-appropriate defaults
    if user:
        # User scope: config lives inside the KB directory
        config_path = kb_path / LOCAL_KB_CONFIG_FILENAME
        config_content = """# User KB Configuration
# This file marks this directory as your personal memex knowledge base

# Optional: default tags for entries created here
# default_tags:
#   - personal

# Optional: exclude patterns (glob)
# exclude:
#   - "*.draft.md"
"""
    else:
        # Project scope: config lives at project root with kb_path reference
        config_path = Path.cwd() / ".kbconfig"
        # Calculate relative path from project root to kb directory
        try:
            relative_kb_path = kb_path.relative_to(Path.cwd())
        except ValueError:
            # If kb_path is not under cwd, use absolute path
            relative_kb_path = kb_path
        config_content = f"""# Project KB Configuration
# This file configures the project knowledge base

# Path to the KB directory (required for project-scope KBs)
kb_path: ./{relative_kb_path}

# Optional: default tags for entries created here
# default_tags:
#   - {Path.cwd().name}

# Optional: default write directory for new entries
# primary: guides

# Optional: boost these paths in search (glob patterns)
# boost_paths:
#   - guides/*
#   - reference/*

# Optional: exclude patterns from indexing (glob)
# exclude:
#   - "*.draft.md"
"""
    config_path.write_text(config_content, encoding="utf-8")

    # Output
    scope_label = "user" if user else "project"
    if user:
        files_created = [f"kb/README.md", f"kb/{LOCAL_KB_CONFIG_FILENAME}"]
    else:
        files_created = [f"{kb_path.name}/README.md", ".kbconfig"]

    if as_json:
        output({
            "created": str(kb_path),
            "config": str(config_path),
            "scope": scope_label,
            "files": files_created,
            "hint": f"Use 'mx add' to add entries to this KB"
        }, as_json=True)
    else:
        click.echo(f"✓ Initialized {scope_label} KB at {kb_path}")
        click.echo(f"  Config: {config_path}")
        click.echo()
        click.echo("Next steps:")
        if user:
            click.echo("  mx add --title=\"Entry\" --tags=\"...\" --content=\"...\"")
            click.echo("  mx search \"query\"")
        else:
            click.echo("  mx add --title=\"Entry\" --tags=\"...\" --content=\"...\" --local")
            click.echo("  mx search \"query\"   # Searches local KB first")


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

    \b
    See also:
      mx get  - Read a specific entry by path
      mx list - List entries with optional filters
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
@click.argument("path", required=False)
@click.option("--title", "-t", "by_title", help="Get entry by title instead of path")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON with metadata")
@click.option("--metadata", "-m", is_flag=True, help="Show only metadata")
def get(path: Optional[str], by_title: Optional[str], as_json: bool, metadata: bool):
    """Read a knowledge base entry.

    \b
    Examples:
      mx get tooling/beads-issue-tracker.md
      mx get tooling/beads-issue-tracker.md --json
      mx get tooling/beads-issue-tracker.md --metadata
      mx get --title="Docker Guide"
      mx get -t "Python Tooling"

    \b
    See also:
      mx search - Search entries by query
      mx list   - List entries with optional filters
    """
    from .core import find_entries_by_title, get_entry, get_similar_titles

    # Validate that exactly one of path or --title is provided
    if path and by_title:
        click.echo("Error: Cannot specify both PATH and --title", err=True)
        sys.exit(1)
    if not path and not by_title:
        click.echo("Error: Must specify either PATH or --title", err=True)
        sys.exit(1)

    # If --title is used, find the entry by title
    if by_title:
        matches = run_async(find_entries_by_title(by_title))

        if len(matches) == 0:
            # No exact match - show suggestions
            suggestions = run_async(get_similar_titles(by_title))
            click.echo(f"Error: No entry found with title '{by_title}'", err=True)
            if suggestions:
                click.echo("\nDid you mean:", err=True)
                for suggestion in suggestions:
                    click.echo(f"  - {suggestion}", err=True)
            sys.exit(1)

        if len(matches) > 1:
            # Multiple matches - show candidates
            click.echo(f"Error: Multiple entries found with title '{by_title}':", err=True)
            for match in matches:
                click.echo(f"  - {match['path']}", err=True)
            click.echo("\nUse the full path to specify which entry.", err=True)
            sys.exit(1)

        # Single match - use its path
        path = matches[0]["path"]

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
@click.option("--scope", type=click.Choice(["project", "user"]), help="Target KB scope (default: auto-detect)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def add(
    title: str,
    tags: str,
    category: str,
    content: Optional[str],
    file_path: Optional[str],
    stdin: bool,
    scope: Optional[str],
    as_json: bool,
):
    """Create a new knowledge base entry.

    \b
    Examples:
      mx add --title="My Entry" --tags="foo,bar" --content="# Content here"
      mx add --title="My Entry" --tags="foo,bar" --file=content.md
      cat content.md | mx add --title="My Entry" --tags="foo,bar" --stdin

    \b
    Scope Selection:
      --scope=project  Write to project KB (./kb/)
      --scope=user     Write to user KB (~/.memex/kb/)
      (default)        Auto-detect based on current directory

    \b
    When to use each scope:
      project: Team knowledge, infra docs, shared patterns
      user:    Personal notes, experiments, drafts

    \b
    See also:
      mx append - Append content to existing entry (or create new)
    """
    from .core import add_entry

    # Validate mutual exclusivity of content sources
    sources = sum([bool(content), bool(file_path), stdin])
    if sources > 1:
        click.echo("Error: Only one of --content, --file, or --stdin can be used", err=True)
        sys.exit(1)
    if sources == 0:
        click.echo("Error: Must provide --content, --file, or --stdin", err=True)
        sys.exit(1)

    # Resolve content source
    if stdin:
        content = sys.stdin.read()
    elif file_path:
        content = Path(file_path).read_text()

    tag_list = [t.strip() for t in tags.split(",")]

    try:
        result = run_async(add_entry(title=title, content=content, tags=tag_list, category=category, scope=scope))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        # Include scope in JSON output if explicitly set
        if scope:
            result['scope'] = scope
        output(result, as_json=True)
    else:
        # Show path with scope prefix if explicitly set
        path_display = f"@{scope}/{result['path']}" if scope else result['path']
        click.echo(f"Created: {path_display}")
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

    \b
    See also:
      mx patch  - Apply surgical find-replace edits to an entry
      mx replace - Replace entry content or tags entirely
      mx add    - Create a new entry (never appends)
    """
    from .core import append_entry

    # Validate mutual exclusivity of content sources
    sources = sum([bool(content), bool(file_path), stdin])
    if sources > 1:
        click.echo("Error: Only one of --content, --file, or --stdin can be used", err=True)
        sys.exit(1)
    if sources == 0:
        click.echo("Error: Must provide --content, --file, or --stdin", err=True)
        sys.exit(1)

    # Resolve content source
    if stdin:
        content = sys.stdin.read()
    elif file_path:
        content = Path(file_path).read_text()

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
# Replace Command (formerly 'update')
# ─────────────────────────────────────────────────────────────────────────────


@cli.command(name="replace")
@click.argument("path")
@click.option("--tags", help="New tags (comma-separated)")
@click.option("--content", help="New content (replaces existing)")
@click.option("--file", "-f", "file_path", type=click.Path(exists=True), help="Read content from file")
@click.option("--find", "find_flag", hidden=True, help="(Intent detection)")
@click.option("--replace", "replace_flag", hidden=True, help="(Intent detection)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def replace_cmd(
    path: str,
    tags: Optional[str],
    content: Optional[str],
    file_path: Optional[str],
    find_flag: Optional[str],
    replace_flag: Optional[str],
    as_json: bool,
):
    """Replace content or tags in a knowledge base entry.

    Overwrites entire content or tags. For surgical edits, use 'mx patch'.

    \b
    Examples:
      mx replace path/entry.md --tags="new,tags"
      mx replace path/entry.md --content="New content here"
      mx replace path/entry.md --file=updated-content.md

    \b
    See also:
      mx patch  - Apply surgical find-replace edits (keeps rest of content)
      mx append - Add content to end of entry (doesn't overwrite)
    """
    from .cli_intent import detect_update_intent_mismatch
    from .core import update_entry

    # Check for intent mismatch (wrong command based on flags)
    mismatch = detect_update_intent_mismatch(
        path=path,
        find_text=find_flag,
        replace_text=replace_flag,
    )
    if mismatch:
        click.echo(mismatch.format_error(), err=True)
        sys.exit(1)

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
        click.echo(f"Replaced: {result['path']}")


# Hidden alias for backwards compatibility
@cli.command(name="update", hidden=True)
@click.argument("path")
@click.option("--tags", help="New tags (comma-separated)")
@click.option("--content", help="New content")
@click.option("--file", "-f", "file_path", type=click.Path(exists=True), help="Read content from file")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def update_alias(ctx, path, tags, content, file_path, as_json):
    """(Deprecated: use 'mx replace' instead)"""
    ctx.invoke(replace_cmd, path=path, tags=tags, content=content, file_path=file_path, as_json=as_json)


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
@click.option("--project-only", is_flag=True, help="Only index project KB (exclude user KB)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def reindex(project_only: bool, as_json: bool):
    """Rebuild search indices from all markdown files.

    By default, indexes entries from both project and user KBs.
    Use --project-only to restrict to project KB only.

    \b
    Examples:
      mx reindex                # Index project + user KBs
      mx reindex --project-only # Index project KB only
      mx reindex --json
    """
    from .core import reindex as core_reindex

    if not as_json:
        scope_msg = "project KB" if project_only else "all KBs"
        click.echo(f"Reindexing {scope_msg}...")

    result = run_async(core_reindex(project_only=project_only))

    if as_json:
        output({
            "kb_files": result.kb_files,
            "whoosh_docs": result.whoosh_docs,
            "chroma_docs": result.chroma_docs,
            "project_only": project_only,
        }, as_json=True)
    else:
        click.echo(f"✓ Indexed {result.kb_files} entries, {result.whoosh_docs} keyword docs, {result.chroma_docs} semantic docs")


# ─────────────────────────────────────────────────────────────────────────────
# Context Command Group
# ─────────────────────────────────────────────────────────────────────────────


@cli.group(invoke_without_command=True)
@click.pass_context
def context(ctx):
    """Show or validate project KB context (.kbcontext file).

    The .kbcontext file configures KB behavior for a project:
    - primary: Default directory for new entries
    - paths: Boost these paths in search results
    - default_tags: Suggested tags for new entries

    \b
    Examples:
      mx context            # Show current context
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
            click.echo("Run 'mx init' to create a project KB.")
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
@click.option("--content", "content_flag", hidden=True, help="(Intent detection)")
@click.option("--append", "append_flag", hidden=True, help="(Intent detection)")
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
    content_flag: str | None,
    append_flag: str | None,
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

    \b
    See also:
      mx append  - Append content to existing entry (or create new)
      mx replace - Replace entry content or tags entirely
    """
    from .cli_intent import detect_patch_intent_mismatch
    from .core import patch_entry

    # Check for intent mismatch (wrong command based on flags)
    mismatch = detect_patch_intent_mismatch(
        path=path,
        find_text=find_text if not find_file else "provided",
        replace_text=replace_text,
        content=content_flag,
        append=append_flag,
    )
    if mismatch:
        click.echo(mismatch.format_error(), err=True)
        sys.exit(3)

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
# Quick-Add Helper Functions
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


# ─────────────────────────────────────────────────────────────────────────────
# Quick-Add Command
# ─────────────────────────────────────────────────────────────────────────────


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
    from .core import add_entry, get_valid_categories

    # Resolve content source
    if stdin:
        content = sys.stdin.read()
    elif file_path:
        content = Path(file_path).read_text()
    elif not content:
        click.echo("Error: Provide --content, --file, or --stdin", err=True)
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
        ))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    path = result.get('path') if isinstance(result, dict) else result.path
    click.echo(f"\nCreated: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Templates Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("action", default="list", type=click.Choice(["list", "show"]))
@click.argument("name", required=False)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def templates(action: str, name: Optional[str], as_json: bool):
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
        click.echo("-" * 40)
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
# Info Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def info(as_json: bool):
    """Show knowledge base configuration and stats.

    Shows all active KBs (project + user) with entry counts.

    \b
    Examples:
      mx info
      mx info --json
    """
    from .config import (
        ConfigurationError,
        get_index_root,
        get_kb_root,
        get_kb_roots,
        get_project_kb_root,
        get_user_kb_root,
    )
    from .core import get_valid_categories

    try:
        primary_kb = get_kb_root()
        index_root = get_index_root()
    except ConfigurationError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    # Get all active KBs
    project_kb = get_project_kb_root()
    user_kb = get_user_kb_root()
    all_kbs = get_kb_roots()

    # Count entries per KB
    def count_entries(kb_path):
        if not kb_path or not kb_path.exists():
            return 0
        return sum(1 for p in kb_path.rglob("*.md") if not p.name.startswith((".", "_")))

    project_count = count_entries(project_kb)
    user_count = count_entries(user_kb)
    total_count = project_count + user_count

    categories = get_valid_categories()

    # Build payload
    kbs_info = []
    if project_kb:
        kbs_info.append({"scope": "project", "path": str(project_kb), "entries": project_count})
    if user_kb:
        kbs_info.append({"scope": "user", "path": str(user_kb), "entries": user_count})

    payload = {
        "primary_kb": str(primary_kb),
        "index_root": str(index_root),
        "kbs": kbs_info,
        "total_entries": total_count,
        "categories": categories,
    }

    if as_json:
        output(payload, as_json=True)
        return

    click.echo("Memex Info")
    click.echo("=" * 40)
    click.echo(f"Primary KB: {primary_kb}")
    click.echo(f"Index Root: {index_root}")
    click.echo()
    click.echo("Active KBs:")
    if project_kb:
        click.echo(f"  project:  {project_kb} ({project_count} entries)")
    else:
        click.echo("  project:  (none)")
    if user_kb:
        click.echo(f"  user:     {user_kb} ({user_count} entries)")
    else:
        click.echo("  user:     (none)")
    click.echo()
    click.echo(f"Total:      {total_count} entries")
    if categories:
        click.echo(f"Categories: {', '.join(categories)}")


@cli.command("config")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def config_alias(ctx, as_json: bool):
    """Alias for mx info."""
    ctx.invoke(info, as_json=as_json)


# ─────────────────────────────────────────────────────────────────────────────
# History Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--limit", "-n", default=10, help="Max entries to show")
@click.option("--rerun", "-r", type=int, help="Re-execute search at position N (1=most recent)")
@click.option("--clear", is_flag=True, help="Clear all search history")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def history(limit: int, rerun: Optional[int], clear: bool, as_json: bool):
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
# Batch Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--file", "-f", "file_path",
    type=click.Path(exists=True),
    help="Read commands from file instead of stdin",
)
@click.option(
    "--continue-on-error/--stop-on-error",
    default=True,
    help="Continue processing after errors (default: continue)",
)
def batch(file_path: Optional[str], continue_on_error: bool):
    """Execute multiple KB operations in a single invocation.

    Reads commands from stdin (or --file) and executes them sequentially.
    Output is always JSON with per-operation results.

    \b
    Supported commands:
      add --title='...' --tags='...' [--category=...] [--content='...']
      update <path> [--tags='...'] [--content='...'] [--append]
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
    try:
        from .batch import run_batch
    except ImportError:
        click.echo(json.dumps({
            "error": "Batch module not available",
            "hint": "The batch module needs to be restored from v0.1.0"
        }), err=True)
        sys.exit(1)

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
            json.dumps({"error": "No commands provided"}), err=True
        )
        sys.exit(1)

    result = run_async(run_batch(commands, continue_on_error=continue_on_error))
    click.echo(result.model_dump_json(indent=2))

    # Exit with error if any commands failed
    if result.failed > 0:
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Session-Log Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command("session-log")
@click.option("--message", "-m", help="Session summary message")
@click.option(
    "--file", "-f", "file_path",
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
    message: Optional[str],
    file_path: Optional[str],
    stdin: bool,
    entry: Optional[str],
    tags: Optional[str],
    links: Optional[str],
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
    try:
        from .core import log_session as core_log_session
    except ImportError:
        click.echo("Error: log_session function not available in core", err=True)
        click.echo("This function needs to be restored from v0.1.0", err=True)
        sys.exit(1)

    # Validate message source (mutually exclusive)
    content_sources = sum([bool(message), bool(file_path), stdin])
    if content_sources == 0:
        click.echo("Error: Provide --message, --file, or --stdin", err=True)
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
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        output(result.model_dump(), as_json=True)
    else:
        click.echo(f"Logged to: {result.path}")
        if result.project:
            click.echo(f"Project: {result.project}")


# ─────────────────────────────────────────────────────────────────────────────
# Summarize Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--dry-run", is_flag=True, help="Preview changes without writing")
@click.option("--limit", type=int, help="Maximum entries to process")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def summarize(dry_run: bool, limit: Optional[int], as_json: bool):
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
    try:
        from .core import generate_descriptions
    except ImportError:
        click.echo("Error: generate_descriptions function not available in core", err=True)
        click.echo("This function needs to be restored from v0.1.0", err=True)
        sys.exit(1)

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
                click.echo(f"Generated descriptions for {len(updated)} entries:")
                for r in updated[:10]:
                    click.echo(f"  - {r['path']}")
                if len(updated) > 10:
                    click.echo(f"  ... and {len(updated) - 10} more")

        if skipped:
            click.echo(f"\nSkipped {len(skipped)} entries (no content to summarize)")

        if errors:
            click.echo(f"\n{len(errors)} errors:")
            for r in errors[:5]:
                click.echo(f"  - {r['path']}: {r.get('reason', 'Unknown error')}")


# ─────────────────────────────────────────────────────────────────────────────
# Schema Command (Agent Introspection)
# ─────────────────────────────────────────────────────────────────────────────


def _build_schema() -> dict:
    """Build the complete CLI schema with agent-friendly metadata.

    Returns a dict containing all commands, their options, related commands,
    and common mistakes for agent introspection.
    """
    schema = {
        "version": "0.1.0",
        "description": "Token-efficient CLI for memex knowledge base",
        "commands": {
            "search": {
                "description": "Search the knowledge base with hybrid keyword + semantic search",
                "aliases": [],
                "arguments": [
                    {"name": "query", "required": True, "description": "Search query text"}
                ],
                "options": [
                    {"name": "--tags", "short": "-t", "type": "string", "description": "Filter by tags (comma-separated)"},
                    {"name": "--mode", "type": "choice", "choices": ["hybrid", "keyword", "semantic"], "default": "hybrid", "description": "Search mode"},
                    {"name": "--limit", "short": "-n", "type": "integer", "default": 10, "description": "Max results"},
                    {"name": "--min-score", "type": "float", "description": "Minimum score threshold (0.0-1.0)"},
                    {"name": "--content", "short": "-c", "type": "flag", "description": "Include full content in results"},
                    {"name": "--strict", "type": "flag", "description": "Disable semantic fallback for keyword mode"},
                    {"name": "--terse", "type": "flag", "description": "Output paths only (one per line)"},
                    {"name": "--full-titles", "type": "flag", "description": "Show full titles without truncation"},
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["get", "list"],
                "common_mistakes": {
                    "empty query": "Query cannot be empty. Provide a non-whitespace search term.",
                    "--tags without value": "Tags must be comma-separated, e.g., --tags=infra,docker",
                },
                "examples": [
                    "mx search \"deployment\"",
                    "mx search \"docker\" --tags=infrastructure",
                    "mx search \"api\" --mode=semantic --limit=5",
                ],
            },
            "get": {
                "description": "Read a knowledge base entry by path",
                "aliases": [],
                "arguments": [
                    {"name": "path", "required": True, "description": "Path to entry relative to KB root"}
                ],
                "options": [
                    {"name": "--json", "type": "flag", "description": "Output as JSON with metadata"},
                    {"name": "--metadata", "short": "-m", "type": "flag", "description": "Show only metadata"},
                ],
                "related": ["search", "list"],
                "common_mistakes": {
                    "absolute path": "Use relative path from KB root, not absolute filesystem path",
                    "missing .md extension": "Include the .md extension: 'tooling/entry.md' not 'tooling/entry'",
                },
                "examples": [
                    "mx get tooling/beads-issue-tracker.md",
                    "mx get tooling/beads-issue-tracker.md --metadata",
                ],
            },
            "add": {
                "description": "Create a new knowledge base entry",
                "aliases": [],
                "arguments": [],
                "options": [
                    {"name": "--title", "short": "-t", "type": "string", "required": True, "description": "Entry title"},
                    {"name": "--tags", "type": "string", "required": True, "description": "Tags (comma-separated)"},
                    {"name": "--category", "short": "-c", "type": "string", "description": "Category/directory"},
                    {"name": "--content", "type": "string", "description": "Content (or use --file/--stdin)"},
                    {"name": "--file", "short": "-f", "type": "path", "description": "Read content from file"},
                    {"name": "--stdin", "type": "flag", "description": "Read content from stdin"},
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["append", "update"],
                "common_mistakes": {
                    "missing content source": "Must provide --content, --file, or --stdin",
                    "tags without value": "Tags are required: --tags=\"tag1,tag2\"",
                },
                "examples": [
                    "mx add --title=\"My Entry\" --tags=\"foo,bar\" --content=\"# Content\"",
                    "mx add --title=\"My Entry\" --tags=\"foo,bar\" --file=content.md",
                ],
            },
            "append": {
                "description": "Append content to existing entry by title, or create new if not found",
                "aliases": [],
                "arguments": [
                    {"name": "title", "required": True, "description": "Title of entry to append to (case-insensitive)"}
                ],
                "options": [
                    {"name": "--content", "short": "-c", "type": "string", "description": "Content to append"},
                    {"name": "--file", "short": "-f", "type": "path", "description": "Read content from file"},
                    {"name": "--stdin", "type": "flag", "description": "Read content from stdin"},
                    {"name": "--tags", "short": "-t", "type": "string", "description": "Tags (required for new entries)"},
                    {"name": "--category", "type": "string", "description": "Category for new entries"},
                    {"name": "--directory", "short": "-d", "type": "string", "description": "Directory for new entries"},
                    {"name": "--no-create", "type": "flag", "description": "Error if entry not found"},
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["add", "update", "patch"],
                "common_mistakes": {
                    "using path instead of title": "append takes title, not path. Use 'mx append \"Entry Title\"' not 'mx append path/entry.md'",
                    "missing content": "Must provide --content, --file, or --stdin",
                },
                "examples": [
                    "mx append \"Daily Log\" --content=\"Session summary\"",
                    "mx append \"API Docs\" --file=api.md --tags=\"api,docs\"",
                ],
            },
            "replace": {
                "description": "Replace content or tags in an existing entry (overwrites)",
                "aliases": ["update"],
                "arguments": [
                    {"name": "path", "required": True, "description": "Path to entry relative to KB root"}
                ],
                "options": [
                    {"name": "--tags", "type": "string", "description": "New tags (comma-separated)"},
                    {"name": "--content", "type": "string", "description": "New content (replaces existing)"},
                    {"name": "--file", "short": "-f", "type": "path", "description": "Read content from file"},
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["patch", "append"],
                "common_mistakes": {
                    "confusing with append": "replace overwrites content. Use 'mx append' to add to existing content.",
                    "confusing with patch": "replace overwrites entire content. Use 'mx patch' for surgical find-replace.",
                },
                "examples": [
                    "mx replace path/entry.md --tags=\"new,tags\"",
                    "mx replace path/entry.md --file=updated-content.md",
                ],
            },
            "patch": {
                "description": "Apply surgical find-replace edits to a KB entry",
                "aliases": [],
                "arguments": [
                    {"name": "path", "required": True, "description": "Path to entry relative to KB root"}
                ],
                "options": [
                    {"name": "--find", "type": "string", "description": "Exact text to find and replace"},
                    {"name": "--replace", "type": "string", "description": "Replacement text"},
                    {"name": "--find-file", "type": "path", "description": "Read --find text from file"},
                    {"name": "--replace-file", "type": "path", "description": "Read --replace text from file"},
                    {"name": "--replace-all", "type": "flag", "description": "Replace all occurrences"},
                    {"name": "--dry-run", "type": "flag", "description": "Preview changes without writing"},
                    {"name": "--backup", "type": "flag", "description": "Create .bak backup before patching"},
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["replace", "append"],
                "common_mistakes": {
                    "--find without --replace": "Both --find and --replace are required",
                    "multiple matches without --replace-all": "If text matches multiple times, use --replace-all or provide more context in --find",
                    "using for append": "patch is for replacement. Use 'mx append' to add content to an entry.",
                },
                "exit_codes": {
                    "0": "Success",
                    "1": "Text not found",
                    "2": "Multiple matches (ambiguous, use --replace-all)",
                    "3": "File error (not found, permission, encoding)",
                },
                "examples": [
                    "mx patch tooling/notes.md --find \"old text\" --replace \"new text\"",
                    "mx patch tooling/notes.md --find \"TODO\" --replace \"DONE\" --replace-all",
                ],
            },
            "delete": {
                "description": "Delete a knowledge base entry",
                "aliases": [],
                "arguments": [
                    {"name": "path", "required": True, "description": "Path to entry relative to KB root"}
                ],
                "options": [
                    {"name": "--force", "short": "-f", "type": "flag", "description": "Delete even if has backlinks"},
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": [],
                "common_mistakes": {
                    "deleting with backlinks": "Entries with backlinks require --force. Check backlinks first with 'mx get path.md --metadata'",
                },
                "examples": [
                    "mx delete path/to/entry.md",
                    "mx delete path/to/entry.md --force",
                ],
            },
            "list": {
                "description": "List knowledge base entries with optional filters",
                "aliases": [],
                "arguments": [],
                "options": [
                    {"name": "--tag", "short": "-t", "type": "string", "description": "Filter by tag"},
                    {"name": "--category", "short": "-c", "type": "string", "description": "Filter by category"},
                    {"name": "--limit", "short": "-n", "type": "integer", "default": 20, "description": "Max results"},
                    {"name": "--full-titles", "type": "flag", "description": "Show full titles without truncation"},
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["search", "tree", "tags"],
                "common_mistakes": {
                    "invalid category": "Category must exist in KB. Use 'mx tree' to see valid categories.",
                },
                "examples": [
                    "mx list",
                    "mx list --tag=infrastructure",
                    "mx list --category=tooling --limit=10",
                ],
            },
            "tree": {
                "description": "Display knowledge base directory structure",
                "aliases": [],
                "arguments": [
                    {"name": "path", "required": False, "default": "", "description": "Starting path (default: root)"}
                ],
                "options": [
                    {"name": "--depth", "short": "-d", "type": "integer", "default": 3, "description": "Max depth"},
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["list"],
                "common_mistakes": {},
                "examples": [
                    "mx tree",
                    "mx tree tooling --depth=2",
                ],
            },
            "tags": {
                "description": "List all tags with usage counts",
                "aliases": [],
                "arguments": [],
                "options": [
                    {"name": "--min-count", "type": "integer", "default": 1, "description": "Minimum usage count"},
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["list", "search"],
                "common_mistakes": {},
                "examples": [
                    "mx tags",
                    "mx tags --min-count=3",
                ],
            },
            "health": {
                "description": "Audit knowledge base for problems (orphans, broken links, stale content)",
                "aliases": [],
                "arguments": [],
                "options": [
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["suggest-links", "hubs"],
                "common_mistakes": {},
                "examples": [
                    "mx health",
                    "mx health --json",
                ],
            },
            "hubs": {
                "description": "Show most connected entries (hub notes)",
                "aliases": [],
                "arguments": [],
                "options": [
                    {"name": "--limit", "short": "-n", "type": "integer", "default": 10, "description": "Max results"},
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["health", "suggest-links"],
                "common_mistakes": {},
                "examples": [
                    "mx hubs",
                    "mx hubs --limit=5",
                ],
            },
            "suggest-links": {
                "description": "Suggest entries to link to based on semantic similarity",
                "aliases": [],
                "arguments": [
                    {"name": "path", "required": True, "description": "Path to entry relative to KB root"}
                ],
                "options": [
                    {"name": "--limit", "short": "-n", "type": "integer", "default": 5, "description": "Max suggestions"},
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["health", "hubs"],
                "common_mistakes": {},
                "examples": [
                    "mx suggest-links tooling/my-entry.md",
                ],
            },
            "whats-new": {
                "description": "Show recently created or updated entries",
                "aliases": [],
                "arguments": [],
                "options": [
                    {"name": "--days", "short": "-d", "type": "integer", "default": 30, "description": "Look back N days"},
                    {"name": "--limit", "short": "-n", "type": "integer", "default": 10, "description": "Max results"},
                    {"name": "--project", "short": "-p", "type": "string", "description": "Filter by project name"},
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["list", "search"],
                "common_mistakes": {},
                "examples": [
                    "mx whats-new",
                    "mx whats-new --days=7 --limit=5",
                    "mx whats-new --project=myapp",
                ],
            },
            "prime": {
                "description": "Output agent workflow context for session start (for hooks)",
                "aliases": [],
                "arguments": [],
                "options": [
                    {"name": "--full", "type": "flag", "description": "Force full CLI output"},
                    {"name": "--mcp", "type": "flag", "description": "Force MCP mode (minimal output)"},
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["schema"],
                "common_mistakes": {},
                "examples": [
                    "mx prime",
                    "mx prime --full",
                    "mx prime --mcp",
                ],
            },
            "init": {
                "description": "Initialize a knowledge base (project or user scope)",
                "aliases": [],
                "arguments": [],
                "options": [
                    {"name": "--path", "short": "-p", "type": "path", "description": "Custom location for KB (default: kb/)"},
                    {"name": "--user", "short": "-u", "type": "flag", "description": "Create user-scope KB at ~/.memex/kb/"},
                    {"name": "--force", "short": "-f", "type": "flag", "description": "Reinitialize existing KB"},
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["add", "search"],
                "common_mistakes": {},
                "examples": [
                    "mx init",
                    "mx init --user",
                    "mx init --path docs/kb",
                    "mx init --force",
                ],
            },
            "reindex": {
                "description": "Rebuild search indices from all markdown files",
                "aliases": [],
                "arguments": [],
                "options": [],
                "related": ["search"],
                "common_mistakes": {
                    "running unnecessarily": "Only needed after bulk imports or if search seems stale. Normal operations auto-index.",
                },
                "examples": [
                    "mx reindex",
                ],
            },
            "context": {
                "description": "Show or validate project KB context (.kbcontext file)",
                "aliases": [],
                "arguments": [],
                "subcommands": ["show", "validate"],
                "options": [],
                "related": ["init", "add", "search"],
                "common_mistakes": {},
                "examples": [
                    "mx context",
                    "mx context show",
                    "mx context validate",
                ],
            },
            "schema": {
                "description": "Output CLI schema with agent-friendly metadata for introspection",
                "aliases": [],
                "arguments": [],
                "options": [
                    {"name": "--command", "short": "-c", "type": "string", "description": "Show schema for specific command only"},
                    {"name": "--compact", "type": "flag", "description": "Minimal output (commands and options only)"},
                ],
                "related": ["prime"],
                "common_mistakes": {},
                "examples": [
                    "mx schema",
                    "mx schema --command=patch",
                    "mx schema --compact",
                ],
            },
        },
        "global_options": [
            {"name": "--json-errors", "type": "flag", "description": "Output errors as JSON (for programmatic use)"},
            {"name": "--version", "type": "flag", "description": "Show version"},
            {"name": "--help", "type": "flag", "description": "Show help"},
        ],
        "workflows": {
            "search_and_read": {
                "description": "Find and read an entry",
                "steps": ["mx search \"query\"", "mx get path/from/results.md"],
            },
            "create_entry": {
                "description": "Create a new KB entry",
                "steps": ["mx add --title=\"Title\" --tags=\"tag1,tag2\" --content=\"...\""],
            },
            "surgical_edit": {
                "description": "Make precise edits to existing content",
                "steps": ["mx get path.md  # Read current content", "mx patch path.md --find \"old\" --replace \"new\""],
            },
            "append_to_log": {
                "description": "Add content to an ongoing log entry",
                "steps": ["mx append \"Log Title\" --content=\"New entry...\""],
            },
        },
    }
    return schema


@cli.command()
@click.option("--command", "-c", "command_name", help="Show schema for specific command only")
@click.option("--compact", is_flag=True, help="Minimal output (commands and options only)")
def schema(command_name: Optional[str], compact: bool):
    """Output CLI schema with agent-friendly metadata for introspection.

    Provides structured JSON describing all commands, their options,
    related commands, and common mistakes. Designed for agent tooling
    to enable proactive error avoidance.

    \b
    Schema includes:
    - All commands with their arguments and options
    - Related commands (cross-references)
    - Common mistakes and how to avoid them
    - Example invocations
    - Recommended workflows

    \b
    Examples:
      mx schema                    # Full schema
      mx schema --command=patch    # Schema for patch command only
      mx schema --compact          # Minimal output
    """
    full_schema = _build_schema()

    if command_name:
        # Show specific command only
        if command_name not in full_schema["commands"]:
            click.echo(f"Error: Unknown command '{command_name}'", err=True)
            available = ", ".join(sorted(full_schema["commands"].keys()))
            click.echo(f"Available commands: {available}", err=True)
            sys.exit(1)

        result = {
            "command": command_name,
            **full_schema["commands"][command_name],
        }
        output(result, as_json=True)
    elif compact:
        # Minimal output - just commands and their options
        compact_schema = {
            "version": full_schema["version"],
            "commands": {},
        }
        for cmd, data in full_schema["commands"].items():
            compact_schema["commands"][cmd] = {
                "description": data["description"],
                "arguments": data.get("arguments", []),
                "options": [opt["name"] for opt in data.get("options", [])],
            }
        output(compact_schema, as_json=True)
    else:
        # Full schema
        output(full_schema, as_json=True)


# ─────────────────────────────────────────────────────────────────────────────
# Publishing
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--kb-root", "-k",
    type=click.Path(exists=True),
    help="KB source directory (overrides .kbcontext and MEMEX_KB_ROOT)",
)
@click.option(
    "--global", "use_global",
    is_flag=True,
    help="Use global MEMEX_KB_ROOT (required when no --kb-root or project_kb)",
)
@click.option(
    "--yes", "-y",
    is_flag=True,
    help="Skip confirmation prompt when publishing from global KB",
)
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
    "--title", "-t",
    default="Memex",
    help="Site title for header and page titles (default: Memex)",
)
@click.option(
    "--index", "-i", "index_entry",
    default=None,
    help="Path to entry to use as landing page (e.g., guides/welcome)",
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
@click.pass_context
def publish(
    ctx: click.Context,
    kb_root: str | None,
    use_global: bool,
    yes: bool,
    output_dir: str,
    base_url: str,
    title: str,
    index_entry: str | None,
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
    KB source resolution (in order):
      1. --kb-root flag (explicit path)
      2. project_kb in .kbcontext (relative to context file)
      3. --global flag required to use MEMEX_KB_ROOT

    \b
    Base URL resolution:
      1. --base-url flag (explicit)
      2. publish_base_url in .kbcontext (auto-applied)

    Use --base-url when hosting at a subdirectory (e.g., user.github.io/repo).
    Without it, links will 404. Configure in .kbcontext to avoid repeating:

    \b
      # .kbcontext
      project_kb: ./kb
      publish_base_url: /repo-name

    \b
    Examples:
      mx publish -o docs                   # Uses .kbcontext settings
      mx publish --kb-root ./kb -o docs    # Explicit KB source
      mx publish --global -o docs          # Use MEMEX_KB_ROOT
      mx publish --base-url /my-kb         # Subdirectory hosting
    """
    from .config import get_kb_root
    from .context import get_kb_context
    from .core import publish as core_publish

    # Get context early - used for multiple settings
    context = get_kb_context()

    # Resolve KB source with safety guardrails
    resolved_kb: Path | None = None
    source_description = ""

    if kb_root:
        # Explicit --kb-root flag takes priority
        resolved_kb = Path(kb_root).resolve()
        source_description = "--kb-root flag"
    elif context and context.project_kb and context.source_file:
        # Try .kbcontext project_kb
        project_kb_path = (context.source_file.parent / context.project_kb).resolve()
        if project_kb_path.exists():
            resolved_kb = project_kb_path
            source_description = ".kbcontext project_kb"

    # No local KB found - require --global flag for safety
    if not resolved_kb:
        if not use_global:
            click.echo("Error: No project KB found", err=True)
            click.echo("", err=True)
            click.echo("Options:", err=True)
            click.echo("  - Add 'project_kb: ./kb' to .kbcontext", err=True)
            click.echo("  - Use --kb-root ./path/to/kb", err=True)
            click.echo("  - Use --global to publish from MEMEX_KB_ROOT", err=True)
            sys.exit(1)

        resolved_kb = get_kb_root()
        source_description = "MEMEX_KB_ROOT (--global)"

        # Warn and require confirmation when publishing from global KB
        if not yes:
            click.echo(f"Warning: You are about to publish content from your global KB at {resolved_kb}", err=True)
            click.echo("This may include organizational or private content.", err=True)
            click.echo("", err=True)
            if not click.confirm("Continue?", default=False):
                click.echo("Aborted.", err=True)
                sys.exit(0)

    # Resolve base_url from context if not specified via CLI
    resolved_base_url = base_url
    if not resolved_base_url and context and context.publish_base_url:
        resolved_base_url = context.publish_base_url

    # Show confirmation message
    click.echo(f"Publishing from: {resolved_kb} (via {source_description})")
    if resolved_base_url:
        click.echo(f"Base URL: {resolved_base_url}")

    try:
        result = run_async(core_publish(
            output_dir=output_dir,
            base_url=resolved_base_url,
            site_title=title,
            index_entry=index_entry,
            include_drafts=include_drafts,
            include_archived=include_archived,
            clean=not no_clean,
            kb_root=resolved_kb,
        ))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
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
