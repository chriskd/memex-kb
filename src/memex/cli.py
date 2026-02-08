#!/usr/bin/env python3
"""
mx: CLI for memex knowledge base

Usage:
    mx search "query"              # Search entries
    mx get path/to/entry.md        # Read an entry
    mx add --title="..." --tags=.. # Create entry
    mx tree                        # Browse structure
    mx health                      # Audit KB health
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import sys
from datetime import datetime
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast

import click
from click.exceptions import ClickException, UsageError

from . import __version__ as MEMEX_VERSION

# Lazy imports to speed up CLI startup
# The heavy imports (chromadb, sentence-transformers) only load when needed


def run_async(coro):
    """Run async function synchronously."""
    return asyncio.run(coro)


def decode_escape_sequences(s: str) -> str:
    """Decode escape sequences in a string (e.g., \\n -> newline, \\t -> tab).

    Uses Python's unicode_escape codec to interpret common escape sequences.
    Only applies to CLI --content values where users expect shell-like behavior.
    """
    import codecs

    # Use unicode_escape to decode, but we need to encode to bytes first
    # since unicode_escape works on bytes
    return codecs.decode(s, "unicode_escape")


def _has_yaml_frontmatter(content: str) -> bool:
    lines = content.splitlines()
    start_index = None
    for i, line in enumerate(lines):
        if line.strip() == "":
            continue
        if line.strip() != "---":
            return False
        start_index = i + 1
        break

    if start_index is None:
        return False

    for line in lines[start_index : start_index + 200]:
        if line.strip() in ("---", "..."):
            return True
    return False


def _try_parse_iso_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _strip_yaml_frontmatter(raw_content: str) -> tuple[bool, str]:
    """Best-effort strip a leading YAML frontmatter block from content.

    Returns (had_frontmatter, body). If no leading frontmatter is detected,
    returns (False, raw_content).
    """
    if not _has_yaml_frontmatter(raw_content):
        return False, raw_content

    lines = raw_content.splitlines(keepends=True)

    # Find the first non-empty line and ensure it is a frontmatter start marker.
    i = 0
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i >= len(lines) or lines[i].strip() != "---":
        return False, raw_content

    # Find the closing marker within a reasonable bound.
    j = i + 1
    end = min(len(lines), i + 1 + 400)
    while j < end:
        if lines[j].strip() in ("---", "..."):
            body = "".join(lines[j + 1 :])
            return True, body
        j += 1

    # If we couldn't find the end marker, don't strip anything.
    return True, raw_content


def _extract_input_frontmatter_overrides(
    raw_content: str,
) -> tuple[bool, dict[str, Any] | None, str]:
    """If raw_content starts with YAML frontmatter, strip it and return safe overrides.

    We preserve only optional fields to avoid surprising behavior (title/tags remain CLI-owned).
    """
    had_frontmatter, body = _strip_yaml_frontmatter(raw_content)
    if not had_frontmatter:
        return False, None, raw_content

    try:
        import frontmatter as fm

        post = fm.loads(raw_content)
    except Exception:
        # If parsing fails, keep stripping behavior but don't try to merge overrides.
        return True, None, body

    meta = post.metadata if isinstance(post.metadata, dict) else {}
    body = post.content

    overrides: dict[str, Any] = {}

    # Optional fields we can safely carry forward.
    if isinstance(meta.get("description"), str) and meta["description"].strip():
        overrides["description"] = meta["description"].strip()

    aliases = meta.get("aliases")
    if isinstance(aliases, list) and all(isinstance(a, str) for a in aliases):
        overrides["aliases"] = [a.strip() for a in aliases if a.strip()]

    status = meta.get("status")
    if isinstance(status, str) and status.strip():
        overrides["status"] = status.strip()

    created = _try_parse_iso_datetime(meta.get("created"))
    if created is not None:
        overrides["created"] = created

    updated = _try_parse_iso_datetime(meta.get("updated"))
    if updated is not None:
        overrides["updated"] = updated

    contributors = meta.get("contributors")
    if isinstance(contributors, list) and all(isinstance(c, str) for c in contributors):
        overrides["contributors"] = [c.strip() for c in contributors if c.strip()]

    # Preserve typed relations / semantic links if present and well-formed.
    # If invalid, ignore; add_entry will still write valid frontmatter.
    try:
        from .models import RelationLink, SemanticLink

        rels = meta.get("relations")
        if isinstance(rels, list):
            parsed: list[RelationLink] = []
            for item in rels:
                if isinstance(item, dict):
                    parsed.append(RelationLink.model_validate(item))
            if parsed:
                overrides["relations"] = parsed

        sem = meta.get("semantic_links")
        if isinstance(sem, list):
            parsed_sem: list[SemanticLink] = []
            for item in sem:
                if isinstance(item, dict):
                    parsed_sem.append(SemanticLink.model_validate(item))
            if parsed_sem:
                overrides["semantic_links"] = parsed_sem
    except Exception:
        pass

    return True, (overrides or None), body


if TYPE_CHECKING:
    from .models import RelationLink


def _parse_relations_inputs(
    relation_items: tuple[str, ...],
    relations_json: str | None,
) -> list[RelationLink] | None:
    from .models import RelationLink

    relations: list[RelationLink] | None = None
    if relation_items:
        relations = []
        for item in relation_items:
            if "=" not in item:
                click.echo("Error: --relation must be in path=type format", err=True)
                sys.exit(1)
            path_value, type_value = item.split("=", 1)
            if not path_value.strip() or not type_value.strip():
                click.echo("Error: --relation must be in path=type format", err=True)
                sys.exit(1)
            relations.append(RelationLink(path=path_value.strip(), type=type_value.strip()))

    if relations_json:
        try:
            relations_data = json.loads(relations_json)
            if not isinstance(relations_data, list):
                click.echo("Error: --relations must be a JSON array", err=True)
                sys.exit(1)
            if relations is None:
                relations = []
            for i, relation_data in enumerate(relations_data):
                if not isinstance(relation_data, dict):
                    click.echo(f"Error: --relations[{i}] must be a JSON object", err=True)
                    sys.exit(1)
                missing = [f for f in ("path", "type") if f not in relation_data]
                if missing:
                    click.echo(
                        f"Error: --relations[{i}] missing required fields: {', '.join(missing)}",
                        err=True,
                    )
                    sys.exit(1)
                relations.append(
                    RelationLink(
                        path=relation_data["path"],
                        type=relation_data["type"],
                    )
                )
        except json.JSONDecodeError as e:
            click.echo(f"Error: --relations is not valid JSON: {e}", err=True)
            sys.exit(1)

    return relations


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


def _json_dumps(data: Any, *, compact: bool) -> str:
    if compact:
        return json.dumps(data, separators=(",", ":"), default=str)
    return json.dumps(data, indent=2, default=str)


def _emit_json(
    payload: dict[str, Any],
    *,
    compact: bool,
    max_bytes: int | None = None,
) -> None:
    """Emit JSON, optionally compact and best-effort bounded by max_bytes.

    The bounding logic is intentionally conservative: it only truncates list-valued
    fields commonly used by agent hooks ("entries", "recent_entries").
    """
    if max_bytes is not None and not compact:
        raise UsageError("--max-bytes requires compact JSON output")

    if max_bytes is None:
        click.echo(_json_dumps(payload, compact=compact))
        return

    # Best-effort truncation for bounded hook output.
    import copy

    data = copy.deepcopy(payload)
    dropped: dict[str, int] = {}

    def _size() -> int:
        return len(_json_dumps(data, compact=True).encode("utf-8"))

    if _size() <= max_bytes:
        click.echo(_json_dumps(data, compact=True))
        return

    for field in ("recent_entries", "entries"):
        value = data.get(field)
        if not isinstance(value, list) or not value:
            continue
        original_len = len(value)
        while value and _size() > max_bytes:
            value.pop()
        if len(value) != original_len:
            dropped[field] = original_len - len(value)

    if dropped:
        data["truncated"] = dropped

    if _size() > max_bytes:
        raise UsageError("--max-bytes is too small to fit the requested payload")

    click.echo(_json_dumps(data, compact=True))


def _handle_error(
    ctx: click.Context,
    error: Exception,
    fallback_message: str | None = None,
    exit_code: int = 1,
) -> NoReturn:
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
            suggestion = error.details.get("suggestion") if error.details else None
            if suggestion:
                click.echo(f"Hint: {_normalize_error_message(str(suggestion))}", err=True)
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
    if "category" in message_lower and (
        "required" in message_lower or "not found" in message_lower
    ):
        return ErrorCode.INVALID_CATEGORY
    if "tag" in message_lower and "required" in message_lower:
        return ErrorCode.INVALID_TAGS
    if "index" in message_lower and "unavailable" in message_lower:
        return ErrorCode.INDEX_UNAVAILABLE
    if "semantic" in message_lower and (
        "unavailable" in message_lower or "not available" in message_lower
    ):
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
    if ".kbconfig" in message.lower():
        lines.append("No .kbconfig primary set. Update .kbconfig or pass --category.")
    if tags:
        lines.append(f"Your tags: {', '.join(tags)}")
    if suggestion:
        lines.append(f"Suggested: --category={suggestion}")
    elif matches:
        lines.append(f"Tags matched categories: {', '.join(matches)}")
    if valid_categories:
        lines.append(f"Available categories: {', '.join(valid_categories)}")
    lines.append('Example: mx add --title="..." --tags="..." --category=... --content="..."')
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
                    "suggestion": "Provide --category or set primary in .kbconfig",
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


def format_json_error(code: str, message: str, details: dict | None = None) -> str:
    """Format an error as JSON for --json-errors output."""
    error: dict[str, dict[str, object]] = {"error": {"code": code, "message": message}}
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
                    raise UsageError(f"No such command '{cmd_name}'. Did you mean '{matches[0]}'?")
            raise

    def invoke(self, ctx):
        """Override invoke to catch and format errors."""
        try:
            return super().invoke(ctx)
        except ClickException as e:
            if ctx.params.get("json_errors"):
                code = get_error_code_for_exception(e)
                click.echo(format_json_error(code, e.format_message()), err=True)
                raise SystemExit(1)
            raise

    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        **extra: Any,
    ) -> Any:
        """Override main to catch errors during argument parsing.

        This catches errors that happen before invoke() is called,
        such as invalid option types or missing required arguments.
        """
        # Click's global options generally must come before subcommands, but focusgroup
        # users frequently wrote: `mx search "q" --json-errors`.
        #
        # When --json-errors is present anywhere, we:
        # - normalize args to move it to the front (so Click parses it as a global flag)
        # - run with standalone_mode=False so Click raises exceptions instead of printing
        #   plain text and calling sys.exit().
        argv = list(args) if args is not None else list(sys.argv[1:])
        json_errors_requested = "--json-errors" in argv

        if not json_errors_requested:
            return super().main(args, prog_name, complete_var, standalone_mode, **extra)

        # Normalize misplaced --json-errors to be a true global flag.
        if "--json-errors" in argv:
            argv = [a for a in argv if a != "--json-errors"]
            argv.insert(0, "--json-errors")

        try:
            return super().main(
                argv,
                prog_name,
                complete_var,
                standalone_mode=False,
                **extra,
            )
        except ClickException as e:
            code = get_error_code_for_exception(e)
            click.echo(format_json_error(code, e.format_message()), err=True)
            raise SystemExit(1)
        except SystemExit:
            # In case a subcommand calls sys.exit explicitly, preserve it.
            raise
        except Exception as e:
            click.echo(format_json_error("INTERNAL_ERROR", str(e)), err=True)
            raise SystemExit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Path Suggestions (Get UX)
# ─────────────────────────────────────────────────────────────────────────────


def _suggest_similar_paths(path: str, limit: int = 5) -> list[str]:
    """Suggest similar KB paths when a path-based lookup fails.

    Best-effort: if KB isn't configured (or scope KB not found), returns [].
    """
    from .config import (
        ConfigurationError,
        get_kb_root,
        get_project_kb_root,
        get_user_kb_root,
        parse_scoped_path,
    )

    scope, relative = parse_scoped_path(path)
    try:
        if scope == "project":
            kb_root = get_project_kb_root()
            if not kb_root:
                return []
        elif scope == "user":
            kb_root = get_user_kb_root()
            if not kb_root:
                return []
        else:
            kb_root = get_kb_root()
    except ConfigurationError:
        return []

    if not kb_root.exists():
        return []

    candidates: list[str] = []
    try:
        for p in kb_root.rglob("*.md"):
            try:
                rel = str(p.relative_to(kb_root))
            except Exception:
                continue
            if rel.startswith(".kb-indices/") or "/.kb-indices/" in rel:
                continue
            if rel.startswith(".") or "/." in rel:
                continue
            if rel.startswith("_") or "/_" in rel:
                continue
            candidates.append(rel)
    except Exception:
        return []

    target = relative
    target_alt = target[:-3] if target.endswith(".md") else f"{target}.md"

    matches = difflib.get_close_matches(target, candidates, n=limit, cutoff=0.6)
    if len(matches) < limit:
        more = difflib.get_close_matches(target_alt, candidates, n=limit, cutoff=0.6)
        for m in more:
            if m not in matches:
                matches.append(m)
                if len(matches) >= limit:
                    break
    return matches[:limit]


def _require_kb_configured(ctx: click.Context, scope: str | None = None) -> None:
    """Fail fast with friendly guidance when no KB is configured.

    Many read-only commands can otherwise look like an empty KB on cold start.
    """
    from .config import ConfigurationError, get_kb_root, get_kb_root_by_scope

    try:
        if scope:
            get_kb_root_by_scope(scope)
        else:
            get_kb_root()
    except ConfigurationError as exc:
        _handle_error(ctx, exc)


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

    # Search is an optional extra; don't push users into a crash loop on first run.
    import importlib.util

    has_keyword_search = importlib.util.find_spec("whoosh") is not None

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
    _output_status(kb_root, context, entries, project_name, has_keyword_search)


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
            entries = run_async(core_whats_new(days=14, limit=limit, scope=project))
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
    has_keyword_search: bool = False,
) -> None:
    """Output the status display.

    Args:
        kb_root: KB root path (None if not configured).
        context: Loaded KBContext (None if no .kbconfig).
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
        lines.append("Tip: Run 'mx prime' for a short agent walkthrough.")
    else:
        lines.append("KB Root: NOT CONFIGURED")
        lines.append("")
        lines.append("Set MEMEX_USER_KB_ROOT environment variable or run 'mx init'")
        lines.append("to point to your knowledge base directory.")

    # Start here section
    lines.append("")
    lines.append("Start here")
    lines.append("-" * 40)
    lines.append("  mx prime            Quick agent walkthrough")
    if not kb_root:
        lines.append("  mx init             Create project KB in ./kb")
        lines.append("  mx init --user      Create personal KB in ~/.memex/kb")
    lines.append("  mx info             KB paths + categories")
    lines.append("  mx context show     .kbconfig (primary + default tags)")
    lines.append('  mx add --title="..." --tags="..." --category=... --content="..."')
    if has_keyword_search:
        lines.append('  mx search "query"   Search entries')
    else:
        lines.append("  mx doctor           Check search deps + install hint")
    lines.append("  mx get path/to/entry.md")

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
            date_str = str(e.get("activity_date", ""))
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
        lines.append("  mx prime           Quick agent onboarding")
        lines.append("  mx init            Create project KB in ./kb")
        lines.append("  mx --help           Show all commands")
    else:
        lines.append("  mx prime            Quick agent onboarding")
        if entries:
            # KB has content
            if has_keyword_search:
                lines.append('  mx search "query"   Search the knowledge base')
            else:
                lines.append("  mx doctor           Check search deps + install hint")
            lines.append("  mx whats-new        Recent changes")
            lines.append("  mx tree             Browse structure")
        else:
            # Empty KB
            lines.append('  mx add --title="..." --tags="..." --category=...  Add first entry')
            lines.append("  mx tree             Browse structure")

        if not context:
            lines.append("  mx context show     Inspect project .kbconfig")

        lines.append("  mx --help           Show all commands")

    # Output
    click.echo("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Main CLI Group
# ─────────────────────────────────────────────────────────────────────────────


@click.group(cls=JsonErrorGroup, invoke_without_command=True)
@click.version_option(version=MEMEX_VERSION, prog_name="mx")
@click.option(
    "--json-errors",
    "json_errors",
    is_flag=True,
    help="Output errors as JSON (for programmatic use)",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    envvar="MEMEX_QUIET",
    help="Suppress warnings, show only errors and essential output",
)
@click.pass_context
def cli(ctx: click.Context, json_errors: bool, quiet: bool):
    """mx: CLI for memex knowledge base.

    Search, browse, and manage KB entries.

    \b
    Quick start:
      mx search "deployment"         # Find entries (keyword; semantic if installed)
      mx get path/to/entry.md        # Read an entry
      mx tree                        # Browse structure
      mx health                      # Check KB health
      Tip: For semantic search: uv tool install 'memex-kb[search]'

    \b
    Create content:
      mx add --title="Title" --tags="a,b" --category=... --content="..."
      mx quick-add --content="..."           # Suggest title/tags/category
      mx ingest notes.md                     # Import file, adds frontmatter
      mx append "Existing Title" --content="append this"
      Note: if --category is omitted and no .kbconfig primary exists, mx add defaults
            to KB root (.) and prints a warning.

    \b
    Modify content:
      mx patch path.md --find "old text" --replace "new text"
      mx replace path.md --tags="new,tags"   # Overwrite tags/content

    \b
    New to mx:
      mx prime                              # Agent onboarding + required frontmatter
      mx info                               # KB paths + categories
      mx context show                       # .kbconfig (primary + default tags)

    \b
    Browse vs recency:
      mx list --tags=foo                    # Filtered list
      mx tree                               # Directory structure
      mx whats-new --days=7                 # Recent changes

    \b
    Agent helpers:
      mx schema --compact                   # CLI schema for LLMs
      mx batch < commands.txt               # Multiple commands, one run

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


@cli.command("help")
@click.argument("command", required=False)
@click.pass_context
def help_cmd(ctx: click.Context, command: str | None):
    """Show help for a command (alias: mx --help).

    Examples:
      mx help
      mx help search
    """
    parent = ctx.parent or ctx
    if not command:
        click.echo(parent.get_help())
        return

    cmd = parent.command.get_command(parent, command)
    if cmd is None:
        raise UsageError(f"No such command '{command}'. Try 'mx --help' for a list.")
    click.echo(cmd.get_help(parent))


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def doctor(ctx: click.Context, as_json: bool):
    """Check installation, KB config, and optional dependency availability."""
    import importlib.util
    import sys

    from .config import ConfigurationError, get_kb_root

    deps = {
        "whoosh": importlib.util.find_spec("whoosh") is not None,
        "chromadb": importlib.util.find_spec("chromadb") is not None,
        "sentence_transformers": importlib.util.find_spec("sentence_transformers") is not None,
    }

    kb_root = None
    kb_configured = True
    try:
        kb_root = str(get_kb_root())
    except ConfigurationError:
        kb_configured = False

    install_hint = "uv tool install 'memex-kb[search]' (recommended) or pip install 'memex-kb[search]'"
    data = {
        "version": MEMEX_VERSION,
        "python": sys.executable,
        "kb_configured": kb_configured,
        "kb_root": kb_root,
        "deps": deps,
        "suggestion": None,
    }

    if not deps["whoosh"] or not deps["chromadb"] or not deps["sentence_transformers"]:
        data["suggestion"] = f"Install search deps for full functionality: {install_hint}"

    if as_json:
        output(data, as_json=True)
        return

    click.echo("mx doctor")
    click.echo("=" * 40)
    click.echo(f"version: {MEMEX_VERSION}")
    click.echo(f"python:  {sys.executable}")
    click.echo(f"kb:      {kb_root if kb_configured else '(not configured)'}")
    click.echo("")
    click.echo("deps")
    click.echo("-" * 40)
    click.echo(f"whoosh:               {'OK' if deps['whoosh'] else 'MISSING'}")
    click.echo(f"chromadb:             {'OK' if deps['chromadb'] else 'MISSING'}")
    click.echo(f"sentence-transformers: {'OK' if deps['sentence_transformers'] else 'MISSING'}")
    if data["suggestion"]:
        click.echo("")
        click.echo(f"Next step: {data['suggestion']}")


# ─────────────────────────────────────────────────────────────────────────────
# Prime Command (Agent Context Injection)
# ─────────────────────────────────────────────────────────────────────────────

PRIME_OUTPUT = """# Memex Knowledge Base

Search org knowledge before reinventing. Add discoveries for future agents.

## 5-minute onboarding (new KB)

1) `mx init` - create project KB in ./kb + .kbconfig
   (or `mx init --user` for personal KB in ~/.memex/kb)
2) `mx add --title="First Entry" --tags="docs" --category=guides --content="Hello KB"` - create entry
   Tip: set `.kbconfig` `primary: guides` to make `--category` optional.
3) `mx list --limit=5` - confirm entry path
4) `mx get guides/first-entry.md` - confirm read path
5) `mx health` - audit (orphans = entries with no incoming links; common early)

## Scope / config checks

- `mx info` - active KB paths + categories
- `mx context show` - project .kbconfig (primary + default tags)

## Required frontmatter

- `title`
- `tags`
- `created` (ISO timestamp or YYYY-MM-DD)

README.md inside the KB should include frontmatter or be excluded (prefix "_" or move it).
Quick fix: add frontmatter or rename it to _README.md.

Example frontmatter:

```yaml
---
title: My Entry
tags: [tag1, tag2]
created: 2026-02-06T00:00:00Z
---
```

## CLI Quick Reference

```bash
# Search (hybrid keyword + semantic)
mx search "deployment"              # Find entries
mx search "docker" --tags=infra     # Filter by tag
mx search "api" --mode=semantic     # Semantic only

# Read entries
mx get path/to/entry.md             # Full entry
mx get path/to/entry.md --metadata  # Just metadata

# Browse
mx tree                             # Directory structure
mx list --tags=infrastructure       # Filter by tag
mx tags                             # List all tags with counts
mx whats-new --days=7               # Recent changes
mx whats-new --scope=project        # Project KB only

# Contribute
mx add --title="My Entry" --tags="foo,bar" --category=tooling --content="..."
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
created: 2024-01-15T10:30:00
---

# Entry Title

Content with [[bidirectional links]] to other entries.
```

Use `[[path/to/entry.md|Display Text]]` for links.
"""


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option(
    "--compact",
    is_flag=True,
    help="Compact JSON output (omit large markdown content). Requires --json.",
)
@click.option(
    "--max-entries",
    type=int,
    default=4,
    show_default=True,
    help="Maximum relevant entries to include.",
)
@click.option(
    "--max-recent",
    type=int,
    default=5,
    show_default=True,
    help="Maximum recent entries to include.",
)
@click.option(
    "--max-bytes",
    type=int,
    help="Best-effort output size bound for compact JSON (bytes). Requires --compact.",
)
def prime(as_json: bool, compact: bool, max_entries: int, max_recent: int, max_bytes: int | None):
    """Output agent workflow context for session start.

    Designed for Claude Code hooks (SessionStart, PreCompact) to prevent
    agents from forgetting KB workflow after context compaction.

    \b
    Examples:
      mx prime              # Output context
      mx prime --json       # Output as JSON
      mx prime --json --compact  # Compact JSON for agent hooks
    """
    from .session_context import build_session_context

    if compact and not as_json:
        raise UsageError("--compact requires --json")
    if max_bytes is not None and not compact:
        raise UsageError("--max-bytes requires --compact")

    content = PRIME_OUTPUT
    session_result = build_session_context(max_entries=max_entries, recent_limit=max_recent)
    if session_result:
        content = session_result.content

    # Always compute a small context snapshot for JSON output (and as a fallback for
    # PRIME_OUTPUT when KB discovery works but session-context could not render).
    kb_root_value: str | None = None
    kb_scope_value: str | None = None
    primary_category_value: str | None = None
    context_file_value: str | None = None
    try:
        from .config import get_kb_root, get_project_kb_root, get_user_kb_root
        from .context import get_kb_context

        kb_root = get_kb_root()
        kb_root_value = str(kb_root)

        project_kb = get_project_kb_root()
        user_kb = get_user_kb_root()
        if project_kb and kb_root == project_kb:
            kb_scope_value = "project"
        elif user_kb and kb_root == user_kb:
            kb_scope_value = "user"

        kb_ctx = get_kb_context()
        if kb_ctx:
            primary_category_value = kb_ctx.primary or None
            context_file_value = str(kb_ctx.source_file) if kb_ctx.source_file else None
    except Exception:
        pass

    if session_result is None and kb_root_value:
        content = (
            content
            + "\n\n## Context snapshot\n\n"
            + f"write_kb={kb_root_value}"
            + (f" | scope={kb_scope_value}" if kb_scope_value else "")
            + (f" | primary={primary_category_value}" if primary_category_value else " | primary=(not set)")
            + (f" | context={context_file_value}" if context_file_value else "")
            + "\n"
        )

    # Help agents understand limitations when optional search deps aren't installed.
    import importlib.util

    deps = {
        "whoosh": importlib.util.find_spec("whoosh") is not None,
        "chromadb": importlib.util.find_spec("chromadb") is not None,
        "sentence_transformers": importlib.util.find_spec("sentence_transformers") is not None,
    }
    missing = [k for k, ok in deps.items() if not ok]
    if missing:
        content = (
            content
            + "\n\n## Search Dependencies\n\n"
            + f"Search deps are missing ({', '.join(missing)}).\n"
            + "Fallback: use `mx list --tags=<tag>` and `mx tree`.\n"
            + "Install: uv tool install 'memex-kb[search]' (recommended) or pip install 'memex-kb[search]'\n"
        )

    if as_json:
        if compact:
            payload: dict[str, Any] = {
                "kb_root": kb_root_value,
                "kb_scope": kb_scope_value,
                "primary_category": primary_category_value,
                "context_file": context_file_value,
                "missing_search_deps": missing,
            }
            if session_result:
                payload["project"] = session_result.project
                payload["entries"] = session_result.entries
                payload["recent_entries"] = session_result.recent_entries
                payload["cached"] = session_result.cached
            _emit_json(payload, compact=True, max_bytes=max_bytes)
        else:
            payload = {
                "content": content,
                "kb_root": kb_root_value,
                "kb_scope": kb_scope_value,
                "primary_category": primary_category_value,
                "context_file": context_file_value,
            }
            if session_result:
                payload["project"] = session_result.project
                payload["entries"] = session_result.entries
                payload["recent_entries"] = session_result.recent_entries
                payload["cached"] = session_result.cached
            output(payload, as_json=True)
    else:
        click.echo(content)


# ─────────────────────────────────────────────────────────────────────────────
# Session Context (Project-Relevant Hook Output)
# ─────────────────────────────────────────────────────────────────────────────


@cli.command("session-context")
@click.option(
    "--max-entries",
    type=int,
    default=4,
    show_default=True,
    help="Maximum relevant entries to include.",
)
@click.option(
    "--max-recent",
    type=int,
    default=5,
    show_default=True,
    help="Maximum recent entries to include.",
)
@click.option(
    "--install",
    is_flag=True,
    help="Update .claude/settings.json with mx session-context SessionStart hook",
)
@click.option(
    "--install-path",
    type=click.Path(path_type=Path),
    help="Custom path for Claude settings.json (implies --install).",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option(
    "--compact",
    is_flag=True,
    help="Compact JSON output (omit large markdown content). Requires --json.",
)
@click.option(
    "--max-bytes",
    type=int,
    help="Best-effort output size bound for compact JSON (bytes). Requires --compact.",
)
def session_context_command(
    max_entries: int,
    max_recent: int,
    install: bool,
    install_path: Path | None,
    as_json: bool,
    compact: bool,
    max_bytes: int | None,
):
    """Output dynamic project-relevant KB context for session hooks.

    Use --install to write the Claude Code SessionStart hook.
    """
    from .session_context import (
        build_session_context,
        default_settings_path,
        install_session_hook,
    )

    if compact and not as_json:
        raise UsageError("--compact requires --json")
    if max_bytes is not None and not compact:
        raise UsageError("--max-bytes requires --compact")

    if install or install_path:
        target = install_path or default_settings_path(Path.cwd())
        try:
            installed_path = install_session_hook(target)
        except ValueError as exc:
            raise ClickException(str(exc)) from exc
        payload = {
            "installed": True,
            "settings_path": str(installed_path),
            "command": "mx session-context",
        }
        if as_json:
            if compact:
                _emit_json(payload, compact=True, max_bytes=max_bytes)
            else:
                output(payload, as_json=True)
        else:
            click.echo(f"✓ Updated Claude settings at {installed_path}")
            click.echo('SessionStart hook set to: "mx session-context"')
        return

    result = build_session_context(max_entries=max_entries, recent_limit=max_recent)
    if not result:
        return

    if as_json:
        if compact:
            _emit_json(
                {
                    "project": result.project,
                    "entries": result.entries,
                    "recent_entries": result.recent_entries,
                    "cached": result.cached,
                },
                compact=True,
                max_bytes=max_bytes,
            )
        else:
            output(
                {
                    "project": result.project,
                    "entries": result.entries,
                    "recent_entries": result.recent_entries,
                    "content": result.content,
                    "cached": result.cached,
                },
                as_json=True,
            )
    else:
        click.echo(result.content)


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
def init(path: str | None, user: bool, force: bool, as_json: bool):
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
    from .frontmatter import build_frontmatter, create_new_metadata

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
    else:
        kb_path = Path(path) if path else Path.cwd() / LOCAL_KB_DIR

    # Check if already exists
    if kb_path.exists():
        if not force:
            scope_label = "User" if user else "Project"
            if as_json:
                output(
                    {
                        "error": f"{scope_label} KB already exists at {kb_path}",
                        "hint": "Use --force to reinitialize",
                    },
                    as_json=True,
                )
            else:
                click.echo(f"Error: {scope_label} KB already exists at {kb_path}", err=True)
                click.echo("Use --force to reinitialize.", err=True)
            sys.exit(1)

    # Create directory structure
    kb_path.mkdir(parents=True, exist_ok=True)

    # Default write directory for new entries so mx add doesn't warn on a fresh KB.
    default_primary = "inbox"
    (kb_path / default_primary).mkdir(parents=True, exist_ok=True)

    # Create README with scope-appropriate content
    readme_path = kb_path / "README.md"
    if user:
        readme_metadata = create_new_metadata(
            title="User Knowledge Base",
            tags=["kb", "meta", "user"],
        )
        readme_frontmatter = build_frontmatter(readme_metadata)
        readme_body = """# User Knowledge Base

This directory contains your personal knowledge base entries managed by `mx`.
This KB is available everywhere and is not shared with collaborators.

## Usage

```bash
mx add --title="Entry" --tags="tag1,tag2" --content="..." --scope=user
mx search "query" --scope=user
mx list --scope=user
```

## Structure

Entries are Markdown files with YAML frontmatter:

```markdown
---
title: Entry Title
tags: [tag1, tag2]
created: 2024-01-15T10:30:00
---

# Entry Title

Your content here.
```

## Scope

User KB entries are personal and available in all projects.
They are stored at ~/.memex/kb/ and are not committed to git.
"""
        readme_content = f"{readme_frontmatter}{readme_body}"
    else:
        readme_metadata = create_new_metadata(
            title="Project Knowledge Base",
            tags=["kb", "meta", "project"],
        )
        readme_frontmatter = build_frontmatter(readme_metadata)
        readme_body = """# Project Knowledge Base

This directory contains project-specific knowledge base entries managed by `mx`.
Commit this directory to share knowledge with collaborators.

## Usage

```bash
mx add --title="Entry" --tags="tag1,tag2" --content="..." --scope=project
mx search "query" --scope=project
mx list --scope=project
```

## Structure

Entries are Markdown files with YAML frontmatter:

```markdown
---
title: Entry Title
tags: [tag1, tag2]
created: 2024-01-15T10:30:00
---

# Entry Title

Your content here.
```

## Integration

Project KB entries take precedence over global KB entries in search results.
This keeps project-specific knowledge close to the code.
"""
        readme_content = f"{readme_frontmatter}{readme_body}"
    readme_path.write_text(readme_content, encoding="utf-8")

    # Create config file with scope-appropriate defaults
    if user:
        # User scope: config lives inside the KB directory
        config_path = kb_path / LOCAL_KB_CONFIG_FILENAME
        config_content = f"""# User KB Configuration
# This file marks this directory as your personal memex knowledge base

# Default write directory for new entries (relative to KB root)
primary: {default_primary}

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

# Default write directory for new entries (relative to KB root)
primary: {default_primary}

# Optional: boost these paths in search (glob patterns)
# boost_paths:
#   - {default_primary}/*
#   - reference/*

# Optional: exclude patterns from indexing (glob)
# exclude:
#   - "*.draft.md"
"""
    config_path.write_text(config_content, encoding="utf-8")

    # Output
    scope_label = "user" if user else "project"
    if user:
        files_created = ["kb/README.md", f"kb/{LOCAL_KB_CONFIG_FILENAME}"]
    else:
        files_created = [f"{kb_path.name}/README.md", ".kbconfig"]

    if as_json:
        output(
            {
                "created": str(kb_path),
                "config": str(config_path),
                "scope": scope_label,
                "files": files_created,
                "hint": "Use 'mx add' to add entries to this KB",
            },
            as_json=True,
        )
    else:
        click.echo(f"✓ Initialized {scope_label} KB at {kb_path}")
        click.echo(f"  Config: {config_path}")
        click.echo()
        click.echo("Next steps:")
        if user:
            click.echo('  mx add --title="Entry" --tags="..." --content="..." --scope=user')
            click.echo('  mx search "query" --scope=user')
        else:
            click.echo('  mx add --title="Entry" --tags="..." --content="..." --scope=project')
            click.echo('  mx search "query" --scope=project')


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
@click.option("--tag", "--tags", "tags", help="Filter by tags (comma-separated)")
@click.option("--mode", type=click.Choice(["hybrid", "keyword", "semantic"]), default="hybrid")
@click.option("--limit", "-n", default=10, type=click.IntRange(min=1), help="Max results")
@click.option(
    "--min-score",
    type=click.FloatRange(min=0.0, max=1.0),
    default=None,
    help=("Minimum score threshold (0.0-1.0). Scores: >=0.7 high, 0.4-0.7 moderate, <0.4 weak"),
)
@click.option("--content", is_flag=True, help="Include full content in results")
@click.option("--strict", is_flag=True, help="Disable semantic fallback for keyword mode")
@click.option("--terse", is_flag=True, help="Output paths only (one per line)")
@click.option("--full-titles", is_flag=True, help="Show full titles without truncation")
@click.option("--scope", type=click.Choice(["project", "user"]), help="Limit to specific KB scope")
@click.option(
    "--include-neighbors",
    is_flag=True,
    help="Include semantically linked entries, typed relations, and wikilinks",
)
@click.option(
    "--neighbor-depth",
    type=click.IntRange(min=1, max=5),
    default=1,
    help="Max hops for neighbor traversal (default 1)",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    tags: str | None,
    mode: str,
    limit: int,
    min_score: float | None,
    content: bool,
    strict: bool,
    terse: bool,
    full_titles: bool,
    scope: str | None,
    include_neighbors: bool,
    neighbor_depth: int,
    as_json: bool,
):
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

    The --include-neighbors flag enables graph-aware search by including entries
    that are semantically linked, connected via typed relations, or referenced by
    wikilinks from the direct matches. Use --neighbor-depth to control how many
    hops to traverse (default 1).

    \b
    Examples:
      mx search "deployment"
      mx search "docker" --tags=infrastructure
      mx search "api" --mode=semantic --limit=5
      mx search "config" --min-score=0.5          # Only confident results
      mx search "query" --strict                  # No semantic fallback
      mx search "query" --scope=project           # Project KB only
      mx search "query" --include-neighbors       # Include linked entries
      mx search "query" --include-neighbors --neighbor-depth=2

    \b
    See also:
      mx get  - Read a specific entry by path
      mx list - List entries with optional filters
    """
    from .config import ConfigurationError, get_kb_root
    from .core import expand_search_with_neighbors
    from .core import search as core_search

    try:
        get_kb_root()  # Validate KB is configured
    except ConfigurationError as exc:
        _handle_error(ctx, exc)

    # Validate query is not empty
    if not query or not query.strip():
        raise UsageError("Query cannot be empty.")

    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    # Cast mode to literal type (validated by click.Choice above)
    mode_literal = cast(Literal["hybrid", "keyword", "semantic"], mode)

    try:
        result = run_async(
            core_search(
                query=query,
                limit=limit,
                mode=mode_literal,
                tags=tag_list,
                include_content=content,
                strict=strict,
                scope=scope,
            )
        )
    except Exception as exc:
        # Most common case for first-time users: optional search deps not installed.
        try:
            from .errors import ErrorCode, MemexError

            if isinstance(exc, MemexError) and exc.code == ErrorCode.DEPENDENCY_MISSING:
                if exc.details.get("feature") == "search":
                    details = dict(exc.details or {})
                    suggestion = str(details.get("suggestion") or "").strip()
                    fallback = "Tip: Use 'mx list --tags=<tag>' as a fallback when search deps are unavailable."
                    details["suggestion"] = f"{suggestion}\n{fallback}" if suggestion else fallback
                    exc = MemexError(exc.code, exc.message, details)
        except Exception:
            pass
        _handle_error(ctx, exc, fallback_message="Search failed.")

    # Record search in history
    from . import search_history

    search_history.record_search(
        query=query,
        result_count=len(result.results),
        mode=mode,
        tags=tag_list,
    )

    # Apply min_score filter if specified
    filtered_results = result.results
    if min_score is not None:
        filtered_results = [r for r in result.results if r.score >= min_score]

    # Expand with neighbors if requested
    if include_neighbors:
        expanded_results = run_async(
            expand_search_with_neighbors(
                results=filtered_results,
                depth=neighbor_depth,
                include_content=content,
            )
        )

        if as_json:
            # JSON output with is_neighbor and linked_from fields
            results_data = []
            for item in expanded_results:
                result_item = {
                    "path": item["path"],
                    "title": item["title"],
                    "score": item["score"],
                    "confidence": _score_confidence(item["score"]),
                    "is_neighbor": item["is_neighbor"],
                }
                if item["is_neighbor"]:
                    result_item["linked_from"] = item.get("linked_from")
                if content and item.get("content"):
                    result_item["content"] = item["content"]
                else:
                    result_item["snippet"] = item.get("snippet", "")
                results_data.append(result_item)
            output({"results": results_data}, as_json=True)
        elif terse:
            for item in expanded_results:
                prefix = "[N] " if item["is_neighbor"] else ""
                click.echo(f"{prefix}{item['path']}")
        else:
            if not expanded_results:
                click.echo("No results found.")
                return

            rows = []
            for item in expanded_results:
                neighbor_mark = "*" if item["is_neighbor"] else ""
                rows.append(
                    {
                        "path": item["path"],
                        "title": item["title"],
                        "score": f"{item['score']:.2f}",
                        "conf": _score_confidence_short(item["score"]),
                        "nbr": neighbor_mark,
                    }
                )
            title_width = 10000 if full_titles else 30
            columns = ["path", "title", "score", "conf", "nbr"]
            widths = {"path": 40, "title": title_width}
            click.echo(format_table(rows, columns, widths))

            # Show full content below table when --content flag is used
            if content:
                click.echo("\n" + "=" * 60)
                for item in expanded_results:
                    neighbor_label = " [neighbor]" if item["is_neighbor"] else ""
                    click.echo(f"\n## {item['path']}{neighbor_label}")
                    click.echo("-" * 40)
                    if item.get("content"):
                        click.echo(item["content"])
                    else:
                        click.echo(item.get("snippet", ""))
                    click.echo()
    else:
        # Original output without neighbor expansion
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
                    click.echo(
                        f"No results above score threshold {min_score:.2f}. "
                        f"({len(result.results)} results filtered out)"
                    )
                else:
                    click.echo("No results found.")
                return

            rows = [
                {
                    "path": r.path,
                    "title": r.title,
                    "score": f"{r.score:.2f}",
                    "conf": _score_confidence_short(r.score),
                }
                for r in filtered_results
            ]
            title_width = 10000 if full_titles else 30
            click.echo(
                format_table(
                    rows,
                    ["path", "title", "score", "conf"],
                    {"path": 40, "title": title_width},
                )
            )

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
@click.option("--title", "by_title", help="Get entry by title instead of path")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON with metadata")
@click.option("--metadata", "-m", is_flag=True, help="Show only metadata")
@click.pass_context
def get(ctx: click.Context, path: str | None, by_title: str | None, as_json: bool, metadata: bool):
    """Read a knowledge base entry.

    \b
    Examples:
      mx get guides/quick-start.md
      mx get guides/quick-start.md --json
      mx get guides/quick-start.md --metadata
      mx get --title="Docker Guide"
      mx get --title "Python Tooling"

    \b
    See also:
      mx search - Search entries by query
      mx list   - List entries with optional filters
    """
    from .config import ConfigurationError
    from .errors import MemexError
    from .core import find_entries_by_title, get_entry, get_similar_titles

    # Validate that exactly one of path or --title is provided
    if path and by_title:
        raise UsageError("Cannot specify both PATH and --title")
    if not path and not by_title:
        raise UsageError("Must specify either PATH or --title")

    # If --title is used, find the entry by title
    if by_title:
        try:
            matches = run_async(find_entries_by_title(by_title))
        except ConfigurationError as exc:
            # Surface KB configuration guidance (no traceback, no generic fallback).
            _handle_error(ctx, exc)

        if len(matches) == 0:
            # No exact match - show suggestions
            try:
                suggestions = run_async(get_similar_titles(by_title))
            except ConfigurationError as exc:
                _handle_error(ctx, exc)
            if ctx.obj and ctx.obj.get("json_errors"):
                err = MemexError.entry_not_found(
                    by_title,
                    suggestion="Use mx list or mx get <path> for an exact path.",
                )
                if suggestions:
                    err.details["suggestions"] = suggestions
                _handle_error(ctx, err)
            else:
                click.echo(f"Error: No entry found with title '{by_title}'", err=True)
                if suggestions:
                    click.echo("\nDid you mean:", err=True)
                    for suggestion in suggestions:
                        click.echo(f"  - {suggestion}", err=True)
                sys.exit(1)

        if len(matches) > 1:
            # Multiple matches - show candidates
            candidate_paths = [m.get("path") for m in matches if m.get("path")]
            if ctx.obj and ctx.obj.get("json_errors"):
                err = MemexError.ambiguous_match(by_title, candidate_paths)
                _handle_error(ctx, err)
            else:
                click.echo(f"Error: Multiple entries found with title '{by_title}':", err=True)
                for match in matches:
                    click.echo(f"  - {match['path']}", err=True)
                click.echo("\nUse the full path to specify which entry.", err=True)
                sys.exit(1)

        # Single match - use its path
        path = matches[0]["path"]

    # At this point path is guaranteed to be set (either from arg or from matches)
    assert path is not None, "path must be set by this point"

    try:
        entry = run_async(get_entry(path=path))
    except ConfigurationError as exc:
        # Avoid misleading "Get failed." fallback; show standard KB guidance.
        _handle_error(ctx, exc)
    except Exception as e:
        message = str(e)
        if "not found" in message.lower():
            suggestions = _suggest_similar_paths(path)
            if ctx.obj and ctx.obj.get("json_errors"):
                err = MemexError.entry_not_found(
                    path,
                    suggestion="Use mx list to browse entries, or run mx tree to explore structure.",
                )
                if suggestions:
                    err.details["suggestions"] = suggestions
                _handle_error(ctx, err)
            else:
                click.echo(f"Error: {e}", err=True)
                if suggestions:
                    click.echo("\nDid you mean:", err=True)
                    for s in suggestions:
                        click.echo(f"  - {s}", err=True)
                    click.echo("\nTip: mx list --full-titles may help identify the right entry.", err=True)
                sys.exit(1)
        _handle_error(ctx, e, fallback_message="Get failed.")

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
# Relations Graph Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("path", required=False)
@click.option(
    "--depth",
    type=click.IntRange(min=0, max=5),
    default=1,
    help="Hops to traverse (default 1)",
)
@click.option(
    "--direction",
    type=click.Choice(["outgoing", "incoming", "both"]),
    default="both",
    help="Edge direction to traverse (default both)",
)
@click.option(
    "--origin",
    "origins",
    multiple=True,
    type=click.Choice(["wikilink", "semantic", "relations"]),
    help="Filter by edge origin (repeatable)",
)
@click.option("--type", "relation_types", multiple=True, help="Filter by relation type")
@click.option("--scope", type=click.Choice(["project", "user"]), help="Limit to KB scope")
@click.option("--graph", "full_graph", is_flag=True, help="Output full graph (JSON only)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def relations(
    ctx: click.Context,
    path: str | None,
    depth: int,
    direction: str,
    origins: tuple[str, ...],
    relation_types: tuple[str, ...],
    scope: str | None,
    full_graph: bool,
    as_json: bool,
):
    """Query the unified relations graph."""
    from .relations_graph import ensure_relations_graph, query_relations_graph

    _require_kb_configured(ctx, scope=scope)
    if full_graph:
        if not as_json:
            click.echo("Use --json with --graph to output the full relations graph.", err=True)
            sys.exit(1)
        graph = ensure_relations_graph(scope=scope)
        output(graph.model_dump(), as_json=True)
        return

    if not path:
        click.echo("Error: PATH is required unless --graph is specified.", err=True)
        sys.exit(1)

    assert path is not None  # narrowing for type checker after sys.exit

    origin_set = (
        cast(set[Literal["wikilink", "relations", "semantic"]], set(origins))
        if origins
        else None
    )
    type_set = set(relation_types) if relation_types else None

    try:
        result = query_relations_graph(
            path=path,
            depth=depth,
            direction=cast(Literal["outgoing", "incoming", "both"], direction),
            origin=origin_set,
            relation_types=type_set,
            scope=scope,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        output(result.model_dump(), as_json=True)
        return

    click.echo(f"Root:  {result.root}")
    click.echo(f"Nodes: {len(result.nodes)}")
    click.echo(f"Edges: {len(result.edges)}")

    if not result.edges:
        return

    rows = []
    for edge in result.edges:
        rows.append(
            {
                "source": edge.source,
                "target": edge.target,
                "origin": edge.origin,
                "type": edge.relation_type or "",
                "score": f"{edge.score:.3f}" if edge.score is not None else "",
            }
        )

    click.echo()
    click.echo(
        format_table(
            rows,
            ["source", "target", "origin", "type", "score"],
            {"source": 40, "target": 40},
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Relations Edit Commands
# ─────────────────────────────────────────────────────────────────────────────


@cli.command("relations-add")
@click.argument("path")
@click.option(
    "--relation",
    "relation_items",
    multiple=True,
    help="Typed relation as path=type (repeatable)",
)
@click.option(
    "--relations",
    "relations_json",
    help=(
        'Typed relations as JSON array (e.g., \'[{"path": "ref/other.md", "type": "implements"}]\')'
    ),
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def relations_add(
    path: str,
    relation_items: tuple[str, ...],
    relations_json: str | None,
    as_json: bool,
):
    """Add typed relations to an entry without replacing full frontmatter."""
    from .core import update_entry_relations

    relations = _parse_relations_inputs(relation_items, relations_json)
    if not relations:
        click.echo("Error: Provide at least one --relation or --relations entry", err=True)
        sys.exit(1)

    try:
        result = run_async(update_entry_relations(path=path, add=relations))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        output(result, as_json=True)
    else:
        scope = result.get("scope")
        path_display = f"@{scope}/{result['path']}" if scope else result["path"]
        click.echo(f"Updated relations: {path_display}")
        click.echo(f"  Added: {len(result.get('added', []))}")
        click.echo(f"  Total: {result.get('total', 0)}")


@cli.command("relations-remove")
@click.argument("path")
@click.option(
    "--relation",
    "relation_items",
    multiple=True,
    help="Typed relation as path=type (repeatable)",
)
@click.option(
    "--relations",
    "relations_json",
    help=(
        'Typed relations as JSON array (e.g., \'[{"path": "ref/other.md", "type": "implements"}]\')'
    ),
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def relations_remove(
    path: str,
    relation_items: tuple[str, ...],
    relations_json: str | None,
    as_json: bool,
):
    """Remove typed relations from an entry without replacing full frontmatter."""
    from .core import update_entry_relations

    relations = _parse_relations_inputs(relation_items, relations_json)
    if not relations:
        click.echo("Error: Provide at least one --relation or --relations entry", err=True)
        sys.exit(1)

    try:
        result = run_async(update_entry_relations(path=path, remove=relations))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        output(result, as_json=True)
    else:
        scope = result.get("scope")
        path_display = f"@{scope}/{result['path']}" if scope else result["path"]
        click.echo(f"Updated relations: {path_display}")
        click.echo(f"  Removed: {len(result.get('removed', []))}")
        click.echo(f"  Total: {result.get('total', 0)}")


# ─────────────────────────────────────────────────────────────────────────────
# Relations Lint Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command("relations-lint")
@click.option("--scope", type=click.Choice(["project", "user"]), help="Limit to KB scope")
@click.option("--strict", is_flag=True, help="Exit non-zero if issues are found")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def relations_lint(ctx: click.Context, scope: str | None, strict: bool, as_json: bool):
    """Lint typed relation types against the canonical taxonomy."""
    from .config import ConfigurationError, get_kb_root
    from .core import lint_relation_types as core_lint_relation_types

    try:
        get_kb_root()  # Validate KB is configured
    except ConfigurationError as exc:
        _handle_error(ctx, exc)

    result = run_async(core_lint_relation_types(scope=scope))

    if as_json:
        output(result, as_json=True)
    else:
        summary = result.get("summary", {})
        canonical_types = result.get("canonical_types", {}) or {}
        issues = result.get("issues", [])

        unknown_count = summary.get("unknown_count", 0)
        inconsistent_count = summary.get("inconsistent_count", 0)
        missing_count = summary.get("missing_count", 0)
        total_issues = unknown_count + inconsistent_count + missing_count

        click.echo("Relations Type Lint")
        click.echo("=" * 40)
        click.echo(f"Entries scanned: {summary.get('entries_scanned', 0)}")
        click.echo(f"Relations scanned: {summary.get('relations_scanned', 0)}")

        if canonical_types:
            canonical_list = ", ".join(sorted(canonical_types.keys()))
            click.echo(f"Canonical types: {canonical_list}")

        if total_issues == 0:
            click.echo("\n✓ No relation type issues found")
        else:
            click.echo(f"\n⚠ Issues found: {total_issues}")
            if unknown_count:
                click.echo(f"  - Unknown types: {unknown_count}")
            if inconsistent_count:
                click.echo(f"  - Inconsistent types: {inconsistent_count}")
            if missing_count:
                click.echo(f"  - Missing types: {missing_count}")

            max_show = 20
            issue_labels = [
                ("unknown", "Unknown types"),
                ("inconsistent", "Inconsistent types"),
                ("missing", "Missing types"),
            ]

            for issue_key, label in issue_labels:
                subset = [issue for issue in issues if issue.get("issue") == issue_key]
                if not subset:
                    continue

                click.echo(f"\n{label} ({len(subset)}):")
                for issue in subset[:max_show]:
                    scope_label = issue.get("scope")
                    path = issue.get("path", "")
                    path_display = f"@{scope_label}/{path}" if scope_label else path
                    target = issue.get("target", "")
                    rel_type = issue.get("type", "")
                    suggestion = issue.get("suggestion")

                    if issue_key == "inconsistent" and suggestion:
                        click.echo(
                            f"  - {path_display} -> {target} (type: {rel_type}, use {suggestion})"
                        )
                    elif issue_key == "missing":
                        click.echo(f"  - {path_display} -> {target} (type missing)")
                    else:
                        click.echo(f"  - {path_display} -> {target} (type: {rel_type})")

                if len(subset) > max_show:
                    click.echo(f"  ... and {len(subset) - max_show} more")

        if strict and total_issues:
            sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Add Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--title", required=True, help="Entry title")
@click.option("--tag", "--tags", "tags", required=True, help="Tags (comma-separated)")
@click.option("--category", default="", help="Category/directory")
@click.option("--content", help="Content (or use --file/--stdin)")
@click.option(
    "--file",
    "-f",
    "file_path",
    type=click.Path(exists=True),
    help="Read content from file",
)
@click.option("--stdin", is_flag=True, help="Read content from stdin")
@click.option(
    "--scope",
    type=click.Choice(["project", "user"]),
    help="Target KB scope (default: auto-detect)",
)
@click.option(
    "--semantic-links",
    "semantic_links_json",
    help=(
        "Semantic links as JSON array "
        '(e.g., \'[{"path": "ref/other.md", "score": 0.8, "reason": "related"}]\')'
    ),
)
@click.option(
    "--relation",
    "relation_items",
    multiple=True,
    help="Typed relation as path=type (repeatable)",
)
@click.option(
    "--relations",
    "relations_json",
    help=(
        'Typed relations as JSON array (e.g., \'[{"path": "ref/other.md", "type": "implements"}]\')'
    ),
)
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
    scope: str | None,
    semantic_links_json: str | None,
    relation_items: tuple[str, ...],
    relations_json: str | None,
    as_json: bool,
):
    """Create a new knowledge base entry.

    \b
    Examples:
      mx add --title="My Entry" --tags="foo,bar" --category=guides --content="# Content here"
      mx add --title="My Entry" --tags="foo,bar" --category=guides --file=content.md
      cat content.md | mx add --title="My Entry" --tags="foo,bar" --category=guides --stdin
      mx add --title="My Entry" --tags="foo" --category=guides --content="..." \\
        --semantic-links='[{"path": "ref/other.md", "score": 0.8, "reason": "related"}]'

    \b
    Category behavior:
      - If --category is omitted and .kbconfig sets primary, primary is used.
      - If --category is omitted and no primary is set, mx writes to KB root (.) and warns.

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
    Semantic Links:
      --semantic-links accepts a JSON array of link objects with:
        path:   Target entry path (required)
        score:  Similarity score 0-1 (required)
        reason: How link was discovered (required)

    \b
    Relations:
      --relation accepts path=type pairs (repeatable)
      --relations accepts a JSON array of relation objects with:
        path: Target entry path (required)
        type: Relation type (required)

    \b
    See also:
      mx append - Append content to existing entry (or create new)
    """
    from .context import get_kb_context
    from .core import add_entry
    from .errors import MemexError
    from .models import SemanticLink

    local_warnings: list[str] = []
    metadata_overrides: dict[str, Any] | None = None

    # Validate mutual exclusivity of content sources
    sources = sum([bool(content), bool(file_path), stdin])
    if sources > 1:
        _handle_error(
            ctx,
            MemexError.validation_error("Only one of --content, --file, or --stdin can be used"),
        )
    if sources == 0:
        _handle_error(ctx, MemexError.validation_error("Must provide --content, --file, or --stdin"))

    # Resolve content source
    if stdin:
        raw = sys.stdin.read()
        had_frontmatter, metadata_overrides, content = _extract_input_frontmatter_overrides(raw)
        if had_frontmatter:
            click.echo(
                "Warning: input already has YAML frontmatter; mx add will merge optional fields "
                "(e.g., description/status/aliases) and avoid duplicating frontmatter.",
                err=True,
            )
    elif file_path:
        raw = Path(file_path).read_text()
        had_frontmatter, metadata_overrides, content = _extract_input_frontmatter_overrides(raw)
        if had_frontmatter:
            click.echo(
                "Warning: input file already has YAML frontmatter; mx add will merge optional fields "
                "(e.g., description/status/aliases) and avoid duplicating frontmatter.",
                err=True,
            )
    elif content:
        # Decode escape sequences in --content (e.g., \n -> newline)
        raw = decode_escape_sequences(content)
        _, metadata_overrides, content = _extract_input_frontmatter_overrides(raw)

    # Content is guaranteed to be set by one of the branches above (validated earlier)
    assert content is not None, "content must be set by this point"

    tag_list = [t.strip() for t in tags.split(",")]
    context = get_kb_context()
    if not category and not (context and context.primary):
        warning = (
            "No --category provided; defaulting to KB root (.). "
            "Use --category=. to be explicit. "
            "Set a default in .kbconfig (see 'mx context show') or move the file later "
            "and run 'mx reindex'."
        )
        if as_json:
            local_warnings.append(warning)
        else:
            click.echo(f"Warning: {warning}", err=True)

    # Parse semantic links JSON if provided
    semantic_links: list[SemanticLink] | None = None
    if semantic_links_json:
        try:
            links_data = json.loads(semantic_links_json)
            if not isinstance(links_data, list):
                _handle_error(ctx, MemexError.validation_error("--semantic-links must be a JSON array"))
            semantic_links = []
            for i, link_data in enumerate(links_data):
                if not isinstance(link_data, dict):
                    _handle_error(
                        ctx,
                        MemexError.validation_error(f"--semantic-links[{i}] must be a JSON object"),
                    )
                missing = [f for f in ("path", "score", "reason") if f not in link_data]
                if missing:
                    _handle_error(
                        ctx,
                        MemexError.validation_error(
                            f"--semantic-links[{i}] missing required fields: {', '.join(missing)}",
                            details={"missing_fields": missing},
                        ),
                    )
                try:
                    semantic_links.append(
                        SemanticLink(
                            path=link_data["path"],
                            score=float(link_data["score"]),
                            reason=link_data["reason"],
                        )
                    )
                except (ValueError, TypeError) as e:
                    _handle_error(
                        ctx,
                        MemexError.validation_error(f"--semantic-links[{i}] invalid: {e}"),
                    )
        except json.JSONDecodeError as e:
            _handle_error(ctx, MemexError.validation_error(f"--semantic-links is not valid JSON: {e}"))

    relations = _parse_relations_inputs(relation_items, relations_json)

    try:
        result = run_async(
            add_entry(
                title=title,
                content=content,
                tags=tag_list,
                category=category,
                scope=scope,
                semantic_links=semantic_links,
                relations=relations,
                metadata_overrides=metadata_overrides,
            )
        )
    except Exception as e:
        _handle_add_error(ctx, e, tag_list)

    core_warnings: list[str] = []
    if isinstance(result, dict) and isinstance(result.get("warnings"), list):
        core_warnings = [str(w) for w in (result.get("warnings") or [])]

    combined_warnings = local_warnings + core_warnings
    if combined_warnings:
        result["warnings"] = combined_warnings

    if as_json:
        # Include scope in JSON output if explicitly set
        if scope:
            result["scope"] = scope
        output(result, as_json=True)
    else:
        quiet = ctx.obj.get("quiet", False) if ctx.obj else False
        if combined_warnings and not quiet:
            for w in combined_warnings:
                click.echo(f"Warning: {w}", err=True)

        # Show path with scope prefix if explicitly set
        path_display = f"@{scope}/{result['path']}" if scope else result["path"]
        click.echo(f"Created: {path_display}")
        if result.get("suggested_links"):
            click.echo("\nSuggested links:")
            for link in result["suggested_links"][:5]:
                click.echo(f"  - {link['path']} ({link['score']:.2f})")
        if result.get("suggested_tags"):
            click.echo("\nSuggested tags:")
            for tag in result["suggested_tags"][:5]:
                click.echo(f"  - {tag['tag']} ({tag['reason']})")

_log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Ingest Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--title", help="Override title (default: extract from H1 or filename)")
@click.option("--tag", "--tags", "tags", help="Tags (comma-separated, default: untagged)")
@click.option("--directory", help="Target directory within KB")
@click.option("--scope", type=click.Choice(["project", "user"]), help="Target KB scope")
@click.option("--dry-run", is_flag=True, help="Show what would be done without making changes")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def ingest(
    file: str,
    title: str | None,
    tags: str | None,
    directory: str | None,
    scope: str | None,
    dry_run: bool,
    as_json: bool,
):
    """Ingest a markdown file into the knowledge base.

    Takes an existing markdown file, prepends frontmatter if missing,
    and moves it to the KB directory if it's not already there.

    \b
    Examples:
      mx ingest notes.md                          # Ingest with auto-detected title/tags
      mx ingest draft.md --title="My Entry"       # Override title
      mx ingest doc.md --tags="api,docs"          # Set tags
      mx ingest doc.md --directory="guides"       # Place in guides/
      mx ingest doc.md --dry-run                  # Preview changes

    \b
    Behavior:
      - If file has frontmatter: preserves existing metadata
      - If file lacks frontmatter: prepends with auto-detected title
      - If file is outside KB: moves it into KB directory
      - If file is inside KB: updates in place

    \b
    Scope Selection:
      --scope=project  Ingest to project KB (./kb/)
      --scope=user     Ingest to user KB (~/.memex/kb/)
      (default)        Auto-detect based on current directory
    """
    from .core import ingest_file

    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    try:
        result = run_async(
            ingest_file(
                file_path=file,
                title=title,
                tags=tag_list,
                directory=directory,
                scope=scope,
                dry_run=dry_run,
            )
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        output(result.model_dump(), as_json=True)
    else:
        if dry_run:
            click.echo("[dry-run] Would ingest:")
        else:
            click.echo("Ingested:")

        click.echo(f"  Path: {result.path}")
        click.echo(f"  Title: {result.title}")
        click.echo(f"  Tags: {', '.join(result.tags)}")

        if result.moved:
            click.echo(f"  Moved from: {result.original_path}")
        if result.frontmatter_added:
            click.echo("  Frontmatter: added")

        if result.suggested_tags:
            click.echo("\nSuggested tags to add:")
            for tag in result.suggested_tags[:5]:
                click.echo(f"  - {tag['tag']} ({tag['reason']})")


# ─────────────────────────────────────────────────────────────────────────────
# Append Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("title")
@click.option("--content", help="Content to append (or use --file/--stdin)")
@click.option(
    "--file",
    "-f",
    "file_path",
    type=click.Path(exists=True),
    help="Read content from file",
)
@click.option("--stdin", is_flag=True, help="Read content from stdin")
@click.option("--tag", "--tags", "tags", help="Tags (comma-separated, required for new entries)")
@click.option("--category", help="Category for new entries")
@click.option("--no-create", is_flag=True, help="Error if entry not found (don't create)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def append(
    title: str,
    content: str | None,
    file_path: str | None,
    stdin: bool,
    tags: str | None,
    category: str | None,
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
    elif content:
        # Decode escape sequences in --content (e.g., \n -> newline)
        content = decode_escape_sequences(content)

    # Content is guaranteed to be set by one of the branches above (validated earlier)
    assert content is not None, "content must be set by this point"

    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    try:
        result = run_async(
            append_entry(
                title=title,
                content=content,
                tags=tag_list,
                category=category or "",
                no_create=no_create,
            )
        )
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

        if result.get("suggested_links"):
            click.echo("\nSuggested links:")
            for link in result["suggested_links"][:5]:
                click.echo(f"  - {link['path']} ({link['score']:.2f})")
        if result.get("suggested_tags"):
            click.echo("\nSuggested tags:")
            for tag in result["suggested_tags"][:5]:
                click.echo(f"  - {tag['tag']} ({tag['reason']})")


# ─────────────────────────────────────────────────────────────────────────────
# Replace Command (formerly 'update')
# ─────────────────────────────────────────────────────────────────────────────


@cli.command(name="replace")
@click.argument("path")
@click.option("--tag", "--tags", "tags", help="New tags (comma-separated)")
@click.option("--content", help="New content (replaces existing)")
@click.option(
    "--file",
    "-f",
    "file_path",
    type=click.Path(exists=True),
    help="Read content from file",
)
@click.option("--find", "find_flag", hidden=True, help="(Intent detection)")
@click.option("--replace", "replace_flag", hidden=True, help="(Intent detection)")
@click.option(
    "--semantic-links",
    "semantic_links_json",
    help=(
        "Semantic links as JSON array "
        '(e.g., \'[{"path": "ref/other.md", "score": 0.8, "reason": "related"}]\')'
    ),
)
@click.option(
    "--relation",
    "relation_items",
    multiple=True,
    help="Typed relation as path=type (repeatable)",
)
@click.option(
    "--relations",
    "relations_json",
    help=(
        'Typed relations as JSON array (e.g., \'[{"path": "ref/other.md", "type": "implements"}]\')'
    ),
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def replace_cmd(
    path: str,
    tags: str | None,
    content: str | None,
    file_path: str | None,
    find_flag: str | None,
    replace_flag: str | None,
    semantic_links_json: str | None,
    relation_items: tuple[str, ...],
    relations_json: str | None,
    as_json: bool,
):
    """Replace content or tags in a knowledge base entry.

    Overwrites entire content or tags. For surgical edits, use 'mx patch'.

    \b
    Examples:
      mx replace path/entry.md --tags="new,tags"
      mx replace path/entry.md --content="New content here"
      mx replace path/entry.md --file=updated-content.md
      mx replace path/entry.md --semantic-links='[{"path": "ref/related.md", "score": 0.9, \
"reason": "manual"}]'

    \b
    Semantic Links:
      --semantic-links accepts a JSON array of link objects with:
        path:   Target entry path (required)
        score:  Similarity score 0-1 (required)
        reason: How link was discovered (required)

    \b
    Relations:
      --relation accepts path=type pairs (repeatable)
      --relations accepts a JSON array of relation objects with:
        path: Target entry path (required)
        type: Relation type (required)

    \b
    See also:
      mx patch  - Apply surgical find-replace edits (keeps rest of content)
      mx append - Add content to end of entry (doesn't overwrite)
    """
    from .cli_intent import detect_update_intent_mismatch
    from .core import update_entry
    from .models import SemanticLink

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
    elif content:
        # Decode escape sequences in --content (e.g., \n -> newline)
        content = decode_escape_sequences(content)

    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    # Parse semantic links JSON if provided
    semantic_links: list[SemanticLink] | None = None
    if semantic_links_json:
        try:
            links_data = json.loads(semantic_links_json)
            if not isinstance(links_data, list):
                click.echo("Error: --semantic-links must be a JSON array", err=True)
                sys.exit(1)
            semantic_links = []
            for i, link_data in enumerate(links_data):
                if not isinstance(link_data, dict):
                    click.echo(f"Error: --semantic-links[{i}] must be a JSON object", err=True)
                    sys.exit(1)
                missing = [f for f in ("path", "score", "reason") if f not in link_data]
                if missing:
                    click.echo(
                        f"Error: --semantic-links[{i}] missing required fields: "
                        f"{', '.join(missing)}",
                        err=True,
                    )
                    sys.exit(1)
                try:
                    semantic_links.append(
                        SemanticLink(
                            path=link_data["path"],
                            score=float(link_data["score"]),
                            reason=link_data["reason"],
                        )
                    )
                except (ValueError, TypeError) as e:
                    click.echo(f"Error: --semantic-links[{i}] invalid: {e}", err=True)
                    sys.exit(1)
        except json.JSONDecodeError as e:
            click.echo(f"Error: --semantic-links is not valid JSON: {e}", err=True)
            sys.exit(1)

    relations = _parse_relations_inputs(relation_items, relations_json)

    try:
        result = run_async(
            update_entry(
                path=path,
                content=content,
                tags=tag_list,
                semantic_links=semantic_links,
                relations=relations,
            )
        )
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
@click.option("--tag", "--tags", "tags", help="New tags (comma-separated)")
@click.option("--content", help="New content")
@click.option(
    "--file",
    "-f",
    "file_path",
    type=click.Path(exists=True),
    help="Read content from file",
)
@click.option("--semantic-links", "semantic_links_json", help="Semantic links as JSON array")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def update_alias(ctx, path, tags, content, file_path, semantic_links_json, as_json):
    """(Deprecated: use 'mx replace' instead)"""
    ctx.invoke(
        replace_cmd,
        path=path,
        tags=tags,
        content=content,
        file_path=file_path,
        semantic_links_json=semantic_links_json,
        as_json=as_json,
    )


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
@click.option("--scope", type=click.Choice(["project", "user"]), help="Limit to specific KB scope")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def tree(ctx: click.Context, path: str, depth: int, scope: str | None, as_json: bool):
    """Display knowledge base directory structure.

    \b
    Examples:
      mx tree
      mx tree tooling --depth=2
      mx tree --scope=project              # Project KB only
    """
    from .core import tree as core_tree

    _require_kb_configured(ctx, scope=scope)
    result = run_async(core_tree(path=path, depth=depth, scope=scope))

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
@click.option("--tag", "--tags", help="Filter by tag")
@click.option("--category", help="Filter by category")
@click.option("--limit", "-n", default=20, help="Max results")
@click.option("--full-titles", is_flag=True, help="Show full titles without truncation")
@click.option("--scope", type=click.Choice(["project", "user"]), help="Limit to specific KB scope")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def list_entries(
    ctx: click.Context,
    tag: str | None,
    category: str | None,
    limit: int,
    full_titles: bool,
    scope: str | None,
    as_json: bool,
):
    """List knowledge base entries.

    \b
    Examples:
      mx list
      mx list --tags=tooling
      mx list --category=infrastructure --limit=10
      mx list --scope=project              # Project KB only
    """
    from .core import get_valid_categories
    from .core import list_entries as core_list_entries

    _require_kb_configured(ctx, scope=scope)
    try:
        result = run_async(core_list_entries(tag=tag, category=category, limit=limit, scope=scope))
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
                click.echo(
                    f"Error: Invalid category '{category}'. No categories exist yet.",
                    err=True,
                )
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
        if not full_titles:
            try:
                if any(len(str(e.get("title", ""))) > 40 for e in result):
                    click.echo("\nTip: Use --full-titles to see full values")
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# What's New Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command("whats-new")
@click.option("--days", "-d", default=30, help="Look back N days")
@click.option("--limit", "-n", default=10, help="Max results")
@click.option("--scope", type=click.Choice(["project", "user"]), help="Limit to specific KB scope")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def whats_new(ctx: click.Context, days: int, limit: int, scope: str | None, as_json: bool):
    """Show recently created or updated entries.

    \b
    Examples:
      mx whats-new
      mx whats-new --days=7 --limit=5
      mx whats-new --scope=project       # Project KB only
    """
    from .core import whats_new as core_whats_new

    _require_kb_configured(ctx, scope=scope)
    result = run_async(core_whats_new(days=days, limit=limit, scope=scope))

    if as_json:
        output(result, as_json=True)
    else:
        if not result:
            click.echo(f"No entries created or updated in the last {days} days.")
            return

        rows = [
            {"path": e["path"], "title": e["title"], "date": str(e["activity_date"])}
            for e in result
        ]
        click.echo(format_table(rows, ["path", "title", "date"], {"path": 40, "title": 30}))


# ─────────────────────────────────────────────────────────────────────────────
# Health Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def health(ctx: click.Context, as_json: bool):
    """Audit knowledge base for problems.

    Checks for orphaned entries, broken links, stale content, empty directories.

    \b
    Examples:
      mx health
      mx health --json
    """
    from .config import ConfigurationError, get_kb_root
    from .core import health as core_health

    try:
        get_kb_root()  # Validate KB is configured
    except ConfigurationError as exc:
        _handle_error(ctx, exc)

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
            click.echo(
                "  Note: orphans have no incoming links yet (no [[wikilinks]] or relations pointing at them)."
            )
            click.echo(
                "  Fix: add a link from an index/hub entry, or use `mx suggest-links path/to/entry.md` to connect."
            )
            click.echo("  Ignore: it's fine in a new KB; rerun after you add a few links.")
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
@click.pass_context
def tags(ctx: click.Context, min_count: int, as_json: bool):
    """List all tags with usage counts.

    \b
    Examples:
      mx tags
      mx tags --min-count=3
    """
    from .config import ConfigurationError, get_kb_root
    from .core import tags as core_tags

    try:
        get_kb_root()  # Validate KB is configured
    except ConfigurationError as exc:
        _handle_error(ctx, exc)

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
@click.pass_context
def hubs(ctx: click.Context, limit: int, as_json: bool):
    """Show most connected entries (hub notes).

    These are key concepts that many other entries link to.

    \b
    Examples:
      mx hubs
      mx hubs --limit=5
    """
    from .config import ConfigurationError, get_kb_root
    from .core import hubs as core_hubs

    try:
        get_kb_root()  # Validate KB is configured
    except ConfigurationError as exc:
        _handle_error(ctx, exc)

    result = run_async(core_hubs(limit=limit))

    if as_json:
        output(result, as_json=True)
    else:
        if not result:
            click.echo("No hub entries found.")
            return

        rows = [
            {
                "path": h["path"],
                "incoming": h["incoming"],
                "outgoing": h["outgoing"],
                "total": h["total"],
            }
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
@click.option("--scope", type=click.Choice(["project", "user"]), help="Limit to specific KB scope")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def reindex(ctx: click.Context, scope: str | None, as_json: bool):
    """Rebuild search indices from all markdown files.

    By default, indexes entries from both project and user KBs.

    \b
    Examples:
      mx reindex                # Index all KBs
      mx reindex --scope=project # Index project KB only
      mx reindex --json
    """
    from .config import ConfigurationError, get_kb_root
    from .core import reindex as core_reindex

    try:
        get_kb_root()  # Validate KB is configured
    except ConfigurationError as exc:
        _handle_error(ctx, exc)

    if not as_json:
        scope_msg = f"{scope} KB" if scope else "all KBs"
        click.echo(f"Reindexing {scope_msg}...")

    try:
        result = run_async(core_reindex(scope=scope))
    except Exception as exc:
        # Match mx search behavior: no traceback, but preserve the underlying message.
        _handle_error(ctx, exc)

    if as_json:
        output(
            {
                "kb_files": result.kb_files,
                "whoosh_docs": result.whoosh_docs,
                "chroma_docs": result.chroma_docs,
                "scope": scope,
            },
            as_json=True,
        )
    else:
        click.echo(
            f"✓ Indexed {result.kb_files} entries, {result.whoosh_docs} keyword docs, "
            f"{result.chroma_docs} semantic docs"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Context Command Group
# ─────────────────────────────────────────────────────────────────────────────


@cli.group(invoke_without_command=True)
@click.pass_context
def context(ctx):
    """Show or validate project KB context (.kbconfig file).

    The .kbconfig file configures KB behavior for a project:
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

    Searches for .kbconfig file starting from current directory.

    \b
    Examples:
      mx context show
      mx context show --json
    """
    from .context import get_kb_context

    ctx = get_kb_context()

    if ctx is None:
        if as_json:
            output({"found": False, "message": "No .kbconfig file found"}, as_json=True)
        else:
            click.echo("No .kbconfig file found.")
            click.echo("Run 'mx init' to create a project KB.")
        return

    if as_json:
        output(
            {
                "found": True,
                "source_file": str(ctx.source_file) if ctx.source_file else None,
                "primary": ctx.primary,
                "paths": ctx.paths,
                "default_tags": ctx.default_tags,
                "project": ctx.project,
            },
            as_json=True,
        )
    else:
        click.echo(f"Context file: {ctx.source_file}")
        click.echo(f"Primary:      {ctx.primary or '(not set)'}")
        if not ctx.primary:
            click.echo(
                "Hint: Set `primary` in .kbconfig (e.g., `primary: guides`) or pass `--category=...` to `mx add`."
            )
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
    """Validate the current .kbconfig file against the knowledge base.

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
            output({"valid": False, "error": "No .kbconfig file found"}, as_json=True)
        else:
            click.echo("Error: No .kbconfig file found.", err=True)
        sys.exit(1)

    # ctx is guaranteed non-None after the check above
    kb_root = get_kb_root()
    warnings = validate_context(ctx, kb_root)  # type: ignore[arg-type]

    assert ctx is not None
    if as_json:
        output(
            {
                "valid": True,
                "source_file": str(ctx.source_file),
                "warnings": warnings,
            },
            as_json=True,
        )
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
@click.pass_context
def delete(ctx: click.Context, path: str, force: bool, as_json: bool):
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
        _handle_error(ctx, e, fallback_message="Delete failed.")

    if as_json:
        output(result, as_json=True)
    else:
        quiet = ctx.obj.get("quiet", False) if ctx.obj else False
        if not quiet and isinstance(result, dict) and result.get("warnings"):
            for w in result.get("warnings") or []:
                click.echo(f"Warning: {w}", err=True)
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
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]+\b", content.lower())
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
    "--file",
    "-f",
    "file_path",
    type=click.Path(exists=True),
    help="Read content from file",
)
@click.option("--stdin", is_flag=True, help="Read content from stdin")
@click.option("--content", help="Raw content to add")
@click.option("--title", help="Override auto-detected title")
@click.option("--tag", "--tags", "tags", help="Override auto-suggested tags (comma-separated)")
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
        click.echo("Error: Provide --content, --file, or --stdin", err=True)
        sys.exit(1)

    assert content is not None
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

    assert content is not None
    # Auto-generate metadata
    auto_title = title or _extract_title_from_content(content)
    auto_tags = tags.split(",") if tags else _suggest_tags_from_content(content, existing_tags)
    auto_category = category or _suggest_category_from_content(content, valid_categories)

    # Ensure we have at least one tag
    if not auto_tags:
        auto_tags = ["uncategorized"]

    if as_json:
        # In JSON mode, output suggestions and let caller decide
        output(
            {
                "title": auto_title,
                "tags": auto_tags,
                "category": auto_category,
                "content_preview": content[:200] + "..." if len(content) > 200 else content,
                "categories_available": valid_categories,
            },
            as_json=True,
        )
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
        result = run_async(
            add_entry(
                title=auto_title,
                content=content,
                tags=auto_tags,
                category=auto_category,
            )
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    path = result.get("path") if isinstance(result, dict) else result.path
    click.echo(f"\nCreated: {path}")


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
      1. Project: .kbconfig templates: section
      2. User: ~/.config/memex/templates/*.yaml
      3. Built-in: troubleshooting, project, pattern, decision, runbook, api, meeting
    """
    from .templates import get_template, list_templates

    if action == "show":
        if not name:
            click.echo("Usage: mx templates show <name>", err=True)
            sys.exit(1)

        assert name is not None  # Checked above
        template = get_template(name)
        if not template:
            available = ", ".join(t.name for t in list_templates())
            click.echo(f"Unknown template: {name}", err=True)
            click.echo(f"Available: {available}", err=True)
            sys.exit(1)

        assert template is not None  # Checked above
        if as_json:
            output(
                {
                    "name": template.name,
                    "description": template.description,
                    "content": template.content,
                    "suggested_tags": template.suggested_tags,
                    "source": template.source,
                },
                as_json=True,
            )
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
        output(
            [
                {
                    "name": t.name,
                    "description": t.description,
                    "source": t.source,
                    "suggested_tags": t.suggested_tags,
                }
                for t in all_templates
            ],
            as_json=True,
        )
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
@click.pass_context
def info(ctx: click.Context, as_json: bool):
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
        get_project_kb_root,
        get_user_kb_root,
    )
    from .context import get_kb_context
    from .core import get_valid_categories
    from .parser import ParseError, parse_entry

    try:
        primary_kb = get_kb_root()
        index_root = get_index_root()
    except ConfigurationError as exc:
        _handle_error(ctx, exc)

    # Get all active KBs
    project_kb = get_project_kb_root()
    user_kb = get_user_kb_root()

    def scan_counts(kb_path: Path | None) -> tuple[int, int]:
        """Return (parsed_entries, parse_errors) using the same parsing rules as mx health."""
        if not kb_path or not kb_path.exists():
            return (0, 0)
        parsed = 0
        errors = 0
        for md_file in kb_path.rglob("*.md"):
            if md_file.name.startswith("_"):
                continue
            try:
                parse_entry(md_file)
            except ParseError:
                errors += 1
            else:
                parsed += 1
        return (parsed, errors)

    project_count, project_parse_errors = scan_counts(project_kb)
    user_count, user_parse_errors = scan_counts(user_kb)
    total_count = project_count + user_count

    primary_scope: str | None = None
    try:
        if project_kb and primary_kb == project_kb:
            primary_scope = "project"
        elif user_kb and primary_kb == user_kb:
            primary_scope = "user"
    except Exception:
        primary_scope = None

    categories = get_valid_categories()
    context = get_kb_context()
    primary_category = context.primary if context else None
    context_file = str(context.source_file) if context and context.source_file else None

    # Build payload
    kbs_info = []
    if project_kb:
        kbs_info.append(
            {
                "scope": "project",
                "path": str(project_kb),
                "entries": project_count,
                "parse_errors": project_parse_errors,
            }
        )
    if user_kb:
        kbs_info.append(
            {
                "scope": "user",
                "path": str(user_kb),
                "entries": user_count,
                "parse_errors": user_parse_errors,
            }
        )

    payload = {
        "primary_kb": str(primary_kb),
        "primary_scope": primary_scope,
        "index_root": str(index_root),
        "kbs": kbs_info,
        "total_entries": total_count,
        "categories": categories,
        "primary_category": primary_category,
        "context_file": context_file,
    }

    if as_json:
        output(payload, as_json=True)
        return

    click.echo("Memex Info")
    click.echo("=" * 40)
    click.echo(f"Primary KB: {primary_kb}")
    click.echo(f"Primary Scope: {primary_scope or '(unknown)'}")
    click.echo(f"Index Root: {index_root}")
    click.echo()
    click.echo("Active KBs:")
    if project_kb:
        extra = f", {project_parse_errors} parse errors" if project_parse_errors else ""
        click.echo(f"  project:  {project_kb} ({project_count} entries{extra})")
    else:
        click.echo("  project:  (none)")
    if user_kb:
        extra = f", {user_parse_errors} parse errors" if user_parse_errors else ""
        click.echo(f"  user:     {user_kb} ({user_count} entries{extra})")
    else:
        click.echo("  user:     (none)")
    click.echo()
    click.echo(f"Total:      {total_count} entries")
    if categories:
        click.echo(f"Categories: {', '.join(categories)}")
    click.echo(f"Primary:    {primary_category or '(not set)'}")
    if context_file:
        click.echo(f"Context:    {context_file}")
    if not primary_category:
        click.echo(
            "Tip: Set `primary` in .kbconfig to choose a default directory for `mx add`. See: `mx context show`."
        )


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

        assert entry is not None  # type narrowing after sys.exit

        # Re-run the search using the search command logic
        if not as_json:
            click.echo(f"Re-running: {entry.query}")
            if entry.tags:
                click.echo(f"  Tags: {', '.join(entry.tags)}")
            click.echo(f"  Mode: {entry.mode}")
            click.echo()

        # Import and run search
        from .core import search as core_search

        result = run_async(
            core_search(
                query=entry.query,
                limit=10,
                mode=cast(Literal["hybrid", "keyword", "semantic"], entry.mode),
                tags=entry.tags if entry.tags else None,
                include_content=False,
            )
        )

        # Record this re-run in history
        search_history.record_search(
            query=entry.query,
            result_count=len(result.results),
            mode=entry.mode,
            tags=entry.tags if entry.tags else None,
        )

        if as_json:
            output(
                [
                    {"path": r.path, "title": r.title, "score": r.score, "snippet": r.snippet}
                    for r in result.results
                ],
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
        output(
            [
                {
                    "position": i + 1,
                    "query": e.query,
                    "timestamp": e.timestamp.isoformat(),
                    "result_count": e.result_count,
                    "mode": e.mode,
                    "tags": e.tags,
                }
                for i, e in enumerate(entries)
            ],
            as_json=True,
        )
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
def batch(file_path: str | None, continue_on_error: bool):
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
        click.echo(
            json.dumps(
                {
                    "error": "Batch module not available",
                    "hint": "The batch module needs to be restored from a prior release",
                }
            ),
            err=True,
        )
        sys.exit(1)

    # Read input
    if file_path:
        lines = Path(file_path).read_text(encoding="utf-8").strip().split("\n")
    else:
        lines = sys.stdin.read().strip().split("\n")

    # Filter empty lines and comments
    commands = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

    if not commands:
        click.echo(json.dumps({"error": "No commands provided"}), err=True)
        sys.exit(1)

    result = run_async(run_batch(commands, continue_on_error=continue_on_error))
    click.echo(result.model_dump_json(indent=2))

    # Exit with error if any commands failed
    if result.failed > 0:
        sys.exit(1)


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
    try:
        from .core import generate_descriptions
    except ImportError:
        click.echo("Error: generate_descriptions function not available in core", err=True)
        click.echo("This function needs to be restored from a prior release", err=True)
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
# Eval Command
# ─────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--dataset",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("eval/queries.json"),
    show_default=True,
    help="Path to JSON dataset of queries and expected paths",
)
@click.option("--limit", "-k", default=5, type=click.IntRange(min=1), help="Top-k cutoff")
@click.option("--mode", type=click.Choice(["hybrid", "keyword", "semantic"]), default="hybrid")
@click.option("--scope", type=click.Choice(["project", "user"]), help="Limit to specific KB scope")
@click.option("--strict", is_flag=True, help="Use strict semantic threshold")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def eval(
    ctx: click.Context,
    dataset: Path,
    limit: int,
    mode: str,
    scope: str | None,
    strict: bool,
    as_json: bool,
):
    """Evaluate search accuracy against a query dataset."""
    try:
        from .core import search as core_search
        from .evaluation import aggregate_metrics, compute_metrics, load_eval_cases
    except Exception as exc:
        _handle_error(ctx, exc, fallback_message="Eval is unavailable (missing dependencies).")

    if not dataset.exists():
        raise UsageError(f"Dataset not found: {dataset}")

    try:
        cases = load_eval_cases(dataset)
    except ValueError as exc:
        raise UsageError(str(exc)) from exc

    results: list[dict] = []
    for case in cases:
        case_mode = case.mode or mode
        case_scope = case.scope or scope
        case_strict = case.strict if case.strict is not None else strict
        case_tags = case.tags

        try:
            result = run_async(
                core_search(
                    query=case.query,
                    limit=limit,
                    mode=cast(Literal["hybrid", "keyword", "semantic"], case_mode),
                    tags=case_tags,
                    include_content=False,
                    strict=case_strict,
                    scope=case_scope,
                )
            )
        except Exception as exc:
            _handle_error(ctx, exc, fallback_message="Evaluation search failed.")

        result_paths = [item.path for item in result.results]
        metrics = compute_metrics(result_paths, case.expected, limit)
        results.append(
            {
                "query": case.query,
                "expected": case.expected,
                "best_rank": metrics["best_rank"],
                "recall": metrics["recall"],
                "mrr": metrics["mrr"],
                "ndcg": metrics["ndcg"],
                "hit": metrics["hit"],
                "hits": metrics["hits"],
            }
        )

    summary = aggregate_metrics(results, limit)
    summary.update({"mode": mode, "scope": scope, "dataset": str(dataset)})

    if as_json:
        output({"summary": summary, "results": results}, as_json=True)
        return

    click.echo(
        f"Evaluation (k={limit}, mode={mode}, scope={scope or 'all'}) - {dataset}"
    )
    click.echo(
        f"Recall@{limit}: {summary['recall@k']:.2f}  "
        f"MRR: {summary['mrr']:.2f}  "
        f"nDCG@{limit}: {summary['ndcg@k']:.2f}  "
        f"Hit@{limit}: {summary['hit_rate@k']:.2f}  "
        f"Queries: {summary['queries']}"
    )

    rows = []
    for item in results:
        rows.append(
            {
                "query": item["query"],
                "best": item["best_rank"] or "-",
                "hit": "yes" if item["hit"] else "no",
            }
        )
    click.echo(format_table(rows, ["query", "best", "hit"], {"query": 50}))

    misses = [item for item in results if not item["hit"]]
    if misses:
        click.echo("\nMisses:")
        for item in misses:
            click.echo(f"- {item['query']} (expected: {', '.join(item['expected'])})")


# ─────────────────────────────────────────────────────────────────────────────
# Schema Command (Agent Introspection)
# ─────────────────────────────────────────────────────────────────────────────


def _build_schema() -> dict:
    """Build the complete CLI schema with agent-friendly metadata.

    Returns a dict containing all commands, their options, related commands,
    and common mistakes for agent introspection.
    """
    schema = {
        "version": MEMEX_VERSION,
        "description": "Token-efficient CLI for memex knowledge base",
        "commands": {
            "search": {
                "description": "Search the knowledge base with hybrid keyword + semantic search",
                "aliases": [],
                "arguments": [
                    {
                        "name": "query",
                        "required": True,
                        "description": "Search query text",
                    }
                ],
                "options": [
                    {
                        "name": "--tags",
                        "type": "string",
                        "description": "Filter by tags (comma-separated)",
                    },
                    {
                        "name": "--mode",
                        "type": "choice",
                        "choices": ["hybrid", "keyword", "semantic"],
                        "default": "hybrid",
                        "description": "Search mode",
                    },
                    {
                        "name": "--limit",
                        "short": "-n",
                        "type": "integer",
                        "default": 10,
                        "description": "Max results",
                    },
                    {
                        "name": "--min-score",
                        "type": "float",
                        "description": "Minimum score threshold (0.0-1.0)",
                    },
                    {
                        "name": "--content",
                        "type": "flag",
                        "description": "Include full content in results",
                    },
                    {
                        "name": "--strict",
                        "type": "flag",
                        "description": "Disable semantic fallback for keyword mode",
                    },
                    {
                        "name": "--terse",
                        "type": "flag",
                        "description": "Output paths only (one per line)",
                    },
                    {
                        "name": "--full-titles",
                        "type": "flag",
                        "description": "Show full titles without truncation",
                    },
                    {
                        "name": "--scope",
                        "type": "choice",
                        "choices": ["project", "user"],
                        "description": "Limit to specific KB scope",
                    },
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["get", "list"],
                "common_mistakes": {
                    "empty query": "Query cannot be empty. Provide a non-whitespace search term.",
                    "--tags without value": (
                        "Tags must be comma-separated, e.g., --tags=infra,docker"
                    ),
                },
                "examples": [
                    'mx search "deployment"',
                    'mx search "docker" --tags=infrastructure',
                    'mx search "api" --scope=project',
                ],
            },
            "get": {
                "description": "Read a knowledge base entry by path",
                "aliases": [],
                "arguments": [
                    {
                        "name": "path",
                        "required": True,
                        "description": "Path to entry relative to KB root",
                    }
                ],
                "options": [
                    {
                        "name": "--json",
                        "type": "flag",
                        "description": "Output as JSON with metadata",
                    },
                    {
                        "name": "--metadata",
                        "short": "-m",
                        "type": "flag",
                        "description": "Show only metadata",
                    },
                ],
                "related": ["search", "list"],
                "common_mistakes": {
                    "absolute path": "Use relative path from KB root, not absolute filesystem path",
                    "missing .md extension": (
                        "Include the .md extension: 'tooling/entry.md' not 'tooling/entry'"
                    ),
                },
                "examples": [
                    "mx get guides/quick-start.md",
                    "mx get guides/quick-start.md --metadata",
                ],
            },
            "add": {
                "description": "Create a new knowledge base entry",
                "aliases": [],
                "arguments": [],
                "options": [
                    {
                        "name": "--title",
                        "type": "string",
                        "required": True,
                        "description": "Entry title",
                    },
                    {
                        "name": "--tags",
                        "type": "string",
                        "required": True,
                        "description": "Tags (comma-separated)",
                    },
                    {
                        "name": "--category",
                        "type": "string",
                        "description": "Category/directory",
                    },
                    {
                        "name": "--content",
                        "type": "string",
                        "description": "Content (or use --file/--stdin)",
                    },
                    {
                        "name": "--file",
                        "short": "-f",
                        "type": "path",
                        "description": "Read content from file",
                    },
                    {"name": "--stdin", "type": "flag", "description": "Read content from stdin"},
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["append", "update"],
                "common_mistakes": {
                    "missing content source": "Must provide --content, --file, or --stdin",
                    "tags without value": 'Tags are required: --tags="tag1,tag2"',
                },
                "examples": [
                    'mx add --title="My Entry" --tags="foo,bar" --content="# Content"',
                    'mx add --title="My Entry" --tags="foo,bar" --file=content.md',
                ],
            },
            "append": {
                "description": (
                    "Append content to existing entry by title, or create new if not found"
                ),
                "aliases": [],
                "arguments": [
                    {
                        "name": "title",
                        "required": True,
                        "description": "Title of entry to append to (case-insensitive)",
                    }
                ],
                "options": [
                    {
                        "name": "--content",
                        "type": "string",
                        "description": "Content to append",
                    },
                    {
                        "name": "--file",
                        "short": "-f",
                        "type": "path",
                        "description": "Read content from file",
                    },
                    {"name": "--stdin", "type": "flag", "description": "Read content from stdin"},
                    {
                        "name": "--tags",
                        "type": "string",
                        "description": "Tags (required for new entries)",
                    },
                    {
                        "name": "--category",
                        "type": "string",
                        "description": "Category for new entries",
                    },
                    {
                        "name": "--no-create",
                        "type": "flag",
                        "description": "Error if entry not found",
                    },
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["add", "update", "patch"],
                "common_mistakes": {
                    "using path instead of title": (
                        "append takes title, not path. Use 'mx append \"Entry Title\"' "
                        "not 'mx append path/entry.md'"
                    ),
                    "missing content": "Must provide --content, --file, or --stdin",
                },
                "examples": [
                    'mx append "Daily Log" --content="Session summary"',
                    'mx append "API Docs" --file=api.md --tags="api,docs"',
                ],
            },
            "replace": {
                "description": "Replace content or tags in an existing entry (overwrites)",
                "aliases": ["update"],
                "arguments": [
                    {
                        "name": "path",
                        "required": True,
                        "description": "Path to entry relative to KB root",
                    }
                ],
                "options": [
                    {
                        "name": "--tags",
                        "type": "string",
                        "description": "New tags (comma-separated)",
                    },
                    {
                        "name": "--content",
                        "type": "string",
                        "description": "New content (replaces existing)",
                    },
                    {
                        "name": "--file",
                        "short": "-f",
                        "type": "path",
                        "description": "Read content from file",
                    },
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["patch", "append"],
                "common_mistakes": {
                    "confusing with append": (
                        "replace overwrites content. Use 'mx append' to add to existing content."
                    ),
                    "confusing with patch": (
                        "replace overwrites entire content. Use 'mx patch' for surgical "
                        "find-replace."
                    ),
                },
                "examples": [
                    'mx replace path/entry.md --tags="new,tags"',
                    "mx replace path/entry.md --file=updated-content.md",
                ],
            },
            "patch": {
                "description": "Apply surgical find-replace edits to a KB entry",
                "aliases": [],
                "arguments": [
                    {
                        "name": "path",
                        "required": True,
                        "description": "Path to entry relative to KB root",
                    }
                ],
                "options": [
                    {
                        "name": "--find",
                        "type": "string",
                        "description": "Exact text to find and replace",
                    },
                    {
                        "name": "--replace",
                        "type": "string",
                        "description": "Replacement text",
                    },
                    {
                        "name": "--find-file",
                        "type": "path",
                        "description": "Read --find text from file",
                    },
                    {
                        "name": "--replace-file",
                        "type": "path",
                        "description": "Read --replace text from file",
                    },
                    {
                        "name": "--replace-all",
                        "type": "flag",
                        "description": "Replace all occurrences",
                    },
                    {
                        "name": "--dry-run",
                        "type": "flag",
                        "description": "Preview changes without writing",
                    },
                    {
                        "name": "--backup",
                        "type": "flag",
                        "description": "Create .bak backup before patching",
                    },
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["replace", "append"],
                "common_mistakes": {
                    "--find without --replace": "Both --find and --replace are required",
                    "multiple matches without --replace-all": (
                        "If text matches multiple times, use --replace-all or provide more "
                        "context in --find"
                    ),
                    "using for append": (
                        "patch is for replacement. Use 'mx append' to add content to an entry."
                    ),
                },
                "exit_codes": {
                    "0": "Success",
                    "1": "Text not found",
                    "2": "Multiple matches (ambiguous, use --replace-all)",
                    "3": "File error (not found, permission, encoding)",
                },
                "examples": [
                    'mx patch tooling/notes.md --find "old text" --replace "new text"',
                    'mx patch tooling/notes.md --find "TODO" --replace "DONE" --replace-all',
                ],
            },
            "delete": {
                "description": "Delete a knowledge base entry",
                "aliases": [],
                "arguments": [
                    {
                        "name": "path",
                        "required": True,
                        "description": "Path to entry relative to KB root",
                    }
                ],
                "options": [
                    {
                        "name": "--force",
                        "short": "-f",
                        "type": "flag",
                        "description": "Delete even if has backlinks",
                    },
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": [],
                "common_mistakes": {
                    "deleting with backlinks": (
                        "Entries with backlinks require --force. Check backlinks first "
                        "with 'mx get path.md --metadata'"
                    ),
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
                    {"name": "--tags", "type": "string", "description": "Filter by tag"},
                    {
                        "name": "--category",
                        "type": "string",
                        "description": "Filter by category",
                    },
                    {
                        "name": "--limit",
                        "short": "-n",
                        "type": "integer",
                        "default": 20,
                        "description": "Max results",
                    },
                    {
                        "name": "--full-titles",
                        "type": "flag",
                        "description": "Show full titles without truncation",
                    },
                    {
                        "name": "--scope",
                        "type": "choice",
                        "choices": ["project", "user"],
                        "description": "Limit to specific KB scope",
                    },
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["search", "tree", "tags"],
                "common_mistakes": {
                    "invalid category": (
                        "Category must exist in KB. Use 'mx tree' to see valid categories."
                    ),
                },
                "examples": [
                    "mx list",
                    "mx list --tags=infrastructure",
                    "mx list --scope=project",
                ],
            },
            "tree": {
                "description": "Display knowledge base directory structure",
                "aliases": [],
                "arguments": [
                    {
                        "name": "path",
                        "required": False,
                        "default": "",
                        "description": "Starting path (default: root)",
                    }
                ],
                "options": [
                    {
                        "name": "--depth",
                        "short": "-d",
                        "type": "integer",
                        "default": 3,
                        "description": "Max depth",
                    },
                    {
                        "name": "--scope",
                        "type": "choice",
                        "choices": ["project", "user"],
                        "description": "Limit to specific KB scope",
                    },
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["list"],
                "common_mistakes": {},
                "examples": [
                    "mx tree",
                    "mx tree tooling --depth=2",
                    "mx tree --scope=project",
                ],
            },
            "tags": {
                "description": "List all tags with usage counts",
                "aliases": [],
                "arguments": [],
                "options": [
                    {
                        "name": "--min-count",
                        "type": "integer",
                        "default": 1,
                        "description": "Minimum usage count",
                    },
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
                "description": (
                    "Audit knowledge base for problems (orphans, broken links, stale content)"
                ),
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
                    {
                        "name": "--limit",
                        "short": "-n",
                        "type": "integer",
                        "default": 10,
                        "description": "Max results",
                    },
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
                    {
                        "name": "path",
                        "required": True,
                        "description": "Path to entry relative to KB root",
                    }
                ],
                "options": [
                    {
                        "name": "--limit",
                        "short": "-n",
                        "type": "integer",
                        "default": 5,
                        "description": "Max suggestions",
                    },
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
                    {
                        "name": "--days",
                        "short": "-d",
                        "type": "integer",
                        "default": 30,
                        "description": "Look back N days",
                    },
                    {
                        "name": "--limit",
                        "short": "-n",
                        "type": "integer",
                        "default": 10,
                        "description": "Max results",
                    },
                    {
                        "name": "--scope",
                        "type": "choice",
                        "choices": ["project", "user"],
                        "description": "Limit to specific KB scope",
                    },
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["list", "search"],
                "common_mistakes": {},
                "examples": [
                    "mx whats-new",
                    "mx whats-new --days=7 --limit=5",
                    "mx whats-new --scope=project",
                ],
            },
            "prime": {
                "description": "Output agent workflow context for session start (for hooks)",
                "aliases": [],
                "arguments": [],
                "options": [
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["schema"],
                "common_mistakes": {},
                "examples": [
                    "mx prime",
                    "mx prime --json",
                ],
            },
            "init": {
                "description": "Initialize a knowledge base (project or user scope)",
                "aliases": [],
                "arguments": [],
                "options": [
                    {
                        "name": "--path",
                        "short": "-p",
                        "type": "path",
                        "description": "Custom location for KB (default: kb/)",
                    },
                    {
                        "name": "--user",
                        "short": "-u",
                        "type": "flag",
                        "description": "Create user-scope KB at ~/.memex/kb/",
                    },
                    {
                        "name": "--force",
                        "short": "-f",
                        "type": "flag",
                        "description": "Reinitialize existing KB",
                    },
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
                "options": [
                    {
                        "name": "--scope",
                        "type": "choice",
                        "choices": ["project", "user"],
                        "description": "Limit to specific KB scope",
                    },
                    {"name": "--json", "type": "flag", "description": "Output as JSON"},
                ],
                "related": ["search"],
                "common_mistakes": {
                    "running unnecessarily": (
                        "Only needed after bulk imports or if search seems stale. "
                        "Normal operations auto-index."
                    ),
                },
                "examples": [
                    "mx reindex",
                    "mx reindex --scope=project",
                ],
            },
            "context": {
                "description": "Show or validate project KB context (.kbconfig file)",
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
                    {
                        "name": "--command",
                        "short": "-c",
                        "type": "string",
                        "description": "Show schema for specific command only",
                    },
                    {
                        "name": "--compact",
                        "type": "flag",
                        "description": "Minimal output (commands and options only)",
                    },
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
            {
                "name": "--json-errors",
                "type": "flag",
                "description": "Output errors as JSON (for programmatic use)",
            },
            {"name": "--version", "type": "flag", "description": "Show version"},
            {"name": "--help", "type": "flag", "description": "Show help"},
        ],
        "workflows": {
            "search_and_read": {
                "description": "Find and read an entry",
                "steps": ['mx search "query"', "mx get path/from/results.md"],
            },
            "create_entry": {
                "description": "Create a new KB entry",
                "steps": ['mx add --title="Title" --tags="tag1,tag2" --content="..."'],
            },
            "surgical_edit": {
                "description": "Make precise edits to existing content",
                "steps": [
                    "mx get path.md  # Read current content",
                    'mx patch path.md --find "old" --replace "new"',
                ],
            },
            "append_to_log": {
                "description": "Add content to an ongoing log entry",
                "steps": ['mx append "Log Title" --content="New entry..."'],
            },
        },
    }
    return schema


@cli.command()
@click.option("--command", "-c", "command_name", help="Show schema for specific command only")
@click.option("--compact", is_flag=True, help="Minimal output (commands and options only)")
def schema(command_name: str | None, compact: bool):
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


def _setup_github_actions_workflow(
    resolved_kb: Path,
    base_url: str,
    title: str,
    dry_run: bool,
) -> None:
    """Generate GitHub Actions workflow for KB publishing."""
    import subprocess

    # Find git root (run from KB directory to handle nested repos correctly)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
            cwd=resolved_kb,
        )
        git_root = Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        click.echo("Error: Not in a git repository", err=True)
        sys.exit(1)

    # Calculate KB path relative to git root
    try:
        kb_rel_path = resolved_kb.relative_to(git_root)
    except ValueError:
        click.echo(f"Error: KB path {resolved_kb} is not within git repo {git_root}", err=True)
        sys.exit(1)

    # Determine base URL - use shell expansion if not explicitly set
    if base_url:
        base_url_line = f"--base-url {base_url}"
    else:
        # Use GitHub's repo name variable for auto-detection
        # Note: ${...} is shell syntax for parameter expansion
        base_url_line = '--base-url "/${GITHUB_REPOSITORY#*/}"'

    # Generate workflow content
    workflow = f'''name: Publish KB to GitHub Pages

on:
  push:
    branches: [main]
    paths:
      - '{kb_rel_path}/**'
  workflow_dispatch:

# Allow only one concurrent deployment
concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install memex
        run: pip install memex-kb

      - name: Generate static site
        run: |
          mx publish --kb-root ./{kb_rel_path} -o _site {base_url_line} --title "{title}"

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: _site

  deploy:
    needs: build
    runs-on: ubuntu-latest
    permissions:
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: ${{{{ steps.deployment.outputs.page_url }}}}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
'''

    workflow_path = git_root / ".github" / "workflows" / "publish-kb.yml"

    if dry_run:
        click.echo("Would create workflow at:")
        click.echo(f"  {workflow_path}")
        click.echo("")
        click.echo("Workflow content:")
        click.echo("─" * 60)
        click.echo(workflow)
        click.echo("─" * 60)
        return

    # Create directory and write file
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(workflow)

    click.echo(f"Created GitHub Actions workflow: {workflow_path}")
    click.echo("")
    click.echo("Next steps:")
    click.echo("  1. Commit and push the workflow file")
    click.echo("  2. Go to Settings > Pages in your GitHub repo")
    click.echo("  3. Set 'Source' to 'GitHub Actions'")
    click.echo("")
    click.echo("The workflow will run on pushes to main that modify your KB.")


@cli.command()
@click.option(
    "--kb-root",
    "-k",
    type=click.Path(exists=True),
    help="KB source directory (overrides auto-detected KB)",
)
@click.option(
    "--scope",
    "-s",
    type=click.Choice(["project", "user"]),
    help="KB scope: project (from .kbconfig) or user (~/.memex/kb/)",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Skip confirmation prompt when publishing from user KB",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(),
    default="_site",
    help="Output directory (default: _site)",
)
@click.option(
    "--base-url",
    "-b",
    default="",
    help="Base URL for links (e.g., /my-kb for subdirectory hosting)",
)
@click.option(
    "--title",
    default="Memex",
    help="Site title for header and page titles (default: Memex)",
)
@click.option(
    "--index",
    "-i",
    "index_entry",
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
@click.option(
    "--setup-github-actions",
    is_flag=True,
    help="Create GitHub Actions workflow for auto-publishing to GitHub Pages",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview what would be done (for --setup-github-actions)",
)
@click.pass_context
def publish(
    ctx: click.Context,
    kb_root: str | None,
    scope: str | None,
    yes: bool,
    output_dir: str,
    base_url: str,
    title: str,
    index_entry: str | None,
    include_drafts: bool,
    include_archived: bool,
    no_clean: bool,
    as_json: bool,
    setup_github_actions: bool,
    dry_run: bool,
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
      2. --scope flag (project or user)
      3. project_kb in .kbconfig (relative to context file)

    \b
    Base URL resolution:
      1. --base-url flag (explicit)
      2. publish_base_url in .kbconfig (auto-applied)

    \b
    Landing page resolution:
      1. --index flag (explicit)
      2. publish_index_entry in .kbconfig (auto-applied)

    Use --base-url when hosting at a subdirectory (e.g., user.github.io/repo).
    Without it, links will 404. Configure in .kbconfig to avoid repeating:

    \b
      # .kbconfig
      project_kb: ./kb
      publish_base_url: /repo-name
      publish_index_entry: guides/welcome  # Landing page

    \b
    Examples:
      mx publish -o docs                   # Uses .kbconfig settings
      mx publish --kb-root ./kb -o docs    # Explicit KB source
      mx publish --scope=user -o docs      # Publish user KB
      mx publish --base-url /my-kb         # Subdirectory hosting

    \b
    GitHub Actions Setup:
      mx publish --setup-github-actions    # Create CI workflow
      mx publish --setup-github-actions --dry-run  # Preview workflow

    The --setup-github-actions flag creates .github/workflows/publish-kb.yml
    that auto-publishes your KB to GitHub Pages on push to main.
    """
    from .config import get_kb_root_by_scope
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
    elif scope:
        # Explicit --scope flag
        try:
            resolved_kb = get_kb_root_by_scope(scope)
            source_description = f"--scope={scope}"
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

        # Warn and require confirmation when publishing from user KB
        if scope == "user" and not yes:
            click.echo(
                f"Warning: You are about to publish content from your user KB at {resolved_kb}",
                err=True,
            )
            click.echo("This may include personal or private content.", err=True)
            click.echo("", err=True)
            if not click.confirm("Continue?", default=False):
                click.echo("Aborted.", err=True)
                sys.exit(0)
    elif context and context.project_kb and context.source_file:
        # Try .kbconfig project_kb
        project_kb_path = (context.source_file.parent / context.project_kb).resolve()
        if project_kb_path.exists():
            resolved_kb = project_kb_path
            source_description = ".kbconfig project_kb"

    # No KB found - show options
    if not resolved_kb:
        click.echo("Error: No KB found to publish", err=True)
        click.echo("", err=True)
        click.echo("Options:", err=True)
        click.echo("  - Use --scope=project (requires project_kb in .kbconfig)", err=True)
        click.echo("  - Use --scope=user for your personal KB", err=True)
        click.echo("  - Use --kb-root ./path/to/kb for an arbitrary directory", err=True)
        sys.exit(1)

    # Resolve base_url from context if not specified via CLI
    resolved_base_url = base_url
    if not resolved_base_url and context and context.publish_base_url:
        resolved_base_url = context.publish_base_url

    # Resolve index_entry from context if not specified via CLI
    resolved_index_entry = index_entry
    if not resolved_index_entry and context and context.publish_index_entry:
        resolved_index_entry = context.publish_index_entry

    # Handle --setup-github-actions
    if setup_github_actions:
        assert resolved_kb is not None  # Checked above (exits if None)
        _setup_github_actions_workflow(
            resolved_kb=resolved_kb,
            base_url=resolved_base_url,
            title=title,
            dry_run=dry_run,
        )
        return

    # Show confirmation message
    click.echo(f"Publishing from: {resolved_kb} (via {source_description})")
    if resolved_base_url:
        click.echo(f"Base URL: {resolved_base_url}")

    try:
        result = run_async(
            core_publish(
                output_dir=output_dir,
                base_url=resolved_base_url,
                site_title=title,
                index_entry=resolved_index_entry,
                include_drafts=include_drafts,
                include_archived=include_archived,
                clean=not no_clean,
                kb_root=resolved_kb,
            )
        )
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
# Command Aliases (hidden, for convenience)
# ─────────────────────────────────────────────────────────────────────────────


@cli.command("show", hidden=True)
@click.argument("path", required=False)
@click.option("--title", "by_title", help="Get entry by title instead of path")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON with metadata")
@click.option("--metadata", "-m", is_flag=True, help="Show only metadata")
@click.pass_context
def show_alias(ctx, path: str | None, by_title: str | None, as_json: bool, metadata: bool):
    """Alias for mx get."""
    ctx.invoke(get, path=path, by_title=by_title, as_json=as_json, metadata=metadata)


@cli.command("find", hidden=True)
@click.argument("query")
@click.option("--tag", "--tags", "tags", help="Filter by tags (comma-separated)")
@click.option("--mode", type=click.Choice(["hybrid", "keyword", "semantic"]), default="hybrid")
@click.option("--limit", "-n", default=10, type=click.IntRange(min=1), help="Max results")
@click.option(
    "--min-score",
    type=click.FloatRange(min=0.0, max=1.0),
    default=None,
    help="Minimum score threshold (0.0-1.0)",
)
@click.option("--content", is_flag=True, help="Include full content in results")
@click.option("--strict", is_flag=True, help="Disable semantic fallback for keyword mode")
@click.option("--terse", is_flag=True, help="Output paths only (one per line)")
@click.option("--full-titles", is_flag=True, help="Show full titles without truncation")
@click.option("--scope", type=click.Choice(["project", "user"]), help="Limit to specific KB scope")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def find_alias(
    ctx,
    query: str,
    tags: str | None,
    mode: str,
    limit: int,
    min_score: float | None,
    content: bool,
    strict: bool,
    terse: bool,
    full_titles: bool,
    scope: str | None,
    as_json: bool,
):
    """Alias for mx search."""
    ctx.invoke(
        search,
        query=query,
        tags=tags,
        mode=mode,
        limit=limit,
        min_score=min_score,
        content=content,
        strict=strict,
        terse=terse,
        full_titles=full_titles,
        scope=scope,
        as_json=as_json,
    )


@cli.command("recent", hidden=True)
@click.option("--days", "-d", default=30, help="Look back N days")
@click.option("--limit", "-n", default=10, help="Max results")
@click.option("--scope", type=click.Choice(["project", "user"]), help="Limit to specific KB scope")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def recent_alias(ctx, days: int, limit: int, scope: str | None, as_json: bool):
    """Alias for mx whats-new."""
    ctx.invoke(whats_new, days=days, limit=limit, scope=scope, as_json=as_json)


@cli.command("ls", hidden=True)
@click.option("--tag", "--tags", help="Filter by tag")
@click.option("--category", help="Filter by category")
@click.option("--limit", "-n", default=20, help="Max results")
@click.option("--full-titles", is_flag=True, help="Show full titles without truncation")
@click.option("--scope", type=click.Choice(["project", "user"]), help="Limit to specific KB scope")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def ls_alias(
    ctx,
    tag: str | None,
    category: str | None,
    limit: int,
    full_titles: bool,
    scope: str | None,
    as_json: bool,
):
    """Alias for mx list."""
    ctx.invoke(
        list_entries,
        tag=tag,
        category=category,
        limit=limit,
        full_titles=full_titles,
        scope=scope,
        as_json=as_json,
    )


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
