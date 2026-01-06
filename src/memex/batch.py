"""Batch operations for memex CLI.

Handles parsing and execution of multiple KB commands in a single invocation.
Reduces subprocess overhead for agents performing multiple KB operations.

Usage:
    mx batch << 'EOF'
    add --title='Note 1' --tags='a,b' --category=tooling --content='Hello world'
    update tooling/note-1.md --tags='a,b,updated'
    search 'api documentation'
    EOF
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any

from .errors import ErrorCode, MemexError
from .models import BatchOperationResult, BatchResponse


# Supported commands and their required arguments
SUPPORTED_COMMANDS = {
    "add": {"required": ["title", "tags"], "positional": []},
    "update": {"required": [], "positional": ["path"]},
    "upsert": {"required": [], "positional": ["title"]},
    "search": {"required": [], "positional": ["query"]},
    "get": {"required": [], "positional": ["path"]},
    "delete": {"required": [], "positional": ["path"]},
}

# Short option aliases
SHORT_OPTIONS = {
    "-t": "--title",
    "-c": "--content",
    "-d": "--directory",
    "-n": "--limit",
    "-f": "--force",
    "-m": "--metadata",
}


@dataclass
class ParsedCommand:
    """Parsed batch command ready for execution."""

    operation: str  # add, update, search, get, delete, upsert
    args: list[str] = field(default_factory=list)  # Positional arguments
    options: dict[str, str | bool] = field(default_factory=dict)  # Named options


class BatchParseError(Exception):
    """Error parsing batch command."""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.BATCH_PARSE_ERROR):
        self.message = message
        self.code = code
        super().__init__(message)


def parse_batch_command(line: str) -> ParsedCommand:
    """Parse a single batch command line into structured form.

    Args:
        line: A command line like "add --title='My Note' --tags='a,b' --content='Hello'"

    Returns:
        ParsedCommand with operation, positional args, and options.

    Raises:
        BatchParseError: If the command cannot be parsed.
    """
    line = line.strip()
    if not line:
        raise BatchParseError("Empty command")

    try:
        tokens = shlex.split(line)
    except ValueError as e:
        raise BatchParseError(f"Failed to parse command: {e}")

    if not tokens:
        raise BatchParseError("Empty command after parsing")

    operation = tokens[0].lower()
    if operation not in SUPPORTED_COMMANDS:
        raise BatchParseError(
            f"Unknown command: {operation}. Supported: {', '.join(SUPPORTED_COMMANDS.keys())}",
            code=ErrorCode.BATCH_UNKNOWN_COMMAND,
        )

    args: list[str] = []
    options: dict[str, str | bool] = {}

    i = 1
    while i < len(tokens):
        token = tokens[i]

        # Handle short options
        if token in SHORT_OPTIONS:
            token = SHORT_OPTIONS[token]

        if token.startswith("--"):
            # Long option: --key=value or --flag
            if "=" in token:
                key, _, value = token.partition("=")
                key = key.lstrip("-")
                options[key] = value
            else:
                # Boolean flag or option with space-separated value
                key = token.lstrip("-")
                # Check if next token is a value (not another option)
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    options[key] = tokens[i + 1]
                    i += 1
                else:
                    options[key] = True
        elif token.startswith("-") and len(token) == 2:
            # Short option with space-separated value: -t value
            expanded = SHORT_OPTIONS.get(token)
            if expanded:
                key = expanded.lstrip("-")
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    options[key] = tokens[i + 1]
                    i += 1
                else:
                    options[key] = True
            else:
                # Unknown short option, treat as positional
                args.append(token)
        else:
            # Positional argument
            args.append(token)

        i += 1

    return ParsedCommand(operation=operation, args=args, options=options)


def validate_command(cmd: ParsedCommand) -> None:
    """Validate that a parsed command has required arguments.

    Args:
        cmd: The parsed command to validate.

    Raises:
        BatchParseError: If required arguments are missing.
    """
    spec = SUPPORTED_COMMANDS[cmd.operation]

    # Check required options
    for required in spec["required"]:
        if required not in cmd.options:
            raise BatchParseError(
                f"{cmd.operation} requires --{required}",
                code=ErrorCode.BATCH_MISSING_ARGUMENT,
            )

    # Check required positional arguments
    required_positional = spec["positional"]
    if len(cmd.args) < len(required_positional):
        missing = required_positional[len(cmd.args) :]
        raise BatchParseError(
            f"{cmd.operation} requires positional argument(s): {', '.join(missing)}",
            code=ErrorCode.BATCH_MISSING_ARGUMENT,
        )


async def execute_add(cmd: ParsedCommand) -> dict:
    """Execute an add command."""
    from .core import add_entry

    title = str(cmd.options.get("title", ""))
    tags_str = str(cmd.options.get("tags", ""))
    tags = [t.strip() for t in tags_str.split(",") if t.strip()]
    content = str(cmd.options.get("content", ""))
    category = str(cmd.options.get("category", ""))
    directory = cmd.options.get("directory")
    force = cmd.options.get("force", False) is True or cmd.options.get("force") == "true"

    result = await add_entry(
        title=title,
        content=content,
        tags=tags,
        category=category,
        directory=str(directory) if directory else None,
        force=force,
    )

    return {
        "path": result.path,
        "created": result.created,
        "suggested_links": result.suggested_links,
        "suggested_tags": result.suggested_tags,
    }


async def execute_update(cmd: ParsedCommand) -> dict:
    """Execute an update command."""
    from .core import update_entry

    path = cmd.args[0] if cmd.args else ""
    content = cmd.options.get("content")
    tags_str = cmd.options.get("tags")
    tags = [t.strip() for t in str(tags_str).split(",") if t.strip()] if tags_str else None
    append = cmd.options.get("append", False) is True or cmd.options.get("append") == "true"

    result = await update_entry(
        path=path,
        content=str(content) if content else None,
        tags=tags,
        append=append,
    )

    return result


async def execute_upsert(cmd: ParsedCommand) -> dict:
    """Execute an upsert command."""
    from .core import upsert_entry

    title = cmd.args[0] if cmd.args else ""
    content = str(cmd.options.get("content", ""))
    tags_str = cmd.options.get("tags")
    tags = [t.strip() for t in str(tags_str).split(",") if t.strip()] if tags_str else None
    directory = cmd.options.get("directory")
    no_timestamp = cmd.options.get("no-timestamp", False) is True
    replace = cmd.options.get("replace", False) is True or cmd.options.get("replace") == "true"

    result = await upsert_entry(
        title=title,
        content=content,
        tags=tags,
        directory=str(directory) if directory else None,
        append=not replace,
        timestamp=not no_timestamp,
    )

    return {
        "path": result.path,
        "action": result.action,
        "title": result.title,
        "matched_by": result.matched_by,
    }


async def execute_search(cmd: ParsedCommand) -> dict:
    """Execute a search command."""
    from .core import search

    query = cmd.args[0] if cmd.args else ""
    tags_str = cmd.options.get("tags")
    tags = [t.strip() for t in str(tags_str).split(",") if t.strip()] if tags_str else None
    mode = str(cmd.options.get("mode", "hybrid"))
    if mode not in ("hybrid", "keyword", "semantic"):
        mode = "hybrid"
    limit = int(cmd.options.get("limit", 10))
    include_content = cmd.options.get("content", False) is True

    result = await search(
        query=query,
        tags=tags,
        mode=mode,  # type: ignore
        limit=limit,
        include_content=include_content,
    )

    return {
        "results": [
            {
                "path": r.path,
                "title": r.title,
                "score": r.score,
                "match_type": r.match_type,
            }
            for r in result.results
        ],
        "warnings": result.warnings,
    }


async def execute_get(cmd: ParsedCommand) -> dict:
    """Execute a get command."""
    from .core import get_entry

    path = cmd.args[0] if cmd.args else ""
    include_metadata = cmd.options.get("metadata", False) is True

    result = await get_entry(path=path)

    response: dict[str, Any] = {
        "path": result.path,
        "content": result.content,
    }
    if include_metadata:
        response["metadata"] = result.metadata.model_dump()
        response["links"] = result.links
        response["backlinks"] = result.backlinks

    return response


async def execute_delete(cmd: ParsedCommand) -> dict:
    """Execute a delete command."""
    from .core import delete_entry

    path = cmd.args[0] if cmd.args else ""
    force = cmd.options.get("force", False) is True or cmd.options.get("force") == "true"

    result = await delete_entry(path=path, force=force)
    return result


# Command executor dispatch table
COMMAND_EXECUTORS = {
    "add": execute_add,
    "update": execute_update,
    "upsert": execute_upsert,
    "search": execute_search,
    "get": execute_get,
    "delete": execute_delete,
}


async def execute_batch_command(cmd: ParsedCommand) -> tuple[bool, Any]:
    """Execute a parsed batch command.

    Args:
        cmd: The parsed command to execute.

    Returns:
        Tuple of (success: bool, result_or_error: Any).
        On success, result is the operation-specific result dict.
        On failure, result is an error dict with error name, code, and message.
    """
    try:
        validate_command(cmd)
        executor = COMMAND_EXECUTORS[cmd.operation]
        result = await executor(cmd)
        return True, result
    except BatchParseError as e:
        return False, {
            "error": e.code.name if hasattr(e.code, "name") else str(e.code),
            "code": int(e.code),
            "message": e.message,
        }
    except MemexError as e:
        return False, e.to_dict()
    except ValueError as e:
        return False, {
            "error": "VALIDATION_ERROR",
            "code": int(ErrorCode.INVALID_CONTENT),
            "message": str(e),
        }
    except FileNotFoundError as e:
        return False, {
            "error": "ENTRY_NOT_FOUND",
            "code": int(ErrorCode.ENTRY_NOT_FOUND),
            "message": str(e),
        }
    except Exception as e:
        return False, {
            "error": "UNEXPECTED_ERROR",
            "code": 1000,
            "message": str(e),
        }


async def run_batch(
    commands: list[str],
    continue_on_error: bool = True,
) -> BatchResponse:
    """Execute a batch of commands and collect results.

    Args:
        commands: List of command strings to execute.
        continue_on_error: If True, continue processing after errors.
                          If False, stop at first error.

    Returns:
        BatchResponse with total, succeeded, failed counts and per-operation results.
    """
    results: list[BatchOperationResult] = []
    succeeded = 0
    failed = 0

    for index, command in enumerate(commands):
        # Parse the command
        try:
            parsed = parse_batch_command(command)
        except BatchParseError as e:
            failed += 1
            results.append(
                BatchOperationResult(
                    index=index,
                    command=command,
                    success=False,
                    error={
                        "error": e.code.name if hasattr(e.code, "name") else str(e.code),
                        "code": int(e.code),
                        "message": e.message,
                    },
                )
            )
            if not continue_on_error:
                break
            continue

        # Execute the command
        success, result = await execute_batch_command(parsed)

        if success:
            succeeded += 1
            results.append(
                BatchOperationResult(
                    index=index,
                    command=command,
                    success=True,
                    result=result,
                )
            )
        else:
            failed += 1
            results.append(
                BatchOperationResult(
                    index=index,
                    command=command,
                    success=False,
                    error=result,
                )
            )
            if not continue_on_error:
                break

    return BatchResponse(
        total=len(results),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )
