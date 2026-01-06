"""Structured error codes for programmatic handling.

This module provides typed error codes and structured error output for agents
and tools that need to distinguish error types to decide next actions.

Error codes use 1000+ range to avoid collision with system exit codes (0-255).
The CLI maps these to exit code 1 for shell compatibility while providing
full error details via --json-errors.

Usage:
    from memex.errors import MemexError, ErrorCode

    # Raise structured errors
    raise MemexError(
        ErrorCode.DUPLICATE_DETECTED,
        "Entry with title 'My Doc' already exists",
        details={"similar_entries": ["docs/my-doc.md"], "suggestion": "Use --force"}
    )

    # In CLI handlers, catch and format:
    except MemexError as e:
        if json_errors:
            click.echo(e.to_json(), err=True)
        else:
            click.echo(f"Error: {e.message}", err=True)
        sys.exit(1)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class ErrorCode(IntEnum):
    """Structured error codes for memex operations.

    Codes 1000+ avoid collision with system exit codes.
    Grouped by category for easy identification.
    """

    # Success (not an error, but included for completeness)
    SUCCESS = 0

    # Entry errors (1001-1099)
    DUPLICATE_DETECTED = 1001
    ENTRY_NOT_FOUND = 1002
    INVALID_PATH = 1003
    ENTRY_EXISTS = 1004
    PARSE_ERROR = 1005
    AMBIGUOUS_MATCH = 1006

    # Index/search errors (1101-1199)
    INDEX_UNAVAILABLE = 1101
    SEMANTIC_SEARCH_UNAVAILABLE = 1102
    SEARCH_FAILED = 1103

    # Configuration errors (1201-1299)
    KB_NOT_CONFIGURED = 1201
    INVALID_CATEGORY = 1202
    CONTEXT_NOT_FOUND = 1203

    # Validation errors (1301-1399)
    MISSING_REQUIRED_FIELD = 1301
    INVALID_CONTENT = 1302
    INVALID_TAGS = 1303
    VALIDATION_ERROR = 1304  # General input validation (mutually exclusive options, etc.)

    # File operation errors (1401-1499)
    FILE_READ_ERROR = 1401
    FILE_WRITE_ERROR = 1402
    PERMISSION_DENIED = 1403

    # Batch operation errors (1501-1599)
    BATCH_PARSE_ERROR = 1501
    BATCH_UNKNOWN_COMMAND = 1502
    BATCH_MISSING_ARGUMENT = 1503


# Human-readable names for error codes
ERROR_NAMES: dict[ErrorCode, str] = {
    ErrorCode.SUCCESS: "SUCCESS",
    ErrorCode.DUPLICATE_DETECTED: "DUPLICATE_DETECTED",
    ErrorCode.ENTRY_NOT_FOUND: "ENTRY_NOT_FOUND",
    ErrorCode.INVALID_PATH: "INVALID_PATH",
    ErrorCode.ENTRY_EXISTS: "ENTRY_EXISTS",
    ErrorCode.PARSE_ERROR: "PARSE_ERROR",
    ErrorCode.AMBIGUOUS_MATCH: "AMBIGUOUS_MATCH",
    ErrorCode.INDEX_UNAVAILABLE: "INDEX_UNAVAILABLE",
    ErrorCode.SEMANTIC_SEARCH_UNAVAILABLE: "SEMANTIC_SEARCH_UNAVAILABLE",
    ErrorCode.SEARCH_FAILED: "SEARCH_FAILED",
    ErrorCode.KB_NOT_CONFIGURED: "KB_NOT_CONFIGURED",
    ErrorCode.INVALID_CATEGORY: "INVALID_CATEGORY",
    ErrorCode.CONTEXT_NOT_FOUND: "CONTEXT_NOT_FOUND",
    ErrorCode.MISSING_REQUIRED_FIELD: "MISSING_REQUIRED_FIELD",
    ErrorCode.INVALID_CONTENT: "INVALID_CONTENT",
    ErrorCode.INVALID_TAGS: "INVALID_TAGS",
    ErrorCode.VALIDATION_ERROR: "VALIDATION_ERROR",
    ErrorCode.FILE_READ_ERROR: "FILE_READ_ERROR",
    ErrorCode.FILE_WRITE_ERROR: "FILE_WRITE_ERROR",
    ErrorCode.PERMISSION_DENIED: "PERMISSION_DENIED",
    ErrorCode.BATCH_PARSE_ERROR: "BATCH_PARSE_ERROR",
    ErrorCode.BATCH_UNKNOWN_COMMAND: "BATCH_UNKNOWN_COMMAND",
    ErrorCode.BATCH_MISSING_ARGUMENT: "BATCH_MISSING_ARGUMENT",
}


@dataclass
class MemexError(Exception):
    """Structured error with code and details for programmatic handling.

    Attributes:
        code: Numeric error code from ErrorCode enum.
        message: Human-readable error message.
        details: Optional dict with additional context (similar entries, suggestions, etc.).
    """

    code: ErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Set the exception message for standard exception handling
        super().__init__(self.message)

    @property
    def error_name(self) -> str:
        """Get the string name of the error code."""
        return ERROR_NAMES.get(self.code, f"ERROR_{self.code}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        result = {
            "error": self.error_name,
            "code": int(self.code),
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result

    def to_json(self, indent: int | None = None) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def duplicate_detected(
        cls,
        title: str,
        similar_entries: list[str],
        suggestion: str = "Use --force or mx update",
    ) -> "MemexError":
        """Create a DUPLICATE_DETECTED error with standard details."""
        return cls(
            code=ErrorCode.DUPLICATE_DETECTED,
            message=f"Entry with title '{title}' may already exist",
            details={
                "similar_entries": similar_entries,
                "suggestion": suggestion,
            },
        )

    @classmethod
    def entry_not_found(cls, path: str, suggestion: str | None = None) -> "MemexError":
        """Create an ENTRY_NOT_FOUND error."""
        details = {}
        if suggestion:
            details["suggestion"] = suggestion
        return cls(
            code=ErrorCode.ENTRY_NOT_FOUND,
            message=f"Entry not found: {path}",
            details=details,
        )

    @classmethod
    def invalid_path(cls, path: str, reason: str) -> "MemexError":
        """Create an INVALID_PATH error."""
        return cls(
            code=ErrorCode.INVALID_PATH,
            message=f"Invalid path: {path}",
            details={"reason": reason},
        )

    @classmethod
    def ambiguous_match(cls, query: str, matches: list[str]) -> "MemexError":
        """Create an AMBIGUOUS_MATCH error."""
        return cls(
            code=ErrorCode.AMBIGUOUS_MATCH,
            message=f"Ambiguous match for '{query}': {len(matches)} entries found",
            details={
                "matches": matches,
                "suggestion": "Specify full path or use more specific query",
            },
        )

    @classmethod
    def index_unavailable(cls, index_type: str = "full-text") -> "MemexError":
        """Create an INDEX_UNAVAILABLE error."""
        return cls(
            code=ErrorCode.INDEX_UNAVAILABLE,
            message=f"{index_type.title()} index is not available",
            details={"suggestion": "Run 'mx reindex' to rebuild the index"},
        )

    @classmethod
    def semantic_search_unavailable(cls) -> "MemexError":
        """Create a SEMANTIC_SEARCH_UNAVAILABLE error."""
        return cls(
            code=ErrorCode.SEMANTIC_SEARCH_UNAVAILABLE,
            message="Semantic search is not available",
            details={
                "reason": "ChromaDB or sentence-transformers not installed",
                "suggestion": "Install with: pip install 'memex[semantic]'",
            },
        )

    @classmethod
    def kb_not_configured(cls) -> "MemexError":
        """Create a KB_NOT_CONFIGURED error."""
        return cls(
            code=ErrorCode.KB_NOT_CONFIGURED,
            message="Knowledge base not configured",
            details={
                "suggestion": "Set MEMEX_KB_ROOT environment variable or run 'mx init'"
            },
        )

    @classmethod
    def missing_required_field(cls, field: str, suggestion: str | None = None) -> "MemexError":
        """Create a MISSING_REQUIRED_FIELD error."""
        details = {"field": field}
        if suggestion:
            details["suggestion"] = suggestion
        return cls(
            code=ErrorCode.MISSING_REQUIRED_FIELD,
            message=f"Missing required field: {field}",
            details=details,
        )

    @classmethod
    def validation_error(cls, message: str, details: dict[str, Any] | None = None) -> "MemexError":
        """Create a VALIDATION_ERROR for input validation failures."""
        return cls(
            code=ErrorCode.VALIDATION_ERROR,
            message=message,
            details=details or {},
        )


def format_error_json(
    code: ErrorCode,
    message: str,
    details: dict[str, Any] | None = None,
) -> str:
    """Format an error as JSON without raising an exception.

    Useful for converting existing exceptions to structured JSON output.
    """
    error = MemexError(code, message, details or {})
    return error.to_json()
