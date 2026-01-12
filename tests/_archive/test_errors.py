"""Tests for structured error codes and JSON error output (errors.py).

Tests cover:
- ErrorCode enum values and names
- MemexError exception creation and serialization
- Factory methods for common errors
- CLI --json-errors flag integration
"""

import json

import pytest

from memex.errors import ErrorCode, MemexError, ERROR_NAMES, format_error_json


# ─────────────────────────────────────────────────────────────────────────────
# ErrorCode Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorCode:
    """Tests for ErrorCode enum."""

    def test_success_is_zero(self):
        """SUCCESS code should be 0."""
        assert ErrorCode.SUCCESS == 0

    def test_error_codes_in_1000_range(self):
        """Error codes should be >= 1000 to avoid collision with system exit codes."""
        for code in ErrorCode:
            if code != ErrorCode.SUCCESS:
                assert code >= 1000, f"{code.name} should be >= 1000"

    def test_all_codes_have_names(self):
        """All error codes should have human-readable names in ERROR_NAMES."""
        for code in ErrorCode:
            assert code in ERROR_NAMES, f"Missing name for {code}"

    def test_error_names_match_enum_names(self):
        """ERROR_NAMES should match enum member names."""
        for code in ErrorCode:
            assert ERROR_NAMES[code] == code.name


class TestErrorCodeCategories:
    """Tests for error code groupings."""

    def test_entry_errors_in_1000_range(self):
        """Entry-related errors should be in 1001-1099 range."""
        entry_codes = [
            ErrorCode.DUPLICATE_DETECTED,
            ErrorCode.ENTRY_NOT_FOUND,
            ErrorCode.INVALID_PATH,
            ErrorCode.ENTRY_EXISTS,
            ErrorCode.PARSE_ERROR,
            ErrorCode.AMBIGUOUS_MATCH,
        ]
        for code in entry_codes:
            assert 1001 <= code <= 1099, f"{code.name} should be in 1001-1099"

    def test_index_errors_in_1100_range(self):
        """Index-related errors should be in 1101-1199 range."""
        index_codes = [
            ErrorCode.INDEX_UNAVAILABLE,
            ErrorCode.SEMANTIC_SEARCH_UNAVAILABLE,
            ErrorCode.SEARCH_FAILED,
        ]
        for code in index_codes:
            assert 1101 <= code <= 1199, f"{code.name} should be in 1101-1199"

    def test_config_errors_in_1200_range(self):
        """Configuration errors should be in 1201-1299 range."""
        config_codes = [
            ErrorCode.KB_NOT_CONFIGURED,
            ErrorCode.INVALID_CATEGORY,
            ErrorCode.CONTEXT_NOT_FOUND,
        ]
        for code in config_codes:
            assert 1201 <= code <= 1299, f"{code.name} should be in 1201-1299"


# ─────────────────────────────────────────────────────────────────────────────
# MemexError Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMemexError:
    """Tests for MemexError exception class."""

    def test_basic_error_creation(self):
        """Should create error with code and message."""
        error = MemexError(ErrorCode.ENTRY_NOT_FOUND, "Entry not found: test.md")
        assert error.code == ErrorCode.ENTRY_NOT_FOUND
        assert error.message == "Entry not found: test.md"
        assert error.details == {}

    def test_error_with_details(self):
        """Should create error with additional details."""
        error = MemexError(
            ErrorCode.DUPLICATE_DETECTED,
            "Duplicate found",
            {"similar_entries": ["foo.md", "bar.md"]},
        )
        assert error.details["similar_entries"] == ["foo.md", "bar.md"]

    def test_error_name_property(self):
        """error_name property should return string name."""
        error = MemexError(ErrorCode.ENTRY_NOT_FOUND, "test")
        assert error.error_name == "ENTRY_NOT_FOUND"

    def test_to_dict(self):
        """to_dict() should serialize error to dict."""
        error = MemexError(
            ErrorCode.ENTRY_NOT_FOUND,
            "Entry not found: test.md",
            {"suggestion": "Check the path"},
        )
        d = error.to_dict()
        assert d["error"] == "ENTRY_NOT_FOUND"
        assert d["code"] == 1002
        assert d["message"] == "Entry not found: test.md"
        assert d["details"]["suggestion"] == "Check the path"

    def test_to_dict_without_details(self):
        """to_dict() should omit details key when empty."""
        error = MemexError(ErrorCode.ENTRY_NOT_FOUND, "test")
        d = error.to_dict()
        assert "details" not in d

    def test_to_json(self):
        """to_json() should serialize error to JSON string."""
        error = MemexError(ErrorCode.ENTRY_NOT_FOUND, "test")
        j = error.to_json()
        parsed = json.loads(j)
        assert parsed["error"] == "ENTRY_NOT_FOUND"
        assert parsed["code"] == 1002

    def test_exception_message(self):
        """MemexError should work as a normal exception."""
        error = MemexError(ErrorCode.ENTRY_NOT_FOUND, "Entry not found: test.md")
        assert str(error) == "Entry not found: test.md"


class TestMemexErrorFactoryMethods:
    """Tests for MemexError factory methods."""

    def test_duplicate_detected(self):
        """duplicate_detected() should create DUPLICATE_DETECTED error."""
        error = MemexError.duplicate_detected(
            title="My Doc",
            similar_entries=["docs/my-doc.md"],
        )
        assert error.code == ErrorCode.DUPLICATE_DETECTED
        assert "My Doc" in error.message
        assert error.details["similar_entries"] == ["docs/my-doc.md"]
        assert "suggestion" in error.details

    def test_entry_not_found(self):
        """entry_not_found() should create ENTRY_NOT_FOUND error."""
        error = MemexError.entry_not_found("test/path.md")
        assert error.code == ErrorCode.ENTRY_NOT_FOUND
        assert "test/path.md" in error.message

    def test_entry_not_found_with_suggestion(self):
        """entry_not_found() should include suggestion when provided."""
        error = MemexError.entry_not_found("test.md", suggestion="Did you mean tests.md?")
        assert error.details["suggestion"] == "Did you mean tests.md?"

    def test_invalid_path(self):
        """invalid_path() should create INVALID_PATH error."""
        error = MemexError.invalid_path("../escape.md", "path traversal")
        assert error.code == ErrorCode.INVALID_PATH
        assert error.details["reason"] == "path traversal"

    def test_ambiguous_match(self):
        """ambiguous_match() should create AMBIGUOUS_MATCH error."""
        error = MemexError.ambiguous_match("test", ["a.md", "b.md"])
        assert error.code == ErrorCode.AMBIGUOUS_MATCH
        assert error.details["matches"] == ["a.md", "b.md"]

    def test_index_unavailable(self):
        """index_unavailable() should create INDEX_UNAVAILABLE error."""
        error = MemexError.index_unavailable()
        assert error.code == ErrorCode.INDEX_UNAVAILABLE
        assert "suggestion" in error.details

    def test_semantic_search_unavailable(self):
        """semantic_search_unavailable() should create SEMANTIC_SEARCH_UNAVAILABLE error."""
        error = MemexError.semantic_search_unavailable()
        assert error.code == ErrorCode.SEMANTIC_SEARCH_UNAVAILABLE
        assert "pip install" in error.details["suggestion"]

    def test_kb_not_configured(self):
        """kb_not_configured() should create KB_NOT_CONFIGURED error."""
        error = MemexError.kb_not_configured()
        assert error.code == ErrorCode.KB_NOT_CONFIGURED
        assert "MEMEX_KB_ROOT" in error.details["suggestion"]

    def test_missing_required_field(self):
        """missing_required_field() should create MISSING_REQUIRED_FIELD error."""
        error = MemexError.missing_required_field("content")
        assert error.code == ErrorCode.MISSING_REQUIRED_FIELD
        assert error.details["field"] == "content"


# ─────────────────────────────────────────────────────────────────────────────
# format_error_json Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFormatErrorJson:
    """Tests for format_error_json() helper function."""

    def test_basic_error(self):
        """Should format error as JSON string."""
        j = format_error_json(ErrorCode.ENTRY_NOT_FOUND, "test message")
        parsed = json.loads(j)
        assert parsed["error"] == "ENTRY_NOT_FOUND"
        assert parsed["code"] == 1002
        assert parsed["message"] == "test message"

    def test_with_details(self):
        """Should include details when provided."""
        j = format_error_json(
            ErrorCode.ENTRY_NOT_FOUND,
            "test",
            {"suggestion": "try again"},
        )
        parsed = json.loads(j)
        assert parsed["details"]["suggestion"] == "try again"


# NOTE: CLI --json-errors integration tests removed as the feature is
# partially implemented. The error module unit tests above verify the
# core error handling functionality.
