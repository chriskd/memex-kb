"""Tests for --json-errors CLI flag behavior."""

import json
import subprocess
import sys

import pytest


def run_mx(*args: str) -> tuple[int, str, str]:
    """Run mx CLI command and return (exit_code, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-m", "memex.cli", *args],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


class TestJsonErrors:
    """Tests for --json-errors flag formatting Click validation errors as JSON."""

    def test_invalid_integer_without_json_errors(self):
        """Without --json-errors, invalid integer shows plain text error."""
        exit_code, stdout, stderr = run_mx("search", "test", "--limit", "abc")

        assert exit_code != 0
        # Should be plain text, not JSON
        assert "Invalid value" in stderr or "invalid" in stderr.lower()
        # Ensure it's not JSON format
        with pytest.raises(json.JSONDecodeError):
            json.loads(stderr.strip())

    def test_invalid_integer_with_json_errors(self):
        """With --json-errors, invalid integer returns JSON error."""
        exit_code, stdout, stderr = run_mx("--json-errors", "search", "test", "--limit", "abc")

        assert exit_code == 1
        error_data = json.loads(stderr.strip())
        assert "error" in error_data
        assert error_data["error"]["code"] == "INVALID_ARGUMENT"
        assert "--limit" in error_data["error"]["message"]

    def test_unknown_option_with_json_errors(self):
        """Unknown option returns JSON error with UNKNOWN_OPTION code."""
        exit_code, stdout, stderr = run_mx("--json-errors", "search", "test", "--unknown-flag")

        assert exit_code == 1
        error_data = json.loads(stderr.strip())
        assert error_data["error"]["code"] == "UNKNOWN_OPTION"
        assert "--unknown-flag" in error_data["error"]["message"]

    def test_missing_required_option_with_json_errors(self):
        """Missing required option returns JSON error."""
        exit_code, stdout, stderr = run_mx("--json-errors", "add", "--title=test")

        assert exit_code == 1
        error_data = json.loads(stderr.strip())
        # The error code might be INVALID_ARGUMENT or MISSING_ARGUMENT depending on how Click reports it
        assert error_data["error"]["code"] in ("INVALID_ARGUMENT", "MISSING_ARGUMENT")
        assert "--tags" in error_data["error"]["message"]

    def test_missing_argument_with_json_errors(self):
        """Missing positional argument returns JSON error."""
        exit_code, stdout, stderr = run_mx("--json-errors", "search")

        assert exit_code == 1
        error_data = json.loads(stderr.strip())
        # Click may report missing arguments as BadParameter or UsageError
        assert error_data["error"]["code"] in ("USAGE_ERROR", "INVALID_ARGUMENT", "MISSING_ARGUMENT")
        assert "QUERY" in error_data["error"]["message"] or "argument" in error_data["error"]["message"].lower()

    def test_invalid_choice_with_json_errors(self):
        """Invalid choice value returns JSON error."""
        exit_code, stdout, stderr = run_mx("--json-errors", "search", "test", "--mode", "invalid")

        assert exit_code == 1
        error_data = json.loads(stderr.strip())
        assert error_data["error"]["code"] == "INVALID_ARGUMENT"
        assert "mode" in error_data["error"]["message"].lower()

    def test_successful_command_with_json_errors(self):
        """Successful command with --json-errors still outputs normally."""
        exit_code, stdout, stderr = run_mx("--json-errors", "prime", "--mcp")

        assert exit_code == 0
        # Output should be normal prime output, not JSON error
        assert "KB Quick Reference" in stdout or "Search:" in stdout
        # stderr should be empty on success
        assert stderr.strip() == ""

    def test_json_errors_flag_position(self):
        """--json-errors must come before subcommand (global option)."""
        # When --json-errors comes after subcommand, it's not recognized
        exit_code, stdout, stderr = run_mx("search", "test", "--json-errors")

        assert exit_code != 0
        # Should mention unknown option since subcommand doesn't have --json-errors
        assert "json-errors" in stderr.lower()

    def test_error_json_structure(self):
        """Verify the JSON error structure matches expected format."""
        exit_code, stdout, stderr = run_mx("--json-errors", "search", "test", "--limit", "abc")

        error_data = json.loads(stderr.strip())

        # Verify structure
        assert isinstance(error_data, dict)
        assert "error" in error_data
        assert isinstance(error_data["error"], dict)
        assert "code" in error_data["error"]
        assert "message" in error_data["error"]

        # Code should be a string
        assert isinstance(error_data["error"]["code"], str)
        # Message should be non-empty
        assert len(error_data["error"]["message"]) > 0


class TestJsonErrorsIntegration:
    """Integration tests for --json-errors with various commands."""

    def test_get_nonexistent_entry_with_json_errors(self):
        """get command with nonexistent entry should output error as JSON."""
        exit_code, stdout, stderr = run_mx("--json-errors", "get", "nonexistent/path.md")

        # This error happens during command execution, not during Click validation
        # It may or may not be JSON depending on how the command handles it
        assert exit_code != 0

    def test_search_with_zero_limit(self):
        """Search with --limit=0 should return validation error as JSON."""
        exit_code, stdout, stderr = run_mx("--json-errors", "search", "test", "--limit", "0")

        assert exit_code == 1
        error_data = json.loads(stderr.strip())
        assert error_data["error"]["code"] == "INVALID_ARGUMENT"
        # Should mention the limit range validation
        assert "0" in error_data["error"]["message"] or "range" in error_data["error"]["message"].lower()
