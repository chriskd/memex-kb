"""Tests for command typo suggestions in CLI error messages."""

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


class TestCommandTypoSuggestions:
    """Tests for suggesting correct commands when user makes a typo."""

    def test_serach_suggests_search(self):
        """'mx serach' suggests 'search'."""
        exit_code, stdout, stderr = run_mx("serach", "test")

        assert exit_code != 0
        assert "No such command 'serach'" in stderr
        assert "Did you mean 'search'?" in stderr

    def test_ad_suggests_add(self):
        """'mx ad' suggests 'add'."""
        exit_code, stdout, stderr = run_mx("ad")

        assert exit_code != 0
        assert "No such command 'ad'" in stderr
        assert "Did you mean 'add'?" in stderr

    def test_lits_suggests_list(self):
        """'mx lits' suggests 'list'."""
        exit_code, stdout, stderr = run_mx("lits")

        assert exit_code != 0
        assert "No such command 'lits'" in stderr
        assert "Did you mean 'list'?" in stderr

    def test_tre_suggests_tree(self):
        """'mx tre' suggests 'tree'."""
        exit_code, stdout, stderr = run_mx("tre")

        assert exit_code != 0
        assert "No such command 'tre'" in stderr
        assert "Did you mean 'tree'?" in stderr

    def test_healt_suggests_health(self):
        """'mx healt' suggests 'health'."""
        exit_code, stdout, stderr = run_mx("healt")

        assert exit_code != 0
        assert "No such command 'healt'" in stderr
        assert "Did you mean 'health'?" in stderr

    def test_updat_suggests_update(self):
        """'mx updat' suggests 'update'."""
        exit_code, stdout, stderr = run_mx("updat")

        assert exit_code != 0
        assert "No such command 'updat'" in stderr
        assert "Did you mean 'update'?" in stderr

    def test_delte_suggests_delete(self):
        """'mx delte' suggests 'delete'."""
        exit_code, stdout, stderr = run_mx("delte")

        assert exit_code != 0
        assert "No such command 'delte'" in stderr
        assert "Did you mean 'delete'?" in stderr

    def test_no_suggestion_for_completely_wrong_command(self):
        """Completely wrong command should not suggest anything."""
        exit_code, stdout, stderr = run_mx("xyzabc123")

        assert exit_code != 0
        assert "No such command 'xyzabc123'" in stderr
        # Should not have a suggestion for gibberish
        assert "Did you mean" not in stderr

    def test_typo_suggestion_with_json_errors(self):
        """Typo suggestions should work with --json-errors flag."""
        exit_code, stdout, stderr = run_mx("--json-errors", "serach", "test")

        assert exit_code == 1
        error_data = json.loads(stderr.strip())
        assert "error" in error_data
        assert error_data["error"]["code"] == "USAGE_ERROR"
        assert "No such command 'serach'" in error_data["error"]["message"]
        assert "Did you mean 'search'?" in error_data["error"]["message"]

    def test_valid_command_no_suggestion(self):
        """Valid commands should work normally without suggestions."""
        # Use prime --mcp as it doesn't require external resources
        exit_code, stdout, stderr = run_mx("prime", "--mcp")

        assert exit_code == 0
        assert "Did you mean" not in stderr
        assert "No such command" not in stderr
