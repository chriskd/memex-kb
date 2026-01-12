"""Tests for CLI intent detection and command suggestions.

When users pick the wrong command but express clear intent through flags,
the CLI should suggest the correct command.
"""

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


class TestPatchIntentDetection:
    """Tests for intent detection on the patch command."""

    def test_patch_with_content_no_find_suggests_alternatives(self):
        """'mx patch entry.md --content=...' suggests append/update."""
        exit_code, stdout, stderr = run_mx(
            "patch", "entry.md", "--content=new stuff"
        )

        assert exit_code != 0
        assert "--find is required" in stderr
        assert "Did you mean:" in stderr
        assert "append" in stderr.lower()
        assert "update" in stderr.lower()

    def test_patch_with_append_flag_suggests_append_command(self):
        """'mx patch entry.md --append=...' suggests append command."""
        exit_code, stdout, stderr = run_mx(
            "patch", "entry.md", "--append=stuff"
        )

        assert exit_code != 0
        assert "No such option: --append" in stderr
        assert "Did you mean:" in stderr
        assert "mx append" in stderr

    def test_patch_with_find_and_replace_works_normally(self):
        """Normal patch usage should not trigger intent suggestions."""
        # This will fail because entry doesn't exist, but it should NOT
        # show intent mismatch errors
        exit_code, stdout, stderr = run_mx(
            "patch", "nonexistent.md", "--find=old", "--replace=new"
        )

        # Should fail for file-not-found reasons, not intent mismatch
        assert "Did you mean:" not in stderr
        assert "--find is required" not in stderr

    def test_patch_content_suggestion_includes_path(self):
        """Intent suggestion should include the path user provided."""
        exit_code, stdout, stderr = run_mx(
            "patch", "my/custom/path.md", "--content=stuff"
        )

        assert exit_code != 0
        # The suggestion should reference the path for patch command
        assert "my/custom/path.md" in stderr


class TestUpdateIntentDetection:
    """Tests for intent detection on the update command."""

    def test_update_with_find_suggests_patch(self):
        """'mx update entry.md --find=...' suggests patch command."""
        exit_code, stdout, stderr = run_mx(
            "update", "entry.md", "--find=old text"
        )

        assert exit_code != 0
        assert "No such option: --find" in stderr
        assert "Did you mean:" in stderr
        assert "mx patch" in stderr

    def test_update_with_replace_suggests_patch(self):
        """'mx update entry.md --replace=...' suggests patch command."""
        exit_code, stdout, stderr = run_mx(
            "update", "entry.md", "--replace=new text"
        )

        assert exit_code != 0
        assert "No such option: --replace" in stderr
        assert "Did you mean:" in stderr
        assert "mx patch" in stderr

    def test_update_normal_usage_works(self):
        """Normal update usage should not trigger intent suggestions."""
        # This will fail because entry doesn't exist, but it should NOT
        # show intent mismatch errors
        exit_code, stdout, stderr = run_mx(
            "update", "nonexistent.md", "--tags=new,tags"
        )

        # Should fail for file-not-found reasons, not intent mismatch
        assert "Did you mean:" not in stderr
        assert "No such option" not in stderr


class TestIntentSuggestionFormat:
    """Tests for the format of intent suggestions."""

    def test_suggestion_format_has_error_line(self):
        """Suggestion should start with 'Error:' line."""
        exit_code, stdout, stderr = run_mx(
            "patch", "entry.md", "--content=stuff"
        )

        lines = stderr.strip().split("\n")
        assert lines[0].startswith("Error:")

    def test_suggestion_format_has_did_you_mean(self):
        """Suggestion should include 'Did you mean:' section."""
        exit_code, stdout, stderr = run_mx(
            "patch", "entry.md", "--content=stuff"
        )

        assert "Did you mean:" in stderr

    def test_suggestion_format_has_multiple_options(self):
        """Suggestion should offer multiple alternatives when appropriate."""
        exit_code, stdout, stderr = run_mx(
            "patch", "entry.md", "--content=stuff"
        )

        # Should suggest both append and update as alternatives
        lines = stderr.strip().split("\n")
        suggestion_lines = [l for l in lines if l.strip().startswith("-")]
        assert len(suggestion_lines) >= 2
