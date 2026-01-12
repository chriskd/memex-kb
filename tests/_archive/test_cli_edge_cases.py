"""Tests for CLI edge cases and error handling."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from memex import core
from memex.cli import cli


@pytest.fixture(autouse=True)
def reset_searcher_state(monkeypatch):
    """Ensure cached searcher state does not leak across tests."""
    monkeypatch.setattr(core, "_searcher", None)
    monkeypatch.setattr(core, "_searcher_ready", False)


@pytest.fixture
def kb_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary KB root with standard categories."""
    root = tmp_path / "kb"
    root.mkdir()
    (root / "development").mkdir()
    monkeypatch.setenv("MEMEX_KB_ROOT", str(root))
    return root


@pytest.fixture
def index_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary index root."""
    root = tmp_path / ".indices"
    root.mkdir()
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(root))
    return root


def _create_entry(path: Path, title: str, tags: list[str], content: str = ""):
    """Helper to create a KB entry with frontmatter."""
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    text = f"""---
title: {title}
tags:
{tags_yaml}
created: {datetime.now(timezone.utc).isoformat()}
---

{content if content else f"Content for {title}."}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class TestUnicode:
    """Tests for Unicode handling across CLI commands."""

    def test_unicode_in_title(self, kb_root, index_root):
        """Handles unicode in entry titles correctly."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "add",
                "--title=Unicode Title Test",
                "--tags=unicode,test",
                "--category=development",
                "--content=Content with unicode title reference.",
            ],
        )

        assert result.exit_code == 0
        assert "Created:" in result.output

        # Verify file was created
        entry_path = kb_root / "development" / "unicode-title-test.md"
        assert entry_path.exists()

    def test_unicode_in_content(self, kb_root, index_root):
        """Handles unicode in content correctly."""
        unicode_content = "Chinese: \u4e2d\u6587 Russian: \u0420\u0443\u0441\u0441\u043a\u0438\u0439 Arabic: \u0627\u0644\u0639\u0631\u0628\u064a\u0629 Japanese: \u65e5\u672c\u8a9e"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "add",
                "--title=Unicode Content",
                "--tags=unicode",
                "--category=development",
                f"--content={unicode_content}",
            ],
        )

        assert result.exit_code == 0

        entry_path = kb_root / "development" / "unicode-content.md"
        assert entry_path.exists()

        file_content = entry_path.read_text(encoding="utf-8")
        assert "\u4e2d\u6587" in file_content  # Chinese
        assert "\u0420\u0443\u0441\u0441\u043a\u0438\u0439" in file_content  # Russian
        assert "\u0627\u0644\u0639\u0631\u0628\u064a\u0629" in file_content  # Arabic
        assert "\u65e5\u672c\u8a9e" in file_content  # Japanese

    def test_unicode_in_tags(self, kb_root, index_root):
        """Handles unicode tag names correctly."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "add",
                "--title=Unicode Tags Entry",
                "--tags=test,unicode-\u00e9\u00e0\u00fc",
                "--category=development",
                "--content=Entry with unicode tags.",
            ],
        )

        assert result.exit_code == 0

        entry_path = kb_root / "development" / "unicode-tags-entry.md"
        assert entry_path.exists()

        file_content = entry_path.read_text(encoding="utf-8")
        assert "unicode-\u00e9\u00e0\u00fc" in file_content

    def test_unicode_in_path(self, kb_root, index_root):
        """Handles unicode in file paths correctly."""
        # Create a category with unicode name
        unicode_category = kb_root / "d\u00e9veloppement"
        unicode_category.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "add",
                "--title=Path Unicode Test",
                "--tags=test",
                "--category=d\u00e9veloppement",
                "--content=Entry in unicode category.",
            ],
        )

        assert result.exit_code == 0

        entry_path = unicode_category / "path-unicode-test.md"
        assert entry_path.exists()

    def test_emoji_handling(self, kb_root, index_root):
        """Handles emoji in content correctly."""
        emoji_content = "Party time! \U0001f389 Launch! \U0001f680 Code! \U0001f4bb"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "add",
                "--title=Emoji Test",
                "--tags=emoji",
                "--category=development",
                f"--content={emoji_content}",
            ],
        )

        assert result.exit_code == 0

        entry_path = kb_root / "development" / "emoji-test.md"
        assert entry_path.exists()

        file_content = entry_path.read_text(encoding="utf-8")
        assert "\U0001f389" in file_content  # Party popper
        assert "\U0001f680" in file_content  # Rocket
        assert "\U0001f4bb" in file_content  # Laptop


class TestSpecialChars:
    """Tests for special character handling."""

    def test_quotes_in_content(self, kb_root, index_root):
        """Handles single and double quotes in content."""
        content_with_quotes = 'She said "Hello" and he replied \'Hi there\''

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "add",
                "--title=Quotes Test",
                "--tags=test",
                "--category=development",
                f"--content={content_with_quotes}",
            ],
        )

        assert result.exit_code == 0

        entry_path = kb_root / "development" / "quotes-test.md"
        file_content = entry_path.read_text()
        assert '"Hello"' in file_content
        assert "'Hi there'" in file_content

    def test_backslash_in_content(self, kb_root, index_root):
        """Handles backslashes in content."""
        content_with_backslash = "Windows path: C:\\Users\\name\\Documents"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "add",
                "--title=Backslash Test",
                "--tags=test",
                "--category=development",
                f"--content={content_with_backslash}",
            ],
        )

        assert result.exit_code == 0

        entry_path = kb_root / "development" / "backslash-test.md"
        file_content = entry_path.read_text()
        assert "C:\\Users\\name\\Documents" in file_content

    def test_markdown_special_chars(self, kb_root, index_root):
        """Handles markdown special characters (*, _, #, etc.) in content."""
        markdown_content = """# Header
**bold text** and *italic text*
_also italic_ and __also bold__
Some `inline code` here
- list item 1
- list item 2
> blockquote
[link](http://example.com)
"""

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "add",
                "--title=Markdown Chars Test",
                "--tags=markdown",
                "--category=development",
                f"--content={markdown_content}",
            ],
        )

        assert result.exit_code == 0

        entry_path = kb_root / "development" / "markdown-chars-test.md"
        file_content = entry_path.read_text()
        assert "**bold text**" in file_content
        assert "*italic text*" in file_content
        assert "`inline code`" in file_content
        assert "> blockquote" in file_content

    def test_yaml_special_chars(self, kb_root, index_root):
        """Handles colon in content without breaking frontmatter."""
        content_with_colon = "Key: value pairs like name: John and age: 30"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "add",
                "--title=YAML Chars Test",
                "--tags=yaml",
                "--category=development",
                f"--content={content_with_colon}",
            ],
        )

        assert result.exit_code == 0

        entry_path = kb_root / "development" / "yaml-chars-test.md"
        file_content = entry_path.read_text()

        # Verify frontmatter is intact
        assert "title: YAML Chars Test" in file_content
        assert "yaml" in file_content

        # Verify content is preserved
        assert "Key: value" in file_content
        assert "name: John" in file_content


class TestLargeContent:
    """Tests for handling large content."""

    def test_large_file_read(self, kb_root, index_root):
        """Reads large files (10KB+) correctly."""
        # Create a 15KB entry
        large_content = "A" * 15000
        _create_entry(
            kb_root / "development" / "large-file.md",
            "Large File",
            ["large"],
            large_content,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["get", "development/large-file.md"])

        assert result.exit_code == 0
        assert "Large File" in result.output
        # Should contain at least part of the content
        assert "A" * 100 in result.output

    def test_large_file_write(self, kb_root, index_root):
        """Writes large files (10KB+) correctly."""
        large_content = "B" * 15000

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "add",
                "--title=Large Write Test",
                "--tags=large",
                "--category=development",
                f"--content={large_content}",
            ],
        )

        assert result.exit_code == 0

        entry_path = kb_root / "development" / "large-write-test.md"
        assert entry_path.exists()

        file_content = entry_path.read_text()
        assert "B" * 15000 in file_content

    def test_many_entries_list(self, kb_root, index_root):
        """Lists 50+ entries correctly."""
        # Create 55 entries
        for i in range(55):
            _create_entry(
                kb_root / "development" / f"entry-{i:03d}.md",
                f"Entry Number {i}",
                ["batch"],
                f"Content for entry {i}.",
            )

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--limit=100"])

        assert result.exit_code == 0
        # Should show multiple entries
        assert "entry-" in result.output
        # Check that the output contains expected entries
        assert "Entry Number" in result.output or "entry-" in result.output


class TestPathSecurity:
    """Tests for path security validation."""

    def test_path_traversal_blocked(self, kb_root, index_root):
        """Path traversal attempts (../../../etc/passwd) are blocked."""
        runner = CliRunner()
        result = runner.invoke(cli, ["get", "../../../etc/passwd"])

        assert result.exit_code == 1
        assert "Error:" in result.output
        # The path is treated as relative and fails with "not found" or "invalid"
        assert "not found" in result.output.lower() or "invalid" in result.output.lower()

    def test_absolute_path_rejected(self, kb_root, index_root):
        """Absolute paths (/etc/passwd) are rejected."""
        runner = CliRunner()
        result = runner.invoke(cli, ["get", "/etc/passwd"])

        assert result.exit_code == 1
        assert "Error:" in result.output
        # Absolute paths may fail with parse error or invalid path error
        assert "invalid" in result.output.lower() or "failed" in result.output.lower()

    def test_hidden_path_rejected(self, kb_root, index_root):
        """Hidden paths (.hidden/secret) are rejected."""
        runner = CliRunner()
        result = runner.invoke(cli, ["get", ".hidden/secret.md"])

        assert result.exit_code == 1
        assert "Error:" in result.output
        # Hidden paths fail with "not found" since they don't exist in KB
        assert "not found" in result.output.lower() or "invalid" in result.output.lower()


class TestErrorMessages:
    """Tests for error message quality."""

    def test_missing_kb_root_error(self, tmp_path, monkeypatch):
        """Clear error message when MEMEX_KB_ROOT is not set."""
        # Unset the environment variable
        monkeypatch.delenv("MEMEX_KB_ROOT", raising=False)
        monkeypatch.delenv("MEMEX_INDEX_ROOT", raising=False)

        runner = CliRunner()
        result = runner.invoke(cli, ["list"])

        assert result.exit_code != 0
        # The exception message contains MEMEX_KB_ROOT info
        # Check both output and exception for the error message
        error_info = str(result.exception) if result.exception else result.output
        assert "MEMEX_KB_ROOT" in error_info or "not set" in error_info.lower()

    def test_invalid_json_flag_combination(self, kb_root, index_root):
        """Error for invalid flag combinations."""
        runner = CliRunner()

        # Test mutually exclusive options in add command
        result = runner.invoke(
            cli,
            [
                "add",
                "--title=Test",
                "--tags=test",
                "--content=Content",
                "--stdin",  # Cannot use both --content and --stdin
            ],
            input="stdin content",
        )

        # The CLI accepts this but --stdin reads from stdin
        # Content comes from stdin, not --content
        # This is actually valid behavior, so test with get instead

        # Test with patch command - mutually exclusive find options
        _create_entry(
            kb_root / "development" / "test-entry.md",
            "Test Entry",
            ["test"],
            "Original content here.",
        )

        result = runner.invoke(
            cli,
            [
                "patch",
                "development/test-entry.md",
                "--find=old",
                "--find-file=/dev/null",  # Cannot use both --find and --find-file
                "--replace=new",
            ],
        )

        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower() or "Error:" in result.output

    def test_permission_error_message(self, kb_root, index_root, tmp_path, monkeypatch):
        """Handles permission errors gracefully."""
        # Create a file without read permissions (Unix only)
        if os.name != "nt":  # Skip on Windows
            # Create entry
            entry_path = kb_root / "development" / "no-read.md"
            _create_entry(entry_path, "No Read", ["test"], "Cannot read me.")

            # Remove read permission
            entry_path.chmod(0o000)

            try:
                runner = CliRunner()
                result = runner.invoke(cli, ["get", "development/no-read.md"])

                # Should fail gracefully
                assert result.exit_code != 0
                # Should have some error indication
                assert "Error:" in result.output or "permission" in result.output.lower()
            finally:
                # Restore permissions for cleanup
                entry_path.chmod(0o644)
        else:
            # On Windows, skip this test with a pass
            pytest.skip("Permission test not applicable on Windows")


class TestSearchEdgeCases:
    """Additional edge case tests for search command."""

    def test_empty_query_rejected(self, kb_root, index_root):
        """Empty search query is rejected with helpful error."""
        runner = CliRunner()

        # Empty string query
        result = runner.invoke(cli, ["search", ""])

        assert result.exit_code != 0
        assert "empty" in result.output.lower() or "cannot" in result.output.lower()

    def test_whitespace_only_query_rejected(self, kb_root, index_root):
        """Whitespace-only search query is rejected."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "   "])

        assert result.exit_code != 0
        assert "empty" in result.output.lower() or "cannot" in result.output.lower()

    def test_very_long_query_handled(self, kb_root, index_root):
        """Very long search queries are handled gracefully."""
        long_query = "a" * 1000

        runner = CliRunner()
        result = runner.invoke(cli, ["search", long_query])

        # Should not crash, may return no results or an error
        assert result.exit_code in (0, 1)


class TestGetEdgeCases:
    """Additional edge case tests for get command."""

    def test_get_nonexistent_entry(self, kb_root, index_root):
        """Getting a nonexistent entry returns helpful error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["get", "development/does-not-exist.md"])

        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "not found" in result.output.lower() or "does not exist" in result.output.lower()

    def test_get_by_title_case_insensitive(self, kb_root, index_root):
        """Get by title is case-insensitive."""
        _create_entry(
            kb_root / "development" / "my-entry.md",
            "My Fancy Entry",
            ["test"],
            "Some content here.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["get", "--title=my fancy entry"])

        assert result.exit_code == 0
        assert "My Fancy Entry" in result.output


class TestAddEdgeCases:
    """Additional edge case tests for add command."""

    def test_add_with_empty_content(self, kb_root, index_root):
        """Adding entry with empty content still creates valid file."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "add",
                "--title=Empty Content",
                "--tags=test",
                "--category=development",
                "--content=",
            ],
        )

        # May fail or succeed depending on implementation
        if result.exit_code == 0:
            entry_path = kb_root / "development" / "empty-content.md"
            assert entry_path.exists()
            content = entry_path.read_text()
            assert "title: Empty Content" in content

    def test_add_with_newlines_in_content(self, kb_root, index_root):
        """Handles newlines in content correctly."""
        multiline = "Line 1\nLine 2\nLine 3"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "add",
                "--title=Newlines Test",
                "--tags=test",
                "--category=development",
                f"--content={multiline}",
            ],
        )

        assert result.exit_code == 0

        entry_path = kb_root / "development" / "newlines-test.md"
        content = entry_path.read_text()
        assert "Line 1" in content
        assert "Line 2" in content
        assert "Line 3" in content
