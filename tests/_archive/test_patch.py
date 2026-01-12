"""Tests for patch module and patch command.

Tests cover:
- Pure patch logic (find_matches, apply_patch)
- File I/O operations (read_file_safely, write_file_atomically)
- CLI command (argument parsing, output formatting, exit codes)
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from memex.cli import cli
from memex.patch import (
    MatchContext,
    PatchExitCode,
    PatchResult,
    apply_patch,
    find_matches,
    generate_diff,
    read_file_safely,
    write_file_atomically,
)


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


# -------------------------------------------------------------------------
# Test find_matches
# -------------------------------------------------------------------------


class TestFindMatches:
    """Tests for find_matches function."""

    def test_finds_single_match(self):
        """Single occurrence returns one match."""
        content = "Hello world, this is a test."
        matches = find_matches(content, "world")
        assert len(matches) == 1
        assert matches[0].match_number == 1
        assert matches[0].match_text == "world"

    def test_finds_multiple_matches(self):
        """Multiple occurrences return all matches."""
        content = "TODO: fix this\nTODO: test that\nTODO: deploy"
        matches = find_matches(content, "TODO")
        assert len(matches) == 3
        assert [m.match_number for m in matches] == [1, 2, 3]

    def test_no_matches_returns_empty(self):
        """No occurrences returns empty list."""
        content = "Hello world"
        matches = find_matches(content, "foo")
        assert matches == []

    def test_empty_find_string_returns_empty(self):
        """Empty search string returns empty list."""
        content = "Hello world"
        matches = find_matches(content, "")
        assert matches == []

    def test_match_context_extraction(self):
        """Context before/after is correctly extracted."""
        content = "This is before TARGET and this is after."
        matches = find_matches(content, "TARGET")
        assert len(matches) == 1
        assert "before" in matches[0].context_before
        assert "after" in matches[0].context_after

    def test_line_number_calculation(self):
        """Line numbers are 1-indexed and correct."""
        content = "line one\nline two TARGET\nline three"
        matches = find_matches(content, "TARGET")
        assert len(matches) == 1
        assert matches[0].line_number == 2

    def test_multiline_find_string(self):
        """Multi-line search strings work."""
        content = "Start\nfirst\nsecond\nEnd"
        matches = find_matches(content, "first\nsecond")
        assert len(matches) == 1

    def test_non_overlapping_matches(self):
        """Matches are non-overlapping."""
        content = "aaaaaa"
        matches = find_matches(content, "aaa")
        # Should find 2 non-overlapping: positions 0 and 3
        assert len(matches) == 2


class TestMatchContext:
    """Tests for MatchContext formatting."""

    def test_format_preview(self):
        """Preview formatting works."""
        ctx = MatchContext(
            match_number=1,
            start_pos=10,
            line_number=5,
            context_before="before ",
            match_text="TARGET",
            context_after=" after",
        )
        preview = ctx.format_preview()
        assert "Match 1" in preview
        assert "line 5" in preview
        assert "TARGET" in preview

    def test_format_preview_truncates_long_context(self):
        """Long context is truncated."""
        ctx = MatchContext(
            match_number=1,
            start_pos=100,
            line_number=1,
            context_before="x" * 100,
            match_text="TARGET",
            context_after="y" * 100,
        )
        preview = ctx.format_preview()
        assert len(preview) < 200  # Reasonable length

    def test_format_preview_escapes_newlines(self):
        """Newlines are escaped in preview."""
        ctx = MatchContext(
            match_number=1,
            start_pos=10,
            line_number=1,
            context_before="line1\nline2",
            match_text="TARGET",
            context_after="line3\nline4",
        )
        preview = ctx.format_preview()
        assert "\\n" in preview


# -------------------------------------------------------------------------
# Test apply_patch
# -------------------------------------------------------------------------


class TestApplyPatch:
    """Tests for apply_patch function."""

    def test_single_replacement(self):
        """Single match is replaced successfully."""
        result = apply_patch("Hello world", "world", "universe")
        assert result.success
        assert result.exit_code == PatchExitCode.SUCCESS
        assert result.new_content == "Hello universe"
        assert result.replacements_made == 1

    def test_replace_all_multiple(self):
        """All occurrences replaced with replace_all=True."""
        result = apply_patch("TODO TODO TODO", "TODO", "DONE", replace_all=True)
        assert result.success
        assert result.new_content == "DONE DONE DONE"
        assert result.replacements_made == 3

    def test_not_found_error(self):
        """Returns NOT_FOUND exit code when text missing."""
        result = apply_patch("Hello world", "foo", "bar")
        assert not result.success
        assert result.exit_code == PatchExitCode.NOT_FOUND
        assert "not found" in result.message.lower()

    def test_ambiguous_without_replace_all(self):
        """Returns AMBIGUOUS when multiple matches without flag."""
        result = apply_patch("TODO TODO", "TODO", "DONE", replace_all=False)
        assert not result.success
        assert result.exit_code == PatchExitCode.AMBIGUOUS
        assert result.matches_found == 2
        assert len(result.match_contexts) <= 3  # First 3 for preview

    def test_ambiguous_shows_context(self):
        """Ambiguous result includes match contexts."""
        result = apply_patch("fix TODO\nfix TODO\nfix TODO", "TODO", "DONE")
        assert not result.success
        assert result.match_contexts is not None
        assert len(result.match_contexts) == 3

    def test_empty_find_string(self):
        """Empty find_string returns NOT_FOUND."""
        result = apply_patch("Hello", "", "world")
        assert not result.success
        assert result.exit_code == PatchExitCode.NOT_FOUND

    def test_replace_string_empty(self):
        """Empty replace_string effectively deletes."""
        result = apply_patch("Hello world", "world", "")
        assert result.success
        assert result.new_content == "Hello "

    def test_replace_with_same_string(self):
        """Replacing with same string is allowed."""
        result = apply_patch("Hello world", "world", "world")
        assert result.success
        assert result.new_content == "Hello world"


# -------------------------------------------------------------------------
# Test generate_diff
# -------------------------------------------------------------------------


class TestGenerateDiff:
    """Tests for generate_diff function."""

    def test_generates_unified_diff(self):
        """Generates unified diff format."""
        original = "line1\nline2\nline3"
        patched = "line1\nmodified\nline3"
        diff = generate_diff(original, patched, "test.md")
        assert "--- a/test.md" in diff
        assert "+++ b/test.md" in diff
        assert "-line2" in diff
        assert "+modified" in diff

    def test_empty_diff_for_identical(self):
        """No diff output when content identical."""
        content = "same content"
        diff = generate_diff(content, content, "test.md")
        assert diff == ""


# -------------------------------------------------------------------------
# Test file operations
# -------------------------------------------------------------------------


class TestReadFileSafely:
    """Tests for read_file_safely function."""

    def test_read_utf8_file(self, tmp_path):
        """Normal UTF-8 file reads correctly."""
        path = tmp_path / "test.md"
        path.write_text("---\ntitle: Test\n---\n\nBody content", encoding="utf-8")

        frontmatter, body, error = read_file_safely(path)

        assert error is None
        assert "title: Test" in frontmatter
        assert "Body content" in body  # Body may have leading newline

    def test_read_utf8_bom(self, tmp_path):
        """UTF-8 BOM is stripped."""
        path = tmp_path / "test.md"
        # Write with BOM
        path.write_bytes(b"\xef\xbb\xbf---\ntitle: Test\n---\n\nBody")

        frontmatter, body, error = read_file_safely(path)

        assert error is None
        assert frontmatter.startswith("---")  # BOM stripped

    def test_read_non_utf8_fails(self, tmp_path):
        """Non-UTF-8 file returns FILE_ERROR."""
        path = tmp_path / "test.md"
        # Write invalid UTF-8
        path.write_bytes(b"\x80\x81\x82")

        frontmatter, body, error = read_file_safely(path)

        assert error is not None
        assert error.exit_code == PatchExitCode.FILE_ERROR
        assert "UTF-8" in error.message

    def test_read_nonexistent_file(self, tmp_path):
        """Non-existent file returns FILE_ERROR."""
        path = tmp_path / "nonexistent.md"

        frontmatter, body, error = read_file_safely(path)

        assert error is not None
        assert error.exit_code == PatchExitCode.FILE_ERROR
        assert "not found" in error.message.lower()

    def test_frontmatter_body_split(self, tmp_path):
        """Frontmatter and body are split correctly."""
        path = tmp_path / "test.md"
        path.write_text("---\ntitle: Test\ntags:\n  - foo\n---\n\n# Body\n\nContent here")

        frontmatter, body, error = read_file_safely(path)

        assert error is None
        assert "---" in frontmatter
        assert "title: Test" in frontmatter
        assert "# Body" in body
        assert "Content here" in body

    def test_no_frontmatter(self, tmp_path):
        """File without frontmatter is handled."""
        path = tmp_path / "test.md"
        path.write_text("# Just content\n\nNo frontmatter here")

        frontmatter, body, error = read_file_safely(path)

        assert error is None
        assert frontmatter == ""
        assert "Just content" in body


class TestWriteFileAtomically:
    """Tests for write_file_atomically function."""

    def test_write_atomic(self, tmp_path):
        """Write uses temp file + rename."""
        path = tmp_path / "test.md"
        path.write_text("original")

        error = write_file_atomically(path, "---\nfm\n---\n", "new content")

        assert error is None
        assert path.read_text() == "---\nfm\n---\nnew content"

    def test_write_preserves_permissions(self, tmp_path):
        """File permissions preserved after write."""
        path = tmp_path / "test.md"
        path.write_text("original")
        os.chmod(path, 0o644)

        error = write_file_atomically(path, "", "new")

        assert error is None
        assert oct(path.stat().st_mode)[-3:] == "644"

    def test_backup_created(self, tmp_path):
        """Backup file created when backup=True."""
        path = tmp_path / "test.md"
        path.write_text("original content")

        error = write_file_atomically(path, "", "new content", backup=True)

        assert error is None
        backup_path = path.with_suffix(".md.bak")
        assert backup_path.exists()
        assert backup_path.read_text() == "original content"

    def test_creates_new_file(self, tmp_path):
        """Can create a new file."""
        path = tmp_path / "new.md"

        error = write_file_atomically(path, "---\nfm\n---\n", "content")

        assert error is None
        assert path.exists()
        assert "content" in path.read_text()


# -------------------------------------------------------------------------
# Test CLI command
# -------------------------------------------------------------------------


class TestPatchCLI:
    """Tests for 'mx patch' command."""

    @patch("memex.cli.run_async")
    def test_basic_patch(self, mock_run_async, runner):
        """Basic patch via CLI works."""
        mock_run_async.return_value = {
            "success": True,
            "exit_code": 0,
            "message": "Patched entry.md",
            "replacements": 1,
            "path": "entry.md",
        }

        result = runner.invoke(
            cli, ["patch", "entry.md", "--find", "old text", "--replace", "new text"]
        )

        assert result.exit_code == 0
        assert "Patched" in result.output

    @patch("memex.cli.run_async")
    def test_replace_all(self, mock_run_async, runner):
        """--replace-all replaces all occurrences."""
        mock_run_async.return_value = {
            "success": True,
            "exit_code": 0,
            "message": "Replaced 3 occurrences",
            "replacements": 3,
            "path": "entry.md",
        }

        result = runner.invoke(
            cli,
            ["patch", "entry.md", "--find", "TODO", "--replace", "DONE", "--replace-all"],
        )

        assert result.exit_code == 0
        # Verify replace_all was passed
        call_args = mock_run_async.call_args
        assert call_args is not None

    @patch("memex.cli.run_async")
    def test_dry_run(self, mock_run_async, runner):
        """--dry-run shows diff without changes."""
        mock_run_async.return_value = {
            "success": True,
            "exit_code": 0,
            "message": "Dry run - no changes made",
            "replacements": 1,
            "diff": "--- a/entry.md\n+++ b/entry.md\n-old\n+new",
            "path": "entry.md",
        }

        result = runner.invoke(
            cli, ["patch", "entry.md", "--find", "old", "--replace", "new", "--dry-run"]
        )

        assert result.exit_code == 0
        assert "Dry run" in result.output
        assert "---" in result.output  # Diff output

    @patch("memex.cli.run_async")
    def test_json_output(self, mock_run_async, runner):
        """--json produces valid JSON output."""
        mock_run_async.return_value = {
            "success": True,
            "exit_code": 0,
            "message": "Patched",
            "replacements": 1,
            "path": "entry.md",
        }

        result = runner.invoke(
            cli, ["patch", "entry.md", "--find", "x", "--replace", "y", "--json"]
        )

        assert result.exit_code == 0
        import json

        data = json.loads(result.output)
        assert data["success"] is True

    @patch("memex.cli.run_async")
    def test_exit_code_not_found(self, mock_run_async, runner):
        """Exit code 1 for text not found."""
        mock_run_async.return_value = {
            "success": False,
            "exit_code": 1,
            "message": "Text not found: foo",
        }

        result = runner.invoke(
            cli, ["patch", "entry.md", "--find", "foo", "--replace", "bar"]
        )

        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    @patch("memex.cli.run_async")
    def test_exit_code_ambiguous(self, mock_run_async, runner):
        """Exit code 2 for ambiguous matches."""
        mock_run_async.return_value = {
            "success": False,
            "exit_code": 2,
            "message": "Found 3 matches",
            "match_contexts": [
                {"match_number": 1, "line_number": 5, "preview": "Match 1 (line 5)..."},
                {"match_number": 2, "line_number": 10, "preview": "Match 2 (line 10)..."},
            ],
        }

        result = runner.invoke(
            cli, ["patch", "entry.md", "--find", "TODO", "--replace", "DONE"]
        )

        assert result.exit_code == 2
        assert "matches" in result.output.lower()

    def test_missing_find_option(self, runner):
        """Missing --find returns exit code 3."""
        result = runner.invoke(cli, ["patch", "entry.md", "--replace", "bar"])

        assert result.exit_code == 3
        assert "Must provide --find" in result.output

    def test_missing_replace_option(self, runner):
        """Missing --replace returns exit code 3."""
        result = runner.invoke(cli, ["patch", "entry.md", "--find", "foo"])

        assert result.exit_code == 3
        assert "Must provide --replace" in result.output

    def test_mutual_exclusivity_find(self, runner, tmp_path):
        """--find and --find-file are mutually exclusive."""
        find_file = tmp_path / "find.txt"
        find_file.write_text("find text")

        result = runner.invoke(
            cli,
            [
                "patch",
                "entry.md",
                "--find",
                "text",
                "--find-file",
                str(find_file),
                "--replace",
                "bar",
            ],
        )

        assert result.exit_code == 3
        assert "mutually exclusive" in result.output.lower()

    def test_mutual_exclusivity_replace(self, runner, tmp_path):
        """--replace and --replace-file are mutually exclusive."""
        replace_file = tmp_path / "replace.txt"
        replace_file.write_text("replace text")

        result = runner.invoke(
            cli,
            [
                "patch",
                "entry.md",
                "--find",
                "foo",
                "--replace",
                "text",
                "--replace-file",
                str(replace_file),
            ],
        )

        assert result.exit_code == 3
        assert "mutually exclusive" in result.output.lower()

    @patch("memex.cli.run_async")
    def test_find_file_option(self, mock_run_async, runner, tmp_path):
        """--find-file reads find text from file."""
        find_file = tmp_path / "find.txt"
        find_file.write_text("multi\nline\nfind")

        mock_run_async.return_value = {
            "success": True,
            "exit_code": 0,
            "replacements": 1,
            "path": "entry.md",
        }

        result = runner.invoke(
            cli,
            ["patch", "entry.md", "--find-file", str(find_file), "--replace", "replacement"],
        )

        assert result.exit_code == 0
        # Verify the multi-line content was passed
        call_args = mock_run_async.call_args
        assert call_args is not None

    @patch("memex.cli.run_async")
    def test_replace_file_option(self, mock_run_async, runner, tmp_path):
        """--replace-file reads replace text from file."""
        replace_file = tmp_path / "replace.txt"
        replace_file.write_text("multi\nline\nreplace")

        mock_run_async.return_value = {
            "success": True,
            "exit_code": 0,
            "replacements": 1,
            "path": "entry.md",
        }

        result = runner.invoke(
            cli,
            ["patch", "entry.md", "--find", "foo", "--replace-file", str(replace_file)],
        )

        assert result.exit_code == 0

    @patch("memex.cli.run_async")
    def test_backup_flag(self, mock_run_async, runner):
        """--backup flag is passed to core function."""
        mock_run_async.return_value = {
            "success": True,
            "exit_code": 0,
            "replacements": 1,
            "path": "entry.md",
        }

        result = runner.invoke(
            cli, ["patch", "entry.md", "--find", "x", "--replace", "y", "--backup"]
        )

        assert result.exit_code == 0


# -------------------------------------------------------------------------
# Test PatchResult
# -------------------------------------------------------------------------


class TestPatchResult:
    """Tests for PatchResult data class."""

    def test_to_dict_success(self):
        """to_dict for successful result."""
        result = PatchResult(
            success=True,
            exit_code=PatchExitCode.SUCCESS,
            message="Replaced 1 occurrence(s)",
            matches_found=1,
            replacements_made=1,
            new_content="patched content",
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["exit_code"] == 0
        assert d["message"] == "Replaced 1 occurrence(s)"

    def test_to_dict_with_contexts(self):
        """to_dict includes match contexts when present."""
        result = PatchResult(
            success=False,
            exit_code=PatchExitCode.AMBIGUOUS,
            message="Found 2 matches",
            matches_found=2,
            match_contexts=[
                MatchContext(1, 10, 5, "before", "TARGET", "after"),
                MatchContext(2, 50, 15, "before2", "TARGET", "after2"),
            ],
        )

        d = result.to_dict()

        assert "match_contexts" in d
        assert len(d["match_contexts"]) == 2

    def test_to_dict_with_diff(self):
        """to_dict includes diff when present."""
        result = PatchResult(
            success=True,
            exit_code=PatchExitCode.SUCCESS,
            message="Dry run",
            diff="--- a/file\n+++ b/file",
        )

        d = result.to_dict()

        assert d["diff"] == "--- a/file\n+++ b/file"
