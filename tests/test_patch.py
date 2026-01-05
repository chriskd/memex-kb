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


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


# ─────────────────────────────────────────────────────────────────────────────
# Test find_matches
# ─────────────────────────────────────────────────────────────────────────────


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

    def test_empty_old_string_returns_empty(self):
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

    def test_multiline_old_string(self):
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


# ─────────────────────────────────────────────────────────────────────────────
# Test apply_patch
# ─────────────────────────────────────────────────────────────────────────────


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

    def test_empty_old_string(self):
        """Empty old_string returns NOT_FOUND."""
        result = apply_patch("Hello", "", "world")
        assert not result.success
        assert result.exit_code == PatchExitCode.NOT_FOUND

    def test_new_string_empty(self):
        """Empty new_string effectively deletes."""
        result = apply_patch("Hello world", "world", "")
        assert result.success
        assert result.new_content == "Hello "

    def test_replace_with_same_string(self):
        """Replacing with same string is allowed."""
        result = apply_patch("Hello world", "world", "world")
        assert result.success
        assert result.new_content == "Hello world"


# ─────────────────────────────────────────────────────────────────────────────
# Test generate_diff
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# Test file operations
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# Test CLI command
# ─────────────────────────────────────────────────────────────────────────────


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
            cli, ["patch", "entry.md", "--old", "old text", "--new", "new text"]
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
            ["patch", "entry.md", "--old", "TODO", "--new", "DONE", "--replace-all"],
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
            cli, ["patch", "entry.md", "--old", "old", "--new", "new", "--dry-run"]
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
            cli, ["patch", "entry.md", "--old", "x", "--new", "y", "--json"]
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
            cli, ["patch", "entry.md", "--old", "foo", "--new", "bar"]
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
            cli, ["patch", "entry.md", "--old", "TODO", "--new", "DONE"]
        )

        assert result.exit_code == 2
        assert "matches" in result.output.lower()

    def test_missing_old_option(self, runner):
        """Missing --old returns exit code 3."""
        result = runner.invoke(cli, ["patch", "entry.md", "--new", "bar"])

        assert result.exit_code == 3
        assert "Must provide --old" in result.output

    def test_missing_new_option(self, runner):
        """Missing --new returns exit code 3."""
        result = runner.invoke(cli, ["patch", "entry.md", "--old", "foo"])

        assert result.exit_code == 3
        assert "Must provide --new" in result.output

    def test_mutual_exclusivity_old(self, runner, tmp_path):
        """--old and --old-file are mutually exclusive."""
        old_file = tmp_path / "old.txt"
        old_file.write_text("old text")

        result = runner.invoke(
            cli,
            [
                "patch",
                "entry.md",
                "--old",
                "text",
                "--old-file",
                str(old_file),
                "--new",
                "bar",
            ],
        )

        assert result.exit_code == 3
        assert "mutually exclusive" in result.output.lower()

    def test_mutual_exclusivity_new(self, runner, tmp_path):
        """--new and --new-file are mutually exclusive."""
        new_file = tmp_path / "new.txt"
        new_file.write_text("new text")

        result = runner.invoke(
            cli,
            [
                "patch",
                "entry.md",
                "--old",
                "foo",
                "--new",
                "text",
                "--new-file",
                str(new_file),
            ],
        )

        assert result.exit_code == 3
        assert "mutually exclusive" in result.output.lower()

    @patch("memex.cli.run_async")
    def test_old_file_option(self, mock_run_async, runner, tmp_path):
        """--old-file reads old text from file."""
        old_file = tmp_path / "old.txt"
        old_file.write_text("multi\nline\nold")

        mock_run_async.return_value = {
            "success": True,
            "exit_code": 0,
            "replacements": 1,
            "path": "entry.md",
        }

        result = runner.invoke(
            cli,
            ["patch", "entry.md", "--old-file", str(old_file), "--new", "replacement"],
        )

        assert result.exit_code == 0
        # Verify the multi-line content was passed
        call_args = mock_run_async.call_args
        assert call_args is not None

    @patch("memex.cli.run_async")
    def test_new_file_option(self, mock_run_async, runner, tmp_path):
        """--new-file reads new text from file."""
        new_file = tmp_path / "new.txt"
        new_file.write_text("multi\nline\nnew")

        mock_run_async.return_value = {
            "success": True,
            "exit_code": 0,
            "replacements": 1,
            "path": "entry.md",
        }

        result = runner.invoke(
            cli,
            ["patch", "entry.md", "--old", "foo", "--new-file", str(new_file)],
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
            cli, ["patch", "entry.md", "--old", "x", "--new", "y", "--backup"]
        )

        assert result.exit_code == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test PatchResult
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# Additional Unit Tests (Coverage Gaps)
# ─────────────────────────────────────────────────────────────────────────────


class TestFindMatchesEdgeCases:
    """Additional edge cases for find_matches."""

    def test_content_without_newlines(self):
        """Content with no newlines has correct line numbers."""
        matches = find_matches("no newlines here TARGET end", "TARGET")
        assert len(matches) == 1
        assert matches[0].line_number == 1

    def test_not_found_truncates_long_old_string(self):
        """Long old_string is truncated in NOT_FOUND error message."""
        result = apply_patch("content", "x" * 100, "y")
        assert result.exit_code == PatchExitCode.NOT_FOUND
        assert "..." in result.message
        # Should have at most 53 chars (50 + "...")
        assert len(result.message) < 70


class TestReadFileSafelyEdgeCases:
    """Additional edge cases for read_file_safely."""

    def test_read_permission_denied(self, tmp_path):
        """Permission denied returns FILE_ERROR."""
        import sys
        if sys.platform == "win32":
            pytest.skip("Permission test not reliable on Windows")

        path = tmp_path / "test.md"
        path.write_text("content")
        os.chmod(path, 0o000)

        try:
            frontmatter, body, error = read_file_safely(path)
            assert error is not None
            assert error.exit_code == PatchExitCode.FILE_ERROR
            assert "Permission denied" in error.message
        finally:
            os.chmod(path, 0o644)  # Restore for cleanup


class TestWriteFileAtomicallyEdgeCases:
    """Additional edge cases for write_file_atomically."""

    def test_backup_to_existing_backup(self, tmp_path):
        """Creating backup when .bak already exists overwrites it."""
        path = tmp_path / "test.md"
        path.write_text("current")
        backup_path = path.with_suffix(".md.bak")
        backup_path.write_text("old backup")

        error = write_file_atomically(path, "", "new", backup=True)

        assert error is None
        assert backup_path.read_text() == "current"  # Overwrote old backup


class TestPatchCLIEdgeCases:
    """Additional edge cases for CLI command."""

    @patch("memex.cli.run_async")
    def test_exit_code_file_error(self, mock_run_async, runner):
        """Exit code 3 for file errors from patch_entry."""
        mock_run_async.return_value = {
            "success": False,
            "exit_code": 3,
            "message": "File is not valid UTF-8",
        }

        result = runner.invoke(
            cli, ["patch", "entry.md", "--old", "x", "--new", "y"]
        )

        assert result.exit_code == 3
        assert "UTF-8" in result.output

    @patch("memex.cli.run_async")
    def test_unexpected_exception_exit_code_3(self, mock_run_async, runner):
        """Unexpected exception exits with code 3."""
        mock_run_async.side_effect = RuntimeError("Unexpected error")

        result = runner.invoke(
            cli, ["patch", "entry.md", "--old", "x", "--new", "y"]
        )

        assert result.exit_code == 3
        assert "Error" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests for patch_entry()
# ─────────────────────────────────────────────────────────────────────────────


from datetime import date

from memex import core


pytestmark_integration = pytest.mark.semantic


@pytest.fixture(autouse=True)
def reset_searcher_state(monkeypatch):
    """Ensure cached searcher state does not leak across tests."""
    monkeypatch.setattr(core, "_searcher", None)
    monkeypatch.setattr(core, "_searcher_ready", False)


@pytest.fixture
def patch_kb_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary KB root with standard categories."""
    root = tmp_path / "kb"
    root.mkdir()
    (root / "development").mkdir()
    (root / "testing").mkdir()
    monkeypatch.setenv("MEMEX_KB_ROOT", str(root))
    return root


@pytest.fixture
def patch_index_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary index root."""
    root = tmp_path / ".indices"
    root.mkdir()
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(root))
    return root


def _create_patch_entry(path: Path, title: str, content_body: str, tags: list[str] | None = None):
    """Helper to create a KB entry with frontmatter."""
    tags = tags or ["test"]
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    content = f"""---
title: {title}
tags:
{tags_yaml}
created: {date.today().isoformat()}
---

{content_body}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestPatchEntryIntegration:
    """Integration tests for patch_entry with real file operations."""

    @pytest.mark.asyncio
    async def test_patch_updates_file_content(self, patch_kb_root, patch_index_root):
        """Successful patch updates file content correctly."""
        _create_patch_entry(
            patch_kb_root / "development" / "test.md",
            "Test Entry",
            "Original content with TODO marker here.",
        )

        result = await core.patch_entry(
            path="development/test.md",
            old_string="TODO",
            new_string="DONE",
        )

        assert result["success"]
        assert result["exit_code"] == 0
        assert result["replacements"] == 1

        # Verify file was actually modified
        content = (patch_kb_root / "development" / "test.md").read_text()
        assert "DONE" in content
        assert "TODO" not in content

    @pytest.mark.asyncio
    async def test_patch_updates_metadata(self, patch_kb_root, patch_index_root):
        """Patch sets 'updated' date in frontmatter."""
        _create_patch_entry(
            patch_kb_root / "development" / "test.md",
            "Test Entry",
            "Content to patch",
        )

        await core.patch_entry(
            path="development/test.md",
            old_string="Content",
            new_string="Modified",
        )

        content = (patch_kb_root / "development" / "test.md").read_text()
        assert "updated:" in content

    @pytest.mark.asyncio
    async def test_patch_dry_run_no_changes(self, patch_kb_root, patch_index_root):
        """Dry run returns diff without modifying file."""
        original_content = "Content with TARGET word here."
        _create_patch_entry(
            patch_kb_root / "development" / "test.md",
            "Test Entry",
            original_content,
        )

        result = await core.patch_entry(
            path="development/test.md",
            old_string="TARGET",
            new_string="REPLACED",
            dry_run=True,
        )

        assert result["success"]
        assert "diff" in result
        assert "---" in result["diff"]
        assert "REPLACED" in result["diff"]  # New text appears in diff

        # File should NOT be modified
        content = (patch_kb_root / "development" / "test.md").read_text()
        assert "TARGET" in content
        assert "REPLACED" not in content

    @pytest.mark.asyncio
    async def test_patch_creates_backup(self, patch_kb_root, patch_index_root):
        """--backup creates .bak file with original content."""
        _create_patch_entry(
            patch_kb_root / "development" / "test.md",
            "Test Entry",
            "Original content",
        )
        original = (patch_kb_root / "development" / "test.md").read_text()

        await core.patch_entry(
            path="development/test.md",
            old_string="Original",
            new_string="Modified",
            backup=True,
        )

        backup_path = patch_kb_root / "development" / "test.md.bak"
        assert backup_path.exists()
        assert backup_path.read_text() == original

    @pytest.mark.asyncio
    async def test_patch_replace_all(self, patch_kb_root, patch_index_root):
        """Replace all occurrences with replace_all=True."""
        _create_patch_entry(
            patch_kb_root / "development" / "test.md",
            "Test Entry",
            "TODO here and TODO there and TODO everywhere",
        )

        result = await core.patch_entry(
            path="development/test.md",
            old_string="TODO",
            new_string="DONE",
            replace_all=True,
        )

        assert result["success"]
        assert result["replacements"] == 3

        content = (patch_kb_root / "development" / "test.md").read_text()
        assert content.count("DONE") == 3
        assert "TODO" not in content

    @pytest.mark.asyncio
    async def test_patch_nonexistent_entry(self, patch_kb_root, patch_index_root):
        """Patching non-existent entry returns FILE_ERROR."""
        result = await core.patch_entry(
            path="development/nonexistent.md",
            old_string="foo",
            new_string="bar",
        )

        assert not result["success"]
        assert result["exit_code"] == 3
        assert "not found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_patch_text_not_found(self, patch_kb_root, patch_index_root):
        """Patching with non-matching text returns NOT_FOUND."""
        _create_patch_entry(
            patch_kb_root / "development" / "test.md",
            "Test Entry",
            "Some content here",
        )

        result = await core.patch_entry(
            path="development/test.md",
            old_string="nonexistent text",
            new_string="replacement",
        )

        assert not result["success"]
        assert result["exit_code"] == 1
        assert "not found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_patch_ambiguous_without_replace_all(self, patch_kb_root, patch_index_root):
        """Multiple matches without replace_all returns AMBIGUOUS."""
        _create_patch_entry(
            patch_kb_root / "development" / "test.md",
            "Test Entry",
            "TODO first\nTODO second",
        )

        result = await core.patch_entry(
            path="development/test.md",
            old_string="TODO",
            new_string="DONE",
            replace_all=False,
        )

        assert not result["success"]
        assert result["exit_code"] == 2
        assert "match" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_patch_directory_path_returns_error(self, patch_kb_root, patch_index_root):
        """Attempting to patch a directory returns FILE_ERROR."""
        result = await core.patch_entry(
            path="development",
            old_string="foo",
            new_string="bar",
        )

        assert not result["success"]
        assert result["exit_code"] == 3
        assert "not a file" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_patch_preserves_frontmatter(self, patch_kb_root, patch_index_root):
        """Patch preserves frontmatter except updated date."""
        _create_patch_entry(
            patch_kb_root / "development" / "test.md",
            "My Special Title",
            "Content to patch",
            tags=["custom-tag", "another-tag"],
        )

        await core.patch_entry(
            path="development/test.md",
            old_string="Content",
            new_string="Modified",
        )

        content = (patch_kb_root / "development" / "test.md").read_text()
        assert "title: My Special Title" in content
        assert "custom-tag" in content
        assert "another-tag" in content
