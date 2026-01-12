"""Tests for mx delete CLI command."""

import json
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
    for category in ("development", "architecture"):
        (root / category).mkdir()
    monkeypatch.setenv("MEMEX_KB_ROOT", str(root))
    return root


@pytest.fixture
def index_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary index root."""
    root = tmp_path / ".indices"
    root.mkdir()
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(root))
    return root


def _create_entry(path: Path, title: str, tags: list[str], content: str = "Some content."):
    """Helper to create a KB entry with frontmatter."""
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    text = f"""---
title: {title}
tags:
{tags_yaml}
created: {datetime.now(timezone.utc).isoformat()}
---

{content}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class TestDeleteBasic:
    """Basic delete command tests."""

    def test_delete_removes_file(self, kb_root, index_root):
        """Deleting an entry actually removes the file from disk."""
        entry_path = kb_root / "development" / "to-delete.md"
        _create_entry(entry_path, "To Delete", ["test"], "Content to delete.")

        assert entry_path.exists()

        runner = CliRunner()
        result = runner.invoke(cli, ["delete", "development/to-delete.md"])

        assert result.exit_code == 0
        assert not entry_path.exists()

    def test_delete_nonexistent_fails(self, kb_root, index_root):
        """Deleting a nonexistent entry returns an error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["delete", "development/does-not-exist.md"])

        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "not found" in result.output.lower()

    def test_delete_returns_success_message(self, kb_root, index_root):
        """Successful delete shows a confirmation message."""
        entry_path = kb_root / "development" / "success-msg.md"
        _create_entry(entry_path, "Success Message", ["test"], "Content here.")

        runner = CliRunner()
        result = runner.invoke(cli, ["delete", "development/success-msg.md"])

        assert result.exit_code == 0
        assert "Deleted:" in result.output
        assert "development/success-msg.md" in result.output


class TestDeleteBacklinks:
    """Tests for delete behavior with backlinks."""

    def test_delete_with_backlinks_fails(self, kb_root, index_root):
        """Deleting an entry with backlinks fails without --force."""
        # Entry A links to Entry B
        _create_entry(
            kb_root / "development" / "entry-a.md",
            "Entry A",
            ["test"],
            "Links to [[development/entry-b|Entry B]]",
        )
        _create_entry(
            kb_root / "development" / "entry-b.md",
            "Entry B",
            ["test"],
            "Some content",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["delete", "development/entry-b.md"])

        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "backlink" in result.output.lower()

    def test_delete_force_with_backlinks_succeeds(self, kb_root, index_root):
        """Deleting an entry with backlinks succeeds with --force."""
        # Entry A links to Entry B
        _create_entry(
            kb_root / "development" / "entry-a.md",
            "Entry A",
            ["test"],
            "Links to [[development/entry-b|Entry B]]",
        )
        entry_b = kb_root / "development" / "entry-b.md"
        _create_entry(entry_b, "Entry B", ["test"], "Some content")

        runner = CliRunner()
        result = runner.invoke(cli, ["delete", "development/entry-b.md", "--force"])

        assert result.exit_code == 0
        assert not entry_b.exists()
        assert "Deleted:" in result.output

    def test_delete_force_short_flag(self, kb_root, index_root):
        """The -f short flag works for force delete."""
        # Entry A links to Entry B
        _create_entry(
            kb_root / "development" / "entry-a.md",
            "Entry A",
            ["test"],
            "Links to [[development/entry-b|Entry B]]",
        )
        entry_b = kb_root / "development" / "entry-b.md"
        _create_entry(entry_b, "Entry B", ["test"], "Some content")

        runner = CliRunner()
        result = runner.invoke(cli, ["delete", "development/entry-b.md", "-f"])

        assert result.exit_code == 0
        assert not entry_b.exists()

    def test_delete_shows_backlinks_in_error(self, kb_root, index_root):
        """Error message lists which entries have backlinks to the target."""
        # Multiple entries link to entry-target
        _create_entry(
            kb_root / "development" / "linker-one.md",
            "Linker One",
            ["test"],
            "Links to [[development/entry-target|Target]]",
        )
        _create_entry(
            kb_root / "development" / "entry-target.md",
            "Target Entry",
            ["test"],
            "I am the target.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["delete", "development/entry-target.md"])

        assert result.exit_code == 1
        assert "backlink" in result.output.lower()
        # The error should mention the linking entry
        assert "linker-one" in result.output.lower() or "development/linker-one" in result.output.lower()


class TestDeleteJsonOutput:
    """Tests for --json output format."""

    def test_delete_json_returns_deleted_path(self, kb_root, index_root):
        """JSON output includes the deleted path."""
        entry_path = kb_root / "development" / "json-delete.md"
        _create_entry(entry_path, "JSON Delete", ["test"], "Content here.")

        runner = CliRunner()
        result = runner.invoke(cli, ["delete", "development/json-delete.md", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "deleted" in data
        assert data["deleted"] == "development/json-delete.md"

    def test_delete_json_returns_had_backlinks(self, kb_root, index_root):
        """JSON output shows backlinks info when force deleting."""
        # Entry A links to Entry B
        _create_entry(
            kb_root / "development" / "linker.md",
            "Linker",
            ["test"],
            "Links to [[development/target|Target]]",
        )
        _create_entry(
            kb_root / "development" / "target.md",
            "Target",
            ["test"],
            "Target content",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["delete", "development/target.md", "--force", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "had_backlinks" in data
        assert isinstance(data["had_backlinks"], list)
        assert len(data["had_backlinks"]) >= 1

    def test_delete_json_error_format(self, kb_root, index_root):
        """Errors are also output to stderr, not breaking JSON."""
        runner = CliRunner()
        result = runner.invoke(cli, ["delete", "nonexistent.md", "--json"])

        assert result.exit_code == 1
        # Error should go to stderr
        assert "Error:" in result.output

    def test_delete_json_no_backlinks(self, kb_root, index_root):
        """JSON output shows empty backlinks for entries without backlinks."""
        entry_path = kb_root / "development" / "no-backlinks.md"
        _create_entry(entry_path, "No Backlinks", ["test"], "Standalone entry.")

        runner = CliRunner()
        result = runner.invoke(cli, ["delete", "development/no-backlinks.md", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "had_backlinks" in data
        assert data["had_backlinks"] == []


class TestDeleteValidation:
    """Tests for path validation in delete command."""

    def test_delete_directory_path_fails(self, kb_root, index_root):
        """Cannot delete a directory, only files."""
        runner = CliRunner()
        result = runner.invoke(cli, ["delete", "development"])

        assert result.exit_code == 1
        assert "Error:" in result.output
        # Should indicate it's not a file or is a directory
        assert "not a file" in result.output.lower() or "directory" in result.output.lower()

    def test_delete_non_md_file_fails(self, kb_root, index_root):
        """Only .md files can be deleted."""
        # Create a non-markdown file
        txt_file = kb_root / "development" / "notes.txt"
        txt_file.write_text("Some text content")

        runner = CliRunner()
        result = runner.invoke(cli, ["delete", "development/notes.txt"])

        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "markdown" in result.output.lower() or ".md" in result.output.lower()

    def test_delete_path_outside_kb_fails(self, kb_root, index_root):
        """Path traversal attempts are blocked."""
        runner = CliRunner()
        result = runner.invoke(cli, ["delete", "../outside.md"])

        assert result.exit_code == 1
        assert "Error:" in result.output
        # Should indicate invalid path
        assert "invalid" in result.output.lower() or "escape" in result.output.lower()

    def test_delete_hidden_path_fails(self, kb_root, index_root):
        """Paths with hidden directories are blocked."""
        runner = CliRunner()
        result = runner.invoke(cli, ["delete", ".hidden/file.md"])

        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "invalid" in result.output.lower()

    def test_delete_absolute_path_fails(self, kb_root, index_root):
        """Absolute paths are blocked."""
        runner = CliRunner()
        result = runner.invoke(cli, ["delete", "/etc/passwd"])

        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "invalid" in result.output.lower()
