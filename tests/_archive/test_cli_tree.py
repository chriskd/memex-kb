"""Tests for mx tree CLI command."""

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
    """Create a temporary KB root with nested structure for tree tests."""
    root = tmp_path / "kb"
    root.mkdir()
    # Create nested structure for tree tests
    (root / "development").mkdir()
    (root / "development" / "python").mkdir()
    (root / "development" / "python" / "frameworks").mkdir()
    (root / "development" / "python" / "frameworks" / "deep").mkdir()
    (root / "architecture").mkdir()
    (root / "devops").mkdir()
    monkeypatch.setenv("MEMEX_KB_ROOT", str(root))
    return root


@pytest.fixture
def index_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary index root."""
    root = tmp_path / ".indices"
    root.mkdir()
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(root))
    return root


def _create_entry(path: Path, title: str, tags: list[str]):
    """Helper to create a KB entry with frontmatter."""
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    text = f"""---
title: {title}
tags:
{tags_yaml}
created: {datetime.now(timezone.utc).isoformat()}
---

Content for {title}.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class TestTreeBasic:
    """Basic tree command functionality tests."""

    def test_tree_shows_directory_structure(self, kb_root, index_root):
        """Tree command displays directories in tree format."""
        runner = CliRunner()
        result = runner.invoke(cli, ["tree"])

        assert result.exit_code == 0
        # Should show directories with tree formatting
        assert "development/" in result.output
        assert "architecture/" in result.output
        assert "devops/" in result.output
        # Should have directory count in summary
        assert "directories" in result.output

    def test_tree_shows_file_titles(self, kb_root, index_root):
        """Tree command shows entry titles, not just filenames."""
        _create_entry(
            kb_root / "development" / "python-guide.md",
            "Python Development Guide",
            ["python", "guide"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["tree"])

        assert result.exit_code == 0
        # Should show filename with title in parentheses
        assert "python-guide.md" in result.output
        assert "Python Development Guide" in result.output

    def test_tree_shows_counts(self, kb_root, index_root):
        """Tree command shows entry counts at the end."""
        _create_entry(
            kb_root / "development" / "entry1.md",
            "Entry One",
            ["dev"],
        )
        _create_entry(
            kb_root / "architecture" / "entry2.md",
            "Entry Two",
            ["arch"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["tree"])

        assert result.exit_code == 0
        # Should show count summary
        assert "directories" in result.output
        assert "files" in result.output

    def test_tree_from_subdirectory(self, kb_root, index_root):
        """Tree command can start from a subdirectory."""
        _create_entry(
            kb_root / "development" / "python" / "flask.md",
            "Flask Framework",
            ["python", "flask"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["tree", "development"])

        assert result.exit_code == 0
        # Should show python subdirectory
        assert "python/" in result.output
        # Should not show top-level siblings
        assert "architecture/" not in result.output


class TestTreeDepth:
    """Tests for tree depth control."""

    def test_tree_depth_1_shows_top_level(self, kb_root, index_root):
        """Depth 1 shows only immediate children."""
        _create_entry(
            kb_root / "development" / "python" / "deep.md",
            "Deep Entry",
            ["python"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["tree", "--depth=1"])

        assert result.exit_code == 0
        # Top-level directories visible
        assert "development/" in result.output
        # Nested content should NOT be visible at depth 1
        assert "python/" not in result.output
        assert "deep.md" not in result.output

    def test_tree_depth_2_shows_nested(self, kb_root, index_root):
        """Depth 2 shows two levels of nesting."""
        _create_entry(
            kb_root / "development" / "python" / "frameworks" / "django.md",
            "Django Framework",
            ["python", "django"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["tree", "--depth=2"])

        assert result.exit_code == 0
        # Level 1
        assert "development/" in result.output
        # Level 2
        assert "python/" in result.output
        # Level 3 should not be visible
        assert "frameworks/" not in result.output
        assert "django.md" not in result.output

    def test_tree_default_depth_3(self, kb_root, index_root):
        """Default depth is 3, showing 3 levels."""
        _create_entry(
            kb_root / "development" / "python" / "frameworks" / "deep" / "very-deep.md",
            "Very Deep Entry",
            ["python"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["tree"])

        assert result.exit_code == 0
        # Level 1, 2, 3 should be visible
        assert "development/" in result.output
        assert "python/" in result.output
        assert "frameworks/" in result.output
        # Level 4 should not be visible (default depth=3)
        assert "deep/" not in result.output
        assert "very-deep.md" not in result.output

    def test_tree_depth_limits_output(self, kb_root, index_root):
        """Deeper content is not shown when depth limit is reached."""
        _create_entry(
            kb_root / "development" / "python" / "frameworks" / "deep" / "hidden.md",
            "Hidden Entry",
            ["python"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["tree", "-d", "2"])

        assert result.exit_code == 0
        # Level 3 and beyond not visible
        assert "frameworks/" not in result.output
        assert "deep/" not in result.output
        assert "hidden.md" not in result.output
        assert "Hidden Entry" not in result.output


class TestTreeJsonOutput:
    """Tests for tree JSON output."""

    def test_tree_json_structure(self, kb_root, index_root):
        """JSON output has proper structure with tree, directories, files."""
        _create_entry(
            kb_root / "development" / "entry.md",
            "Dev Entry",
            ["dev"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["tree", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        assert "tree" in data
        assert "directories" in data
        assert "files" in data
        assert isinstance(data["tree"], dict)
        assert isinstance(data["directories"], int)
        assert isinstance(data["files"], int)

    def test_tree_json_includes_type(self, kb_root, index_root):
        """JSON output includes type field for directories and files."""
        _create_entry(
            kb_root / "development" / "entry.md",
            "Dev Entry",
            ["dev"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["tree", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        tree = data["tree"]

        # Directory should have _type: directory
        assert "development" in tree
        assert tree["development"]["_type"] == "directory"

        # File should have _type: file
        assert "entry.md" in tree["development"]
        assert tree["development"]["entry.md"]["_type"] == "file"

    def test_tree_json_includes_title(self, kb_root, index_root):
        """JSON output includes title for files."""
        _create_entry(
            kb_root / "development" / "python-guide.md",
            "Python Development Guide",
            ["python"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["tree", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        tree = data["tree"]

        file_entry = tree["development"]["python-guide.md"]
        assert file_entry["title"] == "Python Development Guide"

    def test_tree_json_includes_children(self, kb_root, index_root):
        """JSON directories have children as nested keys."""
        _create_entry(
            kb_root / "development" / "python" / "flask.md",
            "Flask Guide",
            ["python"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["tree", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        tree = data["tree"]

        # development has python as child
        assert "development" in tree
        assert "python" in tree["development"]
        assert tree["development"]["python"]["_type"] == "directory"
        # python has flask.md as child
        assert "flask.md" in tree["development"]["python"]


class TestTreeValidation:
    """Tests for tree command validation and error handling."""

    def test_tree_nonexistent_path_fails(self, kb_root, index_root):
        """Tree command fails with error for nonexistent path."""
        runner = CliRunner()
        result = runner.invoke(cli, ["tree", "nonexistent/path"])

        assert result.exit_code != 0
        # Exception is raised with path info
        assert result.exception is not None
        assert "nonexistent/path" in str(result.exception)

    def test_tree_file_path_fails(self, kb_root, index_root):
        """Tree command fails when given a file path instead of directory."""
        _create_entry(
            kb_root / "development" / "entry.md",
            "Dev Entry",
            ["dev"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["tree", "development/entry.md"])

        assert result.exit_code != 0
        # Exception indicates it's not a directory
        assert result.exception is not None
        assert "not a directory" in str(result.exception).lower()


class TestTreeEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_tree_empty_directory(self, kb_root, index_root):
        """Tree handles empty directories gracefully."""
        # architecture directory exists but is empty
        runner = CliRunner()
        result = runner.invoke(cli, ["tree", "architecture"])

        assert result.exit_code == 0
        # Should still show summary (0 directories, 0 files within architecture)
        assert "0 directories" in result.output
        assert "0 files" in result.output

    def test_tree_empty_kb(self, tmp_path, monkeypatch):
        """Tree handles completely empty KB gracefully."""
        empty_root = tmp_path / "empty_kb"
        empty_root.mkdir()
        monkeypatch.setenv("MEMEX_KB_ROOT", str(empty_root))

        index_root = tmp_path / ".indices"
        index_root.mkdir()
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        # Reset searcher
        monkeypatch.setattr(core, "_searcher", None)
        monkeypatch.setattr(core, "_searcher_ready", False)

        runner = CliRunner()
        result = runner.invoke(cli, ["tree"])

        assert result.exit_code == 0
        assert "0 directories" in result.output
        assert "0 files" in result.output

    def test_tree_hidden_files_excluded(self, kb_root, index_root):
        """Hidden files and directories are excluded from tree."""
        # Create hidden directory and file
        hidden_dir = kb_root / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "secret.md").write_text("---\ntitle: Secret\ntags: []\n---\n")
        (kb_root / "development" / ".dotfile.md").write_text("---\ntitle: Dot\ntags: []\n---\n")

        runner = CliRunner()
        result = runner.invoke(cli, ["tree", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Hidden directory should not appear
        assert ".hidden" not in data["tree"]
        # Hidden file should not appear
        assert ".dotfile.md" not in data["tree"].get("development", {})

    def test_tree_underscore_files_excluded(self, kb_root, index_root):
        """Files/directories starting with underscore are excluded."""
        # Create underscore prefixed items
        underscore_dir = kb_root / "_drafts"
        underscore_dir.mkdir()
        (underscore_dir / "draft.md").write_text("---\ntitle: Draft\ntags: []\n---\n")

        runner = CliRunner()
        result = runner.invoke(cli, ["tree", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Underscore directory should not appear
        assert "_drafts" not in data["tree"]

    def test_tree_short_depth_flag(self, kb_root, index_root):
        """Short -d flag works for depth option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["tree", "-d", "1"])

        assert result.exit_code == 0
        # At depth 1, nested directories should not be visible
        assert "python/" not in result.output

    def test_tree_json_with_depth(self, kb_root, index_root):
        """JSON output respects depth limit."""
        _create_entry(
            kb_root / "development" / "python" / "frameworks" / "django.md",
            "Django Framework",
            ["python"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["tree", "--json", "--depth=2"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        tree = data["tree"]

        # Level 1 and 2 should be present
        assert "development" in tree
        assert "python" in tree["development"]
        # Level 3 should not be present (depth=2 means 2 levels)
        assert "frameworks" not in tree["development"]["python"]
