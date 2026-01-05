"""Integration tests for CLI with real temporary knowledge base.

These tests use actual file system operations with temporary directories
to verify end-to-end CLI functionality.
"""

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from memex.cli import cli


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_kb(tmp_path, monkeypatch):
    """Create a real temporary knowledge base with structure."""
    kb_root = tmp_path / "kb"
    kb_root.mkdir()

    # Create directories
    (kb_root / "tooling").mkdir()
    (kb_root / "projects").mkdir()
    (kb_root / "projects" / "myapp").mkdir()

    # Create a sample entry
    sample_entry = """---
title: Sample Entry
tags:
  - tooling
  - documentation
created: 2025-01-01
---

# Sample Entry

This is a sample knowledge base entry for testing.

## Section 1

Some content here.

## Section 2

More content with [[internal-link.md]] reference.
"""
    (kb_root / "tooling" / "sample.md").write_text(sample_entry)

    # Create another entry
    another_entry = """---
title: Another Entry
tags:
  - infrastructure
created: 2025-01-02
---

# Another Entry

Content for another entry with link to [[tooling/sample.md]].
"""
    (kb_root / "projects" / "myapp" / "readme.md").write_text(another_entry)

    # Set environment
    monkeypatch.setenv("MEMEX_KB_ROOT", str(kb_root))

    return kb_root


class TestBasicKBOperations:
    """Test basic KB read operations with real files."""

    def test_info_command(self, temp_kb, runner):
        """Test info command shows KB configuration."""
        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        assert str(temp_kb) in result.output or "kb" in result.output.lower()

    def test_info_json_output(self, temp_kb, runner):
        """Test info command with JSON output."""
        result = runner.invoke(cli, ["info", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "kb_root" in data or "path" in data or "root" in data

    def test_tree_command(self, temp_kb, runner):
        """Test tree command shows directory structure."""
        result = runner.invoke(cli, ["tree"])

        assert result.exit_code == 0
        assert "tooling" in result.output
        assert "projects" in result.output

    def test_tree_with_depth(self, temp_kb, runner):
        """Test tree command with depth limit."""
        result = runner.invoke(cli, ["tree", "--depth", "1"])

        assert result.exit_code == 0
        assert "tooling" in result.output

    def test_list_command(self, temp_kb, runner):
        """Test list command shows entries."""
        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        # Should show our sample entries
        assert "sample" in result.output.lower() or "Sample" in result.output


class TestGetCommand:
    """Test get command with real files."""

    def test_get_entry(self, temp_kb, runner):
        """Test get command reads entry content."""
        result = runner.invoke(cli, ["get", "tooling/sample.md"])

        assert result.exit_code == 0
        assert "Sample Entry" in result.output
        assert "This is a sample" in result.output

    def test_get_metadata_only(self, temp_kb, runner):
        """Test get command with metadata only flag."""
        result = runner.invoke(cli, ["get", "tooling/sample.md", "--metadata"])

        assert result.exit_code == 0
        assert "Sample Entry" in result.output
        assert "tooling" in result.output

    def test_get_json_output(self, temp_kb, runner):
        """Test get command with JSON output."""
        result = runner.invoke(cli, ["get", "tooling/sample.md", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "content" in data or "metadata" in data

    def test_get_nonexistent(self, temp_kb, runner):
        """Test get command with nonexistent file."""
        result = runner.invoke(cli, ["get", "nonexistent.md"])

        assert result.exit_code == 1
        assert "Error" in result.output or "not found" in result.output.lower()


class TestAddCommand:
    """Test add command with real files."""

    def test_add_entry(self, temp_kb, runner):
        """Test adding a new entry."""
        result = runner.invoke(
            cli,
            [
                "add",
                "--title", "New Entry",
                "--tags", "testing,integration",
                "--category", "tooling",
                "--content", "This is new content.",
                "--force",  # Skip duplicate detection
            ],
        )

        assert result.exit_code == 0
        assert "Created" in result.output or "new-entry" in result.output.lower()

        # Verify file was created
        created_files = list((temp_kb / "tooling").glob("*.md"))
        assert len(created_files) >= 2  # original + new

    def test_add_dry_run(self, temp_kb, runner):
        """Test add command with dry-run."""
        result = runner.invoke(
            cli,
            [
                "add",
                "--title", "Dry Run Entry",
                "--tags", "test",
                "--category", "tooling",
                "--content", "Preview content.",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "Preview" in result.output or "dry" in result.output.lower()

        # Verify file was NOT created
        dry_run_files = list((temp_kb / "tooling").glob("*dry*.md"))
        assert len(dry_run_files) == 0

    def test_add_json_output(self, temp_kb, runner):
        """Test add command with JSON output."""
        result = runner.invoke(
            cli,
            [
                "add",
                "--title", "JSON Entry",
                "--tags", "json",
                "--category", "tooling",
                "--content", "JSON content.",
                "--force",  # Skip duplicate detection
                "--json",
            ],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "path" in data or "created" in data


class TestUpdateCommand:
    """Test update command with real files."""

    def test_update_content(self, temp_kb, runner):
        """Test updating entry content."""
        result = runner.invoke(
            cli,
            ["update", "tooling/sample.md", "--content", "# Updated Content\n\nNew body."],
        )

        assert result.exit_code == 0

    def test_update_content_with_tags(self, temp_kb, runner):
        """Test updating entry content and tags together."""
        result = runner.invoke(
            cli,
            [
                "update", "tooling/sample.md",
                "--content", "# New Content\n\nBody.",
                "--tags", "updated,tags",
            ],
        )

        assert result.exit_code == 0


class TestDeleteCommand:
    """Test delete command with real files."""

    def test_delete_entry(self, temp_kb, runner):
        """Test deleting an entry."""
        # First create an entry to delete
        (temp_kb / "tooling" / "to-delete.md").write_text(
            "---\ntitle: To Delete\ntags: [delete]\n---\nContent"
        )

        result = runner.invoke(cli, ["delete", "tooling/to-delete.md"])

        assert result.exit_code == 0
        assert not (temp_kb / "tooling" / "to-delete.md").exists()


class TestTagsCommand:
    """Test tags command with real files."""

    def test_tags_command(self, temp_kb, runner):
        """Test tags command lists tags."""
        result = runner.invoke(cli, ["tags"])

        assert result.exit_code == 0
        # Should show tags from our sample entries
        assert "tooling" in result.output or "documentation" in result.output

    def test_tags_json_output(self, temp_kb, runner):
        """Test tags command with JSON output."""
        result = runner.invoke(cli, ["tags", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)


class TestHealthCommand:
    """Test health command with real files."""

    def test_health_command(self, temp_kb, runner):
        """Test health command runs audit."""
        result = runner.invoke(cli, ["health"])

        assert result.exit_code == 0

    def test_health_json_output(self, temp_kb, runner):
        """Test health command with JSON output."""
        result = runner.invoke(cli, ["health", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)


class TestInitCommand:
    """Test init command for KB bootstrapping."""

    def test_init_creates_kb_structure(self, tmp_path, runner, monkeypatch):
        """Test init creates KB root and default directories."""
        kb_root = tmp_path / "new-kb"

        # Clear environment to simulate fresh setup
        monkeypatch.delenv("MEMEX_KB_ROOT", raising=False)
        monkeypatch.delenv("MEMEX_INDEX_ROOT", raising=False)

        result = runner.invoke(cli, ["init", "--kb-root", str(kb_root), "--no-context"])

        assert result.exit_code == 0
        assert "Memex KB Initialized" in result.output
        assert kb_root.exists()
        assert (kb_root / "projects").exists()
        assert (kb_root / "tooling").exists()
        assert (kb_root / "infrastructure").exists()

    def test_init_with_existing_kb_no_force(self, tmp_path, runner, monkeypatch):
        """Test init exits cleanly when KB already exists."""
        kb_root = tmp_path / "existing-kb"
        kb_root.mkdir()
        (kb_root / "projects").mkdir()  # Make it non-empty

        monkeypatch.delenv("MEMEX_KB_ROOT", raising=False)
        monkeypatch.delenv("MEMEX_INDEX_ROOT", raising=False)

        result = runner.invoke(cli, ["init", "--kb-root", str(kb_root), "--no-context"])

        assert result.exit_code == 0
        assert "already exists" in result.output

    def test_init_with_force_reinitializes(self, tmp_path, runner, monkeypatch):
        """Test init --force reinitializes existing KB."""
        kb_root = tmp_path / "existing-kb"
        kb_root.mkdir()
        (kb_root / "projects").mkdir()

        monkeypatch.delenv("MEMEX_KB_ROOT", raising=False)
        monkeypatch.delenv("MEMEX_INDEX_ROOT", raising=False)

        result = runner.invoke(cli, ["init", "--kb-root", str(kb_root), "--force", "--no-context"])

        assert result.exit_code == 0
        assert "Memex KB Initialized" in result.output
        assert (kb_root / "tooling").exists()
        assert (kb_root / "infrastructure").exists()

    def test_init_uses_env_var_when_set(self, tmp_path, runner, monkeypatch):
        """Test init uses MEMEX_KB_ROOT when set."""
        kb_root = tmp_path / "env-kb"
        monkeypatch.setenv("MEMEX_KB_ROOT", str(kb_root))
        monkeypatch.delenv("MEMEX_INDEX_ROOT", raising=False)

        result = runner.invoke(cli, ["init", "--no-context"])

        assert result.exit_code == 0
        assert kb_root.exists()
        # When MEMEX_KB_ROOT is set, no export instruction needed
        assert 'export MEMEX_KB_ROOT' not in result.output

    def test_init_shows_env_export_instructions(self, tmp_path, runner, monkeypatch):
        """Test init shows environment variable instructions when not set."""
        kb_root = tmp_path / "new-kb"

        monkeypatch.delenv("MEMEX_KB_ROOT", raising=False)
        monkeypatch.delenv("MEMEX_INDEX_ROOT", raising=False)

        result = runner.invoke(cli, ["init", "--kb-root", str(kb_root), "--no-context"])

        assert result.exit_code == 0
        assert "export MEMEX_KB_ROOT" in result.output
        assert "shell profile" in result.output

    def test_init_creates_index_root(self, tmp_path, runner, monkeypatch):
        """Test init creates index root directory."""
        kb_root = tmp_path / "kb"
        index_root = tmp_path / "indices"

        monkeypatch.delenv("MEMEX_KB_ROOT", raising=False)
        monkeypatch.delenv("MEMEX_INDEX_ROOT", raising=False)

        result = runner.invoke(
            cli,
            ["init", "--kb-root", str(kb_root), "--index-root", str(index_root), "--no-context"],
        )

        assert result.exit_code == 0
        assert index_root.exists()

    def test_init_preserves_existing_kbcontext(self, tmp_path, runner, monkeypatch):
        """Test init doesn't overwrite existing .kbcontext when present."""
        kb_root = tmp_path / "kb"
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Create existing .kbcontext with custom content
        existing_content = "# existing context\nprimary: custom/path\n"
        (project_dir / ".kbcontext").write_text(existing_content)

        # Create a .git dir so it's detected as a project
        (project_dir / ".git").mkdir()

        monkeypatch.chdir(project_dir)
        monkeypatch.delenv("MEMEX_KB_ROOT", raising=False)
        monkeypatch.delenv("MEMEX_INDEX_ROOT", raising=False)

        result = runner.invoke(cli, ["init", "--kb-root", str(kb_root)])

        assert result.exit_code == 0
        assert ".kbcontext already exists" in result.output
        # Verify file was NOT overwritten
        assert (project_dir / ".kbcontext").read_text() == existing_content


class TestContextCommands:
    """Test context subcommands with real files."""

    def test_context_show_no_file(self, temp_kb, runner, monkeypatch):
        """Test context show when no .kbcontext exists."""
        # Change to a directory without .kbcontext
        monkeypatch.chdir(temp_kb)

        result = runner.invoke(cli, ["context", "show"])

        assert result.exit_code == 0
        assert "No .kbcontext" in result.output

    def test_context_init(self, temp_kb, runner, monkeypatch):
        """Test context init creates .kbcontext file."""
        monkeypatch.chdir(temp_kb / "projects" / "myapp")

        result = runner.invoke(cli, ["context", "init", "--project", "myapp"])

        assert result.exit_code == 0
        assert "Created" in result.output
        assert (temp_kb / "projects" / "myapp" / ".kbcontext").exists()

    def test_context_init_force(self, temp_kb, runner, monkeypatch):
        """Test context init with --force overwrites existing."""
        project_dir = temp_kb / "projects" / "myapp"
        monkeypatch.chdir(project_dir)

        # Create existing file
        (project_dir / ".kbcontext").write_text("existing")

        result = runner.invoke(cli, ["context", "init", "--force"])

        assert result.exit_code == 0


class TestSearchIntegration:
    """Test search with real indexed content.

    Note: Full search tests require semantic dependencies.
    These tests verify basic keyword search functionality.
    """

    @pytest.mark.skip(reason="Requires full indexing which is slow")
    def test_search_basic(self, temp_kb, runner):
        """Test basic search functionality."""
        result = runner.invoke(cli, ["search", "sample"])

        assert result.exit_code == 0
