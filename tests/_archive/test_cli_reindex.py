"""Tests for mx reindex CLI command."""

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


def _create_entry(path: Path, title: str, tags: list[str]):
    """Helper to create a KB entry."""
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


class TestReindexBasic:
    """Test basic mx reindex functionality."""

    def test_reindex_rebuilds_indices(self, kb_root, index_root):
        """Successfully rebuilds both Whoosh and Chroma indices."""
        # Create some entries to index
        _create_entry(kb_root / "development" / "python-basics.md", "Python Basics", ["python", "tutorial"])
        _create_entry(kb_root / "development" / "testing-guide.md", "Testing Guide", ["testing", "pytest"])

        runner = CliRunner()
        result = runner.invoke(cli, ["reindex"])

        assert result.exit_code == 0
        assert "Indexed" in result.output

    def test_reindex_shows_progress_message(self, kb_root, index_root):
        """Shows 'Reindexing knowledge base...' message during reindex."""
        runner = CliRunner()
        result = runner.invoke(cli, ["reindex"])

        assert result.exit_code == 0
        assert "Reindexing knowledge base..." in result.output

    def test_reindex_shows_completion_summary(self, kb_root, index_root):
        """Shows completion message with counts after reindexing."""
        _create_entry(kb_root / "development" / "entry-one.md", "Entry One", ["tag1"])
        _create_entry(kb_root / "development" / "entry-two.md", "Entry Two", ["tag2"])

        runner = CliRunner()
        result = runner.invoke(cli, ["reindex"])

        assert result.exit_code == 0
        # Check for completion format: "Indexed N entries, M keyword docs, P semantic docs"
        assert "Indexed" in result.output
        assert "entries" in result.output
        assert "keyword docs" in result.output
        assert "semantic docs" in result.output


class TestReindexCounts:
    """Test that reindex counts are correct."""

    def test_reindex_counts_kb_files(self, kb_root, index_root):
        """Counts total KB files processed correctly."""
        # Create exactly 3 entries
        _create_entry(kb_root / "development" / "file-a.md", "File A", ["test"])
        _create_entry(kb_root / "development" / "file-b.md", "File B", ["test"])
        _create_entry(kb_root / "development" / "file-c.md", "File C", ["test"])

        runner = CliRunner()
        result = runner.invoke(cli, ["reindex", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["kb_files"] == 3

    def test_reindex_counts_whoosh_docs(self, kb_root, index_root):
        """Counts Whoosh index documents correctly."""
        _create_entry(kb_root / "development" / "whoosh-test-1.md", "Whoosh Test 1", ["whoosh"])
        _create_entry(kb_root / "development" / "whoosh-test-2.md", "Whoosh Test 2", ["whoosh"])

        runner = CliRunner()
        result = runner.invoke(cli, ["reindex", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        # Each KB file should have a corresponding Whoosh document
        assert data["whoosh_docs"] >= data["kb_files"]

    def test_reindex_counts_chroma_docs(self, kb_root, index_root):
        """Counts Chroma (semantic) index documents correctly."""
        _create_entry(kb_root / "development" / "chroma-test-1.md", "Chroma Test 1", ["semantic"])
        _create_entry(kb_root / "development" / "chroma-test-2.md", "Chroma Test 2", ["semantic"])

        runner = CliRunner()
        result = runner.invoke(cli, ["reindex", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        # Chroma may have multiple docs per file (chunked), but should have at least one per file
        assert data["chroma_docs"] >= data["kb_files"]


class TestReindexJsonOutput:
    """Test mx reindex --json output format."""

    def test_reindex_json_structure(self, kb_root, index_root):
        """JSON output has proper structure with all required fields."""
        _create_entry(kb_root / "development" / "json-test.md", "JSON Test", ["json"])

        runner = CliRunner()
        result = runner.invoke(cli, ["reindex", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Verify all required keys exist
        assert "kb_files" in data
        assert "whoosh_docs" in data
        assert "chroma_docs" in data

    def test_reindex_json_includes_stats(self, kb_root, index_root):
        """JSON output includes kb_files, whoosh_docs, and chroma_docs as integers."""
        _create_entry(kb_root / "development" / "stats-test.md", "Stats Test", ["stats"])

        runner = CliRunner()
        result = runner.invoke(cli, ["reindex", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # All values should be integers
        assert isinstance(data["kb_files"], int)
        assert isinstance(data["whoosh_docs"], int)
        assert isinstance(data["chroma_docs"], int)

        # Should have at least 1 file
        assert data["kb_files"] >= 1


class TestReindexEdgeCases:
    """Test mx reindex edge cases."""

    def test_reindex_empty_kb(self, kb_root, index_root):
        """Handles empty KB gracefully with zero counts."""
        runner = CliRunner()
        result = runner.invoke(cli, ["reindex"])

        assert result.exit_code == 0
        assert "Indexed" in result.output

        # Also verify with JSON
        result_json = runner.invoke(cli, ["reindex", "--json"])
        assert result_json.exit_code == 0
        data = json.loads(result_json.output)

        assert data["kb_files"] == 0
        assert data["whoosh_docs"] == 0
        assert data["chroma_docs"] == 0
