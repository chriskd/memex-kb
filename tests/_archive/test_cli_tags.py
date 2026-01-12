"""Tests for mx tags CLI command."""

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


class TestTagsBasic:
    """Test basic mx tags functionality."""

    def test_tags_lists_all_tags(self, kb_root, index_root):
        """Shows all tags used in KB."""
        _create_entry(kb_root / "development" / "entry1.md", "Entry 1", ["python", "testing"])
        _create_entry(kb_root / "development" / "entry2.md", "Entry 2", ["python", "docs"])
        _create_entry(kb_root / "development" / "entry3.md", "Entry 3", ["javascript"])

        runner = CliRunner()
        result = runner.invoke(cli, ["tags"])

        assert result.exit_code == 0
        assert "python" in result.output
        assert "testing" in result.output
        assert "docs" in result.output
        assert "javascript" in result.output

    def test_tags_shows_counts(self, kb_root, index_root):
        """Shows usage count for each tag."""
        _create_entry(kb_root / "development" / "entry1.md", "Entry 1", ["python", "testing"])
        _create_entry(kb_root / "development" / "entry2.md", "Entry 2", ["python", "testing"])
        _create_entry(kb_root / "development" / "entry3.md", "Entry 3", ["python", "docs"])

        runner = CliRunner()
        result = runner.invoke(cli, ["tags"])

        assert result.exit_code == 0
        # python: 3 entries
        assert "python: 3" in result.output
        # testing: 2 entries
        assert "testing: 2" in result.output
        # docs: 1 entry
        assert "docs: 1" in result.output

    def test_tags_sorted_by_count(self, kb_root, index_root):
        """Most used tags first."""
        _create_entry(kb_root / "development" / "entry1.md", "Entry 1", ["python", "testing"])
        _create_entry(kb_root / "development" / "entry2.md", "Entry 2", ["python", "testing"])
        _create_entry(kb_root / "development" / "entry3.md", "Entry 3", ["python", "docs"])

        runner = CliRunner()
        result = runner.invoke(cli, ["tags"])

        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        # First line should be python (count 3)
        assert "python" in lines[0]
        # Second line should be testing (count 2)
        assert "testing" in lines[1]
        # docs should come after (count 1)
        assert "docs" in lines[2]


class TestTagsMinCount:
    """Test --min-count filtering."""

    def test_tags_min_count_filters(self, kb_root, index_root):
        """--min-count=3 excludes tags with count < 3."""
        _create_entry(kb_root / "development" / "entry1.md", "Entry 1", ["python", "testing"])
        _create_entry(kb_root / "development" / "entry2.md", "Entry 2", ["python", "testing"])
        _create_entry(kb_root / "development" / "entry3.md", "Entry 3", ["python", "docs"])

        runner = CliRunner()
        result = runner.invoke(cli, ["tags", "--min-count=3"])

        assert result.exit_code == 0
        # Only python has count >= 3
        assert "python" in result.output
        assert "testing" not in result.output
        assert "docs" not in result.output

    def test_tags_min_count_1_default(self, kb_root, index_root):
        """Default shows all tags (count >= 1)."""
        _create_entry(kb_root / "development" / "entry1.md", "Entry 1", ["python"])
        _create_entry(kb_root / "development" / "entry2.md", "Entry 2", ["javascript"])
        _create_entry(kb_root / "development" / "entry3.md", "Entry 3", ["rust"])

        runner = CliRunner()
        result = runner.invoke(cli, ["tags"])

        assert result.exit_code == 0
        # All tags should appear (each has count 1)
        assert "python" in result.output
        assert "javascript" in result.output
        assert "rust" in result.output

    def test_tags_min_count_high_shows_none(self, kb_root, index_root):
        """Very high min-count shows no tags."""
        _create_entry(kb_root / "development" / "entry1.md", "Entry 1", ["python", "testing"])
        _create_entry(kb_root / "development" / "entry2.md", "Entry 2", ["python", "docs"])

        runner = CliRunner()
        result = runner.invoke(cli, ["tags", "--min-count=100"])

        assert result.exit_code == 0
        assert "No tags found" in result.output


class TestTagsJsonOutput:
    """Test --json output format."""

    def test_tags_json_structure(self, kb_root, index_root):
        """Proper JSON with array of tag objects."""
        _create_entry(kb_root / "development" / "entry1.md", "Entry 1", ["python", "testing"])
        _create_entry(kb_root / "development" / "entry2.md", "Entry 2", ["python", "docs"])

        runner = CliRunner()
        result = runner.invoke(cli, ["tags", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) >= 1
        # Each item should be a dict with tag and count
        for item in data:
            assert isinstance(item, dict)
            assert "tag" in item

    def test_tags_json_includes_count(self, kb_root, index_root):
        """Each tag has 'count' field."""
        _create_entry(kb_root / "development" / "entry1.md", "Entry 1", ["python", "testing"])
        _create_entry(kb_root / "development" / "entry2.md", "Entry 2", ["python", "testing"])
        _create_entry(kb_root / "development" / "entry3.md", "Entry 3", ["python", "docs"])

        runner = CliRunner()
        result = runner.invoke(cli, ["tags", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Find python tag and verify count
        python_tag = next((t for t in data if t["tag"] == "python"), None)
        assert python_tag is not None
        assert python_tag["count"] == 3

        # Find testing tag and verify count
        testing_tag = next((t for t in data if t["tag"] == "testing"), None)
        assert testing_tag is not None
        assert testing_tag["count"] == 2

        # Find docs tag and verify count
        docs_tag = next((t for t in data if t["tag"] == "docs"), None)
        assert docs_tag is not None
        assert docs_tag["count"] == 1


class TestTagsEdgeCases:
    """Test edge cases for mx tags."""

    def test_tags_empty_kb(self, kb_root, index_root):
        """Handles empty KB gracefully."""
        runner = CliRunner()
        result = runner.invoke(cli, ["tags"])

        assert result.exit_code == 0
        assert "No tags found" in result.output

    def test_tags_unicode_tags(self, kb_root, index_root):
        """Handles unicode tag names."""
        _create_entry(kb_root / "development" / "entry1.md", "Entry 1", ["python", "dokumentation"])
        _create_entry(kb_root / "development" / "entry2.md", "Entry 2", ["python"])

        runner = CliRunner()
        result = runner.invoke(cli, ["tags"])

        assert result.exit_code == 0
        assert "python" in result.output
        assert "dokumentation" in result.output

    def test_tags_single_entry(self, kb_root, index_root):
        """Works with just one entry."""
        _create_entry(kb_root / "development" / "single.md", "Single Entry", ["solitary", "unique"])

        runner = CliRunner()
        result = runner.invoke(cli, ["tags"])

        assert result.exit_code == 0
        assert "solitary" in result.output
        assert "unique" in result.output
        # Both should have count 1
        assert "solitary: 1" in result.output
        assert "unique: 1" in result.output
