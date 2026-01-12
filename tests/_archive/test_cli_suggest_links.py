"""Tests for mx suggest-links CLI command."""

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
    (root / "python").mkdir()
    (root / "database").mkdir()
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
    path.write_text(text)


class TestSuggestLinksBasic:
    """Basic functionality tests for suggest-links command."""

    def test_suggest_links_finds_similar(self, kb_root, index_root):
        """suggest-links should suggest semantically similar entries."""
        # Create Python-related entries that should be similar
        _create_entry(
            kb_root / "python" / "basics.md",
            "Python Basics",
            ["python", "programming"],
            "Introduction to Python programming language. Variables, types, functions.",
        )
        _create_entry(
            kb_root / "python" / "advanced.md",
            "Python Advanced",
            ["python", "programming"],
            "Advanced Python programming techniques and patterns. Decorators, generators.",
        )
        _create_entry(
            kb_root / "python" / "testing.md",
            "Python Testing",
            ["python", "testing"],
            "Testing Python applications with pytest. Unit tests, fixtures.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["suggest-links", "python/basics.md"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        # Should suggest at least one of the other Python entries
        assert "python/" in result.output.lower() or "advanced" in result.output.lower() or "testing" in result.output.lower()

    def test_suggest_links_excludes_self(self, kb_root, index_root):
        """suggest-links should not suggest the entry itself."""
        _create_entry(
            kb_root / "python" / "basics.md",
            "Python Basics",
            ["python"],
            "Introduction to Python programming language.",
        )
        _create_entry(
            kb_root / "python" / "advanced.md",
            "Python Advanced",
            ["python"],
            "Advanced Python programming techniques.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["suggest-links", "python/basics.md", "--json"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        data = json.loads(result.output)
        # The requested entry should not appear in suggestions
        paths = [s["path"] for s in data]
        assert "python/basics.md" not in paths

    def test_suggest_links_excludes_existing_links(self, kb_root, index_root):
        """suggest-links should not suggest already-linked entries."""
        # Create entry with an existing link
        _create_entry(
            kb_root / "python" / "basics.md",
            "Python Basics",
            ["python"],
            "Introduction to Python. See also [[python/advanced.md|Advanced Python]].",
        )
        _create_entry(
            kb_root / "python" / "advanced.md",
            "Python Advanced",
            ["python"],
            "Advanced Python programming techniques.",
        )
        _create_entry(
            kb_root / "python" / "testing.md",
            "Python Testing",
            ["python"],
            "Testing Python applications with pytest.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["suggest-links", "python/basics.md", "--json"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        data = json.loads(result.output)
        # Already-linked entry should not appear in suggestions
        paths = [s["path"] for s in data]
        assert "python/advanced.md" not in paths

    def test_suggest_links_shows_reason(self, kb_root, index_root):
        """suggest-links should show why each entry was suggested."""
        _create_entry(
            kb_root / "python" / "basics.md",
            "Python Basics",
            ["python", "programming"],
            "Introduction to Python programming language.",
        )
        _create_entry(
            kb_root / "python" / "advanced.md",
            "Python Advanced",
            ["python", "programming"],
            "Advanced Python programming techniques.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["suggest-links", "python/basics.md"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        # Should show either shared tags or semantic similarity reason
        assert "Shares tags" in result.output or "Semantically similar" in result.output


class TestSuggestLinksLimit:
    """Tests for --limit option."""

    def test_suggest_links_default_limit_5(self, kb_root, index_root):
        """Default limit should be 5 suggestions."""
        # Create more than 5 entries
        for i in range(8):
            _create_entry(
                kb_root / "python" / f"entry{i}.md",
                f"Python Entry {i}",
                ["python"],
                f"Python content number {i}. Programming with Python.",
            )

        runner = CliRunner()
        result = runner.invoke(cli, ["suggest-links", "python/entry0.md", "--json"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        data = json.loads(result.output)
        # Should have at most 5 suggestions
        assert len(data) <= 5

    def test_suggest_links_custom_limit(self, kb_root, index_root):
        """--limit should control max suggestions."""
        # Create several entries
        for i in range(6):
            _create_entry(
                kb_root / "python" / f"entry{i}.md",
                f"Python Entry {i}",
                ["python"],
                f"Python content number {i}. Programming with Python.",
            )

        runner = CliRunner()
        result = runner.invoke(cli, ["suggest-links", "python/entry0.md", "--limit=2", "--json"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        data = json.loads(result.output)
        # Should have at most 2 suggestions
        assert len(data) <= 2


class TestSuggestLinksJsonOutput:
    """Tests for --json output format."""

    def test_suggest_links_json_structure(self, kb_root, index_root):
        """JSON output should be a proper array of suggestion objects."""
        _create_entry(
            kb_root / "python" / "basics.md",
            "Python Basics",
            ["python"],
            "Introduction to Python programming language.",
        )
        _create_entry(
            kb_root / "python" / "advanced.md",
            "Python Advanced",
            ["python"],
            "Advanced Python programming techniques.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["suggest-links", "python/basics.md", "--json"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        data = json.loads(result.output)
        assert isinstance(data, list)
        if data:
            assert "path" in data[0]
            assert "title" in data[0]
            assert "reason" in data[0]

    def test_suggest_links_json_includes_score(self, kb_root, index_root):
        """JSON output should include score for each suggestion."""
        _create_entry(
            kb_root / "python" / "basics.md",
            "Python Basics",
            ["python"],
            "Introduction to Python programming language.",
        )
        _create_entry(
            kb_root / "python" / "advanced.md",
            "Python Advanced",
            ["python"],
            "Advanced Python programming techniques.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["suggest-links", "python/basics.md", "--json"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        data = json.loads(result.output)
        if data:
            assert "score" in data[0]
            assert isinstance(data[0]["score"], (int, float))
            assert 0.0 <= data[0]["score"] <= 1.0


class TestSuggestLinksValidation:
    """Tests for input validation."""

    def test_suggest_links_nonexistent_path_fails(self, kb_root, index_root):
        """suggest-links should error for non-existent path."""
        runner = CliRunner()
        result = runner.invoke(cli, ["suggest-links", "nonexistent/entry.md"])

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_suggest_links_directory_path_fails(self, kb_root, index_root):
        """suggest-links should error for directory path."""
        # Create a directory with an entry inside
        _create_entry(
            kb_root / "python" / "basics.md",
            "Python Basics",
            ["python"],
            "Introduction to Python.",
        )

        runner = CliRunner()
        # Try to get suggestions for a directory instead of a file
        result = runner.invoke(cli, ["suggest-links", "python"])

        assert result.exit_code != 0
        assert "error" in result.output.lower() or "not found" in result.output.lower()


class TestSuggestLinksEdgeCases:
    """Tests for edge cases."""

    def test_suggest_links_no_similar(self, kb_root, index_root):
        """suggest-links should handle case with no similar entries."""
        # Create a single isolated entry
        _create_entry(
            kb_root / "database" / "sql.md",
            "SQL Database Design",
            ["sql", "architecture"],
            "Relational database design principles.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["suggest-links", "database/sql.md"])

        assert result.exit_code == 0
        # Should either show no suggestions or handle gracefully
        assert "No link suggestions" in result.output or result.output.strip() == "" or "Suggested links" in result.output

    def test_suggest_links_all_already_linked(self, kb_root, index_root):
        """suggest-links should handle when all similar entries are already linked."""
        # Create entry that already links to the only similar entry
        _create_entry(
            kb_root / "python" / "basics.md",
            "Python Basics",
            ["python"],
            "Introduction to Python. See [[python/advanced.md|Advanced]] and [[python/testing.md|Testing]].",
        )
        _create_entry(
            kb_root / "python" / "advanced.md",
            "Python Advanced",
            ["python"],
            "Advanced Python programming.",
        )
        _create_entry(
            kb_root / "python" / "testing.md",
            "Python Testing",
            ["python"],
            "Testing Python applications.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["suggest-links", "python/basics.md", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        # Already-linked entries should not appear
        paths = [s["path"] for s in data]
        assert "python/advanced.md" not in paths
        assert "python/testing.md" not in paths


class TestSuggestLinksHelp:
    """Tests for help text."""

    def test_suggest_links_help(self):
        """suggest-links --help should show usage info."""
        runner = CliRunner()
        result = runner.invoke(cli, ["suggest-links", "--help"])

        assert result.exit_code == 0
        assert "--limit" in result.output or "-n" in result.output
        assert "--json" in result.output
        assert "path" in result.output.lower()
