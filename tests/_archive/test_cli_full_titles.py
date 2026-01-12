"""Tests for mx --full-titles flag on search and list commands."""

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


def _create_entry(
    path: Path,
    title: str,
    tags: list[str],
    created: datetime,
):
    """Helper to create a KB entry with frontmatter."""
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    content = f"""---
title: {title}
tags:
{tags_yaml}
created: {created.isoformat()}
---

## Content

Some content here about {title.lower()}.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class TestListFullTitles:
    """Test mx list --full-titles flag."""

    def test_list_truncates_long_titles_by_default(self, kb_root, index_root):
        """By default, long titles are truncated."""
        today = datetime.now(timezone.utc)
        long_title = "This Is A Very Long Title That Should Be Truncated In Normal Output Mode"
        _create_entry(
            kb_root / "development" / "long-entry.md",
            long_title,
            ["python"],
            created=today,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        # Title should be truncated (default limit is 40 chars)
        assert "..." in result.output
        assert long_title not in result.output

    def test_list_full_titles_shows_complete_titles(self, kb_root, index_root):
        """With --full-titles, complete titles are shown."""
        today = datetime.now(timezone.utc)
        long_title = "This Is A Very Long Title That Should Be Truncated In Normal Output Mode"
        _create_entry(
            kb_root / "development" / "long-entry.md",
            long_title,
            ["python"],
            created=today,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--full-titles"])

        assert result.exit_code == 0
        # Full title should be present
        assert long_title in result.output

    def test_list_full_titles_with_filters(self, kb_root, index_root):
        """--full-titles works with other filters."""
        today = datetime.now(timezone.utc)
        long_title = "A Comprehensive Guide To Understanding Complex Architecture Patterns"
        _create_entry(
            kb_root / "architecture" / "guide.md",
            long_title,
            ["architecture", "patterns"],
            created=today,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--category=architecture", "--full-titles"])

        assert result.exit_code == 0
        assert long_title in result.output


class TestSearchFullTitles:
    """Test mx search --full-titles flag."""

    def test_search_truncates_long_titles_by_default(self, kb_root, index_root):
        """By default, long titles are truncated in search results."""
        today = datetime.now(timezone.utc)
        long_title = "Search Implementation Overview With Extended Details"
        _create_entry(
            kb_root / "development" / "search-impl.md",
            long_title,
            ["search", "implementation"],
            created=today,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "search implementation"])

        assert result.exit_code == 0
        # Title should be truncated (default limit is 30 chars for search)
        assert "..." in result.output

    def test_search_full_titles_shows_complete_titles(self, kb_root, index_root):
        """With --full-titles, complete titles are shown in search results."""
        today = datetime.now(timezone.utc)
        long_title = "Search Implementation Overview With Extended Details"
        _create_entry(
            kb_root / "development" / "search-impl.md",
            long_title,
            ["search", "implementation"],
            created=today,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "search implementation", "--full-titles"])

        assert result.exit_code == 0
        # Full title should be present
        assert long_title in result.output

    def test_search_full_titles_with_other_options(self, kb_root, index_root):
        """--full-titles works alongside other search options."""
        today = datetime.now(timezone.utc)
        long_title = "Detailed Python Development Guidelines And Best Practices"
        _create_entry(
            kb_root / "development" / "python-guide.md",
            long_title,
            ["python", "guidelines"],
            created=today,
        )

        runner = CliRunner()
        result = runner.invoke(cli, [
            "search", "python guidelines",
            "--full-titles",
            "--limit=5",
            "--mode=hybrid"
        ])

        assert result.exit_code == 0
        assert long_title in result.output
