"""Tests for mx get --title CLI option."""

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
    for category in ("development", "infrastructure", "guides"):
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
    content: str = "Some content here.",
):
    """Helper to create a KB entry with frontmatter."""
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    created = datetime.now(timezone.utc)
    file_content = f"""---
title: {title}
tags:
{tags_yaml}
created: {created.isoformat()}
---

## Content

{content}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(file_content)


class TestGetByTitle:
    """Test mx get --title functionality."""

    def test_get_by_title_single_match(self, kb_root, index_root):
        """Get entry by title when there's exactly one match."""
        _create_entry(
            kb_root / "infrastructure" / "docker-guide.md",
            "Docker Guide",
            ["docker", "infrastructure"],
            "Docker configuration and usage.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["get", "--title=Docker Guide"])

        assert result.exit_code == 0
        assert "# Docker Guide" in result.output
        assert "Docker configuration and usage." in result.output

    def test_get_by_title_short_option(self, kb_root, index_root):
        """Get entry by title using -t short option."""
        _create_entry(
            kb_root / "guides" / "python-setup.md",
            "Python Setup",
            ["python"],
            "Python environment setup.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["get", "-t", "Python Setup"])

        assert result.exit_code == 0
        assert "# Python Setup" in result.output

    def test_get_by_title_case_insensitive(self, kb_root, index_root):
        """Title matching is case-insensitive."""
        _create_entry(
            kb_root / "development" / "api-docs.md",
            "API Documentation",
            ["api", "docs"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["get", "--title=api documentation"])

        assert result.exit_code == 0
        assert "# API Documentation" in result.output

    def test_get_by_title_no_match_shows_error(self, kb_root, index_root):
        """No match shows error message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["get", "--title=Nonexistent Entry"])

        assert result.exit_code == 1
        assert "Error: No entry found with title 'Nonexistent Entry'" in result.output

    def test_get_by_title_no_match_shows_suggestions(self, kb_root, index_root):
        """No exact match shows fuzzy suggestions."""
        _create_entry(
            kb_root / "infrastructure" / "docker-guide.md",
            "Docker Guide",
            ["docker"],
        )
        _create_entry(
            kb_root / "infrastructure" / "docker-compose.md",
            "Docker Compose Setup",
            ["docker"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["get", "--title=Docker Guid"])

        assert result.exit_code == 1
        assert "Did you mean:" in result.output
        assert "Docker Guide" in result.output

    def test_get_by_title_multiple_matches_shows_candidates(self, kb_root, index_root):
        """Multiple matches show error with candidate paths."""
        _create_entry(
            kb_root / "development" / "getting-started.md",
            "Getting Started",
            ["intro"],
        )
        _create_entry(
            kb_root / "guides" / "getting-started.md",
            "Getting Started",
            ["intro"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["get", "--title=Getting Started"])

        assert result.exit_code == 1
        assert "Error: Multiple entries found with title 'Getting Started':" in result.output
        assert "development/getting-started.md" in result.output
        assert "guides/getting-started.md" in result.output
        assert "Use the full path to specify which entry." in result.output

    def test_get_by_title_with_json_output(self, kb_root, index_root):
        """Get by title with --json flag works."""
        _create_entry(
            kb_root / "development" / "test-entry.md",
            "Test Entry",
            ["testing"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["get", "--title=Test Entry", "--json"])

        assert result.exit_code == 0
        assert '"title": "Test Entry"' in result.output

    def test_get_by_title_with_metadata_flag(self, kb_root, index_root):
        """Get by title with --metadata flag works."""
        _create_entry(
            kb_root / "development" / "meta-entry.md",
            "Meta Entry",
            ["meta", "test"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["get", "--title=Meta Entry", "--metadata"])

        assert result.exit_code == 0
        assert "Title:    Meta Entry" in result.output
        assert "Tags:     meta, test" in result.output


class TestGetValidation:
    """Test mx get argument validation."""

    def test_get_requires_path_or_title(self, kb_root, index_root):
        """Get without path or --title shows error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["get"])

        assert result.exit_code == 1
        assert "Must specify either PATH or --title" in result.output

    def test_get_cannot_use_both_path_and_title(self, kb_root, index_root):
        """Get with both path and --title shows error."""
        _create_entry(
            kb_root / "development" / "entry.md",
            "Entry",
            ["test"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["get", "development/entry.md", "--title=Entry"])

        assert result.exit_code == 1
        assert "Cannot specify both PATH and --title" in result.output

    def test_get_by_path_still_works(self, kb_root, index_root):
        """Traditional path-based get still works."""
        _create_entry(
            kb_root / "development" / "path-entry.md",
            "Path Entry",
            ["test"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["get", "development/path-entry.md"])

        assert result.exit_code == 0
        assert "# Path Entry" in result.output


class TestGetHelpText:
    """Test mx get help text includes --title option."""

    def test_help_shows_title_option(self):
        """Help text shows --title option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["get", "--help"])

        assert result.exit_code == 0
        assert "--title" in result.output or "-t" in result.output
        assert "Get entry by title instead of path" in result.output
