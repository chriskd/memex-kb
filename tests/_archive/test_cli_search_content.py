"""Tests for CLI search --content flag functionality."""

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
    (root / "test").mkdir()
    monkeypatch.setenv("MEMEX_USER_KB_ROOT", str(root))
    return root


@pytest.fixture
def index_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary index root."""
    root = tmp_path / ".indices"
    root.mkdir()
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(root))
    return root


def _create_entry(path: Path, title: str, tags: list[str], content_body: str):
    """Helper to create a KB entry with frontmatter."""
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    created = datetime.now(timezone.utc)
    content = f"""---
title: {title}
tags:
{tags_yaml}
created: {created.isoformat()}
---

{content_body}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class TestSearchContentFlag:
    """Tests for --content flag behavior."""

    def test_json_output_includes_content_when_flag_used(self, kb_root, index_root):
        """JSON output should include 'content' field when --content is used."""
        _create_entry(
            kb_root / "test" / "example.md",
            title="Example Entry",
            tags=["test", "example"],
            content_body="# Main Heading\n\nThis is the full document content with multiple paragraphs.\n\nSecond paragraph here.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "example", "--content", "--json"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        data = json.loads(result.output)
        assert len(data) >= 1
        # When --content is used, we should have 'content' field
        assert "content" in data[0]
        assert "Main Heading" in data[0]["content"]
        assert "full document content" in data[0]["content"]
        # Should NOT have snippet when content is present
        assert "snippet" not in data[0]

    def test_json_output_uses_snippet_without_content_flag(self, kb_root, index_root):
        """JSON output should use 'snippet' field when --content is NOT used."""
        _create_entry(
            kb_root / "test" / "example.md",
            title="Example Entry",
            tags=["test", "example"],
            content_body="# Main Heading\n\nThis is the full document content.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "example", "--json"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        data = json.loads(result.output)
        assert len(data) >= 1
        # When --content is NOT used, we should have 'snippet' field
        assert "snippet" in data[0]
        assert "content" not in data[0]

    def test_table_output_shows_content_section_when_flag_used(self, kb_root, index_root):
        """Table output should show full content section when --content is used."""
        _create_entry(
            kb_root / "test" / "example.md",
            title="Example Entry",
            tags=["test", "example"],
            content_body="# Content Heading\n\nUnique marker text for testing.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "example", "--content"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        # Should contain the table header
        assert "PATH" in result.output
        # Should contain the content section separator
        assert "====" in result.output
        # Should contain the full content
        assert "Content Heading" in result.output
        assert "Unique marker text" in result.output

    def test_table_output_no_content_section_without_flag(self, kb_root, index_root):
        """Table output should NOT show content section when --content is NOT used."""
        _create_entry(
            kb_root / "test" / "example.md",
            title="Example Entry",
            tags=["test", "example"],
            content_body="# Content Heading\n\nSome text here.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "example"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        # Should contain the table
        assert "PATH" in result.output
        # Should NOT contain content section separator
        assert "====" not in result.output


class TestSearchContentFlagHelp:
    """Tests for --content flag help text."""

    def test_content_flag_in_help(self):
        """Search help should include --content option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "--help"])

        assert result.exit_code == 0
        assert "--content" in result.output or "-c" in result.output
        assert "full content" in result.output.lower()
