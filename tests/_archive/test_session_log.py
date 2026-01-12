"""Tests for session-log command (mx session-log / core.log_session)."""

import pytest
from pathlib import Path

from click.testing import CliRunner

from memex import core
from memex.cli import cli


def _create_entry(path: Path, title: str, content: str, tags: list | None = None):
    """Helper to create test entries with proper frontmatter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tags = tags or ["test"]
    frontmatter = f"""---
title: {title}
tags:
{chr(10).join(f'- {t}' for t in tags)}
created: '2024-01-15T10:00:00'
---

"""
    path.write_text(frontmatter + content)


@pytest.fixture
def session_kb_root(tmp_path, monkeypatch):
    """Set up KB root for session-log tests."""
    kb_root = tmp_path / "kb"
    index_root = tmp_path / "index"
    kb_root.mkdir()
    index_root.mkdir()
    monkeypatch.setenv("MEMEX_KB_ROOT", str(kb_root))
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))
    return kb_root


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


class TestSessionLogExplicitPath:
    """Tests for explicit --entry path handling."""

    @pytest.mark.asyncio
    async def test_explicit_path_creates_at_exact_location(self, session_kb_root):
        """When --entry is provided and file doesn't exist, create at exact path."""
        result = await core.log_session(
            message="Test session message",
            entry_path="projects/my-sessions.md",
        )

        assert result.path == "projects/my-sessions.md"
        assert result.action == "created"

        # Verify file was created at the exact specified path
        file_path = session_kb_root / "projects" / "my-sessions.md"
        assert file_path.exists()

        content = file_path.read_text()
        assert "Test session message" in content

    @pytest.mark.asyncio
    async def test_explicit_path_appends_to_existing(self, session_kb_root):
        """When --entry is provided and file exists, append to it."""
        _create_entry(
            session_kb_root / "projects" / "sessions.md",
            "Session Log",
            "# Session Log\n\nExisting content.",
            tags=["sessions"],
        )

        result = await core.log_session(
            message="New session message",
            entry_path="projects/sessions.md",
        )

        assert result.path == "projects/sessions.md"
        assert result.action == "appended"

        content = (session_kb_root / "projects" / "sessions.md").read_text()
        assert "Existing content" in content
        assert "New session message" in content

    def test_explicit_path_via_cli(self, session_kb_root, runner):
        """Test explicit --entry path via CLI creates at exact location."""
        result = runner.invoke(
            cli,
            ["session-log", "--message", "CLI session", "--entry", "custom/path/log.md"],
        )

        assert result.exit_code == 0
        assert "Logged to:" in result.output

        # Verify file created at exact path
        file_path = session_kb_root / "custom" / "path" / "log.md"
        assert file_path.exists()

        content = file_path.read_text()
        assert "CLI session" in content

    def test_explicit_path_with_subdirectory_via_cli(self, session_kb_root, runner):
        """Test explicit --entry creates nested directories as needed."""
        result = runner.invoke(
            cli,
            ["session-log", "-m", "Deep session", "-e", "a/b/c/sessions.md"],
        )

        assert result.exit_code == 0

        file_path = session_kb_root / "a" / "b" / "c" / "sessions.md"
        assert file_path.exists()


class TestSessionLogPathPreservation:
    """Tests to ensure path is used literally, not reinterpreted."""

    @pytest.mark.asyncio
    async def test_path_not_modified_by_project_name(self, session_kb_root):
        """Explicit path should not be modified based on project name."""
        # When no project context exists, project_name becomes "Unknown"
        # The bug was that 'projects/sessions.md' became 'projects/unknown-sessions.md'
        result = await core.log_session(
            message="Test message",
            entry_path="projects/sessions.md",
        )

        # Path should be exactly as specified
        assert result.path == "projects/sessions.md"

        # File should be at the exact path, not at 'unknown-sessions.md'
        assert (session_kb_root / "projects" / "sessions.md").exists()
        assert not (session_kb_root / "projects" / "unknown-sessions.md").exists()

    def test_cli_path_not_modified(self, session_kb_root, runner):
        """CLI --entry path should be used literally."""
        result = runner.invoke(
            cli,
            ["session-log", "-m", "Test", "-e", "my/exact-path.md"],
        )

        assert result.exit_code == 0

        # File should exist at exact path
        assert (session_kb_root / "my" / "exact-path.md").exists()
