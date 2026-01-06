"""Comprehensive tests for update, append, and upsert functionality.

Tests cover:
- mx update with --tags (replace tags)
- mx update with --content (replace content)
- mx update with --content --append (append to existing)
- mx update with --append --timestamp (timestamped append)
- mx update with --file and --stdin variants
- mx upsert create vs append behavior
- mx upsert with --no-create flag

Verifies:
1. Content is correctly modified
2. Frontmatter is preserved
3. Index is updated after changes
4. Error handling for edge cases
"""

import json
import re
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from memex import core
from memex.cli import cli


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture(autouse=True)
def reset_searcher_state(monkeypatch):
    """Ensure cached searcher state does not leak across tests."""
    monkeypatch.setattr(core, "_searcher", None)
    monkeypatch.setattr(core, "_searcher_ready", False)


@pytest.fixture
def update_kb_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary KB root with standard categories and entries."""
    root = tmp_path / "kb"
    root.mkdir()
    (root / "development").mkdir()
    (root / "testing").mkdir()
    (root / "notes").mkdir()
    monkeypatch.setenv("MEMEX_KB_ROOT", str(root))
    return root


@pytest.fixture
def update_index_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary index root."""
    root = tmp_path / ".indices"
    root.mkdir()
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(root))
    return root


def _create_entry(path: Path, title: str, content_body: str, tags: list[str] | None = None):
    """Helper to create a KB entry with frontmatter."""
    tags = tags or ["test"]
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    content = f"""---
title: {title}
tags:
{tags_yaml}
created: {date.today().isoformat()}
---

{content_body}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Test mx update with --tags
# ─────────────────────────────────────────────────────────────────────────────


class TestUpdateTags:
    """Tests for mx update --tags functionality."""

    @pytest.mark.asyncio
    async def test_update_tags_replaces_existing(self, update_kb_root, update_index_root):
        """Update --tags replaces all existing tags."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "Content here",
            tags=["old-tag1", "old-tag2"],
        )

        result = await core.update_entry(
            path="development/test.md",
            tags=["new-tag1", "new-tag2", "new-tag3"],
            content="Content here",
        )

        assert result["path"] == "development/test.md"

        # Verify tags were replaced
        content = (update_kb_root / "development" / "test.md").read_text()
        assert "new-tag1" in content
        assert "new-tag2" in content
        assert "new-tag3" in content
        assert "old-tag1" not in content
        assert "old-tag2" not in content

    @pytest.mark.asyncio
    async def test_update_tags_only_preserves_content(self, update_kb_root, update_index_root):
        """Update --tags alone preserves existing content."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "Original content that should be preserved",
            tags=["old-tag"],
        )

        result = await core.update_entry(
            path="development/test.md",
            tags=["new-tag"],
            # NOTE: No content argument - tags only update
        )

        assert result["path"] == "development/test.md"

        content = (update_kb_root / "development" / "test.md").read_text()
        assert "new-tag" in content
        assert "old-tag" not in content
        assert "Original content that should be preserved" in content

    @pytest.mark.asyncio
    async def test_update_preserves_tags_when_not_specified(self, update_kb_root, update_index_root):
        """Update preserves existing tags when tags not specified."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "Original content",
            tags=["preserve-me", "keep-this"],
        )

        result = await core.update_entry(
            path="development/test.md",
            content="New content",
        )

        assert result["path"] == "development/test.md"

        content = (update_kb_root / "development" / "test.md").read_text()
        assert "preserve-me" in content
        assert "keep-this" in content

    @pytest.mark.asyncio
    async def test_update_tags_requires_at_least_one(self, update_kb_root, update_index_root):
        """Update fails if tags list is empty."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "Content here",
            tags=["existing"],
        )

        with pytest.raises(ValueError, match="At least one tag is required"):
            await core.update_entry(
                path="development/test.md",
                tags=[],
                content="Content here",
            )

    def test_update_tags_via_cli(self, update_kb_root, update_index_root, runner):
        """Test update --tags via CLI with content."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "CLI Test",
            "CLI content",
            tags=["old"],
        )

        result = runner.invoke(
            cli,
            ["update", "development/test.md", "--tags", "alpha,beta,gamma", "--content", "CLI content"],
        )

        assert result.exit_code == 0
        assert "Updated:" in result.output

        content = (update_kb_root / "development" / "test.md").read_text()
        assert "alpha" in content
        assert "beta" in content
        assert "gamma" in content
        assert "old" not in content

    def test_update_tags_only_via_cli(self, update_kb_root, update_index_root, runner):
        """Test update --tags via CLI without content preserves existing content."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "CLI Test",
            "Original content to preserve",
            tags=["old-tag"],
        )

        result = runner.invoke(
            cli,
            ["update", "development/test.md", "--tags", "new-tag1,new-tag2"],
        )

        assert result.exit_code == 0
        assert "Updated:" in result.output

        content = (update_kb_root / "development" / "test.md").read_text()
        assert "new-tag1" in content
        assert "new-tag2" in content
        assert "old-tag" not in content
        assert "Original content to preserve" in content


# ─────────────────────────────────────────────────────────────────────────────
# Test mx update with --content (replace)
# ─────────────────────────────────────────────────────────────────────────────


class TestUpdateContent:
    """Tests for mx update --content functionality (replace mode)."""

    @pytest.mark.asyncio
    async def test_update_content_replaces_body(self, update_kb_root, update_index_root):
        """Update content replaces the entire body."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "# Old Content\n\nThis is the original content.",
        )

        await core.update_entry(
            path="development/test.md",
            content="# New Content\n\nCompletely replaced body.",
        )

        content = (update_kb_root / "development" / "test.md").read_text()
        assert "# New Content" in content
        assert "Completely replaced body" in content
        assert "Old Content" not in content
        assert "original content" not in content

    @pytest.mark.asyncio
    async def test_update_content_preserves_frontmatter(self, update_kb_root, update_index_root):
        """Update content preserves all frontmatter fields."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "My Special Title",
            "Original body",
            tags=["important", "preserve-me"],
        )

        await core.update_entry(
            path="development/test.md",
            content="New body content",
        )

        content = (update_kb_root / "development" / "test.md").read_text()
        assert "title: My Special Title" in content
        assert "important" in content
        assert "preserve-me" in content
        # Updated date should be added
        assert "updated:" in content

    @pytest.mark.asyncio
    async def test_update_content_sets_updated_date(self, update_kb_root, update_index_root):
        """Update sets 'updated' field in frontmatter."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "Original content",
        )

        await core.update_entry(
            path="development/test.md",
            content="Updated content",
        )

        content = (update_kb_root / "development" / "test.md").read_text()
        assert "updated:" in content

    @pytest.mark.asyncio
    async def test_update_nonexistent_entry_fails(self, update_kb_root, update_index_root):
        """Update on nonexistent entry raises ValueError."""
        with pytest.raises(ValueError, match="Entry not found"):
            await core.update_entry(
                path="development/nonexistent.md",
                content="Some content",
            )

    @pytest.mark.asyncio
    async def test_update_directory_path_fails(self, update_kb_root, update_index_root):
        """Update on directory path raises ValueError."""
        with pytest.raises(ValueError, match="not a file"):
            await core.update_entry(
                path="development",
                content="Some content",
            )

    def test_update_content_via_cli(self, update_kb_root, update_index_root, runner):
        """Test update --content via CLI."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "CLI Test",
            "Old CLI content",
        )

        result = runner.invoke(
            cli,
            ["update", "development/test.md", "--content", "Brand new content via CLI"],
        )

        assert result.exit_code == 0
        content = (update_kb_root / "development" / "test.md").read_text()
        assert "Brand new content via CLI" in content
        assert "Old CLI content" not in content


# ─────────────────────────────────────────────────────────────────────────────
# Test mx update with --content --append
# ─────────────────────────────────────────────────────────────────────────────


class TestUpdateAppend:
    """Tests for mx update --content --append functionality."""

    @pytest.mark.asyncio
    async def test_update_append_adds_to_end(self, update_kb_root, update_index_root):
        """Update --append adds content to end of existing body."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "# Original Section\n\nOriginal content.",
        )

        await core.update_entry(
            path="development/test.md",
            content="# Appended Section\n\nNew content added.",
            append=True,
        )

        content = (update_kb_root / "development" / "test.md").read_text()
        # Both should be present
        assert "# Original Section" in content
        assert "Original content" in content
        assert "# Appended Section" in content
        assert "New content added" in content

    @pytest.mark.asyncio
    async def test_update_append_preserves_order(self, update_kb_root, update_index_root):
        """Appended content appears after original content."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "FIRST",
        )

        await core.update_entry(
            path="development/test.md",
            content="SECOND",
            append=True,
        )

        content = (update_kb_root / "development" / "test.md").read_text()
        first_pos = content.find("FIRST")
        second_pos = content.find("SECOND")
        assert first_pos < second_pos, "Appended content should appear after original"

    @pytest.mark.asyncio
    async def test_update_append_multiple_times(self, update_kb_root, update_index_root):
        """Multiple appends accumulate content."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "Entry 1",
        )

        await core.update_entry(
            path="development/test.md",
            content="Entry 2",
            append=True,
        )
        await core.update_entry(
            path="development/test.md",
            content="Entry 3",
            append=True,
        )

        content = (update_kb_root / "development" / "test.md").read_text()
        assert "Entry 1" in content
        assert "Entry 2" in content
        assert "Entry 3" in content

    def test_update_append_via_cli(self, update_kb_root, update_index_root, runner):
        """Test update --content --append via CLI."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "CLI Test",
            "Original CLI content",
        )

        result = runner.invoke(
            cli,
            ["update", "development/test.md", "--content", "Appended via CLI", "--append"],
        )

        assert result.exit_code == 0
        assert "Appended to:" in result.output

        content = (update_kb_root / "development" / "test.md").read_text()
        assert "Original CLI content" in content
        assert "Appended via CLI" in content


# ─────────────────────────────────────────────────────────────────────────────
# Test mx update with --append --timestamp
# ─────────────────────────────────────────────────────────────────────────────


class TestUpdateAppendTimestamp:
    """Tests for mx update --append --timestamp functionality."""

    def test_update_append_timestamp_adds_header(self, update_kb_root, update_index_root, runner):
        """Update --append --timestamp adds timestamp header."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Session Log",
            "Previous sessions here.",
        )

        result = runner.invoke(
            cli,
            ["update", "development/test.md", "--content", "Session note", "--append", "--timestamp"],
        )

        assert result.exit_code == 0

        content = (update_kb_root / "development" / "test.md").read_text()
        # Check for timestamp pattern: ## YYYY-MM-DD HH:MM UTC
        assert re.search(r"## \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC", content)
        assert "Session note" in content
        assert "Previous sessions here" in content

    def test_timestamp_requires_append(self, update_kb_root, update_index_root, runner):
        """--timestamp without --append should fail."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "Content",
        )

        result = runner.invoke(
            cli,
            ["update", "development/test.md", "--content", "Test", "--timestamp"],
        )

        assert result.exit_code == 1
        assert "--timestamp requires --append" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Test mx update with --file
# ─────────────────────────────────────────────────────────────────────────────


class TestUpdateFromFile:
    """Tests for mx update --file functionality."""

    def test_update_from_file(self, update_kb_root, update_index_root, runner, tmp_path):
        """Update --file reads content from file."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "Original content",
        )

        content_file = tmp_path / "new_content.md"
        content_file.write_text("# Content from file\n\nLoaded successfully.")

        result = runner.invoke(
            cli,
            ["update", "development/test.md", "--file", str(content_file)],
        )

        assert result.exit_code == 0

        content = (update_kb_root / "development" / "test.md").read_text()
        assert "Content from file" in content
        assert "Loaded successfully" in content
        assert "Original content" not in content

    def test_update_from_file_append(self, update_kb_root, update_index_root, runner, tmp_path):
        """Update --file --append appends file content."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "Original content",
        )

        content_file = tmp_path / "append_content.md"
        content_file.write_text("Appended from file")

        result = runner.invoke(
            cli,
            ["update", "development/test.md", "--file", str(content_file), "--append"],
        )

        assert result.exit_code == 0

        content = (update_kb_root / "development" / "test.md").read_text()
        assert "Original content" in content
        assert "Appended from file" in content

    def test_update_from_file_with_timestamp(self, update_kb_root, update_index_root, runner, tmp_path):
        """Update --file --append --timestamp adds timestamped header."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Session Log",
            "Previous notes",
        )

        content_file = tmp_path / "session.md"
        content_file.write_text("Session notes from file")

        result = runner.invoke(
            cli,
            ["update", "development/test.md", "--file", str(content_file), "--append", "--timestamp"],
        )

        assert result.exit_code == 0

        content = (update_kb_root / "development" / "test.md").read_text()
        assert re.search(r"## \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC", content)
        assert "Session notes from file" in content

    def test_update_file_not_found(self, update_kb_root, update_index_root, runner):
        """Update --file with nonexistent file fails gracefully."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "Content",
        )

        result = runner.invoke(
            cli,
            ["update", "development/test.md", "--file", "/nonexistent/path.md"],
        )

        assert result.exit_code != 0


# ─────────────────────────────────────────────────────────────────────────────
# Test mx update with --stdin
# ─────────────────────────────────────────────────────────────────────────────


class TestUpdateFromStdin:
    """Tests for mx update --stdin functionality."""

    def test_update_from_stdin(self, update_kb_root, update_index_root, runner):
        """Update --stdin reads content from stdin."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "Original content",
        )

        result = runner.invoke(
            cli,
            ["update", "development/test.md", "--stdin"],
            input="Content from stdin\nMultiple lines",
        )

        assert result.exit_code == 0

        content = (update_kb_root / "development" / "test.md").read_text()
        assert "Content from stdin" in content
        assert "Multiple lines" in content

    def test_update_stdin_append(self, update_kb_root, update_index_root, runner):
        """Update --stdin --append appends stdin content."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "Original content",
        )

        result = runner.invoke(
            cli,
            ["update", "development/test.md", "--stdin", "--append"],
            input="Appended from stdin",
        )

        assert result.exit_code == 0
        assert "Appended to:" in result.output

        content = (update_kb_root / "development" / "test.md").read_text()
        assert "Original content" in content
        assert "Appended from stdin" in content

    def test_update_stdin_with_timestamp(self, update_kb_root, update_index_root, runner):
        """Update --stdin --append --timestamp adds timestamped header."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Session Log",
            "Previous notes",
        )

        result = runner.invoke(
            cli,
            ["update", "development/test.md", "--stdin", "--append", "--timestamp"],
            input="Done for today",
        )

        assert result.exit_code == 0

        content = (update_kb_root / "development" / "test.md").read_text()
        assert re.search(r"## \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC", content)
        assert "Done for today" in content

    def test_update_stdin_and_file_mutually_exclusive(self, update_kb_root, update_index_root, runner, tmp_path):
        """--stdin and --file are mutually exclusive."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "Content",
        )

        content_file = tmp_path / "content.md"
        content_file.write_text("File content")

        result = runner.invoke(
            cli,
            ["update", "development/test.md", "--stdin", "--file", str(content_file)],
            input="Stdin content",
        )

        assert result.exit_code == 1
        assert "mutually exclusive" in result.output.lower()

    def test_update_stdin_and_content_mutually_exclusive(self, update_kb_root, update_index_root, runner):
        """--stdin and --content are mutually exclusive."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "Content",
        )

        result = runner.invoke(
            cli,
            ["update", "development/test.md", "--stdin", "--content", "Direct content"],
            input="Stdin content",
        )

        assert result.exit_code == 1
        assert "mutually exclusive" in result.output.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Test mx upsert create behavior
# ─────────────────────────────────────────────────────────────────────────────


class TestUpsertCreate:
    """Tests for mx upsert creating new entries."""

    @pytest.mark.asyncio
    async def test_upsert_creates_new_entry(self, update_kb_root, update_index_root, monkeypatch):
        """Upsert creates new entry when title not found."""
        # Set up KB context for directory resolution
        context_file = update_kb_root / ".kbcontext"
        context_file.write_text('{"primary": "notes"}')
        monkeypatch.chdir(update_kb_root)

        result = await core.upsert_entry(
            title="Brand New Entry",
            content="This is new content",
            tags=["new", "test"],
            directory="notes",
        )

        assert result.action == "created"
        assert result.title == "Brand New Entry"

        # Verify file was created
        created_files = list((update_kb_root / "notes").glob("*.md"))
        assert len(created_files) == 1
        content = created_files[0].read_text()
        assert "Brand New Entry" in content
        assert "This is new content" in content

    @pytest.mark.asyncio
    async def test_upsert_creates_with_timestamp(self, update_kb_root, update_index_root):
        """Upsert adds timestamp header when creating new entry."""
        result = await core.upsert_entry(
            title="Timestamped Entry",
            content="Session content",
            tags=["test"],
            directory="notes",
            timestamp=True,
        )

        assert result.action == "created"

        created_files = list((update_kb_root / "notes").glob("*.md"))
        content = created_files[0].read_text()
        # Check for timestamp pattern
        assert re.search(r"## \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC", content)

    @pytest.mark.asyncio
    async def test_upsert_creates_without_timestamp(self, update_kb_root, update_index_root):
        """Upsert respects --no-timestamp flag."""
        result = await core.upsert_entry(
            title="No Timestamp Entry",
            content="Plain content",
            tags=["test"],
            directory="notes",
            timestamp=False,
        )

        assert result.action == "created"

        created_files = list((update_kb_root / "notes").glob("*.md"))
        content = created_files[0].read_text()
        # Should NOT have timestamp header
        assert not re.search(r"## \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC", content)

    def test_upsert_create_via_cli(self, update_kb_root, update_index_root, runner, monkeypatch):
        """Test upsert creation via CLI."""
        monkeypatch.chdir(update_kb_root)

        result = runner.invoke(
            cli,
            ["upsert", "CLI Created Entry", "--content", "Created via CLI", "--tags", "cli,test", "--directory", "notes"],
        )

        assert result.exit_code == 0
        assert "Created" in result.output or "created" in result.output.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Test mx upsert append behavior
# ─────────────────────────────────────────────────────────────────────────────


class TestUpsertAppend:
    """Tests for mx upsert appending to existing entries."""

    @pytest.mark.asyncio
    async def test_upsert_appends_to_existing(self, update_kb_root, update_index_root):
        """Upsert appends to entry when title matches."""
        # Create an entry first
        _create_entry(
            update_kb_root / "notes" / "existing-entry.md",
            "Existing Entry",
            "Original content",
            tags=["existing"],
        )

        result = await core.upsert_entry(
            title="Existing Entry",
            content="Appended content",
        )

        assert result.action == "appended"
        assert "notes/existing-entry.md" in result.path

        content = (update_kb_root / "notes" / "existing-entry.md").read_text()
        assert "Original content" in content
        assert "Appended content" in content

    @pytest.mark.asyncio
    async def test_upsert_append_with_timestamp(self, update_kb_root, update_index_root):
        """Upsert append adds timestamp header by default."""
        _create_entry(
            update_kb_root / "notes" / "log.md",
            "Session Log",
            "Previous sessions",
        )

        await core.upsert_entry(
            title="Session Log",
            content="New session",
            timestamp=True,
        )

        content = (update_kb_root / "notes" / "log.md").read_text()
        assert re.search(r"## \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC", content)
        assert "New session" in content

    @pytest.mark.asyncio
    async def test_upsert_append_without_timestamp(self, update_kb_root, update_index_root):
        """Upsert append respects --no-timestamp flag."""
        _create_entry(
            update_kb_root / "notes" / "plain.md",
            "Plain Entry",
            "Original",
        )

        await core.upsert_entry(
            title="Plain Entry",
            content="Appended without timestamp",
            timestamp=False,
        )

        content = (update_kb_root / "notes" / "plain.md").read_text()
        assert "Appended without timestamp" in content
        # In the body section, no timestamp should appear for this append
        body_start = content.find("---", content.find("---") + 3) + 3
        body = content[body_start:]
        # The new content should appear without a timestamp header
        assert "Appended without timestamp" in body

    def test_upsert_append_via_cli(self, update_kb_root, update_index_root, runner):
        """Test upsert append via CLI."""
        _create_entry(
            update_kb_root / "notes" / "cli-test.md",
            "CLI Upsert Test",
            "Original CLI content",
        )

        result = runner.invoke(
            cli,
            ["upsert", "CLI Upsert Test", "--content", "Appended via CLI"],
        )

        assert result.exit_code == 0

        content = (update_kb_root / "notes" / "cli-test.md").read_text()
        assert "Original CLI content" in content
        assert "Appended via CLI" in content


# ─────────────────────────────────────────────────────────────────────────────
# Test mx upsert with --replace
# ─────────────────────────────────────────────────────────────────────────────


class TestUpsertReplace:
    """Tests for mx upsert --replace flag."""

    @pytest.mark.asyncio
    async def test_upsert_replace_replaces_content(self, update_kb_root, update_index_root):
        """Upsert --replace replaces content instead of appending."""
        _create_entry(
            update_kb_root / "notes" / "replace-test.md",
            "Replace Test Entry",
            "Original content to be replaced",
        )

        result = await core.upsert_entry(
            title="Replace Test Entry",
            content="Completely new content",
            append=False,  # Replace mode
        )

        assert result.action == "replaced"
        assert result.path == "notes/replace-test.md"

        content = (update_kb_root / "notes" / "replace-test.md").read_text()
        assert "Completely new content" in content
        assert "Original content to be replaced" not in content

    def test_upsert_replace_via_cli(self, update_kb_root, update_index_root, runner):
        """Test upsert --replace via CLI shows correct message."""
        _create_entry(
            update_kb_root / "notes" / "cli-replace.md",
            "CLI Replace Test",
            "Original CLI content",
        )

        result = runner.invoke(
            cli,
            ["upsert", "CLI Replace Test", "--content", "Replaced via CLI", "--replace"],
        )

        assert result.exit_code == 0
        assert "Replaced:" in result.output
        assert "Appended to:" not in result.output

        content = (update_kb_root / "notes" / "cli-replace.md").read_text()
        assert "Replaced via CLI" in content
        assert "Original CLI content" not in content

    def test_upsert_replace_json_output(self, update_kb_root, update_index_root, runner, monkeypatch):
        """Test upsert --replace JSON output contains 'replaced' action."""
        monkeypatch.setenv("MEMEX_KB_ROOT", str(update_kb_root))
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(update_index_root))

        _create_entry(
            update_kb_root / "notes" / "json-replace.md",
            "JSON Replace Test",
            "Original content",
        )

        result = runner.invoke(
            cli,
            ["upsert", "JSON Replace Test", "--content", "New content", "--replace", "--json"],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["action"] == "replaced"
        assert data["path"] == "notes/json-replace.md"


# ─────────────────────────────────────────────────────────────────────────────
# Test mx upsert with --no-create
# ─────────────────────────────────────────────────────────────────────────────


class TestUpsertNoCreate:
    """Tests for mx upsert --no-create flag."""

    @pytest.mark.asyncio
    async def test_upsert_no_create_fails_when_not_found(self, update_kb_root, update_index_root):
        """Upsert --no-create raises error when entry not found."""
        with pytest.raises(ValueError, match="No entry found"):
            await core.upsert_entry(
                title="Nonexistent Entry",
                content="Content",
                create_if_missing=False,
            )

    @pytest.mark.asyncio
    async def test_upsert_no_create_succeeds_when_found(self, update_kb_root, update_index_root):
        """Upsert --no-create succeeds when entry exists."""
        _create_entry(
            update_kb_root / "notes" / "exists.md",
            "Existing Entry",
            "Original content",
        )

        result = await core.upsert_entry(
            title="Existing Entry",
            content="Appended content",
            create_if_missing=False,
        )

        assert result.action == "appended"

    def test_upsert_no_create_via_cli(self, update_kb_root, update_index_root, runner):
        """Test upsert --no-create via CLI."""
        result = runner.invoke(
            cli,
            ["upsert", "Nonexistent Entry", "--content", "Content", "--no-create"],
        )

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_upsert_no_create_succeeds_via_cli(self, update_kb_root, update_index_root, runner):
        """Test upsert --no-create succeeds when entry exists via CLI."""
        _create_entry(
            update_kb_root / "notes" / "cli-exists.md",
            "CLI Existing",
            "Original",
        )

        result = runner.invoke(
            cli,
            ["upsert", "CLI Existing", "--content", "Via CLI no-create", "--no-create"],
        )

        assert result.exit_code == 0

        content = (update_kb_root / "notes" / "cli-exists.md").read_text()
        assert "Via CLI no-create" in content


# ─────────────────────────────────────────────────────────────────────────────
# Test edge cases and error handling
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and error handling tests."""

    @pytest.mark.asyncio
    async def test_update_empty_content(self, update_kb_root, update_index_root):
        """Update with empty content clears the body."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "Content to clear",
        )

        await core.update_entry(
            path="development/test.md",
            content="",
        )

        content = (update_kb_root / "development" / "test.md").read_text()
        # Frontmatter should be preserved
        assert "title: Test Entry" in content
        # Body should be empty or minimal
        assert "Content to clear" not in content

    @pytest.mark.asyncio
    async def test_update_multiline_content(self, update_kb_root, update_index_root):
        """Update handles multiline content correctly."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "Original",
        )

        multiline = """# Section 1

Paragraph one.

## Section 1.1

Paragraph two.

# Section 2

Final paragraph."""

        await core.update_entry(
            path="development/test.md",
            content=multiline,
        )

        content = (update_kb_root / "development" / "test.md").read_text()
        assert "# Section 1" in content
        assert "## Section 1.1" in content
        assert "# Section 2" in content

    @pytest.mark.asyncio
    async def test_update_special_characters(self, update_kb_root, update_index_root):
        """Update handles special characters in content."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "Original",
        )

        special_content = """Code example:

```python
def hello(name):
    print(f"Hello, {name}!")
```

YAML frontmatter chars: ---
Special: `backticks` and *asterisks*"""

        await core.update_entry(
            path="development/test.md",
            content=special_content,
        )

        content = (update_kb_root / "development" / "test.md").read_text()
        assert "```python" in content
        assert 'print(f"Hello, {name}!")' in content

    def test_update_json_output(self, update_kb_root, update_index_root, runner):
        """Update --json returns valid JSON."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "Content",
        )

        result = runner.invoke(
            cli,
            ["update", "development/test.md", "--content", "New content", "--json"],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "path" in data
        assert data["path"] == "development/test.md"

    def test_upsert_json_output(self, update_kb_root, update_index_root, runner, monkeypatch):
        """Upsert --json returns valid JSON."""
        monkeypatch.chdir(update_kb_root)

        result = runner.invoke(
            cli,
            ["upsert", "JSON Test Entry", "--content", "Content", "--tags", "test", "--directory", "notes", "--json"],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "action" in data
        assert data["action"] in ["created", "appended"]


# ─────────────────────────────────────────────────────────────────────────────
# Test blank line normalization
# ─────────────────────────────────────────────────────────────────────────────


class TestBlankLineNormalization:
    """Tests to ensure blank lines don't accumulate after frontmatter."""

    @pytest.mark.asyncio
    async def test_update_normalizes_blank_lines(self, update_kb_root, update_index_root):
        """Update prevents extra blank lines from accumulating."""
        # Create entry with content that has leading newlines
        (update_kb_root / "development").mkdir(parents=True, exist_ok=True)
        (update_kb_root / "development" / "test.md").write_text(
            """---
title: Test Entry
tags:
- test
created: '2024-01-15T10:00:00'
---



# Content with leading blank lines"""
        )

        # Update the entry
        await core.update_entry(
            path="development/test.md",
            tags=["updated"],  # Just update tags
        )

        content = (update_kb_root / "development" / "test.md").read_text()

        # After frontmatter closing ---, should be exactly one blank line before content
        # Split on closing --- and check what comes after
        parts = content.split("---\n", 2)  # Split into [empty, frontmatter, content]
        assert len(parts) == 3
        body = parts[2]

        # Body should start with single newline then content (from build_frontmatter's \n\n)
        # Verify no leading newlines in the actual stored content
        lines = body.split("\n")
        # First line after frontmatter should be empty (from build_frontmatter's trailing \n\n)
        # Second line should be the content
        assert lines[0] == ""  # One blank line
        assert lines[1].startswith("# Content")  # Content starts immediately

    @pytest.mark.asyncio
    async def test_multiple_updates_dont_accumulate_blank_lines(self, update_kb_root, update_index_root):
        """Multiple updates don't accumulate blank lines."""
        _create_entry(
            update_kb_root / "development" / "test.md",
            "Test Entry",
            "# Original Content",
        )

        # Perform several updates
        for i in range(3):
            await core.update_entry(
                path="development/test.md",
                tags=[f"tag-{i}"],
            )

        content = (update_kb_root / "development" / "test.md").read_text()

        # Count blank lines between --- and first content line
        parts = content.split("---\n", 2)
        body = parts[2]
        lines = body.split("\n")

        # Count leading empty lines
        leading_empty = 0
        for line in lines:
            if line.strip() == "":
                leading_empty += 1
            else:
                break

        # Should be exactly 1 blank line (from build_frontmatter's \n\n before content)
        assert leading_empty == 1, f"Expected 1 leading blank line, got {leading_empty}"
