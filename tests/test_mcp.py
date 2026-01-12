"""Tests for MCP server tool wrappers.

Tests the MCP layer behavior: argument validation, response format, error handling.
Core logic is tested elsewhere - this file tests the MCP wrapper layer.

Target: ~25 tests, runs in <3 seconds.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memex import core, server
from memex.models import KBEntry, QualityReport, SearchResponse


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


async def _call_tool(tool_obj, /, *args, **kwargs):
    """Invoke the wrapped coroutine behind an MCP FunctionTool."""
    bound = tool_obj.fn(*args, **kwargs)
    if callable(bound):
        return await bound()
    return await bound


def _create_entry(
    path: Path,
    title: str,
    tags: list[str],
    created: datetime | None = None,
    content: str = "## Content\n\nSome content here.",
):
    """Create a KB entry with frontmatter."""
    if created is None:
        created = datetime.now(timezone.utc)

    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    full_content = f"""---
title: {title}
tags:
{tags_yaml}
created: {created.isoformat()}
---

{content}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(full_content)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


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
    for category in ("development", "architecture", "devops"):
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


# ─────────────────────────────────────────────────────────────────────────────
# search_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSearchTool:
    """Test search_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_returns_search_response(self, kb_root, index_root):
        """Happy path: returns SearchResponse model."""
        _create_entry(
            kb_root / "development" / "python.md",
            "Python Guide",
            ["python"],
        )

        result = await _call_tool(server.search_tool, query="python", limit=5)

        assert isinstance(result, SearchResponse)
        assert isinstance(result.results, list)

    @pytest.mark.asyncio
    async def test_respects_limit_parameter(self, kb_root, index_root):
        """Limit parameter restricts result count."""
        for i in range(5):
            _create_entry(
                kb_root / "development" / f"entry{i}.md",
                f"Entry {i}",
                ["test"],
            )

        result = await _call_tool(server.search_tool, query="entry", limit=2)

        assert len(result.results) <= 2

    @pytest.mark.asyncio
    async def test_empty_query_handled(self, kb_root, index_root):
        """Empty query returns valid response (not crash)."""
        result = await _call_tool(server.search_tool, query="", limit=5)

        assert isinstance(result, SearchResponse)

    @pytest.mark.asyncio
    async def test_mode_parameter_accepted(self, kb_root, index_root):
        """All search modes are accepted."""
        _create_entry(kb_root / "development" / "test.md", "Test", ["test"])

        for mode in ["hybrid", "keyword", "semantic"]:
            result = await _call_tool(server.search_tool, query="test", mode=mode)
            assert isinstance(result, SearchResponse)


# ─────────────────────────────────────────────────────────────────────────────
# add_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAddTool:
    """Test add_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_creates_entry_returns_path(self, kb_root, index_root):
        """Happy path: creates entry and returns path."""
        result = await _call_tool(
            server.add_tool,
            title="Test Entry",
            content="Test content",
            tags=["test"],
            category="development",
        )

        assert isinstance(result, dict)
        assert "path" in result
        assert (kb_root / result["path"]).exists()

    @pytest.mark.asyncio
    async def test_returns_suggested_links(self, kb_root, index_root):
        """Response includes suggested_links field."""
        result = await _call_tool(
            server.add_tool,
            title="New Entry",
            content="Content",
            tags=["test"],
            category="development",
        )

        assert "suggested_links" in result

    @pytest.mark.asyncio
    async def test_returns_suggested_tags(self, kb_root, index_root):
        """Response includes suggested_tags field."""
        result = await _call_tool(
            server.add_tool,
            title="New Entry",
            content="Content",
            tags=["test"],
            category="development",
        )

        assert "suggested_tags" in result

    @pytest.mark.asyncio
    async def test_empty_tags_rejected(self, kb_root, index_root):
        """Empty tags list raises validation error."""
        with pytest.raises(Exception):  # Pydantic or ValueError
            await _call_tool(
                server.add_tool,
                title="No Tags",
                content="Content",
                tags=[],
                category="development",
            )


# ─────────────────────────────────────────────────────────────────────────────
# get_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGetTool:
    """Test get_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_returns_kb_entry(self, kb_root, index_root):
        """Happy path: returns KBEntry model."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            ["test"],
        )

        result = await _call_tool(server.get_tool, path="development/test.md")

        assert isinstance(result, KBEntry)
        assert result.path == "development/test.md"
        assert result.metadata.title == "Test Entry"

    @pytest.mark.asyncio
    async def test_nonexistent_raises_error(self, kb_root, index_root):
        """Nonexistent entry raises ValueError."""
        with pytest.raises(ValueError, match="Entry not found"):
            await _call_tool(server.get_tool, path="development/nonexistent.md")

    @pytest.mark.asyncio
    async def test_path_traversal_rejected(self, kb_root, index_root):
        """Path traversal attempts are rejected."""
        with pytest.raises(ValueError):
            await _call_tool(server.get_tool, path="../etc/passwd")


# ─────────────────────────────────────────────────────────────────────────────
# update_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestUpdateTool:
    """Test update_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_updates_content(self, kb_root, index_root):
        """Happy path: updates entry content."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            ["test"],
            content="Original content",
        )

        result = await _call_tool(
            server.update_tool,
            path="development/test.md",
            content="Updated content",
        )

        assert result["path"] == "development/test.md"
        assert "Updated content" in (kb_root / "development" / "test.md").read_text()

    @pytest.mark.asyncio
    async def test_nonexistent_raises_error(self, kb_root, index_root):
        """Updating nonexistent entry raises ValueError."""
        with pytest.raises(ValueError, match="Entry not found"):
            await _call_tool(
                server.update_tool,
                path="development/nonexistent.md",
                content="Content",
            )

    @pytest.mark.asyncio
    async def test_no_updates_raises_error(self, kb_root, index_root):
        """Update with no content or section_updates raises error."""
        _create_entry(kb_root / "development" / "test.md", "Test", ["test"])

        with pytest.raises(ValueError, match="Provide new content or section_updates"):
            await _call_tool(
                server.update_tool,
                path="development/test.md",
                content=None,
                tags=None,
                section_updates=None,
            )


# ─────────────────────────────────────────────────────────────────────────────
# list_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestListTool:
    """Test list_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_returns_list(self, kb_root, index_root):
        """Happy path: returns list of entries."""
        _create_entry(kb_root / "development" / "test.md", "Test", ["test"])

        result = await _call_tool(server.list_tool, limit=10)

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_empty_kb_returns_empty_list(self, kb_root, index_root):
        """Empty KB returns empty list."""
        result = await _call_tool(server.list_tool)

        assert result == []

    @pytest.mark.asyncio
    async def test_invalid_category_raises_error(self, kb_root, index_root):
        """Invalid category raises ValueError."""
        with pytest.raises(ValueError, match="Category not found"):
            await _call_tool(server.list_tool, category="nonexistent")


# ─────────────────────────────────────────────────────────────────────────────
# whats_new_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestWhatsNewTool:
    """Test whats_new_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_returns_list(self, kb_root, index_root):
        """Happy path: returns list of recent entries."""
        _create_entry(kb_root / "development" / "test.md", "Test", ["test"])

        result = await _call_tool(server.whats_new_tool, days=30, limit=10)

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_respects_days_parameter(self, kb_root, index_root):
        """Days parameter filters old entries."""
        old_date = datetime.now(timezone.utc) - timedelta(days=60)
        _create_entry(
            kb_root / "development" / "old.md",
            "Old Entry",
            ["test"],
            created=old_date,
        )

        result = await _call_tool(server.whats_new_tool, days=30, limit=10)

        # Old entry should not appear
        paths = [e.get("path", "") for e in result]
        assert "development/old.md" not in paths


# ─────────────────────────────────────────────────────────────────────────────
# delete_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDeleteTool:
    """Test delete_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_deletes_entry(self, kb_root, index_root):
        """Happy path: deletes entry file."""
        _create_entry(kb_root / "development" / "test.md", "Test", ["test"])

        result = await _call_tool(server.delete_tool, path="development/test.md", force=True)

        assert result["deleted"] == "development/test.md"
        assert not (kb_root / "development" / "test.md").exists()

    @pytest.mark.asyncio
    async def test_nonexistent_raises_error(self, kb_root, index_root):
        """Deleting nonexistent entry raises ValueError."""
        with pytest.raises(ValueError, match="Entry not found"):
            await _call_tool(server.delete_tool, path="development/nonexistent.md")


# ─────────────────────────────────────────────────────────────────────────────
# backlinks_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBacklinksTool:
    """Test backlinks_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_returns_list(self, kb_root, index_root):
        """Happy path: returns list of backlinks."""
        _create_entry(kb_root / "development" / "target.md", "Target", ["test"])

        result = await _call_tool(server.backlinks_tool, path="development/target.md")

        assert isinstance(result, list)


# ─────────────────────────────────────────────────────────────────────────────
# tree_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTreeTool:
    """Test tree_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_returns_dict(self, kb_root, index_root):
        """Happy path: returns directory tree dict."""
        result = await _call_tool(server.tree_tool)

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_depth_parameter_accepted(self, kb_root, index_root):
        """Depth parameter is accepted."""
        result = await _call_tool(server.tree_tool, depth=1)

        assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
# tags_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTagsTool:
    """Test tags_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_returns_list(self, kb_root, index_root):
        """Happy path: returns list of tags with counts."""
        _create_entry(kb_root / "development" / "test.md", "Test", ["python", "test"])

        result = await _call_tool(server.tags_tool)

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_empty_kb_returns_empty_list(self, kb_root, index_root):
        """Empty KB returns empty tag list."""
        result = await _call_tool(server.tags_tool)

        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# health_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHealthTool:
    """Test health_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_returns_audit_dict(self, kb_root, index_root):
        """Happy path: returns health audit dict."""
        result = await _call_tool(server.health_tool)

        assert isinstance(result, dict)
        assert "orphans" in result
        assert "broken_links" in result
        assert "stale" in result

    @pytest.mark.asyncio
    async def test_stale_days_parameter_accepted(self, kb_root, index_root):
        """stale_days parameter is accepted."""
        result = await _call_tool(server.health_tool, stale_days=30)

        assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
# quality_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestQualityTool:
    """Test quality_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_returns_quality_report(self, kb_root, index_root):
        """Happy path: returns QualityReport model."""
        result = await _call_tool(server.quality_tool)

        assert isinstance(result, QualityReport)
        assert hasattr(result, "accuracy")
        assert hasattr(result, "total_queries")


# ─────────────────────────────────────────────────────────────────────────────
# suggest_links_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSuggestLinksTool:
    """Test suggest_links_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_returns_list(self, kb_root, index_root):
        """Happy path: returns list of suggestions."""
        _create_entry(kb_root / "development" / "test.md", "Test", ["test"])

        result = await _call_tool(server.suggest_links_tool, path="development/test.md")

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_nonexistent_raises_error(self, kb_root, index_root):
        """Nonexistent entry raises ValueError."""
        with pytest.raises(ValueError, match="Entry not found"):
            await _call_tool(server.suggest_links_tool, path="development/nonexistent.md")


# ─────────────────────────────────────────────────────────────────────────────
# mkdir_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMkdirTool:
    """Test mkdir_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_creates_directory(self, kb_root, index_root):
        """Happy path: creates directory."""
        result = await _call_tool(server.mkdir_tool, path="development/newdir")

        assert isinstance(result, str)
        assert (kb_root / "development" / "newdir").exists()


# ─────────────────────────────────────────────────────────────────────────────
# move_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMoveTool:
    """Test move_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_moves_entry(self, kb_root, index_root):
        """Happy path: moves entry to new location."""
        _create_entry(kb_root / "development" / "source.md", "Source", ["test"])

        result = await _call_tool(
            server.move_tool,
            source="development/source.md",
            destination="architecture/source.md",
        )

        assert isinstance(result, dict)
        assert not (kb_root / "development" / "source.md").exists()
        assert (kb_root / "architecture" / "source.md").exists()


# ─────────────────────────────────────────────────────────────────────────────
# rmdir_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRmdirTool:
    """Test rmdir_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_removes_empty_directory(self, kb_root, index_root):
        """Happy path: removes empty directory."""
        (kb_root / "development" / "emptydir").mkdir()

        result = await _call_tool(server.rmdir_tool, path="development/emptydir")

        assert isinstance(result, str)
        assert not (kb_root / "development" / "emptydir").exists()
