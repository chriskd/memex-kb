"""Tests for search include_content feature."""

from datetime import date
from pathlib import Path

import pytest

from memex import core, server
from memex.models import SearchResponse

pytestmark = pytest.mark.semantic


async def _call_tool(tool_obj, /, *args, **kwargs):
    """Invoke the wrapped coroutine behind an MCP FunctionTool."""
    bound = tool_obj.fn(*args, **kwargs)
    if callable(bound):
        return await bound()
    return await bound


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
    path.write_text(content)


class TestSearchContentDefault:
    """Test default search behavior (content=None)."""

    @pytest.mark.asyncio
    async def test_search_default_no_content(self, kb_root, index_root):
        """By default, content field is None."""
        _create_entry(
            kb_root / "development" / "entry.md",
            "Test Entry",
            "## Section\n\nFull content here that is longer than snippet.",
        )

        await core.reindex()
        response = await _call_tool(server.search_tool, "Test")

        assert isinstance(response, SearchResponse)
        assert len(response.results) >= 1
        assert response.results[0].content is None
        assert response.warnings == []

    @pytest.mark.asyncio
    async def test_search_returns_search_response(self, kb_root, index_root):
        """Search now returns SearchResponse wrapper."""
        _create_entry(
            kb_root / "development" / "entry.md",
            "Test Entry",
            "Content body.",
        )

        await core.reindex()
        response = await _call_tool(server.search_tool, "Test")

        assert isinstance(response, SearchResponse)
        assert hasattr(response, "results")
        assert hasattr(response, "warnings")


class TestSearchWithContent:
    """Test search with include_content=True."""

    @pytest.mark.asyncio
    async def test_include_content_returns_full_text(self, kb_root, index_root):
        """include_content=True populates content field."""
        full_text = "## Section\n\nThis is the full document content that should appear in results."
        _create_entry(
            kb_root / "development" / "entry.md",
            "Test Entry",
            full_text,
        )

        await core.reindex()
        response = await _call_tool(server.search_tool, "Test", include_content=True)

        assert len(response.results) >= 1
        assert response.results[0].content is not None
        assert "full document content" in response.results[0].content

    @pytest.mark.asyncio
    async def test_include_content_preserves_other_fields(self, kb_root, index_root):
        """Content hydration doesn't affect other fields."""
        _create_entry(
            kb_root / "development" / "entry.md",
            "Unique Title",
            "## Section\n\nContent.",
            tags=["python", "testing"],
        )

        await core.reindex()

        # Get results both ways
        without = await _call_tool(server.search_tool, "Unique")
        with_content = await _call_tool(server.search_tool, "Unique", include_content=True)

        assert without.results[0].path == with_content.results[0].path
        assert without.results[0].title == with_content.results[0].title
        assert without.results[0].score == with_content.results[0].score
        assert without.results[0].tags == with_content.results[0].tags

    @pytest.mark.asyncio
    async def test_multiple_results_all_get_content(self, kb_root, index_root):
        """All results get content when include_content=True."""
        for i in range(3):
            _create_entry(
                kb_root / "development" / f"entry{i}.md",
                f"Search Entry {i}",
                f"Content for entry {i} with unique text.",
            )

        await core.reindex()
        response = await _call_tool(
            server.search_tool, "Search Entry", limit=10, include_content=True,
        )

        assert len(response.results) == 3
        for result in response.results:
            assert result.content is not None
            assert "Content for entry" in result.content


class TestSearchContentLimit:
    """Test MAX_CONTENT_RESULTS limit enforcement."""

    @pytest.mark.asyncio
    async def test_limit_warning_when_exceeded(self, kb_root, index_root, monkeypatch):
        """Warning added when results exceed MAX_CONTENT_RESULTS."""
        # Set a low limit for testing (patch in core where it's used)
        monkeypatch.setattr(core, "MAX_CONTENT_RESULTS", 2)

        for i in range(5):
            _create_entry(
                kb_root / "development" / f"entry{i}.md",
                f"Limit Test Entry {i}",
                f"Content for limit test entry {i}.",
            )

        await core.reindex()
        response = await _call_tool(server.search_tool, "Limit Test", limit=5, include_content=True)

        # Should have warning about limit
        assert len(response.warnings) == 1
        assert "limited to 2" in response.warnings[0].lower()

        # Should only return MAX_CONTENT_RESULTS results
        assert len(response.results) == 2

    @pytest.mark.asyncio
    async def test_no_warning_when_under_limit(self, kb_root, index_root, monkeypatch):
        """No warning when results are under MAX_CONTENT_RESULTS."""
        monkeypatch.setattr(core, "MAX_CONTENT_RESULTS", 10)

        for i in range(3):
            _create_entry(
                kb_root / "development" / f"entry{i}.md",
                f"Under Limit Entry {i}",
                f"Content for under limit entry {i}.",
            )

        await core.reindex()
        response = await _call_tool(
            server.search_tool, "Under Limit", limit=5, include_content=True,
        )

        assert response.warnings == []
        assert len(response.results) == 3


class TestSearchContentEdgeCases:
    """Test edge cases for content hydration."""

    @pytest.mark.asyncio
    async def test_missing_file_content_is_none(self, kb_root, index_root):
        """If file was deleted after indexing, content is None."""
        entry_path = kb_root / "development" / "deleted.md"
        _create_entry(entry_path, "Will Be Deleted", "Content.")

        await core.reindex()

        # Delete the file after indexing
        entry_path.unlink()

        response = await _call_tool(server.search_tool, "Deleted", include_content=True)

        # Result may or may not be returned depending on index state
        # But if returned, content should be None
        if response.results:
            assert response.results[0].content is None

    @pytest.mark.asyncio
    async def test_empty_results_no_error(self, kb_root, index_root):
        """No error when hydrating empty results."""
        await core.reindex()
        response = await _call_tool(
            server.search_tool, "nonexistent query xyz", include_content=True,
        )

        assert response.results == []
        assert response.warnings == []
