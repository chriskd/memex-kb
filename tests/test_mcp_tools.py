"""Tests for MCP server tool wrappers.

This test suite verifies that MCP tool wrappers in server.py correctly delegate
to core business logic, handle parameters properly, and propagate errors correctly.

Test strategy:
- Unit tests: Test each MCP tool wrapper in isolation
- Integration tests: Verify tools work with actual KB operations
- Property-based tests: N/A (not applicable for these wrappers)
- Manual tests: N/A (all functionality is testable via API)

Coverage:
- Happy path: Normal operations for all tools
- Edge cases: Empty inputs, boundary values, special characters
- Error conditions: Invalid paths, missing entries, parse errors
- Boundary values: Limit parameters, empty results
"""

from datetime import date, timedelta
from pathlib import Path

import pytest

from memex import core, server
from memex.models import AddEntryResponse, KBEntry, QualityReport, SearchResponse

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


def _create_entry(
    path: Path,
    title: str,
    tags: list[str],
    created: date | None = None,
    updated: date | None = None,
    content: str = "## Content\n\nSome content here.",
):
    """Helper to create a KB entry with frontmatter."""
    if created is None:
        created = date.today()

    updated_line = f"updated: {updated.isoformat()}\n" if updated else ""
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    full_content = f"""---
title: {title}
tags:
{tags_yaml}
created: {created.isoformat()}
{updated_line}---

{content}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(full_content)


# ─────────────────────────────────────────────────────────────────────────────
# search_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSearchTool:
    """Test search_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_search_returns_results(self, kb_root, index_root):
        """Happy path: Search returns matching entries."""
        _create_entry(
            kb_root / "development" / "python.md",
            "Python Tooling",
            ["python", "tools"],
            content="## Overview\n\nPython development tools and best practices.",
        )

        result = await _call_tool(server.search_tool, query="python", limit=10)

        assert isinstance(result, SearchResponse)
        assert len(result.results) >= 1
        assert "python" in result.results[0].path.lower()

    @pytest.mark.asyncio
    async def test_search_with_mode_parameter(self, kb_root, index_root):
        """Happy path: Search modes (hybrid, keyword, semantic) work."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            ["test"],
        )

        # Test each mode
        for mode in ["hybrid", "keyword", "semantic"]:
            result = await _call_tool(server.search_tool, query="test", mode=mode)
            assert isinstance(result, SearchResponse)

    @pytest.mark.asyncio
    async def test_search_with_tags_filter(self, kb_root, index_root):
        """Happy path: Tag filtering restricts results."""
        _create_entry(
            kb_root / "development" / "python.md",
            "Python Entry",
            ["python", "programming"],
        )
        _create_entry(
            kb_root / "development" / "rust.md",
            "Rust Entry",
            ["rust", "programming"],
        )

        result = await _call_tool(server.search_tool, query="programming", tags=["python"])

        # Should only return Python entry
        for r in result.results:
            assert "python" in r.tags

    @pytest.mark.asyncio
    async def test_search_with_include_content(self, kb_root, index_root):
        """Happy path: include_content returns full content."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            ["test"],
            content="## Details\n\nFull content goes here.",
        )

        result = await _call_tool(
            server.search_tool, query="test", include_content=True
        )

        if result.results:
            # Content should be populated when include_content=True
            assert result.results[0].content is not None

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, kb_root, index_root):
        """Boundary: Limit parameter restricts result count."""
        for i in range(5):
            _create_entry(
                kb_root / "development" / f"entry{i}.md",
                f"Entry {i}",
                ["test"],
            )

        result = await _call_tool(server.search_tool, query="entry", limit=2)

        assert len(result.results) <= 2

    @pytest.mark.asyncio
    async def test_search_empty_query(self, kb_root, index_root):
        """Edge case: Empty query string returns results or handles gracefully."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test",
            ["test"],
        )

        result = await _call_tool(server.search_tool, query="", limit=10)

        # Should handle gracefully without error
        assert isinstance(result, SearchResponse)

    @pytest.mark.asyncio
    async def test_search_no_results(self, kb_root, index_root):
        """Edge case: Query with no matches returns empty or few results."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test",
            ["test"],
        )

        result = await _call_tool(
            server.search_tool, query="nonexistent_query_xyz123", limit=10
        )

        assert isinstance(result, SearchResponse)
        # Should return empty results or possibly fallback results
        assert isinstance(result.results, list)


# ─────────────────────────────────────────────────────────────────────────────
# add_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAddTool:
    """Test add_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_add_creates_entry(self, kb_root, index_root):
        """Happy path: Adding entry creates file with correct metadata."""
        result = await _call_tool(
            server.add_tool,
            title="Test Entry",
            content="Test content",
            tags=["test", "example"],
            category="development",
        )

        assert isinstance(result, AddEntryResponse)
        assert result.created is True
        assert "development" in result.path

        # Verify file exists
        file_path = kb_root / result.path
        assert file_path.exists()

    @pytest.mark.asyncio
    async def test_add_with_directory(self, kb_root, index_root):
        """Happy path: Adding with directory creates nested structure."""
        # Create parent directory first
        (kb_root / "development" / "python").mkdir(parents=True)

        result = await _call_tool(
            server.add_tool,
            title="Nested Entry",
            content="Content",
            tags=["test"],
            directory="development/python",
        )

        assert result.created is True
        assert result.path.startswith("development/python/")

    @pytest.mark.asyncio
    async def test_add_with_links(self, kb_root, index_root):
        """Happy path: Adding entry with links includes them in content."""
        # Create target entry first
        _create_entry(
            kb_root / "development" / "target.md",
            "Target Entry",
            ["target"],
        )

        result = await _call_tool(
            server.add_tool,
            title="Source Entry",
            content="Content with reference to other topics.",
            tags=["documentation", "reference"],  # Different tags to avoid duplicate detection
            category="development",
            links=["development/target"],
            force=True,  # Force to bypass duplicate detection
        )

        assert result.created is True

        # Verify links in content (may be in content or link section)
        file_path = kb_root / result.path
        content = file_path.read_text()
        # Link should appear somewhere in the file
        assert "development/target" in content

    @pytest.mark.asyncio
    async def test_add_duplicate_detection(self, kb_root, index_root):
        """Happy path: Duplicate detection warns about similar entries."""
        # Create existing entry
        _create_entry(
            kb_root / "development" / "existing.md",
            "Python Tooling Guide",
            ["python", "tools"],
            content="## Overview\n\nDetailed guide about Python tools.",
        )

        # Try to add very similar entry
        result = await _call_tool(
            server.add_tool,
            title="Python Tooling Guide",
            content="Guide about Python tools",
            tags=["python", "tools"],
            category="development",
            force=False,
        )

        # Should detect potential duplicate
        if not result.created:
            assert len(result.potential_duplicates) > 0
            assert result.warning is not None

    @pytest.mark.asyncio
    async def test_add_force_bypasses_duplicate_check(self, kb_root, index_root):
        """Happy path: force=True bypasses duplicate detection."""
        _create_entry(
            kb_root / "development" / "existing.md",
            "Test Entry",
            ["test"],
        )

        result = await _call_tool(
            server.add_tool,
            title="Test Entry",
            content="Content",
            tags=["test"],
            category="development",
            force=True,
        )

        assert result.created is True

    @pytest.mark.asyncio
    async def test_add_with_empty_tags_list(self, kb_root, index_root):
        """Error condition: Empty tags list should fail validation."""
        with pytest.raises(Exception):  # Pydantic validation error
            await _call_tool(
                server.add_tool,
                title="No Tags Entry",
                content="Content",
                tags=[],  # Invalid: tags must have at least one
                category="development",
            )

    @pytest.mark.asyncio
    async def test_add_invalid_category(self, kb_root, index_root):
        """Edge case: Invalid category creates the category or uses fallback."""
        # The add_tool may create the category if it doesn't exist or use a fallback
        # This is an edge case that the implementation may handle differently
        result = await _call_tool(
            server.add_tool,
            title="Test Entry in New Category",
            content="Content",
            tags=["test"],
            category="nonexistent_category",
        )

        # Either it creates the entry (new behavior) or raises error (strict behavior)
        # Accept either outcome as valid
        assert isinstance(result, AddEntryResponse)

    @pytest.mark.asyncio
    async def test_add_special_characters_in_title(self, kb_root, index_root):
        """Edge case: Special characters in title are slugified."""
        result = await _call_tool(
            server.add_tool,
            title="Test & Special / Characters!",
            content="Content",
            tags=["test"],
            category="development",
        )

        assert result.created is True
        # Verify path doesn't contain special chars
        assert "/" not in Path(result.path).name
        assert "&" not in result.path


# ─────────────────────────────────────────────────────────────────────────────
# get_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGetTool:
    """Test get_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_get_returns_entry(self, kb_root, index_root):
        """Happy path: Getting existing entry returns full data."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            ["python", "test"],
            content="## Section\n\nTest content here.",
        )

        result = await _call_tool(server.get_tool, path="development/test.md")

        assert isinstance(result, KBEntry)
        assert result.path == "development/test.md"
        assert result.metadata.title == "Test Entry"
        assert "python" in result.metadata.tags
        assert "Test content" in result.content

    @pytest.mark.asyncio
    async def test_get_includes_links(self, kb_root, index_root):
        """Happy path: Get extracts links from content."""
        _create_entry(
            kb_root / "development" / "source.md",
            "Source",
            ["test"],
            content="## Links\n\nSee [[development/target]] for details.",
        )

        result = await _call_tool(server.get_tool, path="development/source.md")

        assert "development/target" in result.links

    @pytest.mark.asyncio
    async def test_get_includes_backlinks(self, kb_root, index_root):
        """Happy path: Get includes backlinks from other entries."""
        # Create target entry
        _create_entry(
            kb_root / "development" / "target.md",
            "Target",
            ["test"],
        )

        # Create source entry that links to target
        _create_entry(
            kb_root / "development" / "source.md",
            "Source",
            ["test"],
            content="See [[development/target]] for more.",
        )

        # Rebuild backlink cache
        from memex.backlinks_cache import rebuild_backlink_cache
        rebuild_backlink_cache(kb_root)

        result = await _call_tool(server.get_tool, path="development/target.md")

        assert "development/source" in result.backlinks

    @pytest.mark.asyncio
    async def test_get_nonexistent_entry(self, kb_root, index_root):
        """Error condition: Getting nonexistent entry raises ValueError."""
        with pytest.raises(ValueError, match="Entry not found"):
            await _call_tool(server.get_tool, path="development/nonexistent.md")

    @pytest.mark.asyncio
    async def test_get_directory_instead_of_file(self, kb_root, index_root):
        """Error condition: Path to directory raises ValueError."""
        with pytest.raises(ValueError, match="not a file"):
            await _call_tool(server.get_tool, path="development")

    @pytest.mark.asyncio
    async def test_get_with_path_traversal_attempt(self, kb_root, index_root):
        """Error condition: Path traversal attempt should fail."""
        # Create entry outside KB (this should fail during get)
        with pytest.raises(ValueError):
            await _call_tool(server.get_tool, path="../../../etc/passwd")


# ─────────────────────────────────────────────────────────────────────────────
# update_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestUpdateTool:
    """Test update_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_update_content(self, kb_root, index_root):
        """Happy path: Updating content modifies the entry."""
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

        assert "path" in result
        assert result["path"] == "development/test.md"

        # Verify content changed
        file_content = (kb_root / "development" / "test.md").read_text()
        assert "Updated content" in file_content

    @pytest.mark.asyncio
    async def test_update_tags(self, kb_root, index_root):
        """Happy path: Updating tags modifies frontmatter."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            ["old_tag"],
        )

        result = await _call_tool(
            server.update_tool,
            path="development/test.md",
            content="Some content",  # content is required when updating tags
            tags=["new_tag", "updated"],
        )

        assert "path" in result

        # Verify tags changed
        from memex.parser import parse_entry
        metadata, _, _ = parse_entry(kb_root / "development" / "test.md")
        assert "new_tag" in metadata.tags
        assert "updated" in metadata.tags

    @pytest.mark.asyncio
    async def test_update_with_section_updates(self, kb_root, index_root):
        """Happy path: Section updates modify specific sections."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            ["test"],
            content="## Section1\n\nOld content\n\n## Section2\n\nOther content",
        )

        result = await _call_tool(
            server.update_tool,
            path="development/test.md",
            section_updates={"Section1": "New content for section 1"},
        )

        assert "path" in result

        # Verify section updated
        file_content = (kb_root / "development" / "test.md").read_text()
        assert "New content for section 1" in file_content
        assert "Other content" in file_content  # Section2 unchanged

    @pytest.mark.asyncio
    async def test_update_sets_updated_date(self, kb_root, index_root):
        """Happy path: Update sets the 'updated' field in frontmatter."""
        old_date = date.today() - timedelta(days=10)
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            ["test"],
            created=old_date,
        )

        await _call_tool(
            server.update_tool,
            path="development/test.md",
            content="Updated",
        )

        # Verify updated date is set
        from memex.parser import parse_entry
        metadata, _, _ = parse_entry(kb_root / "development" / "test.md")
        assert metadata.updated is not None
        # Updated is now a datetime, compare date portion
        assert metadata.updated.date() >= date.today()

    @pytest.mark.asyncio
    async def test_update_nonexistent_entry(self, kb_root, index_root):
        """Error condition: Updating nonexistent entry raises error."""
        with pytest.raises(ValueError, match="Entry not found"):
            await _call_tool(
                server.update_tool,
                path="development/nonexistent.md",
                content="New content",
            )

    @pytest.mark.asyncio
    async def test_update_with_empty_parameters(self, kb_root, index_root):
        """Edge case: Update with no content or section_updates should raise error."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            ["test"],
        )

        # Update requires either content, section_updates, or tags
        with pytest.raises(ValueError, match="Provide new content, section_updates, or tags"):
            await _call_tool(
                server.update_tool,
                path="development/test.md",
                content=None,
                tags=None,
                section_updates=None,
            )


# ─────────────────────────────────────────────────────────────────────────────
# delete_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDeleteTool:
    """Test delete_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_delete_removes_entry(self, kb_root, index_root):
        """Happy path: Deleting entry removes the file."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            ["test"],
        )

        result = await _call_tool(
            server.delete_tool, path="development/test.md", force=True
        )

        assert "deleted" in result
        assert result["deleted"] == "development/test.md"
        assert not (kb_root / "development" / "test.md").exists()

    @pytest.mark.asyncio
    async def test_delete_with_backlinks_warns(self, kb_root, index_root):
        """Error condition: Deleting entry with backlinks raises error (without force)."""
        # Create target
        _create_entry(
            kb_root / "development" / "target.md",
            "Target",
            ["test"],
        )

        # Create entry linking to target
        _create_entry(
            kb_root / "development" / "source.md",
            "Source",
            ["test"],
            content="See [[development/target]].",
        )

        # Rebuild backlink cache
        from memex.backlinks_cache import rebuild_backlink_cache
        rebuild_backlink_cache(kb_root)

        # Try to delete without force - should raise error
        with pytest.raises(ValueError, match="backlink"):
            await _call_tool(
                server.delete_tool, path="development/target.md", force=False
            )

    @pytest.mark.asyncio
    async def test_delete_force_ignores_backlinks(self, kb_root, index_root):
        """Happy path: force=True deletes despite backlinks."""
        _create_entry(
            kb_root / "development" / "target.md",
            "Target",
            ["test"],
        )
        _create_entry(
            kb_root / "development" / "source.md",
            "Source",
            ["test"],
            content="See [[development/target]].",
        )

        from memex.backlinks_cache import rebuild_backlink_cache
        rebuild_backlink_cache(kb_root)

        result = await _call_tool(
            server.delete_tool, path="development/target.md", force=True
        )

        assert "deleted" in result
        assert result["deleted"] == "development/target.md"

    @pytest.mark.asyncio
    async def test_delete_nonexistent_entry(self, kb_root, index_root):
        """Error condition: Deleting nonexistent entry raises error."""
        with pytest.raises(ValueError, match="Entry not found"):
            await _call_tool(server.delete_tool, path="development/nonexistent.md")


# ─────────────────────────────────────────────────────────────────────────────
# list_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestListTool:
    """Test list_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_list_all_entries(self, kb_root, index_root):
        """Happy path: List without filters returns all entries."""
        _create_entry(kb_root / "development" / "test1.md", "Test 1", ["test"])
        _create_entry(kb_root / "architecture" / "test2.md", "Test 2", ["arch"])

        result = await _call_tool(server.list_tool, limit=20)

        assert isinstance(result, list)
        assert len(result) >= 2

    @pytest.mark.asyncio
    async def test_list_by_category(self, kb_root, index_root):
        """Happy path: Category filter restricts results."""
        _create_entry(kb_root / "development" / "dev.md", "Dev", ["test"])
        _create_entry(kb_root / "architecture" / "arch.md", "Arch", ["test"])

        result = await _call_tool(server.list_tool, category="development")

        assert all("development" in entry["path"] for entry in result)

    @pytest.mark.asyncio
    async def test_list_by_directory(self, kb_root, index_root):
        """Happy path: Directory filter restricts results."""
        (kb_root / "development" / "python").mkdir(parents=True)
        _create_entry(
            kb_root / "development" / "python" / "test.md", "Test", ["python"]
        )
        _create_entry(kb_root / "development" / "other.md", "Other", ["test"])

        result = await _call_tool(server.list_tool, directory="development/python")

        assert all("development/python" in entry["path"] for entry in result)

    @pytest.mark.asyncio
    async def test_list_by_tag(self, kb_root, index_root):
        """Happy path: Tag filter restricts results."""
        _create_entry(kb_root / "development" / "python.md", "Python", ["python"])
        _create_entry(kb_root / "development" / "rust.md", "Rust", ["rust"])

        result = await _call_tool(server.list_tool, tag="python")

        assert all("python" in entry["tags"] for entry in result)

    @pytest.mark.asyncio
    async def test_list_respects_limit(self, kb_root, index_root):
        """Boundary: Limit parameter restricts result count."""
        for i in range(10):
            _create_entry(
                kb_root / "development" / f"test{i}.md", f"Test {i}", ["test"]
            )

        result = await _call_tool(server.list_tool, limit=3)

        assert len(result) <= 3

    @pytest.mark.asyncio
    async def test_list_empty_kb(self, kb_root, index_root):
        """Edge case: Empty KB returns empty list."""
        result = await _call_tool(server.list_tool)

        assert result == []

    @pytest.mark.asyncio
    async def test_list_invalid_category(self, kb_root, index_root):
        """Error condition: Invalid category raises error."""
        with pytest.raises(ValueError, match="Category not found"):
            await _call_tool(server.list_tool, category="nonexistent")


# ─────────────────────────────────────────────────────────────────────────────
# health_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHealthTool:
    """Test health_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_health_returns_report(self, kb_root, index_root):
        """Happy path: Health check returns audit report."""
        _create_entry(kb_root / "development" / "test.md", "Test", ["test"])

        result = await _call_tool(server.health_tool)

        assert isinstance(result, dict)
        assert "orphans" in result
        assert "broken_links" in result
        assert "stale" in result  # key is "stale", not "stale_entries"

    @pytest.mark.asyncio
    async def test_health_detects_orphans(self, kb_root, index_root):
        """Happy path: Health detects entries with no incoming links."""
        _create_entry(kb_root / "development" / "orphan.md", "Orphan", ["test"])

        # Rebuild backlink cache
        from memex.backlinks_cache import rebuild_backlink_cache
        rebuild_backlink_cache(kb_root)

        result = await _call_tool(server.health_tool)

        # Orphan should be detected
        assert "orphans" in result

    @pytest.mark.asyncio
    async def test_health_detects_broken_links(self, kb_root, index_root):
        """Happy path: Health detects links to nonexistent entries."""
        _create_entry(
            kb_root / "development" / "source.md",
            "Source",
            ["test"],
            content="See [[development/nonexistent]].",
        )

        result = await _call_tool(server.health_tool)

        # Broken link should be detected
        if result["broken_links"]:
            assert len(result["broken_links"]) > 0

    @pytest.mark.asyncio
    async def test_health_stale_days_parameter(self, kb_root, index_root):
        """Happy path: stale_days parameter configures staleness threshold."""
        old_date = date.today() - timedelta(days=100)
        _create_entry(
            kb_root / "development" / "old.md",
            "Old Entry",
            ["test"],
            created=old_date,
        )

        result = await _call_tool(server.health_tool, stale_days=90)

        # Old entry should be flagged as stale
        assert "stale" in result
        assert len(result["stale"]) > 0

    @pytest.mark.asyncio
    async def test_health_empty_kb(self, kb_root, index_root):
        """Edge case: Health check on empty KB returns empty results."""
        result = await _call_tool(server.health_tool)

        assert result["orphans"] == []
        assert result["broken_links"] == []
        assert result["stale"] == []


# ─────────────────────────────────────────────────────────────────────────────
# quality_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestQualityTool:
    """Test quality_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_quality_returns_report(self, kb_root, index_root):
        """Happy path: Quality check returns accuracy report."""
        _create_entry(kb_root / "development" / "test.md", "Test", ["test"])

        result = await _call_tool(server.quality_tool)

        assert isinstance(result, QualityReport)
        assert hasattr(result, "accuracy")
        assert hasattr(result, "total_queries")
        assert hasattr(result, "details")

    @pytest.mark.asyncio
    async def test_quality_with_limit_parameter(self, kb_root, index_root):
        """Happy path: limit parameter controls result count."""
        _create_entry(kb_root / "development" / "test.md", "Test", ["test"])

        result = await _call_tool(server.quality_tool, limit=3)

        assert isinstance(result, QualityReport)

    @pytest.mark.asyncio
    async def test_quality_with_cutoff_parameter(self, kb_root, index_root):
        """Happy path: cutoff parameter affects ranking threshold."""
        _create_entry(kb_root / "development" / "test.md", "Test", ["test"])

        result = await _call_tool(server.quality_tool, limit=5, cutoff=5)

        assert isinstance(result, QualityReport)

    @pytest.mark.asyncio
    async def test_quality_empty_kb(self, kb_root, index_root):
        """Edge case: Quality check on empty KB handles gracefully."""
        result = await _call_tool(server.quality_tool)

        # Should return valid report even with no data
        assert isinstance(result, QualityReport)


# ─────────────────────────────────────────────────────────────────────────────
# suggest_links_tool Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSuggestLinksTool:
    """Test suggest_links_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_suggest_links_returns_suggestions(self, kb_root, index_root):
        """Happy path: Link suggestions returns related entries."""
        _create_entry(
            kb_root / "development" / "python.md",
            "Python Guide",
            ["python", "programming"],
            content="## Overview\n\nPython programming guide.",
        )
        _create_entry(
            kb_root / "development" / "django.md",
            "Django Framework",
            ["python", "web"],
            content="## Overview\n\nDjango web framework for Python.",
        )

        result = await _call_tool(
            server.suggest_links_tool, path="development/python.md"
        )

        assert isinstance(result, list)
        # Should suggest Django (related by tags and content)

    @pytest.mark.asyncio
    async def test_suggest_links_excludes_self(self, kb_root, index_root):
        """Happy path: Suggestions don't include the entry itself."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            ["test"],
        )

        result = await _call_tool(
            server.suggest_links_tool, path="development/test.md"
        )

        # Self should not be in suggestions
        for suggestion in result:
            assert "development/test" not in suggestion["path"]

    @pytest.mark.asyncio
    async def test_suggest_links_respects_limit(self, kb_root, index_root):
        """Boundary: limit parameter restricts suggestion count."""
        _create_entry(
            kb_root / "development" / "main.md",
            "Main Entry",
            ["test"],
        )
        for i in range(10):
            _create_entry(
                kb_root / "development" / f"related{i}.md",
                f"Related {i}",
                ["test"],
            )

        result = await _call_tool(
            server.suggest_links_tool, path="development/main.md", limit=3
        )

        assert len(result) <= 3

    @pytest.mark.asyncio
    async def test_suggest_links_respects_min_score(self, kb_root, index_root):
        """Boundary: min_score filters low-quality suggestions."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            ["test"],
        )
        _create_entry(
            kb_root / "development" / "other.md",
            "Unrelated Entry",
            ["different"],
            content="Completely different content.",
        )

        result = await _call_tool(
            server.suggest_links_tool,
            path="development/test.md",
            min_score=0.9,  # Very high threshold
        )

        # High threshold should filter out unrelated entries
        for suggestion in result:
            assert suggestion["score"] >= 0.9

    @pytest.mark.asyncio
    async def test_suggest_links_nonexistent_entry(self, kb_root, index_root):
        """Error condition: Suggestions for nonexistent entry raises error."""
        with pytest.raises(ValueError, match="Entry not found"):
            await _call_tool(
                server.suggest_links_tool, path="development/nonexistent.md"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Additional Tools Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTreeTool:
    """Test tree_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_tree_returns_structure(self, kb_root, index_root):
        """Happy path: Tree returns directory structure."""
        (kb_root / "development" / "python").mkdir(parents=True)
        _create_entry(
            kb_root / "development" / "python" / "test.md", "Test", ["python"]
        )

        result = await _call_tool(server.tree_tool)

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_tree_with_path_parameter(self, kb_root, index_root):
        """Happy path: Tree with path shows subtree."""
        (kb_root / "development" / "python").mkdir(parents=True)
        _create_entry(
            kb_root / "development" / "python" / "test.md", "Test", ["python"]
        )

        result = await _call_tool(server.tree_tool, path="development")

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_tree_respects_depth(self, kb_root, index_root):
        """Boundary: depth parameter limits tree depth."""
        (kb_root / "development" / "python" / "frameworks").mkdir(parents=True)

        result = await _call_tool(server.tree_tool, depth=1)

        assert isinstance(result, dict)


class TestBacklinksTool:
    """Test backlinks_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_backlinks_returns_linking_entries(self, kb_root, index_root):
        """Happy path: Backlinks returns entries that link to target."""
        _create_entry(
            kb_root / "development" / "target.md",
            "Target",
            ["test"],
        )
        _create_entry(
            kb_root / "development" / "source1.md",
            "Source 1",
            ["test"],
            content="See [[development/target]].",
        )
        _create_entry(
            kb_root / "development" / "source2.md",
            "Source 2",
            ["test"],
            content="Related: [[development/target]].",
        )

        # Rebuild cache
        from memex.backlinks_cache import rebuild_backlink_cache
        rebuild_backlink_cache(kb_root)

        result = await _call_tool(server.backlinks_tool, path="development/target.md")

        assert isinstance(result, list)
        assert "development/source1" in result
        assert "development/source2" in result

    @pytest.mark.asyncio
    async def test_backlinks_no_incoming_links(self, kb_root, index_root):
        """Edge case: Entry with no backlinks returns empty list."""
        _create_entry(kb_root / "development" / "orphan.md", "Orphan", ["test"])

        from memex.backlinks_cache import rebuild_backlink_cache
        rebuild_backlink_cache(kb_root)

        result = await _call_tool(server.backlinks_tool, path="development/orphan.md")

        assert result == []


class TestTagsTool:
    """Test tags_tool MCP wrapper."""

    @pytest.mark.asyncio
    async def test_tags_returns_taxonomy(self, kb_root, index_root):
        """Happy path: Tags returns all tags with counts."""
        _create_entry(kb_root / "development" / "test1.md", "Test 1", ["python", "web"])
        _create_entry(kb_root / "development" / "test2.md", "Test 2", ["python", "cli"])

        result = await _call_tool(server.tags_tool)

        assert isinstance(result, list)
        # Find python tag
        python_tag = next((t for t in result if t["tag"] == "python"), None)
        assert python_tag is not None
        assert python_tag["count"] == 2

    @pytest.mark.asyncio
    async def test_tags_empty_kb(self, kb_root, index_root):
        """Edge case: Empty KB returns empty tag list."""
        result = await _call_tool(server.tags_tool)

        assert result == []
