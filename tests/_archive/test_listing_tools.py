"""Tests for KB listing MCP tools (whats_new)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memex import core, server


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
    created: datetime,
    updated: datetime | None = None,
    source_project: str | None = None,
):
    """Helper to create a KB entry with frontmatter."""
    updated_line = f"updated: {updated.isoformat()}\n" if updated else ""
    source_project_line = f"source_project: {source_project}\n" if source_project else ""
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    content = f"""---
title: {title}
tags:
{tags_yaml}
created: {created.isoformat()}
{updated_line}{source_project_line}---

## Content

Some content here.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class TestWhatsNewTool:
    """Test whats_new MCP tool."""

    @pytest.mark.asyncio
    async def test_whats_new_returns_recent_entries(self, kb_root):
        """Returns entries created/updated within days window."""
        today = datetime.now(timezone.utc)
        old_date = today - timedelta(days=60)

        _create_entry(
            kb_root / "development" / "recent.md",
            "Recent Entry",
            ["python"],
            created=today - timedelta(days=5),
        )
        _create_entry(
            kb_root / "development" / "old.md",
            "Old Entry",
            ["python"],
            created=old_date,
        )

        results = await _call_tool(server.whats_new_tool, days=30)

        assert len(results) == 1
        assert results[0]["title"] == "Recent Entry"
        assert results[0]["activity_type"] == "created"

    @pytest.mark.asyncio
    async def test_whats_new_prefers_updated_over_created(self, kb_root):
        """Updated date takes precedence when both qualify."""
        today = datetime.now(timezone.utc)

        _create_entry(
            kb_root / "development" / "updated.md",
            "Updated Entry",
            ["python"],
            created=today - timedelta(days=20),
            updated=today - timedelta(days=2),
        )

        results = await _call_tool(server.whats_new_tool, days=30)

        assert len(results) == 1
        assert results[0]["activity_type"] == "updated"
        assert results[0]["activity_date"] == (today - timedelta(days=2)).isoformat()

    @pytest.mark.asyncio
    async def test_whats_new_sorts_by_activity_date(self, kb_root):
        """Results are sorted by activity_date descending."""
        today = datetime.now(timezone.utc)

        _create_entry(
            kb_root / "development" / "older.md",
            "Older Entry",
            ["python"],
            created=today - timedelta(days=10),
        )
        _create_entry(
            kb_root / "development" / "newest.md",
            "Newest Entry",
            ["python"],
            created=today - timedelta(days=1),
        )
        _create_entry(
            kb_root / "development" / "middle.md",
            "Middle Entry",
            ["python"],
            created=today - timedelta(days=5),
        )

        results = await _call_tool(server.whats_new_tool, days=30, limit=10)

        titles = [r["title"] for r in results]
        assert titles == ["Newest Entry", "Middle Entry", "Older Entry"]

    @pytest.mark.asyncio
    async def test_whats_new_filters_by_category(self, kb_root):
        """Category filter restricts results (via core function)."""
        today = datetime.now(timezone.utc)

        _create_entry(
            kb_root / "development" / "dev.md",
            "Dev Entry",
            ["python"],
            created=today - timedelta(days=1),
        )
        _create_entry(
            kb_root / "architecture" / "arch.md",
            "Arch Entry",
            ["design"],
            created=today - timedelta(days=1),
        )

        # Use core function directly for category filter (removed from MCP tool)
        results = await core.whats_new(days=30, category="development")

        assert len(results) == 1
        assert results[0]["title"] == "Dev Entry"

    @pytest.mark.asyncio
    async def test_whats_new_filters_by_tag(self, kb_root):
        """Tag filter restricts results (via core function)."""
        today = datetime.now(timezone.utc)

        _create_entry(
            kb_root / "development" / "python.md",
            "Python Entry",
            ["python"],
            created=today - timedelta(days=1),
        )
        _create_entry(
            kb_root / "development" / "rust.md",
            "Rust Entry",
            ["rust"],
            created=today - timedelta(days=1),
        )

        # Use core function directly for tag filter (removed from MCP tool)
        results = await core.whats_new(days=30, tag="rust")

        assert len(results) == 1
        assert results[0]["title"] == "Rust Entry"

    @pytest.mark.asyncio
    async def test_whats_new_respects_limit(self, kb_root):
        """Result count is limited."""
        today = datetime.now(timezone.utc)

        for i in range(5):
            _create_entry(
                kb_root / "development" / f"entry{i}.md",
                f"Entry {i}",
                ["python"],
                created=today - timedelta(days=i),
            )

        results = await _call_tool(server.whats_new_tool, days=30, limit=2)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_whats_new_include_flags(self, kb_root):
        """include_created and include_updated flags work (via core function)."""
        today = datetime.now(timezone.utc)

        _create_entry(
            kb_root / "development" / "new.md",
            "New Entry",
            ["python"],
            created=today - timedelta(days=1),
        )
        _create_entry(
            kb_root / "development" / "updated.md",
            "Updated Entry",
            ["python"],
            created=today - timedelta(days=60),
            updated=today - timedelta(days=1),
        )

        # Only created (use core function directly - removed from MCP tool)
        results_created = await core.whats_new(
            days=30, include_created=True, include_updated=False
        )
        assert len(results_created) == 1
        assert results_created[0]["title"] == "New Entry"

        # Only updated
        results_updated = await core.whats_new(
            days=30, include_created=False, include_updated=True
        )
        assert len(results_updated) == 1
        assert results_updated[0]["title"] == "Updated Entry"

    @pytest.mark.asyncio
    async def test_whats_new_filters_by_project_path(self, kb_root):
        """Project filter matches entries in projects/{project}/ directory."""
        today = datetime.now(timezone.utc)

        # Create projects directory structure
        (kb_root / "projects" / "myapp").mkdir(parents=True)

        _create_entry(
            kb_root / "projects" / "myapp" / "setup.md",
            "MyApp Setup",
            ["setup"],
            created=today - timedelta(days=1),
        )
        _create_entry(
            kb_root / "development" / "other.md",
            "Other Entry",
            ["python"],
            created=today - timedelta(days=1),
        )

        results = await _call_tool(server.whats_new_tool, days=30, project="myapp")

        assert len(results) == 1
        assert results[0]["title"] == "MyApp Setup"

    @pytest.mark.asyncio
    async def test_whats_new_filters_by_project_source_project(self, kb_root):
        """Project filter matches entries with source_project metadata."""
        today = datetime.now(timezone.utc)

        _create_entry(
            kb_root / "development" / "myapp-guide.md",
            "MyApp Guide",
            ["guide"],
            created=today - timedelta(days=1),
            source_project="myapp",
        )
        _create_entry(
            kb_root / "development" / "other.md",
            "Other Entry",
            ["python"],
            created=today - timedelta(days=1),
            source_project="otherapp",
        )

        results = await _call_tool(server.whats_new_tool, days=30, project="myapp")

        assert len(results) == 1
        assert results[0]["title"] == "MyApp Guide"

    @pytest.mark.asyncio
    async def test_whats_new_filters_by_project_tag(self, kb_root):
        """Project filter matches entries with project name in tags."""
        today = datetime.now(timezone.utc)

        _create_entry(
            kb_root / "development" / "myapp-tips.md",
            "MyApp Tips",
            ["myapp", "tips"],
            created=today - timedelta(days=1),
        )
        _create_entry(
            kb_root / "development" / "other.md",
            "Other Entry",
            ["python"],
            created=today - timedelta(days=1),
        )

        results = await _call_tool(server.whats_new_tool, days=30, project="myapp")

        assert len(results) == 1
        assert results[0]["title"] == "MyApp Tips"

    @pytest.mark.asyncio
    async def test_whats_new_project_filter_case_insensitive(self, kb_root):
        """Project filter is case-insensitive."""
        today = datetime.now(timezone.utc)

        _create_entry(
            kb_root / "development" / "entry.md",
            "MyApp Entry",
            ["MyApp"],  # Capitalized tag
            created=today - timedelta(days=1),
        )

        # Query with lowercase
        results = await _call_tool(server.whats_new_tool, days=30, project="myapp")

        assert len(results) == 1
        assert results[0]["title"] == "MyApp Entry"

    @pytest.mark.asyncio
    async def test_whats_new_includes_source_project_in_results(self, kb_root):
        """Results include source_project field."""
        today = datetime.now(timezone.utc)

        _create_entry(
            kb_root / "development" / "entry.md",
            "Entry",
            ["python"],
            created=today - timedelta(days=1),
            source_project="myapp",
        )

        results = await _call_tool(server.whats_new_tool, days=30)

        assert len(results) == 1
        assert results[0]["source_project"] == "myapp"


class TestGetToolViewTracking:
    """Test that get_tool records views."""

    @pytest.mark.asyncio
    async def test_get_tool_records_view(self, kb_root, index_root):
        """get_tool increments view count."""
        from memex.views_tracker import load_views

        _create_entry(
            kb_root / "development" / "entry.md",
            "Test Entry",
            ["python"],
            created=datetime.now(timezone.utc),
        )

        # Call get_tool
        await _call_tool(server.get_tool, "development/entry.md")

        # Check view was recorded
        views = load_views(index_root)
        assert "development/entry.md" in views
        assert views["development/entry.md"].total_views == 1

        # Call again
        await _call_tool(server.get_tool, "development/entry.md")
        views = load_views(index_root)
        assert views["development/entry.md"].total_views == 2
