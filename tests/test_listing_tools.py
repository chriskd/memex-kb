"""Tests for KB listing MCP tools (whats_new, popular)."""

import os
from datetime import date, timedelta
from pathlib import Path

import pytest

from voidlabs_kb import server
from voidlabs_kb.models import ViewStats
from voidlabs_kb.views_tracker import save_views


async def _call_tool(tool_obj, /, *args, **kwargs):
    """Invoke the wrapped coroutine behind an MCP FunctionTool."""
    bound = tool_obj.fn(*args, **kwargs)
    if callable(bound):
        return await bound()
    return await bound


@pytest.fixture(autouse=True)
def reset_searcher_state(monkeypatch):
    """Ensure cached searcher state does not leak across tests."""
    monkeypatch.setattr(server, "_searcher", None)
    monkeypatch.setattr(server, "_searcher_ready", False)


@pytest.fixture
def kb_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary KB root with standard categories."""
    root = tmp_path / "kb"
    root.mkdir()
    for category in ("development", "architecture", "devops"):
        (root / category).mkdir()
    monkeypatch.setenv("KB_ROOT", str(root))
    return root


@pytest.fixture
def index_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary index root."""
    root = tmp_path / ".indices"
    root.mkdir()
    monkeypatch.setenv("INDEX_ROOT", str(root))
    return root


def _create_entry(path: Path, title: str, tags: list[str], created: date, updated: date | None = None):
    """Helper to create a KB entry with frontmatter."""
    updated_line = f"updated: {updated.isoformat()}\n" if updated else ""
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    content = f"""---
title: {title}
tags:
{tags_yaml}
created: {created.isoformat()}
{updated_line}---

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
        today = date.today()
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
        today = date.today()

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
        today = date.today()

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
        """Category filter restricts results."""
        today = date.today()

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

        results = await _call_tool(server.whats_new_tool, days=30, category="development")

        assert len(results) == 1
        assert results[0]["title"] == "Dev Entry"

    @pytest.mark.asyncio
    async def test_whats_new_filters_by_tag(self, kb_root):
        """Tag filter restricts results."""
        today = date.today()

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

        results = await _call_tool(server.whats_new_tool, days=30, tag="rust")

        assert len(results) == 1
        assert results[0]["title"] == "Rust Entry"

    @pytest.mark.asyncio
    async def test_whats_new_respects_limit(self, kb_root):
        """Result count is limited."""
        today = date.today()

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
        """include_created and include_updated flags work."""
        today = date.today()

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

        # Only created
        results_created = await _call_tool(
            server.whats_new_tool, days=30, include_created=True, include_updated=False
        )
        assert len(results_created) == 1
        assert results_created[0]["title"] == "New Entry"

        # Only updated
        results_updated = await _call_tool(
            server.whats_new_tool, days=30, include_created=False, include_updated=True
        )
        assert len(results_updated) == 1
        assert results_updated[0]["title"] == "Updated Entry"


class TestPopularTool:
    """Test popular MCP tool."""

    @pytest.mark.asyncio
    async def test_popular_returns_empty_when_no_views(self, kb_root, index_root):
        """Returns empty list when no views recorded."""
        _create_entry(
            kb_root / "development" / "entry.md",
            "Entry",
            ["python"],
            created=date.today(),
        )

        results = await _call_tool(server.popular_tool)

        assert results == []

    @pytest.mark.asyncio
    async def test_popular_returns_entries_sorted_by_views(self, kb_root, index_root):
        """Entries are sorted by view count descending."""
        from datetime import datetime

        today = date.today()
        now = datetime.now()

        _create_entry(kb_root / "development" / "low.md", "Low Views", ["python"], created=today)
        _create_entry(kb_root / "development" / "high.md", "High Views", ["python"], created=today)
        _create_entry(kb_root / "development" / "mid.md", "Mid Views", ["python"], created=today)

        views = {
            "development/low.md": ViewStats(total_views=5, last_viewed=now),
            "development/high.md": ViewStats(total_views=100, last_viewed=now),
            "development/mid.md": ViewStats(total_views=50, last_viewed=now),
        }
        save_views(views, index_root)

        results = await _call_tool(server.popular_tool, limit=10)

        titles = [r["title"] for r in results]
        assert titles == ["High Views", "Mid Views", "Low Views"]

    @pytest.mark.asyncio
    async def test_popular_includes_view_count(self, kb_root, index_root):
        """Results include view_count field."""
        from datetime import datetime

        today = date.today()
        now = datetime.now()

        _create_entry(kb_root / "development" / "entry.md", "Entry", ["python"], created=today)
        views = {"development/entry.md": ViewStats(total_views=42, last_viewed=now)}
        save_views(views, index_root)

        results = await _call_tool(server.popular_tool)

        assert results[0]["view_count"] == 42

    @pytest.mark.asyncio
    async def test_popular_filters_by_category(self, kb_root, index_root):
        """Category filter restricts results."""
        from datetime import datetime

        today = date.today()
        now = datetime.now()

        _create_entry(kb_root / "development" / "dev.md", "Dev Entry", ["python"], created=today)
        _create_entry(kb_root / "architecture" / "arch.md", "Arch Entry", ["design"], created=today)

        views = {
            "development/dev.md": ViewStats(total_views=10, last_viewed=now),
            "architecture/arch.md": ViewStats(total_views=20, last_viewed=now),
        }
        save_views(views, index_root)

        results = await _call_tool(server.popular_tool, category="development")

        assert len(results) == 1
        assert results[0]["title"] == "Dev Entry"

    @pytest.mark.asyncio
    async def test_popular_filters_by_tag(self, kb_root, index_root):
        """Tag filter restricts results."""
        from datetime import datetime

        today = date.today()
        now = datetime.now()

        _create_entry(kb_root / "development" / "python.md", "Python Entry", ["python"], created=today)
        _create_entry(kb_root / "development" / "rust.md", "Rust Entry", ["rust"], created=today)

        views = {
            "development/python.md": ViewStats(total_views=10, last_viewed=now),
            "development/rust.md": ViewStats(total_views=20, last_viewed=now),
        }
        save_views(views, index_root)

        results = await _call_tool(server.popular_tool, tag="rust")

        assert len(results) == 1
        assert results[0]["title"] == "Rust Entry"

    @pytest.mark.asyncio
    async def test_popular_skips_deleted_entries(self, kb_root, index_root):
        """Entries with views but no file are skipped."""
        from datetime import datetime

        today = date.today()
        now = datetime.now()

        _create_entry(kb_root / "development" / "exists.md", "Exists", ["python"], created=today)

        views = {
            "development/exists.md": ViewStats(total_views=10, last_viewed=now),
            "development/deleted.md": ViewStats(total_views=100, last_viewed=now),
        }
        save_views(views, index_root)

        results = await _call_tool(server.popular_tool)

        assert len(results) == 1
        assert results[0]["title"] == "Exists"


class TestGetToolViewTracking:
    """Test that get_tool records views."""

    @pytest.mark.asyncio
    async def test_get_tool_records_view(self, kb_root, index_root):
        """get_tool increments view count."""
        from voidlabs_kb.views_tracker import load_views

        _create_entry(
            kb_root / "development" / "entry.md",
            "Test Entry",
            ["python"],
            created=date.today(),
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
