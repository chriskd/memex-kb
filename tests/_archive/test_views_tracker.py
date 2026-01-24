"""Tests for KB views tracking."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from memex.models import ViewStats
from memex.views_tracker import (
    cleanup_stale_entries,
    get_popular,
    load_views,
    record_view,
    save_views,
)


@pytest.fixture
def index_root(tmp_path) -> Path:
    """Create a temporary index root."""
    root = tmp_path / ".indices"
    root.mkdir()
    return root


class TestViewsPersistence:
    """Test load/save round-trip for views."""

    def test_load_empty(self, index_root):
        """load_views returns empty dict when file doesn't exist."""
        views = load_views(index_root)
        assert views == {}

    def test_save_and_load_round_trip(self, index_root):
        """Views can be saved and loaded back."""
        now = datetime.now()
        today = now.date().isoformat()

        views = {
            "dev/test.md": ViewStats(
                total_views=42,
                last_viewed=now,
                views_by_day={today: 5, "2024-01-01": 37},
            )
        }

        save_views(views, index_root)
        loaded = load_views(index_root)

        assert "dev/test.md" in loaded
        assert loaded["dev/test.md"].total_views == 42
        assert loaded["dev/test.md"].views_by_day[today] == 5

    def test_load_handles_malformed_json(self, index_root):
        """load_views returns empty dict for malformed JSON."""
        views_file = index_root / "views.json"
        views_file.write_text("not valid json {{{")

        views = load_views(index_root)
        assert views == {}


class TestRecordView:
    """Test view recording functionality."""

    def test_record_view_creates_entry(self, index_root):
        """First view creates new entry."""
        record_view("dev/new.md", index_root)

        views = load_views(index_root)
        assert "dev/new.md" in views
        assert views["dev/new.md"].total_views == 1
        assert views["dev/new.md"].last_viewed is not None

    def test_record_view_increments_count(self, index_root):
        """Multiple views increment count."""
        record_view("dev/test.md", index_root)
        record_view("dev/test.md", index_root)
        record_view("dev/test.md", index_root)

        views = load_views(index_root)
        assert views["dev/test.md"].total_views == 3

    def test_record_view_updates_last_viewed(self, index_root):
        """Each view updates last_viewed timestamp."""
        record_view("dev/test.md", index_root)
        first_view = load_views(index_root)["dev/test.md"].last_viewed

        record_view("dev/test.md", index_root)
        second_view = load_views(index_root)["dev/test.md"].last_viewed

        assert second_view >= first_view

    def test_record_view_buckets_by_day(self, index_root):
        """Views are bucketed by day."""
        today = datetime.now().date().isoformat()

        record_view("dev/test.md", index_root)
        record_view("dev/test.md", index_root)

        views = load_views(index_root)
        assert today in views["dev/test.md"].views_by_day
        assert views["dev/test.md"].views_by_day[today] == 2


class TestGetPopular:
    """Test popular entries retrieval."""

    def test_get_popular_empty(self, index_root):
        """get_popular returns empty list when no views."""
        result = get_popular(limit=10, index_root=index_root)
        assert result == []

    def test_get_popular_sorts_by_total_views(self, index_root):
        """Entries are sorted by total_views descending."""
        now = datetime.now()
        views = {
            "low.md": ViewStats(total_views=5, last_viewed=now),
            "high.md": ViewStats(total_views=100, last_viewed=now),
            "mid.md": ViewStats(total_views=50, last_viewed=now),
        }
        save_views(views, index_root)

        result = get_popular(limit=10, index_root=index_root)

        paths = [path for path, _ in result]
        assert paths == ["high.md", "mid.md", "low.md"]

    def test_get_popular_respects_limit(self, index_root):
        """Result is limited to requested count."""
        now = datetime.now()
        views = {f"entry{i}.md": ViewStats(total_views=i, last_viewed=now) for i in range(10)}
        save_views(views, index_root)

        result = get_popular(limit=3, index_root=index_root)

        assert len(result) == 3

    def test_get_popular_with_days_filter(self, index_root):
        """days parameter filters to recent views only."""
        now = datetime.now()
        today = now.date().isoformat()
        old_date = (now - timedelta(days=60)).date().isoformat()

        views = {
            "recent.md": ViewStats(
                total_views=50,
                last_viewed=now,
                views_by_day={today: 50},
            ),
            "old.md": ViewStats(
                total_views=100,  # Higher total, but old views
                last_viewed=now,
                views_by_day={old_date: 100},
            ),
        }
        save_views(views, index_root)

        # Without filter, old.md wins
        result_all = get_popular(limit=10, days=None, index_root=index_root)
        assert result_all[0][0] == "old.md"

        # With 30-day filter, recent.md wins
        result_recent = get_popular(limit=10, days=30, index_root=index_root)
        assert result_recent[0][0] == "recent.md"


class TestCleanupStaleEntries:
    """Test stale entry cleanup."""

    def test_cleanup_removes_nonexistent_entries(self, index_root):
        """Entries not in valid_paths are removed."""
        now = datetime.now()
        views = {
            "exists.md": ViewStats(total_views=10, last_viewed=now),
            "deleted.md": ViewStats(total_views=20, last_viewed=now),
        }
        save_views(views, index_root)

        removed = cleanup_stale_entries({"exists.md"}, index_root)

        assert removed == 1
        views = load_views(index_root)
        assert "exists.md" in views
        assert "deleted.md" not in views

    def test_cleanup_handles_empty_views(self, index_root):
        """cleanup_stale_entries handles empty views file."""
        removed = cleanup_stale_entries({"exists.md"}, index_root)
        assert removed == 0

    def test_cleanup_handles_all_valid(self, index_root):
        """No entries removed when all are valid."""
        now = datetime.now()
        views = {
            "a.md": ViewStats(total_views=10, last_viewed=now),
            "b.md": ViewStats(total_views=20, last_viewed=now),
        }
        save_views(views, index_root)

        removed = cleanup_stale_entries({"a.md", "b.md", "c.md"}, index_root)

        assert removed == 0
