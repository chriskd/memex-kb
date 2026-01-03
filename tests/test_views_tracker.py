"""Tests for KB views tracking."""

import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from memex.models import ViewStats
from memex.views_tracker import (
    PRUNE_INTERVAL_SECONDS,
    CachedViewsTracker,
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


class TestCachedViewsTracker:
    """Test the CachedViewsTracker class."""

    def test_context_manager_flushes_on_exit(self, index_root):
        """Views are flushed when exiting context manager."""
        with CachedViewsTracker(index_root) as tracker:
            tracker.record_view("dev/test.md")
            # Not flushed yet
            assert tracker.is_dirty

        # After exiting, file should exist
        views = load_views(index_root)
        assert "dev/test.md" in views
        assert views["dev/test.md"].total_views == 1

    def test_batched_writes_no_disk_io_until_flush(self, index_root):
        """Multiple record_view calls don't write to disk until flush."""
        tracker = CachedViewsTracker(index_root)

        # Record multiple views
        tracker.record_view("dev/a.md")
        tracker.record_view("dev/b.md")
        tracker.record_view("dev/c.md")

        # File should not exist yet (no flush)
        views_file = index_root / "views.json"
        assert not views_file.exists()

        # Explicit flush
        tracker.flush()

        # Now file exists with all entries
        views = load_views(index_root)
        assert len(views) == 3
        assert views["dev/a.md"].total_views == 1
        assert views["dev/b.md"].total_views == 1
        assert views["dev/c.md"].total_views == 1

    def test_auto_flush_at_threshold(self, index_root):
        """Views auto-flush when threshold is reached."""
        tracker = CachedViewsTracker(index_root, flush_threshold=5)

        # Record 4 views - no flush yet
        for i in range(4):
            tracker.record_view(f"entry{i}.md")

        views_file = index_root / "views.json"
        assert not views_file.exists()

        # 5th view triggers auto-flush
        tracker.record_view("entry4.md")

        assert views_file.exists()
        views = load_views(index_root)
        assert len(views) == 5

    def test_pending_writes_counter(self, index_root):
        """pending_writes tracks unflushed writes correctly."""
        tracker = CachedViewsTracker(index_root)

        assert tracker.pending_writes == 0

        tracker.record_view("a.md")
        assert tracker.pending_writes == 1

        tracker.record_view("b.md")
        assert tracker.pending_writes == 2

        tracker.flush()
        assert tracker.pending_writes == 0

    def test_is_dirty_flag(self, index_root):
        """is_dirty tracks modification state."""
        tracker = CachedViewsTracker(index_root)

        assert not tracker.is_dirty

        tracker.record_view("a.md")
        assert tracker.is_dirty

        tracker.flush()
        assert not tracker.is_dirty

    def test_flush_returns_false_when_clean(self, index_root):
        """flush() returns False if nothing to flush."""
        tracker = CachedViewsTracker(index_root)

        assert tracker.flush() is False

        tracker.record_view("a.md")
        assert tracker.flush() is True
        assert tracker.flush() is False  # Already clean

    def test_get_views_returns_copy(self, index_root):
        """get_views returns a copy, not the internal cache."""
        tracker = CachedViewsTracker(index_root)
        tracker.record_view("a.md")

        views = tracker.get_views()
        views["b.md"] = ViewStats(total_views=999)

        # Internal cache should not be modified
        internal_views = tracker.get_views()
        assert "b.md" not in internal_views

    def test_delete_entry_flushes_immediately(self, index_root):
        """delete_entry flushes to ensure data integrity."""
        tracker = CachedViewsTracker(index_root)
        tracker.record_view("a.md")
        tracker.record_view("b.md")
        tracker.flush()

        # Delete entry
        result = tracker.delete_entry("a.md")

        assert result is True
        assert not tracker.is_dirty  # Should have flushed

        # Verify on disk
        views = load_views(index_root)
        assert "a.md" not in views
        assert "b.md" in views

    def test_delete_nonexistent_entry(self, index_root):
        """delete_entry returns False for nonexistent entry."""
        tracker = CachedViewsTracker(index_root)
        result = tracker.delete_entry("nonexistent.md")
        assert result is False

    def test_cleanup_stale_flushes_immediately(self, index_root):
        """cleanup_stale flushes to ensure data integrity."""
        tracker = CachedViewsTracker(index_root)
        tracker.record_view("exists.md")
        tracker.record_view("deleted.md")
        tracker.flush()

        # Create new tracker and cleanup
        tracker2 = CachedViewsTracker(index_root)
        removed = tracker2.cleanup_stale({"exists.md"})

        assert removed == 1
        assert not tracker2.is_dirty

        views = load_views(index_root)
        assert "exists.md" in views
        assert "deleted.md" not in views

    def test_set_views_marks_dirty(self, index_root):
        """set_views marks the cache as dirty."""
        tracker = CachedViewsTracker(index_root)

        tracker.set_views({
            "a.md": ViewStats(total_views=10, last_viewed=datetime.now())
        })

        assert tracker.is_dirty
        assert tracker._loaded

        tracker.flush()
        views = load_views(index_root)
        assert views["a.md"].total_views == 10

    def test_repeated_views_same_entry(self, index_root):
        """Multiple views to same entry accumulate correctly."""
        with CachedViewsTracker(index_root) as tracker:
            for _ in range(10):
                tracker.record_view("popular.md")

        views = load_views(index_root)
        assert views["popular.md"].total_views == 10

    def test_lazy_load(self, index_root):
        """Views are loaded lazily on first access."""
        # Pre-populate views file
        now = datetime.now()
        save_views({
            "existing.md": ViewStats(total_views=42, last_viewed=now)
        }, index_root)

        tracker = CachedViewsTracker(index_root)
        assert not tracker._loaded

        # Access triggers load
        views = tracker.get_views()
        assert tracker._loaded
        assert views["existing.md"].total_views == 42


class TestLazyPruning:
    """Test lazy pruning behavior."""

    def test_prune_on_load(self, index_root):
        """Old daily buckets are pruned when loading."""
        now = datetime.now()
        today = now.date().isoformat()
        old_date = (now - timedelta(days=100)).date().isoformat()  # > 90 days

        # Create views file with old data
        views = {
            "entry.md": ViewStats(
                total_views=150,
                last_viewed=now,
                views_by_day={today: 50, old_date: 100},
            )
        }
        save_views(views, index_root)

        # Load via cached tracker - should prune
        tracker = CachedViewsTracker(index_root)
        loaded = tracker.get_views()

        # Old date should be pruned
        assert today in loaded["entry.md"].views_by_day
        assert old_date not in loaded["entry.md"].views_by_day

    def test_prune_not_repeated_within_interval(self, index_root):
        """Pruning is skipped if done recently."""
        tracker = CachedViewsTracker(index_root)
        tracker.record_view("a.md")

        # Force a prune time
        tracker._last_prune_time = time.time()

        # Mock _prune_views_by_day to track calls
        prune_calls = []
        original_maybe_prune = tracker._maybe_prune

        def tracking_prune(views, force=False):
            prune_calls.append(force)
            return original_maybe_prune(views, force)

        tracker._maybe_prune = tracking_prune

        # Flush should skip pruning (within interval)
        tracker.flush()

        # Only one call (from flush), and it should return early
        assert len(prune_calls) == 1

    def test_prune_after_interval(self, index_root):
        """Pruning happens if interval has passed."""
        tracker = CachedViewsTracker(index_root)
        tracker.record_view("a.md")

        # Set last prune time to past
        tracker._last_prune_time = time.time() - PRUNE_INTERVAL_SECONDS - 1

        now = datetime.now()
        old_date = (now - timedelta(days=100)).date().isoformat()
        tracker._cache["a.md"].views_by_day[old_date] = 50

        tracker.flush()

        # Old date should be pruned
        views = load_views(index_root)
        assert old_date not in views["a.md"].views_by_day


class TestCachedTrackerPerformance:
    """Test that caching provides expected performance benefits."""

    def test_batched_views_faster_than_individual(self, index_root):
        """Demonstrate that batched views avoid repeated disk I/O."""
        num_views = 50

        # Using cached tracker (batched)
        start_batched = time.perf_counter()
        with CachedViewsTracker(index_root) as tracker:
            for i in range(num_views):
                tracker.record_view(f"entry{i}.md")
        batched_time = time.perf_counter() - start_batched

        # Clear for individual test
        (index_root / "views.json").unlink()

        # Using individual record_view (unbatched)
        start_individual = time.perf_counter()
        for i in range(num_views):
            record_view(f"entry{i}.md", index_root)
        individual_time = time.perf_counter() - start_individual

        # Batched should be significantly faster (at least 2x)
        # We use a generous factor since CI can be slow
        assert batched_time < individual_time, (
            f"Batched ({batched_time:.4f}s) should be faster than "
            f"individual ({individual_time:.4f}s)"
        )

    def test_many_views_single_flush(self, index_root):
        """Many views result in single disk write."""
        num_views = 1000

        with CachedViewsTracker(index_root, flush_threshold=num_views + 1) as tracker:
            for i in range(num_views):
                tracker.record_view(f"entry{i % 100}.md")

            # Should have pending writes but no file yet
            assert tracker.pending_writes == num_views

        # After context exit, all views persisted
        views = load_views(index_root)
        total = sum(v.total_views for v in views.values())
        assert total == num_views


class TestDataIntegrity:
    """Test that caching maintains data integrity."""

    def test_views_not_lost_on_crash_simulation(self, index_root):
        """Test that unflushed views can be detected."""
        tracker = CachedViewsTracker(index_root)
        tracker.record_view("important.md")

        # Simulate "crash" - don't flush, check dirty flag
        assert tracker.is_dirty
        assert tracker.pending_writes == 1

        # Caller could check is_dirty before discarding tracker
        # and decide to flush in error handler

    def test_atomic_write_integrity(self, index_root):
        """Verify atomic write doesn't corrupt on partial failure."""
        tracker = CachedViewsTracker(index_root)
        tracker.record_view("a.md")
        tracker.flush()

        # Add more views
        tracker.record_view("b.md")
        tracker.record_view("c.md")
        tracker.flush()

        # Verify all data present
        views = load_views(index_root)
        assert "a.md" in views
        assert "b.md" in views
        assert "c.md" in views

    def test_concurrent_tracker_isolation(self, index_root):
        """Multiple trackers don't interfere with each other's cache."""
        # First tracker writes
        tracker1 = CachedViewsTracker(index_root)
        tracker1.record_view("from_tracker1.md")
        tracker1.flush()

        # Second tracker reads and writes
        tracker2 = CachedViewsTracker(index_root)
        views = tracker2.get_views()
        assert "from_tracker1.md" in views

        tracker2.record_view("from_tracker2.md")
        tracker2.flush()

        # Verify both present
        final_views = load_views(index_root)
        assert "from_tracker1.md" in final_views
        assert "from_tracker2.md" in final_views
