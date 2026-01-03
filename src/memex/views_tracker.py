"""Persistent view tracking for KB entries."""

from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import get_index_root
from .models import ViewStats

VIEWS_FILENAME = "views.json"
SCHEMA_VERSION = 1
PRUNE_DAYS = 90  # Keep daily buckets for this many days
PRUNE_INTERVAL_SECONDS = 86400  # Only prune once per day (24 hours)
DEFAULT_FLUSH_THRESHOLD = 100  # Flush after this many pending writes


def _views_path(index_root: Path | None = None) -> Path:
    """Get path to views.json, creating directory if needed."""
    root = index_root or get_index_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / VIEWS_FILENAME


def _prune_views_by_day(views_by_day: dict[str, int], cutoff_date: str) -> dict[str, int]:
    """Filter out daily entries older than cutoff_date."""
    return {day: count for day, count in views_by_day.items() if day >= cutoff_date}


class CachedViewsTracker:
    """Write-through cached view tracker with batched disk writes.

    This class maintains an in-memory cache of view statistics and batches
    writes to disk for better performance. Views are flushed to disk:
    - When flush() is explicitly called
    - When the pending write count exceeds flush_threshold
    - When exiting a context manager (with statement)

    Pruning of old daily buckets is done lazily - only on load or if more
    than PRUNE_INTERVAL_SECONDS have passed since the last prune.

    Usage:
        # As context manager (recommended):
        with CachedViewsTracker(index_root) as tracker:
            tracker.record_view("dev/test.md")
            tracker.record_view("dev/other.md")
        # Views are automatically flushed on exit

        # Manual usage:
        tracker = CachedViewsTracker(index_root)
        tracker.record_view("dev/test.md")
        tracker.flush()  # Explicit flush
    """

    def __init__(
        self,
        index_root: Path | None = None,
        flush_threshold: int = DEFAULT_FLUSH_THRESHOLD,
    ):
        """Initialize the cached tracker.

        Args:
            index_root: Optional override for index storage location.
            flush_threshold: Number of pending writes before auto-flush.
        """
        self._index_root = index_root
        self._flush_threshold = flush_threshold
        self._cache: dict[str, ViewStats] | None = None
        self._dirty = False
        self._pending_writes = 0
        self._last_prune_time: float = 0.0
        self._loaded = False

    def __enter__(self) -> CachedViewsTracker:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, flushing any pending writes."""
        self.flush()

    def _ensure_loaded(self) -> None:
        """Lazily load views from disk if not already loaded."""
        if self._loaded:
            return

        self._cache = self._load_views_from_disk()
        self._loaded = True

    def _load_views_from_disk(self) -> dict[str, ViewStats]:
        """Load view statistics from disk.

        Returns:
            Dict mapping entry paths to their ViewStats.
        """
        path = _views_path(self._index_root)
        if not path.exists():
            return {}

        try:
            payload: dict[str, Any] = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

        # Handle schema migration if needed
        version = payload.get("schema_version", 1)
        if version != SCHEMA_VERSION:
            return {}  # Reset on schema change

        views: dict[str, ViewStats] = {}
        for entry_path, data in payload.get("views", {}).items():
            try:
                views[entry_path] = ViewStats(
                    total_views=data.get("total_views", 0),
                    last_viewed=datetime.fromisoformat(data["last_viewed"])
                    if data.get("last_viewed")
                    else None,
                    views_by_day=data.get("views_by_day", {}),
                )
            except (KeyError, ValueError, TypeError):
                continue  # Skip malformed entries

        # Prune on load
        self._maybe_prune(views, force=True)

        return views

    def _maybe_prune(
        self, views: dict[str, ViewStats], force: bool = False
    ) -> bool:
        """Prune old daily buckets if enough time has passed.

        Args:
            views: The views dict to prune in-place.
            force: If True, prune regardless of time elapsed.

        Returns:
            True if pruning was performed, False otherwise.
        """
        now = time.time()
        if not force and (now - self._last_prune_time) < PRUNE_INTERVAL_SECONDS:
            return False

        cutoff_date = (datetime.now() - timedelta(days=PRUNE_DAYS)).date().isoformat()

        for stats in views.values():
            pruned = _prune_views_by_day(stats.views_by_day, cutoff_date)
            if len(pruned) != len(stats.views_by_day):
                stats.views_by_day = pruned

        self._last_prune_time = now
        return True

    def _save_views_to_disk(self, views: dict[str, ViewStats]) -> None:
        """Save view statistics to disk.

        Uses atomic write pattern (write to temp, then rename).
        Does NOT prune on save - pruning is done lazily on load.
        """
        path = _views_path(self._index_root)

        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "views": {},
        }

        for entry_path, stats in views.items():
            payload["views"][entry_path] = {
                "total_views": stats.total_views,
                "last_viewed": stats.last_viewed.isoformat() if stats.last_viewed else None,
                "views_by_day": stats.views_by_day,
            }

        # Atomic write
        dir_path = path.parent
        with tempfile.NamedTemporaryFile(
            mode="w", dir=dir_path, delete=False, suffix=".tmp"
        ) as f:
            json.dump(payload, f, indent=2)
            temp_path = Path(f.name)

        temp_path.rename(path)

    def get_views(self) -> dict[str, ViewStats]:
        """Get current views (from cache if loaded).

        Returns:
            Dict mapping entry paths to their ViewStats.
        """
        self._ensure_loaded()
        return self._cache.copy() if self._cache else {}

    def record_view(self, path: str) -> None:
        """Record a view for the given entry path.

        Args:
            path: Relative path to the KB entry (e.g., "development/python.md")
        """
        self._ensure_loaded()

        now = datetime.now()
        today = now.date().isoformat()

        if path not in self._cache:
            self._cache[path] = ViewStats()

        stats = self._cache[path]
        stats.total_views += 1
        stats.last_viewed = now
        stats.views_by_day[today] = stats.views_by_day.get(today, 0) + 1

        self._dirty = True
        self._pending_writes += 1

        # Auto-flush if threshold reached
        if self._pending_writes >= self._flush_threshold:
            self.flush()

    def flush(self) -> bool:
        """Flush pending writes to disk.

        Returns:
            True if writes were flushed, False if cache was clean.
        """
        if not self._dirty or self._cache is None:
            return False

        # Maybe prune before saving (lazy prune)
        self._maybe_prune(self._cache)

        self._save_views_to_disk(self._cache)
        self._dirty = False
        self._pending_writes = 0
        return True

    def delete_entry(self, path: str) -> bool:
        """Remove view record for a single entry.

        Args:
            path: Path to the entry to remove.

        Returns:
            True if entry was found and removed, False otherwise.
        """
        self._ensure_loaded()

        if path not in self._cache:
            return False

        del self._cache[path]
        self._dirty = True
        self._pending_writes += 1

        # Flush immediately for deletes (data integrity)
        self.flush()
        return True

    def cleanup_stale(self, valid_paths: set[str]) -> int:
        """Remove view records for entries that no longer exist.

        Args:
            valid_paths: Set of paths that currently exist in the KB.

        Returns:
            Count of removed entries.
        """
        self._ensure_loaded()

        if not self._cache:
            return 0

        stale_paths = set(self._cache.keys()) - valid_paths
        if not stale_paths:
            return 0

        for path in stale_paths:
            del self._cache[path]

        self._dirty = True
        # Flush immediately for cleanup (data integrity)
        self.flush()
        return len(stale_paths)

    def set_views(self, views: dict[str, ViewStats]) -> None:
        """Set the views cache directly (for testing or bulk operations).

        Args:
            views: Dict mapping entry paths to their ViewStats.
        """
        self._cache = views.copy()
        self._loaded = True
        self._dirty = True

    @property
    def is_dirty(self) -> bool:
        """Return True if there are unflushed changes."""
        return self._dirty

    @property
    def pending_writes(self) -> int:
        """Return the number of pending writes since last flush."""
        return self._pending_writes


# Global tracker instance for module-level functions
_global_tracker: CachedViewsTracker | None = None


def _get_tracker(index_root: Path | None = None) -> CachedViewsTracker:
    """Get or create the global tracker instance.

    Note: For repeated operations, prefer using CachedViewsTracker directly
    as a context manager for better control over flushing.
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CachedViewsTracker(index_root)
    return _global_tracker


def reset_global_tracker() -> None:
    """Reset the global tracker (for testing)."""
    global _global_tracker
    if _global_tracker is not None:
        _global_tracker.flush()
    _global_tracker = None


# Legacy module-level API functions (maintain backward compatibility)


def load_views(index_root: Path | None = None) -> dict[str, ViewStats]:
    """Load view statistics from disk.

    Returns:
        Dict mapping entry paths to their ViewStats.
    """
    path = _views_path(index_root)
    if not path.exists():
        return {}

    try:
        payload: dict[str, Any] = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    # Handle schema migration if needed
    version = payload.get("schema_version", 1)
    if version != SCHEMA_VERSION:
        return {}  # Reset on schema change

    views: dict[str, ViewStats] = {}
    for entry_path, data in payload.get("views", {}).items():
        try:
            views[entry_path] = ViewStats(
                total_views=data.get("total_views", 0),
                last_viewed=datetime.fromisoformat(data["last_viewed"])
                if data.get("last_viewed")
                else None,
                views_by_day=data.get("views_by_day", {}),
            )
        except (KeyError, ValueError, TypeError):
            continue  # Skip malformed entries

    return views


def save_views(views: dict[str, ViewStats], index_root: Path | None = None) -> None:
    """Save view statistics to disk.

    Uses atomic write pattern (write to temp, then rename).
    """
    path = _views_path(index_root)

    # Prune old daily buckets
    cutoff_date = (datetime.now() - timedelta(days=PRUNE_DAYS)).date().isoformat()

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "views": {},
    }

    for entry_path, stats in views.items():
        # Filter out old daily entries
        pruned_by_day = {
            day: count
            for day, count in stats.views_by_day.items()
            if day >= cutoff_date
        }

        payload["views"][entry_path] = {
            "total_views": stats.total_views,
            "last_viewed": stats.last_viewed.isoformat() if stats.last_viewed else None,
            "views_by_day": pruned_by_day,
        }

    # Atomic write
    dir_path = path.parent
    with tempfile.NamedTemporaryFile(
        mode="w", dir=dir_path, delete=False, suffix=".tmp"
    ) as f:
        json.dump(payload, f, indent=2)
        temp_path = Path(f.name)

    temp_path.rename(path)


def record_view(path: str, index_root: Path | None = None) -> None:
    """Record a view for the given entry path.

    Note: For repeated calls, prefer using CachedViewsTracker as a context
    manager for better performance:

        with CachedViewsTracker(index_root) as tracker:
            for path in paths:
                tracker.record_view(path)

    Args:
        path: Relative path to the KB entry (e.g., "development/python.md")
        index_root: Optional override for index storage location
    """
    # For single calls, use full read-modify-write for backward compatibility
    # This maintains the original behavior but callers can use CachedViewsTracker
    # for batched operations
    views = load_views(index_root)

    now = datetime.now()
    today = now.date().isoformat()

    if path not in views:
        views[path] = ViewStats()

    stats = views[path]
    stats.total_views += 1
    stats.last_viewed = now
    stats.views_by_day[today] = stats.views_by_day.get(today, 0) + 1

    save_views(views, index_root)


def get_popular(
    limit: int = 10,
    days: int | None = None,
    index_root: Path | None = None,
) -> list[tuple[str, ViewStats]]:
    """Get most viewed entries, optionally within time window.

    Args:
        limit: Maximum entries to return.
        days: If set, only count views from the last N days.
        index_root: Optional override for index storage location.

    Returns:
        List of (path, ViewStats) tuples, sorted by view count descending.
    """
    views = load_views(index_root)

    if not views:
        return []

    if days is not None:
        # Filter to time window
        cutoff_date = (datetime.now() - timedelta(days=days)).date().isoformat()

        def windowed_count(stats: ViewStats) -> int:
            return sum(
                count
                for day, count in stats.views_by_day.items()
                if day >= cutoff_date
            )

        sorted_views = sorted(
            views.items(),
            key=lambda x: windowed_count(x[1]),
            reverse=True,
        )
    else:
        # Use total_views
        sorted_views = sorted(
            views.items(),
            key=lambda x: x[1].total_views,
            reverse=True,
        )

    return sorted_views[:limit]


def delete_entry_views(
    path: str,
    index_root: Path | None = None,
) -> bool:
    """Remove view record for a single entry.

    Args:
        path: Path to the entry to remove.
        index_root: Optional override for index storage location.

    Returns:
        True if entry was found and removed, False otherwise.
    """
    views = load_views(index_root)

    if path not in views:
        return False

    del views[path]
    save_views(views, index_root)
    return True


def cleanup_stale_entries(
    valid_paths: set[str],
    index_root: Path | None = None,
) -> int:
    """Remove view records for entries that no longer exist.

    Args:
        valid_paths: Set of paths that currently exist in the KB.
        index_root: Optional override for index storage location.

    Returns:
        Count of removed entries.
    """
    views = load_views(index_root)

    if not views:
        return 0

    stale_paths = set(views.keys()) - valid_paths
    if not stale_paths:
        return 0

    for path in stale_paths:
        del views[path]

    save_views(views, index_root)
    return len(stale_paths)
