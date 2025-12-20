"""Persistent view tracking for KB entries."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import get_index_root
from .models import ViewStats

VIEWS_FILENAME = "views.json"
SCHEMA_VERSION = 1
PRUNE_DAYS = 90  # Keep daily buckets for this many days


def _views_path(index_root: Path | None = None) -> Path:
    """Get path to views.json, creating directory if needed."""
    root = index_root or get_index_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / VIEWS_FILENAME


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

    Args:
        path: Relative path to the KB entry (e.g., "development/python.md")
        index_root: Optional override for index storage location
    """
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
