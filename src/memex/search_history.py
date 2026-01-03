"""Persistent search history tracking for KB searches."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import get_index_root
from .models import SearchHistoryEntry

HISTORY_FILENAME = "search_history.json"
SCHEMA_VERSION = 1
MAX_HISTORY_ENTRIES = 100  # Keep last N searches
PRUNE_DAYS = 30  # Remove entries older than this


def _history_path(index_root: Path | None = None) -> Path:
    """Get path to search_history.json, creating directory if needed."""
    root = index_root or get_index_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / HISTORY_FILENAME


def load_history(index_root: Path | None = None) -> list[SearchHistoryEntry]:
    """Load search history from disk.

    Returns:
        List of SearchHistoryEntry, most recent first.
    """
    path = _history_path(index_root)
    if not path.exists():
        return []

    try:
        payload: dict[str, Any] = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    # Handle schema migration if needed
    version = payload.get("schema_version", 1)
    if version != SCHEMA_VERSION:
        return []  # Reset on schema change

    entries: list[SearchHistoryEntry] = []
    for data in payload.get("history", []):
        try:
            entries.append(
                SearchHistoryEntry(
                    query=data["query"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    result_count=data.get("result_count", 0),
                    mode=data.get("mode", "hybrid"),
                    tags=data.get("tags", []),
                )
            )
        except (KeyError, ValueError, TypeError):
            continue  # Skip malformed entries

    return entries


def save_history(
    entries: list[SearchHistoryEntry], index_root: Path | None = None
) -> None:
    """Save search history to disk.

    Uses atomic write pattern (write to temp, then rename).
    Prunes old entries before saving.
    """
    path = _history_path(index_root)

    # Prune old entries
    cutoff = datetime.now() - timedelta(days=PRUNE_DAYS)
    entries = [e for e in entries if e.timestamp >= cutoff]

    # Limit to max entries
    entries = entries[:MAX_HISTORY_ENTRIES]

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "history": [
            {
                "query": e.query,
                "timestamp": e.timestamp.isoformat(),
                "result_count": e.result_count,
                "mode": e.mode,
                "tags": e.tags,
            }
            for e in entries
        ],
    }

    # Atomic write
    dir_path = path.parent
    with tempfile.NamedTemporaryFile(
        mode="w", dir=dir_path, delete=False, suffix=".tmp"
    ) as f:
        json.dump(payload, f, indent=2)
        temp_path = Path(f.name)

    temp_path.rename(path)


def record_search(
    query: str,
    result_count: int = 0,
    mode: str = "hybrid",
    tags: list[str] | None = None,
    index_root: Path | None = None,
) -> None:
    """Record a search query to history.

    Args:
        query: The search query string.
        result_count: Number of results returned.
        mode: Search mode used (hybrid, keyword, semantic).
        tags: Tag filters applied.
        index_root: Optional override for index storage location.
    """
    entries = load_history(index_root)

    entry = SearchHistoryEntry(
        query=query,
        timestamp=datetime.now(),
        result_count=result_count,
        mode=mode,
        tags=tags or [],
    )

    # Prepend new entry (most recent first)
    entries.insert(0, entry)

    save_history(entries, index_root)


def get_recent(
    limit: int = 10,
    index_root: Path | None = None,
) -> list[SearchHistoryEntry]:
    """Get most recent searches.

    Args:
        limit: Maximum entries to return.
        index_root: Optional override for index storage location.

    Returns:
        List of SearchHistoryEntry, most recent first.
    """
    entries = load_history(index_root)
    return entries[:limit]


def get_by_index(
    index: int,
    index_root: Path | None = None,
) -> SearchHistoryEntry | None:
    """Get a search history entry by index (1-based).

    Args:
        index: 1-based index into history (1 = most recent).
        index_root: Optional override for index storage location.

    Returns:
        SearchHistoryEntry or None if index out of range.
    """
    entries = load_history(index_root)
    if index < 1 or index > len(entries):
        return None
    return entries[index - 1]


def clear_history(index_root: Path | None = None) -> int:
    """Clear all search history.

    Args:
        index_root: Optional override for index storage location.

    Returns:
        Number of entries cleared.
    """
    entries = load_history(index_root)
    count = len(entries)
    save_history([], index_root)
    return count
