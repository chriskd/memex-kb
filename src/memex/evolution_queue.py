"""Queue management for memory evolution.

This module handles the queue file for non-blocking memory evolution.
Work is queued during `add_entry()` and processed by `mx evolve`.

Queue format: JSONL file at {kb_root}/.indices/evolution_queue.jsonl
Each line: {"new_entry": "...", "neighbor": "...", "score": 0.8, "queued_at": "..."}
"""

from __future__ import annotations

import fcntl
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import get_kb_root

log = logging.getLogger(__name__)

QUEUE_FILENAME = "evolution_queue.jsonl"


@dataclass
class QueueItem:
    """A single item in the evolution queue."""

    new_entry: str
    """Path to the newly added entry."""

    neighbor: str
    """Path to the neighbor entry that needs evolution."""

    score: float
    """Similarity score between entries (0.0-1.0)."""

    queued_at: datetime
    """When this item was queued."""

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "new_entry": self.new_entry,
            "neighbor": self.neighbor,
            "score": self.score,
            "queued_at": self.queued_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> QueueItem:
        """Create from dict (parsed JSON)."""
        queued_at = data.get("queued_at")
        if isinstance(queued_at, str):
            # Parse ISO format timestamp
            queued_at = datetime.fromisoformat(queued_at)
        elif queued_at is None:
            queued_at = datetime.now(UTC)

        return cls(
            new_entry=data["new_entry"],
            neighbor=data["neighbor"],
            score=float(data.get("score", 0.0)),
            queued_at=queued_at,
        )


def get_queue_path(kb_root: Path | None = None) -> Path:
    """Get the path to the evolution queue file.

    Args:
        kb_root: KB root directory. If None, auto-discovers.

    Returns:
        Path to the queue file (may not exist yet).
    """
    if kb_root is None:
        kb_root = get_kb_root()
    indices_dir = kb_root / ".indices"
    indices_dir.mkdir(exist_ok=True)
    return indices_dir / QUEUE_FILENAME


def queue_evolution(
    new_entry_path: str,
    neighbors: list[tuple[str, float]],
    kb_root: Path | None = None,
) -> int:
    """Queue evolution work for later processing.

    Appends items to the queue file with file locking for concurrent safety.

    Args:
        new_entry_path: Path to the newly added entry.
        neighbors: List of (neighbor_path, score) tuples to evolve.
        kb_root: KB root directory. If None, auto-discovers.

    Returns:
        Number of items queued.
    """
    if not neighbors:
        return 0

    queue_path = get_queue_path(kb_root)
    now = datetime.now(UTC)

    items = [
        QueueItem(
            new_entry=new_entry_path,
            neighbor=neighbor_path,
            score=score,
            queued_at=now,
        )
        for neighbor_path, score in neighbors
    ]

    # Append to queue with file locking
    with open(queue_path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            for item in items:
                f.write(json.dumps(item.to_dict()) + "\n")
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    log.info("Queued %d items for evolution: %s", len(items), new_entry_path)
    return len(items)


def read_queue(kb_root: Path | None = None) -> list[QueueItem]:
    """Read all items from the evolution queue.

    Args:
        kb_root: KB root directory. If None, auto-discovers.

    Returns:
        List of QueueItems, ordered by queue time (oldest first).
    """
    queue_path = get_queue_path(kb_root)

    if not queue_path.exists():
        return []

    items = []
    with open(queue_path, "r", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    items.append(QueueItem.from_dict(data))
                except (json.JSONDecodeError, KeyError) as e:
                    log.warning("Skipping malformed queue line: %s", e)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    # Sort by queue time (oldest first)
    items.sort(key=lambda x: x.queued_at)
    return items


def remove_from_queue(
    items_to_remove: list[QueueItem],
    kb_root: Path | None = None,
) -> int:
    """Remove processed items from the queue.

    Uses file locking to safely read-modify-write the queue file.

    Args:
        items_to_remove: Items that have been processed.
        kb_root: KB root directory. If None, auto-discovers.

    Returns:
        Number of items removed.
    """
    if not items_to_remove:
        return 0

    queue_path = get_queue_path(kb_root)

    if not queue_path.exists():
        return 0

    # Build a set of items to remove (by new_entry + neighbor)
    remove_keys = {
        (item.new_entry, item.neighbor)
        for item in items_to_remove
    }

    # Read, filter, and rewrite with exclusive lock
    with open(queue_path, "r+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            # Read all lines
            lines = f.readlines()

            # Filter out removed items
            remaining = []
            removed_count = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    key = (data.get("new_entry"), data.get("neighbor"))
                    if key in remove_keys:
                        removed_count += 1
                        continue
                    remaining.append(line)
                except json.JSONDecodeError:
                    # Keep malformed lines (don't silently delete data)
                    remaining.append(line)

            # Rewrite the file
            f.seek(0)
            f.truncate()
            for line in remaining:
                f.write(line + "\n")

            return removed_count
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def clear_queue(kb_root: Path | None = None) -> int:
    """Clear all items from the queue.

    Args:
        kb_root: KB root directory. If None, auto-discovers.

    Returns:
        Number of items cleared.
    """
    queue_path = get_queue_path(kb_root)

    if not queue_path.exists():
        return 0

    # Count items before clearing
    items = read_queue(kb_root)
    count = len(items)

    # Truncate with exclusive lock
    with open(queue_path, "w", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.truncate(0)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    return count


@dataclass
class QueueStats:
    """Statistics about the evolution queue."""

    count: int
    """Number of items in the queue."""

    oldest_at: datetime | None
    """Timestamp of oldest item, or None if empty."""

    newest_at: datetime | None
    """Timestamp of newest item, or None if empty."""

    unique_new_entries: int
    """Number of unique new entries in queue."""

    unique_neighbors: int
    """Number of unique neighbors to evolve."""


def queue_stats(kb_root: Path | None = None) -> QueueStats:
    """Get statistics about the evolution queue.

    Args:
        kb_root: KB root directory. If None, auto-discovers.

    Returns:
        QueueStats with count, timestamps, and unique entry counts.
    """
    items = read_queue(kb_root)

    if not items:
        return QueueStats(
            count=0,
            oldest_at=None,
            newest_at=None,
            unique_new_entries=0,
            unique_neighbors=0,
        )

    # Items are sorted oldest-first by read_queue
    oldest = items[0].queued_at
    newest = items[-1].queued_at

    unique_new = {item.new_entry for item in items}
    unique_neighbors = {item.neighbor for item in items}

    return QueueStats(
        count=len(items),
        oldest_at=oldest,
        newest_at=newest,
        unique_new_entries=len(unique_new),
        unique_neighbors=len(unique_neighbors),
    )
