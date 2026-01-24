"""Tests for evolution queue module."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from memex.evolution_queue import (
    QueueItem,
    QueueStats,
    clear_queue,
    get_queue_path,
    queue_evolution,
    queue_stats,
    read_queue,
    remove_from_queue,
)


class TestQueueItem:
    """Tests for QueueItem dataclass."""

    def test_to_dict(self):
        """QueueItem.to_dict() produces valid JSON-serializable dict."""
        now = datetime.now(UTC)
        item = QueueItem(
            new_entry="path/new.md",
            neighbor="path/neighbor.md",
            score=0.85,
            queued_at=now,
        )
        d = item.to_dict()
        assert d["new_entry"] == "path/new.md"
        assert d["neighbor"] == "path/neighbor.md"
        assert d["score"] == 0.85
        assert d["queued_at"] == now.isoformat()

    def test_from_dict(self):
        """QueueItem.from_dict() parses dict correctly."""
        data = {
            "new_entry": "path/new.md",
            "neighbor": "path/neighbor.md",
            "score": 0.75,
            "queued_at": "2024-01-15T10:30:00+00:00",
        }
        item = QueueItem.from_dict(data)
        assert item.new_entry == "path/new.md"
        assert item.neighbor == "path/neighbor.md"
        assert item.score == 0.75
        assert item.queued_at.year == 2024

    def test_from_dict_missing_timestamp(self):
        """QueueItem.from_dict() uses current time if queued_at missing."""
        data = {
            "new_entry": "path/new.md",
            "neighbor": "path/neighbor.md",
            "score": 0.8,
        }
        item = QueueItem.from_dict(data)
        assert item.queued_at is not None
        # Should be recent
        assert datetime.now(UTC) - item.queued_at < timedelta(seconds=5)


class TestQueueOperations:
    """Tests for queue file operations."""

    @pytest.fixture
    def tmp_kb(self, tmp_path, monkeypatch):
        """Create a temporary KB directory."""
        kb_path = tmp_path / "kb"
        kb_path.mkdir()
        (kb_path / ".kbconfig").write_text("kb_path: .")
        indices = kb_path / ".indices"
        indices.mkdir()

        # Patch get_kb_root to return our temp KB
        monkeypatch.setenv("MEMEX_SKIP_PROJECT_KB", "")
        monkeypatch.chdir(kb_path)

        return kb_path

    def test_get_queue_path(self, tmp_kb):
        """get_queue_path returns path in .indices."""
        path = get_queue_path(tmp_kb)
        assert path.parent == tmp_kb / ".indices"
        assert path.name == "evolution_queue.jsonl"

    def test_queue_evolution_empty_list(self, tmp_kb):
        """queue_evolution with empty list returns 0."""
        count = queue_evolution("new.md", [], tmp_kb)
        assert count == 0

    def test_queue_evolution_writes_jsonl(self, tmp_kb):
        """queue_evolution writes items to queue file."""
        neighbors = [
            ("neighbor1.md", 0.85),
            ("neighbor2.md", 0.72),
        ]
        count = queue_evolution("new_entry.md", neighbors, tmp_kb)
        assert count == 2

        # Verify file contents
        queue_path = get_queue_path(tmp_kb)
        assert queue_path.exists()

        lines = queue_path.read_text().strip().split("\n")
        assert len(lines) == 2

        item1 = json.loads(lines[0])
        assert item1["new_entry"] == "new_entry.md"
        assert item1["neighbor"] == "neighbor1.md"
        assert item1["score"] == 0.85

    def test_read_queue_empty(self, tmp_kb):
        """read_queue returns empty list for empty/missing queue."""
        items = read_queue(tmp_kb)
        assert items == []

    def test_read_queue_parses_items(self, tmp_kb):
        """read_queue parses items from queue file."""
        queue_evolution("new1.md", [("n1.md", 0.8)], tmp_kb)
        queue_evolution("new2.md", [("n2.md", 0.7)], tmp_kb)

        items = read_queue(tmp_kb)
        assert len(items) == 2
        assert items[0].new_entry == "new1.md"
        assert items[1].new_entry == "new2.md"

    def test_read_queue_sorted_by_time(self, tmp_kb):
        """read_queue returns items sorted by queued_at (oldest first)."""
        # Write items with specific timestamps
        queue_path = get_queue_path(tmp_kb)
        older = datetime(2024, 1, 1, tzinfo=UTC)
        newer = datetime(2024, 1, 2, tzinfo=UTC)

        with open(queue_path, "w") as f:
            # Write newer first
            f.write(json.dumps({
                "new_entry": "newer.md",
                "neighbor": "n.md",
                "score": 0.8,
                "queued_at": newer.isoformat(),
            }) + "\n")
            f.write(json.dumps({
                "new_entry": "older.md",
                "neighbor": "n.md",
                "score": 0.7,
                "queued_at": older.isoformat(),
            }) + "\n")

        items = read_queue(tmp_kb)
        assert len(items) == 2
        # Older should be first
        assert items[0].new_entry == "older.md"
        assert items[1].new_entry == "newer.md"

    def test_read_queue_skips_malformed(self, tmp_kb):
        """read_queue skips malformed lines."""
        queue_path = get_queue_path(tmp_kb)
        with open(queue_path, "w") as f:
            f.write("not valid json\n")
            f.write(json.dumps({
                "new_entry": "valid.md",
                "neighbor": "n.md",
                "score": 0.8,
                "queued_at": datetime.now(UTC).isoformat(),
            }) + "\n")

        items = read_queue(tmp_kb)
        assert len(items) == 1
        assert items[0].new_entry == "valid.md"

    def test_remove_from_queue(self, tmp_kb):
        """remove_from_queue removes specified items."""
        queue_evolution("new1.md", [("n1.md", 0.8), ("n2.md", 0.7)], tmp_kb)
        queue_evolution("new2.md", [("n3.md", 0.6)], tmp_kb)

        items = read_queue(tmp_kb)
        assert len(items) == 3

        # Remove first two items
        to_remove = [items[0], items[1]]
        removed = remove_from_queue(to_remove, tmp_kb)
        assert removed == 2

        # Check remaining
        remaining = read_queue(tmp_kb)
        assert len(remaining) == 1
        assert remaining[0].neighbor == "n3.md"

    def test_remove_from_queue_empty_list(self, tmp_kb):
        """remove_from_queue with empty list returns 0."""
        queue_evolution("new.md", [("n.md", 0.8)], tmp_kb)
        removed = remove_from_queue([], tmp_kb)
        assert removed == 0

        # Queue should be unchanged
        items = read_queue(tmp_kb)
        assert len(items) == 1

    def test_clear_queue(self, tmp_kb):
        """clear_queue removes all items."""
        queue_evolution("new1.md", [("n1.md", 0.8)], tmp_kb)
        queue_evolution("new2.md", [("n2.md", 0.7)], tmp_kb)

        count = clear_queue(tmp_kb)
        assert count == 2

        items = read_queue(tmp_kb)
        assert len(items) == 0

    def test_clear_queue_empty(self, tmp_kb):
        """clear_queue on empty queue returns 0."""
        count = clear_queue(tmp_kb)
        assert count == 0


class TestQueueStats:
    """Tests for queue_stats function."""

    @pytest.fixture
    def tmp_kb(self, tmp_path, monkeypatch):
        """Create a temporary KB directory."""
        kb_path = tmp_path / "kb"
        kb_path.mkdir()
        (kb_path / ".kbconfig").write_text("kb_path: .")
        indices = kb_path / ".indices"
        indices.mkdir()

        monkeypatch.setenv("MEMEX_SKIP_PROJECT_KB", "")
        monkeypatch.chdir(kb_path)

        return kb_path

    def test_stats_empty_queue(self, tmp_kb):
        """queue_stats returns zeros for empty queue."""
        stats = queue_stats(tmp_kb)
        assert stats.count == 0
        assert stats.oldest_at is None
        assert stats.newest_at is None
        assert stats.unique_new_entries == 0
        assert stats.unique_neighbors == 0

    def test_stats_with_items(self, tmp_kb):
        """queue_stats computes correct statistics."""
        # Add items from 2 sources to 3 neighbors
        queue_evolution("new1.md", [("n1.md", 0.8), ("n2.md", 0.7)], tmp_kb)
        queue_evolution("new2.md", [("n2.md", 0.6), ("n3.md", 0.5)], tmp_kb)

        stats = queue_stats(tmp_kb)
        assert stats.count == 4
        assert stats.unique_new_entries == 2
        assert stats.unique_neighbors == 3  # n1, n2, n3
        assert stats.oldest_at is not None
        assert stats.newest_at is not None
