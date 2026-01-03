"""Tests for search history tracking."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from memex.models import SearchHistoryEntry
from memex.search_history import (
    MAX_HISTORY_ENTRIES,
    PRUNE_DAYS,
    clear_history,
    get_by_index,
    get_recent,
    load_history,
    record_search,
    save_history,
)


@pytest.fixture
def index_root(tmp_path) -> Path:
    """Create a temporary index root."""
    root = tmp_path / ".indices"
    root.mkdir()
    return root


class TestHistoryPersistence:
    """Test load/save round-trip for search history."""

    def test_load_empty(self, index_root):
        """load_history returns empty list when file doesn't exist."""
        entries = load_history(index_root)
        assert entries == []

    def test_save_and_load_round_trip(self, index_root):
        """History entries can be saved and loaded back."""
        now = datetime.now()
        entries = [
            SearchHistoryEntry(
                query="test query",
                timestamp=now,
                result_count=5,
                mode="hybrid",
                tags=["tag1", "tag2"],
            )
        ]

        save_history(entries, index_root)
        loaded = load_history(index_root)

        assert len(loaded) == 1
        assert loaded[0].query == "test query"
        assert loaded[0].result_count == 5
        assert loaded[0].mode == "hybrid"
        assert loaded[0].tags == ["tag1", "tag2"]

    def test_load_handles_malformed_json(self, index_root):
        """load_history returns empty list for malformed JSON."""
        history_file = index_root / "search_history.json"
        history_file.write_text("not valid json {{{")

        entries = load_history(index_root)
        assert entries == []

    def test_save_prunes_old_entries(self, index_root):
        """save_history removes entries older than PRUNE_DAYS."""
        now = datetime.now()
        old = now - timedelta(days=PRUNE_DAYS + 1)

        entries = [
            SearchHistoryEntry(query="recent", timestamp=now, result_count=1),
            SearchHistoryEntry(query="old", timestamp=old, result_count=1),
        ]

        save_history(entries, index_root)
        loaded = load_history(index_root)

        assert len(loaded) == 1
        assert loaded[0].query == "recent"

    def test_save_limits_entries(self, index_root):
        """save_history keeps at most MAX_HISTORY_ENTRIES."""
        now = datetime.now()
        entries = [
            SearchHistoryEntry(
                query=f"query{i}",
                timestamp=now - timedelta(seconds=i),
                result_count=i,
            )
            for i in range(MAX_HISTORY_ENTRIES + 10)
        ]

        save_history(entries, index_root)
        loaded = load_history(index_root)

        assert len(loaded) == MAX_HISTORY_ENTRIES
        # First entry should be query0 (most recent)
        assert loaded[0].query == "query0"


class TestRecordSearch:
    """Test search recording functionality."""

    def test_record_search_creates_entry(self, index_root):
        """Recording a search creates a new entry."""
        record_search("test query", result_count=10, index_root=index_root)

        entries = load_history(index_root)
        assert len(entries) == 1
        assert entries[0].query == "test query"
        assert entries[0].result_count == 10

    def test_record_search_prepends(self, index_root):
        """New searches are prepended (most recent first)."""
        record_search("first", index_root=index_root)
        record_search("second", index_root=index_root)
        record_search("third", index_root=index_root)

        entries = load_history(index_root)
        assert len(entries) == 3
        assert entries[0].query == "third"
        assert entries[1].query == "second"
        assert entries[2].query == "first"

    def test_record_search_with_mode(self, index_root):
        """Search mode is recorded correctly."""
        record_search("test", mode="semantic", index_root=index_root)

        entries = load_history(index_root)
        assert entries[0].mode == "semantic"

    def test_record_search_with_tags(self, index_root):
        """Tags are recorded correctly."""
        record_search("test", tags=["infra", "docker"], index_root=index_root)

        entries = load_history(index_root)
        assert entries[0].tags == ["infra", "docker"]

    def test_record_search_none_tags(self, index_root):
        """None tags results in empty list."""
        record_search("test", tags=None, index_root=index_root)

        entries = load_history(index_root)
        assert entries[0].tags == []


class TestGetRecent:
    """Test recent history retrieval."""

    def test_get_recent_empty(self, index_root):
        """get_recent returns empty list when no history."""
        result = get_recent(limit=10, index_root=index_root)
        assert result == []

    def test_get_recent_respects_limit(self, index_root):
        """Result is limited to requested count."""
        for i in range(10):
            record_search(f"query{i}", index_root=index_root)

        result = get_recent(limit=3, index_root=index_root)
        assert len(result) == 3

    def test_get_recent_order(self, index_root):
        """Results are returned most recent first."""
        record_search("first", index_root=index_root)
        record_search("second", index_root=index_root)
        record_search("third", index_root=index_root)

        result = get_recent(limit=10, index_root=index_root)
        assert result[0].query == "third"
        assert result[2].query == "first"


class TestGetByIndex:
    """Test retrieval by position index."""

    def test_get_by_index_valid(self, index_root):
        """get_by_index returns correct entry for valid index."""
        record_search("first", index_root=index_root)
        record_search("second", index_root=index_root)
        record_search("third", index_root=index_root)

        # 1 = most recent
        assert get_by_index(1, index_root).query == "third"
        assert get_by_index(2, index_root).query == "second"
        assert get_by_index(3, index_root).query == "first"

    def test_get_by_index_zero_returns_none(self, index_root):
        """get_by_index returns None for index 0."""
        record_search("test", index_root=index_root)
        assert get_by_index(0, index_root) is None

    def test_get_by_index_out_of_range(self, index_root):
        """get_by_index returns None for out of range index."""
        record_search("test", index_root=index_root)
        assert get_by_index(100, index_root) is None

    def test_get_by_index_empty_history(self, index_root):
        """get_by_index returns None when history is empty."""
        assert get_by_index(1, index_root) is None


class TestClearHistory:
    """Test history clearing."""

    def test_clear_history_removes_all(self, index_root):
        """clear_history removes all entries."""
        for i in range(5):
            record_search(f"query{i}", index_root=index_root)

        count = clear_history(index_root)
        assert count == 5

        entries = load_history(index_root)
        assert entries == []

    def test_clear_history_empty(self, index_root):
        """clear_history returns 0 when already empty."""
        count = clear_history(index_root)
        assert count == 0


class TestDataIntegrity:
    """Test data integrity edge cases."""

    def test_malformed_entry_skipped(self, index_root):
        """Malformed entries in file are skipped."""
        import json

        history_file = index_root / "search_history.json"
        history_file.write_text(json.dumps({
            "schema_version": 1,
            "history": [
                {"query": "valid", "timestamp": datetime.now().isoformat(), "result_count": 1},
                {"query": "missing_timestamp"},  # Malformed
                {"query": "another_valid", "timestamp": datetime.now().isoformat()},
            ]
        }))

        entries = load_history(index_root)
        assert len(entries) == 2
        assert entries[0].query == "valid"
        assert entries[1].query == "another_valid"

    def test_schema_version_mismatch_clears(self, index_root):
        """Different schema version results in empty history."""
        import json

        history_file = index_root / "search_history.json"
        history_file.write_text(json.dumps({
            "schema_version": 999,
            "history": [
                {"query": "old_format", "timestamp": datetime.now().isoformat()},
            ]
        }))

        entries = load_history(index_root)
        assert entries == []
