"""Tests for mx history CLI command."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from memex import core, search_history
from memex.cli import cli
from memex.models import SearchHistoryEntry


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
    monkeypatch.setenv("MEMEX_USER_KB_ROOT", str(root))
    return root


@pytest.fixture
def index_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary index root."""
    root = tmp_path / ".indices"
    root.mkdir()
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(root))
    return root


def _create_test_entry(kb_root: Path, title: str, tags: str, category: str = "development") -> Path:
    """Helper to create a test KB entry."""
    slug = title.lower().replace(" ", "-")
    path = kb_root / category / f"{slug}.md"
    tag_list = tags.split(",")
    tags_yaml = "\n  - ".join(tag_list)
    content = f"""---
title: {title}
tags:
  - {tags_yaml}
created: 2024-01-01T00:00:00
---

# {title}

Content for {title}.
"""
    path.write_text(content)
    return path


class TestHistoryBasicFunctionality:
    """Test basic mx history functionality."""

    def test_history_shows_recent_searches(self, kb_root, index_root):
        """Displays recent searches when history exists."""
        # Record some searches directly
        search_history.record_search(
            query="python testing",
            result_count=5,
            mode="hybrid",
            tags=None,
            index_root=index_root,
        )
        search_history.record_search(
            query="docker deployment",
            result_count=3,
            mode="keyword",
            tags=["devops"],
            index_root=index_root,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["history"])

        assert result.exit_code == 0
        assert "Recent searches:" in result.output
        assert "python testing" in result.output
        assert "docker deployment" in result.output
        assert "hybrid" in result.output
        assert "keyword" in result.output
        assert "5 results" in result.output
        assert "3 results" in result.output

    def test_history_shows_empty_message_when_no_history(self, kb_root, index_root):
        """Shows informative message when no search history exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["history"])

        assert result.exit_code == 0
        assert "No search history" in result.output

    def test_history_limit_restricts_entries(self, kb_root, index_root):
        """--limit restricts number of entries shown."""
        # Record more searches than the limit
        for i in range(5):
            search_history.record_search(
                query=f"query {i}",
                result_count=i,
                mode="hybrid",
                index_root=index_root,
            )

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "--limit", "2"])

        assert result.exit_code == 0
        # Should only show 2 most recent
        assert "query 4" in result.output
        assert "query 3" in result.output
        assert "query 2" not in result.output
        assert "query 1" not in result.output
        assert "query 0" not in result.output

    def test_history_limit_short_flag(self, kb_root, index_root):
        """-n short flag works for limit."""
        for i in range(5):
            search_history.record_search(
                query=f"search {i}",
                result_count=i,
                mode="hybrid",
                index_root=index_root,
            )

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "-n", "3"])

        assert result.exit_code == 0
        assert "search 4" in result.output
        assert "search 3" in result.output
        assert "search 2" in result.output
        assert "search 1" not in result.output

    def test_history_clear_removes_all_entries(self, kb_root, index_root):
        """--clear removes all search history."""
        # Record some searches
        for i in range(3):
            search_history.record_search(
                query=f"to be cleared {i}",
                result_count=i,
                mode="hybrid",
                index_root=index_root,
            )

        runner = CliRunner()

        # Verify history exists
        result = runner.invoke(cli, ["history"])
        assert "to be cleared" in result.output

        # Clear history
        result = runner.invoke(cli, ["history", "--clear"])
        assert result.exit_code == 0
        assert "Cleared 3 search history entries" in result.output

        # Verify history is empty
        result = runner.invoke(cli, ["history"])
        assert "No search history" in result.output


class TestHistoryRerunFunctionality:
    """Test mx history --rerun functionality."""

    def test_rerun_executes_search_at_position(self, kb_root, index_root):
        """--rerun N re-executes the Nth search."""
        # Create a test entry to find
        _create_test_entry(kb_root, "Python Guide", "python,guide")

        # Record a search
        search_history.record_search(
            query="python",
            result_count=1,
            mode="hybrid",
            index_root=index_root,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "--rerun", "1"])

        assert result.exit_code == 0
        assert "Re-running: python" in result.output
        assert "Mode: hybrid" in result.output

    def test_rerun_short_flag(self, kb_root, index_root):
        """-r short flag works for rerun."""
        _create_test_entry(kb_root, "Docker Setup", "docker,devops")

        search_history.record_search(
            query="docker",
            result_count=1,
            mode="keyword",
            index_root=index_root,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "-r", "1"])

        assert result.exit_code == 0
        assert "Re-running: docker" in result.output
        assert "Mode: keyword" in result.output

    def test_rerun_with_invalid_index_shows_error(self, kb_root, index_root):
        """--rerun with invalid index shows error."""
        # Record only one search
        search_history.record_search(
            query="only one",
            result_count=1,
            mode="hybrid",
            index_root=index_root,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "--rerun", "5"])

        assert result.exit_code == 1
        assert "Error: No search at position 5" in result.output

    def test_rerun_with_zero_index_shows_error(self, kb_root, index_root):
        """--rerun 0 shows error (1-based indexing)."""
        search_history.record_search(
            query="test query",
            result_count=1,
            mode="hybrid",
            index_root=index_root,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "--rerun", "0"])

        assert result.exit_code == 1
        assert "Error: No search at position 0" in result.output

    def test_rerun_with_empty_history_shows_error(self, kb_root, index_root):
        """--rerun with empty history shows error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["history", "--rerun", "1"])

        assert result.exit_code == 1
        assert "Error: No search at position 1" in result.output

    def test_rerun_preserves_tags_filter(self, kb_root, index_root):
        """--rerun preserves original tag filters."""
        _create_test_entry(kb_root, "DevOps Guide", "devops,infrastructure")

        search_history.record_search(
            query="guide",
            result_count=1,
            mode="hybrid",
            tags=["devops"],
            index_root=index_root,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "--rerun", "1"])

        assert result.exit_code == 0
        assert "Re-running: guide" in result.output
        assert "Tags: devops" in result.output


class TestHistoryStorage:
    """Test history storage and persistence."""

    def test_searches_are_recorded_to_history(self, kb_root, index_root):
        """Searches are properly recorded to history."""
        # Record a search
        search_history.record_search(
            query="recorded query",
            result_count=7,
            mode="semantic",
            tags=["test", "storage"],
            index_root=index_root,
        )

        # Verify it's in history
        entries = search_history.get_recent(limit=1, index_root=index_root)
        assert len(entries) == 1
        assert entries[0].query == "recorded query"
        assert entries[0].result_count == 7
        assert entries[0].mode == "semantic"
        assert entries[0].tags == ["test", "storage"]

    def test_history_persists_across_invocations(self, kb_root, index_root):
        """History persists across separate CLI invocations."""
        runner = CliRunner()

        # First invocation - record a search
        search_history.record_search(
            query="persistent query",
            result_count=2,
            mode="hybrid",
            index_root=index_root,
        )

        # Second invocation - check history
        result = runner.invoke(cli, ["history"])

        assert result.exit_code == 0
        assert "persistent query" in result.output

    def test_history_file_location(self, kb_root, index_root):
        """History is stored in the correct location."""
        search_history.record_search(
            query="location test",
            result_count=1,
            mode="hybrid",
            index_root=index_root,
        )

        history_file = index_root / "search_history.json"
        assert history_file.exists()

        # Verify content
        data = json.loads(history_file.read_text())
        assert "history" in data
        assert len(data["history"]) == 1
        assert data["history"][0]["query"] == "location test"


class TestHistoryOutputFormat:
    """Test history output formatting."""

    def test_output_shows_query_timestamp_result_count(self, kb_root, index_root):
        """Output shows query, timestamp, and result count."""
        search_history.record_search(
            query="format test",
            result_count=42,
            mode="hybrid",
            index_root=index_root,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["history"])

        assert result.exit_code == 0
        assert "format test" in result.output
        assert "42 results" in result.output
        # Timestamp should be in format like "2024-01-15 14:30"
        assert "hybrid" in result.output

    def test_json_output_structure(self, kb_root, index_root):
        """--json returns proper JSON structure."""
        search_history.record_search(
            query="json test",
            result_count=10,
            mode="semantic",
            tags=["api", "docs"],
            index_root=index_root,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "--json"])

        assert result.exit_code == 0

        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 1

        entry = data[0]
        assert entry["position"] == 1
        assert entry["query"] == "json test"
        assert entry["result_count"] == 10
        assert entry["mode"] == "semantic"
        assert entry["tags"] == ["api", "docs"]
        assert "timestamp" in entry

    def test_json_output_empty_history(self, kb_root, index_root):
        """--json with empty history returns empty array."""
        runner = CliRunner()
        result = runner.invoke(cli, ["history", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == []

    def test_output_shows_tags_when_present(self, kb_root, index_root):
        """Output shows tags when they were used in search."""
        search_history.record_search(
            query="tagged search",
            result_count=5,
            mode="hybrid",
            tags=["python", "testing"],
            index_root=index_root,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["history"])

        assert result.exit_code == 0
        assert "tags: python, testing" in result.output

    def test_output_shows_no_results_text(self, kb_root, index_root):
        """Output shows 'no results' when result_count is 0."""
        search_history.record_search(
            query="empty search",
            result_count=0,
            mode="hybrid",
            index_root=index_root,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["history"])

        assert result.exit_code == 0
        assert "no results" in result.output

    def test_output_shows_tip_for_rerun(self, kb_root, index_root):
        """Output shows tip about --rerun usage."""
        search_history.record_search(
            query="tip test",
            result_count=1,
            mode="hybrid",
            index_root=index_root,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["history"])

        assert result.exit_code == 0
        assert "mx history --rerun" in result.output


class TestHistoryEdgeCases:
    """Test edge cases for history command."""

    def test_history_with_special_characters_in_query(self, kb_root, index_root):
        """Handles special characters in queries correctly."""
        special_queries = [
            "query with 'quotes'",
            'query with "double quotes"',
            "query with <angle> brackets",
            "query with & ampersand",
            "query with unicode: cafe",
        ]

        for query in special_queries:
            search_history.record_search(
                query=query,
                result_count=1,
                mode="hybrid",
                index_root=index_root,
            )

        runner = CliRunner()
        result = runner.invoke(cli, ["history"])

        assert result.exit_code == 0
        for query in special_queries:
            assert query in result.output

    def test_history_max_entries_limit(self, kb_root, index_root):
        """History respects max entries limit (100)."""
        # Record more than max entries
        for i in range(110):
            search_history.record_search(
                query=f"overflow query {i}",
                result_count=i,
                mode="hybrid",
                index_root=index_root,
            )

        # Load history and verify limit
        entries = search_history.load_history(index_root)
        assert len(entries) <= 100

        # Most recent should be preserved
        assert entries[0].query == "overflow query 109"

    def test_history_with_very_long_query(self, kb_root, index_root):
        """Handles very long queries correctly."""
        long_query = "a" * 500

        search_history.record_search(
            query=long_query,
            result_count=1,
            mode="hybrid",
            index_root=index_root,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["history"])

        assert result.exit_code == 0
        # Should contain at least part of the long query
        assert "a" * 50 in result.output

    def test_history_ordering_most_recent_first(self, kb_root, index_root):
        """History entries are ordered most recent first."""
        search_history.record_search(
            query="first query",
            result_count=1,
            mode="hybrid",
            index_root=index_root,
        )
        search_history.record_search(
            query="second query",
            result_count=2,
            mode="hybrid",
            index_root=index_root,
        )
        search_history.record_search(
            query="third query",
            result_count=3,
            mode="hybrid",
            index_root=index_root,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "--json"])

        data = json.loads(result.output)
        assert data[0]["query"] == "third query"
        assert data[1]["query"] == "second query"
        assert data[2]["query"] == "first query"


class TestHistoryErrorCases:
    """Test error handling for history command."""

    def test_error_when_no_kb_root_configured(self, tmp_path, monkeypatch):
        """Error when MEMEX_USER_KB_ROOT is not configured."""
        # Unset the environment variable
        monkeypatch.delenv("MEMEX_USER_KB_ROOT", raising=False)
        monkeypatch.delenv("MEMEX_INDEX_ROOT", raising=False)

        runner = CliRunner()
        result = runner.invoke(cli, ["history"])

        # Should fail due to missing configuration
        assert result.exit_code != 0

    def test_error_when_no_index_root_configured(self, kb_root, monkeypatch):
        """Error when MEMEX_INDEX_ROOT is not configured."""
        monkeypatch.delenv("MEMEX_INDEX_ROOT", raising=False)

        runner = CliRunner()
        result = runner.invoke(cli, ["history"])

        # Should fail due to missing index root
        assert result.exit_code != 0

    def test_handles_corrupted_history_file(self, kb_root, index_root):
        """Gracefully handles corrupted history file."""
        # Write invalid JSON to history file
        history_file = index_root / "search_history.json"
        history_file.write_text("{ invalid json }")

        runner = CliRunner()
        result = runner.invoke(cli, ["history"])

        # Should not crash, treat as empty history
        assert result.exit_code == 0
        assert "No search history" in result.output

    def test_handles_malformed_history_entries(self, kb_root, index_root):
        """Gracefully handles malformed entries in history file."""
        history_file = index_root / "search_history.json"
        history_file.write_text(json.dumps({
            "schema_version": 1,
            "history": [
                {"query": "valid", "timestamp": datetime.now().isoformat(), "result_count": 1},
                {"missing_required_fields": True},  # Malformed entry
                {"query": "also valid", "timestamp": datetime.now().isoformat()},
            ]
        }))

        runner = CliRunner()
        result = runner.invoke(cli, ["history"])

        # Should skip malformed entries and show valid ones
        assert result.exit_code == 0
        assert "valid" in result.output


class TestHistoryRerunWithJson:
    """Test --rerun combined with --json output."""

    def test_rerun_with_json_output(self, kb_root, index_root):
        """--rerun combined with --json returns JSON results."""
        _create_test_entry(kb_root, "API Documentation", "api,docs")

        search_history.record_search(
            query="api",
            result_count=1,
            mode="hybrid",
            index_root=index_root,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "--rerun", "1", "--json"])

        assert result.exit_code == 0
        # Output contains "Re-running:" header followed by JSON
        assert "Re-running: api" in result.output

        # Extract the JSON portion (everything after the header lines)
        output = result.output
        json_start = output.find("[")
        if json_start == -1:
            json_start = output.find("{")
        assert json_start != -1, "No JSON found in output"

        json_str = output[json_start:]
        data = json.loads(json_str)
        assert isinstance(data, list)
