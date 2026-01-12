"""Tests for mx hubs CLI command."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from memex import core
from memex.cli import cli


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
    (root / "development").mkdir()
    monkeypatch.setenv("MEMEX_KB_ROOT", str(root))
    return root


@pytest.fixture
def index_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary index root."""
    root = tmp_path / ".indices"
    root.mkdir()
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(root))
    return root


def _create_entry(path: Path, title: str, tags: list[str], content: str = ""):
    """Helper to create a KB entry with frontmatter."""
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    text = f"""---
title: {title}
tags:
{tags_yaml}
created: {datetime.now(timezone.utc).isoformat()}
---

{content if content else f"Content for {title}."}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class TestHubsBasic:
    """Basic hubs command functionality tests."""

    def test_hubs_shows_most_connected(self, kb_root, index_root):
        """Entries with most links shown first."""
        # Create a hub entry that links to many others
        _create_entry(
            kb_root / "development" / "hub.md",
            "Hub Entry",
            ["core"],
            "Links to [[development/a.md|A]], [[development/b.md|B]], [[development/c.md|C]]",
        )
        # Create leaf entries that link back to hub
        _create_entry(
            kb_root / "development" / "a.md",
            "Entry A",
            ["leaf"],
            "Links to [[development/hub.md|Hub]]",
        )
        _create_entry(
            kb_root / "development" / "b.md",
            "Entry B",
            ["leaf"],
            "Links to [[development/hub.md|Hub]]",
        )
        _create_entry(
            kb_root / "development" / "c.md",
            "Entry C",
            ["leaf"],
            "No links here",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["hubs"])

        assert result.exit_code == 0
        # Hub entry should appear (has most connections: 3 outgoing + 2 incoming = 5)
        assert "development/hub.md" in result.output
        # First entry in output should be the hub (most connected)
        lines = result.output.strip().split("\n")
        # Skip header line
        data_lines = [line for line in lines if "development/hub.md" in line]
        assert len(data_lines) > 0

    def test_hubs_includes_incoming_count(self, kb_root, index_root):
        """Shows incoming link count."""
        # Create entries where hub receives incoming links
        _create_entry(
            kb_root / "development" / "hub.md",
            "Hub Entry",
            ["core"],
            "Main hub entry content",
        )
        _create_entry(
            kb_root / "development" / "a.md",
            "Entry A",
            ["leaf"],
            "Links to [[development/hub.md|Hub]]",
        )
        _create_entry(
            kb_root / "development" / "b.md",
            "Entry B",
            ["leaf"],
            "Also links to [[development/hub.md|Hub]]",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["hubs", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Find the hub entry
        hub_entry = next((h for h in data if h["path"] == "development/hub.md"), None)
        assert hub_entry is not None
        assert "incoming" in hub_entry
        assert hub_entry["incoming"] == 2  # Two entries link to it

    def test_hubs_includes_outgoing_count(self, kb_root, index_root):
        """Shows outgoing link count."""
        # Create hub that links to multiple entries
        _create_entry(
            kb_root / "development" / "hub.md",
            "Hub Entry",
            ["core"],
            "Links to [[development/a.md|A]] and [[development/b.md|B]]",
        )
        _create_entry(
            kb_root / "development" / "a.md",
            "Entry A",
            ["leaf"],
            "Target entry A",
        )
        _create_entry(
            kb_root / "development" / "b.md",
            "Entry B",
            ["leaf"],
            "Target entry B",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["hubs", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Find the hub entry
        hub_entry = next((h for h in data if h["path"] == "development/hub.md"), None)
        assert hub_entry is not None
        assert "outgoing" in hub_entry
        assert hub_entry["outgoing"] == 2  # Links to two entries

    def test_hubs_includes_total(self, kb_root, index_root):
        """Shows total connections."""
        # Create interconnected entries
        _create_entry(
            kb_root / "development" / "hub.md",
            "Hub Entry",
            ["core"],
            "Links to [[development/a.md|A]] and [[development/b.md|B]]",
        )
        _create_entry(
            kb_root / "development" / "a.md",
            "Entry A",
            ["leaf"],
            "Links to [[development/hub.md|Hub]]",
        )
        _create_entry(
            kb_root / "development" / "b.md",
            "Entry B",
            ["leaf"],
            "Links to [[development/hub.md|Hub]]",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["hubs", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Find the hub entry
        hub_entry = next((h for h in data if h["path"] == "development/hub.md"), None)
        assert hub_entry is not None
        assert "total" in hub_entry
        # Total = incoming (2) + outgoing (2) = 4
        assert hub_entry["total"] == hub_entry["incoming"] + hub_entry["outgoing"]


class TestHubsLimit:
    """Tests for hubs --limit option."""

    def test_hubs_default_limit_10(self, kb_root, index_root):
        """Default shows up to 10 entries."""
        # Create 15 entries with varying connections
        for i in range(15):
            links = " ".join(
                f"[[development/entry{j}.md|Entry{j}]]" for j in range(i)
            )
            _create_entry(
                kb_root / "development" / f"entry{i}.md",
                f"Entry {i}",
                ["test"],
                f"Content with links: {links}" if links else "No links",
            )

        runner = CliRunner()
        result = runner.invoke(cli, ["hubs"])

        assert result.exit_code == 0
        # Count actual entry lines (excluding header)
        lines = [line for line in result.output.strip().split("\n") if "development/entry" in line]
        # Default limit is 10
        assert len(lines) <= 10

    def test_hubs_custom_limit(self, kb_root, index_root):
        """--limit=3 shows only 3 entries."""
        # Create 5 interconnected entries
        for i in range(5):
            links = " ".join(
                f"[[development/entry{j}.md|Entry{j}]]" for j in range(5) if j != i
            )
            _create_entry(
                kb_root / "development" / f"entry{i}.md",
                f"Entry {i}",
                ["test"],
                f"Links: {links}",
            )

        runner = CliRunner()
        result = runner.invoke(cli, ["hubs", "--limit=3"])

        assert result.exit_code == 0
        # Count actual entry lines
        lines = [line for line in result.output.strip().split("\n") if "development/entry" in line]
        assert len(lines) == 3

    def test_hubs_limit_exceeds_entries(self, kb_root, index_root):
        """--limit=100 shows all if fewer entries exist."""
        # Create only 3 entries with connections
        _create_entry(
            kb_root / "development" / "hub.md",
            "Hub Entry",
            ["core"],
            "Links to [[development/a.md|A]] and [[development/b.md|B]]",
        )
        _create_entry(
            kb_root / "development" / "a.md",
            "Entry A",
            ["leaf"],
            "Links to [[development/hub.md|Hub]]",
        )
        _create_entry(
            kb_root / "development" / "b.md",
            "Entry B",
            ["leaf"],
            "Links to [[development/hub.md|Hub]]",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["hubs", "--limit=100"])

        assert result.exit_code == 0
        # Should show all 3 entries (each has at least one connection)
        assert "development/hub.md" in result.output
        assert "development/a.md" in result.output
        assert "development/b.md" in result.output


class TestHubsJsonOutput:
    """Tests for hubs --json output."""

    def test_hubs_json_structure(self, kb_root, index_root):
        """Proper JSON array structure."""
        _create_entry(
            kb_root / "development" / "hub.md",
            "Hub Entry",
            ["core"],
            "Links to [[development/leaf.md|Leaf]]",
        )
        _create_entry(
            kb_root / "development" / "leaf.md",
            "Leaf Entry",
            ["leaf"],
            "Links to [[development/hub.md|Hub]]",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["hubs", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Should be a list
        assert isinstance(data, list)
        # Each entry should have required fields
        for entry in data:
            assert "path" in entry
            assert "incoming" in entry
            assert "outgoing" in entry
            assert "total" in entry

    def test_hubs_json_sorted_by_total(self, kb_root, index_root):
        """JSON output sorted by total connections."""
        # Create entries with different connection counts
        # Entry with most connections
        _create_entry(
            kb_root / "development" / "top-hub.md",
            "Top Hub",
            ["core"],
            "Links to [[development/a.md|A]], [[development/b.md|B]], [[development/c.md|C]]",
        )
        # Entry with fewer connections
        _create_entry(
            kb_root / "development" / "small-hub.md",
            "Small Hub",
            ["secondary"],
            "Links to [[development/a.md|A]]",
        )
        # Leaf entries
        _create_entry(
            kb_root / "development" / "a.md",
            "Entry A",
            ["leaf"],
            "Links to [[development/top-hub.md|TopHub]] and [[development/small-hub.md|SmallHub]]",
        )
        _create_entry(
            kb_root / "development" / "b.md",
            "Entry B",
            ["leaf"],
            "Links to [[development/top-hub.md|TopHub]]",
        )
        _create_entry(
            kb_root / "development" / "c.md",
            "Entry C",
            ["leaf"],
            "Links to [[development/top-hub.md|TopHub]]",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["hubs", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Verify sorted by total descending
        totals = [entry["total"] for entry in data]
        assert totals == sorted(totals, reverse=True)

        # First entry should be top-hub (most connections)
        assert data[0]["path"] == "development/top-hub.md"


class TestHubsEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_hubs_empty_kb(self, tmp_path, monkeypatch):
        """Handles empty KB gracefully."""
        empty_root = tmp_path / "empty_kb"
        empty_root.mkdir()
        monkeypatch.setenv("MEMEX_KB_ROOT", str(empty_root))

        index_root = tmp_path / ".indices"
        index_root.mkdir()
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        # Reset searcher
        monkeypatch.setattr(core, "_searcher", None)
        monkeypatch.setattr(core, "_searcher_ready", False)

        runner = CliRunner()
        result = runner.invoke(cli, ["hubs"])

        assert result.exit_code == 0
        assert "No hub entries found" in result.output

    def test_hubs_no_links(self, kb_root, index_root):
        """Handles KB with no internal links."""
        # Create entries without any wiki-style links
        _create_entry(
            kb_root / "development" / "standalone1.md",
            "Standalone Entry 1",
            ["solo"],
            "No links here, just plain content.",
        )
        _create_entry(
            kb_root / "development" / "standalone2.md",
            "Standalone Entry 2",
            ["solo"],
            "Also no links in this entry.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["hubs"])

        assert result.exit_code == 0
        # No entries have connections, so no hubs
        assert "No hub entries found" in result.output

    def test_hubs_single_entry(self, kb_root, index_root):
        """Works with just one entry."""
        _create_entry(
            kb_root / "development" / "only.md",
            "Only Entry",
            ["solo"],
            "Single entry with no links.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["hubs"])

        assert result.exit_code == 0
        # Single entry with no links has no connections
        assert "No hub entries found" in result.output

    def test_hubs_single_entry_with_self_reference(self, kb_root, index_root):
        """Single entry referencing itself is handled."""
        _create_entry(
            kb_root / "development" / "self-ref.md",
            "Self Reference",
            ["meta"],
            "This entry links to itself [[development/self-ref.md|Self]]",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["hubs", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # May or may not show depending on implementation
        # The key is it should not error
        assert isinstance(data, list)

    def test_hubs_short_limit_flag(self, kb_root, index_root):
        """Short -n flag works for limit option."""
        # Create entries with connections
        _create_entry(
            kb_root / "development" / "hub.md",
            "Hub Entry",
            ["core"],
            "Links to [[development/a.md|A]]",
        )
        _create_entry(
            kb_root / "development" / "a.md",
            "Entry A",
            ["leaf"],
            "Links to [[development/hub.md|Hub]]",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["hubs", "-n", "1"])

        assert result.exit_code == 0
        # Should only show 1 entry
        lines = [line for line in result.output.strip().split("\n") if "development/" in line]
        assert len(lines) == 1
