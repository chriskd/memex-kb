"""Tests for mx health CLI command."""

import json
from datetime import datetime, timedelta, timezone
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
    """Create a temporary KB root with standard structure."""
    root = tmp_path / "kb"
    root.mkdir()
    (root / "development").mkdir()
    (root / "architecture").mkdir()
    (root / "empty_category").mkdir()  # For empty dir test
    monkeypatch.setenv("MEMEX_KB_ROOT", str(root))
    return root


@pytest.fixture
def index_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary index root."""
    root = tmp_path / ".indices"
    root.mkdir()
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(root))
    return root


def _create_entry(
    path: Path,
    title: str,
    tags: list[str],
    content: str = "",
    created: datetime | None = None,
    updated: datetime | None = None,
):
    """Helper to create a KB entry with frontmatter."""
    created = created or datetime.now(timezone.utc)
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    updated_line = f"\nupdated: {updated.isoformat()}" if updated else ""
    text = f"""---
title: {title}
tags:
{tags_yaml}
created: {created.isoformat()}{updated_line}
---

{content if content else f"Content for {title}."}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class TestHealthBasic:
    """Basic health command functionality tests."""

    def test_health_shows_score(self, kb_root, index_root):
        """Health command displays a health score (0-100)."""
        _create_entry(
            kb_root / "development" / "entry.md",
            "Test Entry",
            ["test"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["health"])

        assert result.exit_code == 0
        assert "Health Score:" in result.output
        assert "/100" in result.output

    def test_health_shows_orphans(self, kb_root, index_root):
        """Health command identifies entries with no links in/out."""
        # Create an orphan entry (no links to it, no links from it)
        _create_entry(
            kb_root / "development" / "orphan.md",
            "Orphan Entry",
            ["test"],
            "Standalone content with no links.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["health"])

        assert result.exit_code == 0
        # Should show orphan warning
        assert "Orphaned entries" in result.output or "orphaned" in result.output.lower()
        assert "orphan.md" in result.output

    def test_health_shows_broken_links(self, kb_root, index_root):
        """Health command finds links to non-existent entries."""
        # Create entry with broken link
        _create_entry(
            kb_root / "development" / "has-broken-link.md",
            "Has Broken Link",
            ["test"],
            "Links to [[nonexistent.md|Missing Entry]]",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["health"])

        assert result.exit_code == 0
        assert "Broken links" in result.output or "broken" in result.output.lower()
        assert "nonexistent" in result.output.lower()

    def test_health_shows_stale_entries(self, kb_root, index_root):
        """Health command finds very old entries."""
        # Create a stale entry (older than 90 days)
        old_date = datetime.now(timezone.utc) - timedelta(days=365)
        _create_entry(
            kb_root / "development" / "stale-entry.md",
            "Stale Entry",
            ["test"],
            "Old content.",
            created=old_date,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["health"])

        assert result.exit_code == 0
        assert "Stale entries" in result.output or "stale" in result.output.lower()
        assert "stale-entry.md" in result.output

    def test_health_shows_empty_dirs(self, kb_root, index_root):
        """Health command finds empty directories."""
        # empty_category is already created in the fixture with no files

        runner = CliRunner()
        result = runner.invoke(cli, ["health"])

        assert result.exit_code == 0
        assert "Empty directories" in result.output or "empty" in result.output.lower()
        assert "empty_category" in result.output


class TestHealthScore:
    """Tests for health score calculation."""

    def test_health_score_100_when_clean(self, kb_root, index_root):
        """Perfect KB gets score of 100."""
        # Create two entries that link to each other (no orphans)
        # Note: Link path must match actual file location
        _create_entry(
            kb_root / "development" / "entry-a.md",
            "Entry A",
            ["test"],
            "Links to [[architecture/entry-b|Entry B]]",
        )
        _create_entry(
            kb_root / "architecture" / "entry-b.md",
            "Entry B",
            ["test"],
            "Links to [[development/entry-a|Entry A]]",
        )
        # Remove the empty category directory
        (kb_root / "empty_category").rmdir()

        runner = CliRunner()
        result = runner.invoke(cli, ["health", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["summary"]["health_score"] == 100

    def test_health_score_decreases_with_issues(self, kb_root, index_root):
        """Issues lower the health score."""
        # Create multiple issues: orphan + broken link + stale + empty dir
        old_date = datetime.now(timezone.utc) - timedelta(days=365)
        _create_entry(
            kb_root / "development" / "orphan.md",
            "Orphan Entry",
            ["test"],
            "Standalone.",
            created=old_date,
        )
        _create_entry(
            kb_root / "development" / "broken.md",
            "Broken Links",
            ["test"],
            "Links to [[missing1|M1]] and [[missing2|M2]]",
            created=old_date,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["health", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        # Score should be lower than 100 due to issues
        assert data["summary"]["health_score"] < 100

    def test_health_score_min_zero(self, kb_root, index_root):
        """Score never goes below 0."""
        # Create many issues to try to push score negative
        old_date = datetime.now(timezone.utc) - timedelta(days=365)
        for i in range(20):
            _create_entry(
                kb_root / "development" / f"orphan-{i}.md",
                f"Orphan {i}",
                ["test"],
                f"Links to [[missing-{i}|Missing {i}]]",
                created=old_date,
            )
            # Create more empty directories
            (kb_root / f"empty-{i}").mkdir()

        runner = CliRunner()
        result = runner.invoke(cli, ["health", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["summary"]["health_score"] >= 0


class TestHealthJsonOutput:
    """Tests for --json output format."""

    def test_health_json_structure(self, kb_root, index_root):
        """JSON output has proper structure with score, issues."""
        _create_entry(
            kb_root / "development" / "entry.md",
            "Test Entry",
            ["test"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["health", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Check top-level structure
        assert "orphans" in data
        assert "broken_links" in data
        assert "stale" in data
        assert "empty_dirs" in data
        assert "summary" in data

        # Check summary structure
        assert "health_score" in data["summary"]
        assert "total_issues" in data["summary"]

    def test_health_json_includes_summary(self, kb_root, index_root):
        """JSON output has summary with counts for all issue types."""
        _create_entry(
            kb_root / "development" / "entry.md",
            "Test Entry",
            ["test"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["health", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        summary = data["summary"]
        assert "health_score" in summary
        assert "total_issues" in summary
        assert "orphans_count" in summary
        assert "broken_links_count" in summary
        assert "stale_count" in summary
        assert "empty_dirs_count" in summary
        assert "total_entries" in summary

    def test_health_json_includes_all_issues(self, kb_root, index_root):
        """JSON output lists each issue with details."""
        # Create entries with various issues
        old_date = datetime.now(timezone.utc) - timedelta(days=365)
        _create_entry(
            kb_root / "development" / "orphan.md",
            "Orphan Entry",
            ["test"],
            "No links.",
        )
        _create_entry(
            kb_root / "development" / "broken.md",
            "Broken Link Entry",
            ["test"],
            "Links to [[nonexistent|Missing]]",
        )
        _create_entry(
            kb_root / "development" / "stale.md",
            "Stale Entry",
            ["test"],
            "Old content.",
            created=old_date,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["health", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Verify orphans list has path and title
        assert len(data["orphans"]) > 0
        orphan = data["orphans"][0]
        assert "path" in orphan
        assert "title" in orphan

        # Verify broken_links list has source and broken_link
        assert len(data["broken_links"]) > 0
        broken = data["broken_links"][0]
        assert "source" in broken
        assert "broken_link" in broken

        # Verify stale list has path, title, and days info
        assert len(data["stale"]) > 0
        stale = data["stale"][0]
        assert "path" in stale
        assert "title" in stale
        assert "days_old" in stale

        # Verify empty_dirs is a list of strings
        assert isinstance(data["empty_dirs"], list)


class TestHealthEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_health_empty_kb(self, tmp_path, monkeypatch):
        """Health command handles empty KB gracefully."""
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
        result = runner.invoke(cli, ["health"])

        assert result.exit_code == 0
        assert "Health Score:" in result.output

    def test_health_many_broken_links(self, kb_root, index_root):
        """Health command handles many broken links correctly."""
        # Create entry with many broken links
        links = " ".join([f"[[missing-{i}|M{i}]]" for i in range(50)])
        _create_entry(
            kb_root / "development" / "many-broken.md",
            "Many Broken Links",
            ["test"],
            f"Content with {links}",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["health", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        # Should have captured all broken links
        assert data["summary"]["broken_links_count"] >= 50
        # Score should be low but not negative
        assert data["summary"]["health_score"] >= 0
        assert data["summary"]["health_score"] < 100

    def test_health_all_connected(self, kb_root, index_root):
        """Health command handles well-linked KB correctly."""
        # Create a well-connected KB with no orphans
        _create_entry(
            kb_root / "development" / "hub.md",
            "Hub Entry",
            ["test"],
            "[[development/spoke-a|A]] [[development/spoke-b|B]] [[development/spoke-c|C]]",
        )
        _create_entry(
            kb_root / "development" / "spoke-a.md",
            "Spoke A",
            ["test"],
            "Links back to [[development/hub|Hub]]",
        )
        _create_entry(
            kb_root / "development" / "spoke-b.md",
            "Spoke B",
            ["test"],
            "Links back to [[development/hub|Hub]]",
        )
        _create_entry(
            kb_root / "development" / "spoke-c.md",
            "Spoke C",
            ["test"],
            "Links back to [[development/hub|Hub]]",
        )
        # Remove empty directories
        (kb_root / "empty_category").rmdir()
        (kb_root / "architecture").rmdir()

        runner = CliRunner()
        result = runner.invoke(cli, ["health", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # No orphans - all entries are linked
        assert data["summary"]["orphans_count"] == 0
        # No broken links
        assert data["summary"]["broken_links_count"] == 0
        # No empty directories
        assert data["summary"]["empty_dirs_count"] == 0
        # High health score
        assert data["summary"]["health_score"] == 100

    def test_health_report_format(self, kb_root, index_root):
        """Health command output includes report header and sections."""
        _create_entry(
            kb_root / "development" / "entry.md",
            "Test Entry",
            ["test"],
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["health"])

        assert result.exit_code == 0
        assert "Knowledge Base Health Report" in result.output
        assert "=" in result.output  # Separator line

    def test_health_clean_kb_shows_checkmarks(self, kb_root, index_root):
        """Clean KB shows checkmarks for passing checks."""
        # Create well-connected entries
        _create_entry(
            kb_root / "development" / "a.md",
            "Entry A",
            ["test"],
            "Links to [[development/b|B]]",
        )
        _create_entry(
            kb_root / "development" / "b.md",
            "Entry B",
            ["test"],
            "Links to [[development/a|A]]",
        )
        # Remove empty directories
        (kb_root / "empty_category").rmdir()
        (kb_root / "architecture").rmdir()

        runner = CliRunner()
        result = runner.invoke(cli, ["health"])

        assert result.exit_code == 0
        # Should show checkmarks for clean sections
        assert "No orphaned entries" in result.output
        assert "No broken links" in result.output

    def test_health_total_entries_count(self, kb_root, index_root):
        """Health JSON includes correct total entries count."""
        _create_entry(kb_root / "development" / "a.md", "Entry A", ["test"])
        _create_entry(kb_root / "development" / "b.md", "Entry B", ["test"])
        _create_entry(kb_root / "architecture" / "c.md", "Entry C", ["test"])

        runner = CliRunner()
        result = runner.invoke(cli, ["health", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["summary"]["total_entries"] == 3

    def test_health_stale_with_updated_date(self, kb_root, index_root):
        """Entry with recent updated date is not considered stale."""
        old_created = datetime.now(timezone.utc) - timedelta(days=365)
        recent_updated = datetime.now(timezone.utc) - timedelta(days=10)

        _create_entry(
            kb_root / "development" / "recently-updated.md",
            "Recently Updated",
            ["test"],
            "Content was updated.",
            created=old_created,
            updated=recent_updated,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["health", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        # Should not be stale because it was recently updated
        stale_paths = [s["path"] for s in data["stale"]]
        assert "development/recently-updated.md" not in stale_paths
