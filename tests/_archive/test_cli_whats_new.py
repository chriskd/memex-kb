"""Tests for mx whats-new CLI command."""

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
    """Create a temporary KB root with standard categories."""
    root = tmp_path / "kb"
    root.mkdir()
    (root / "development").mkdir()
    (root / "projects" / "myapp").mkdir(parents=True)
    (root / "projects" / "otherapp").mkdir(parents=True)
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
    created: datetime | None = None,
    updated: datetime | None = None,
    source_project: str | None = None,
):
    """Helper to create a KB entry with frontmatter."""
    created = created or datetime.now(timezone.utc)
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    frontmatter = f"""title: {title}
tags:
{tags_yaml}
created: {created.isoformat()}"""
    if updated:
        frontmatter += f"\nupdated: {updated.isoformat()}"
    if source_project:
        frontmatter += f"\nsource_project: {source_project}"

    text = f"""---
{frontmatter}
---

Content for {title}.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class TestWhatsNewBasic:
    """Test basic whats-new functionality."""

    def test_whats_new_shows_recent_entries(self, kb_root, index_root):
        """whats-new shows recently created entries."""
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)

        _create_entry(
            kb_root / "development" / "recent.md",
            "Recent Entry",
            ["python"],
            created=yesterday,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["whats-new"])

        assert result.exit_code == 0
        assert "Recent Entry" in result.output
        assert "recent.md" in result.output

    def test_whats_new_shows_created_and_updated(self, kb_root, index_root):
        """whats-new shows both created and updated entries."""
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)
        yesterday = now - timedelta(days=1)

        # Entry created recently
        _create_entry(
            kb_root / "development" / "new.md",
            "New Entry",
            ["python"],
            created=yesterday,
        )

        # Entry created a while ago but updated recently
        _create_entry(
            kb_root / "development" / "updated.md",
            "Updated Entry",
            ["python"],
            created=week_ago,
            updated=yesterday,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["whats-new"])

        assert result.exit_code == 0
        assert "New Entry" in result.output
        assert "Updated Entry" in result.output

    def test_whats_new_sorted_by_date(self, kb_root, index_root):
        """whats-new results are sorted by date, most recent first."""
        now = datetime.now(timezone.utc)
        three_days_ago = now - timedelta(days=3)
        one_day_ago = now - timedelta(days=1)

        _create_entry(
            kb_root / "development" / "older.md",
            "Older Entry",
            ["python"],
            created=three_days_ago,
        )
        _create_entry(
            kb_root / "development" / "newer.md",
            "Newer Entry",
            ["python"],
            created=one_day_ago,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["whats-new", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 2
        # Newer should be first
        assert data[0]["title"] == "Newer Entry"
        assert data[1]["title"] == "Older Entry"

    def test_whats_new_shows_activity_type(self, kb_root, index_root):
        """whats-new shows activity type (created/updated) in JSON output."""
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)
        yesterday = now - timedelta(days=1)

        _create_entry(
            kb_root / "development" / "new.md",
            "New Entry",
            ["python"],
            created=yesterday,
        )

        _create_entry(
            kb_root / "development" / "updated.md",
            "Updated Entry",
            ["python"],
            created=week_ago,
            updated=yesterday,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["whats-new", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        activity_types = {e["title"]: e["activity_type"] for e in data}
        assert activity_types["New Entry"] == "created"
        assert activity_types["Updated Entry"] == "updated"


class TestWhatsNewDays:
    """Test --days option for whats-new."""

    def test_whats_new_default_30_days(self, kb_root, index_root):
        """Default looks back 30 days."""
        now = datetime.now(timezone.utc)
        within_30 = now - timedelta(days=25)
        beyond_30 = now - timedelta(days=35)

        _create_entry(
            kb_root / "development" / "within.md",
            "Within 30 Days",
            ["python"],
            created=within_30,
        )
        _create_entry(
            kb_root / "development" / "beyond.md",
            "Beyond 30 Days",
            ["python"],
            created=beyond_30,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["whats-new"])

        assert result.exit_code == 0
        assert "Within 30 Days" in result.output
        assert "Beyond 30 Days" not in result.output

    def test_whats_new_custom_days(self, kb_root, index_root):
        """--days=7 looks back only 7 days."""
        now = datetime.now(timezone.utc)
        within_7 = now - timedelta(days=5)
        beyond_7 = now - timedelta(days=10)

        _create_entry(
            kb_root / "development" / "within.md",
            "Within 7 Days",
            ["python"],
            created=within_7,
        )
        _create_entry(
            kb_root / "development" / "beyond.md",
            "Beyond 7 Days",
            ["python"],
            created=beyond_7,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["whats-new", "--days=7"])

        assert result.exit_code == 0
        assert "Within 7 Days" in result.output
        assert "Beyond 7 Days" not in result.output

    def test_whats_new_excludes_old_entries(self, kb_root, index_root):
        """Entries older than the window are excluded."""
        now = datetime.now(timezone.utc)
        month_ago = now - timedelta(days=35)

        _create_entry(
            kb_root / "development" / "old.md",
            "Old Entry",
            ["python"],
            created=month_ago,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["whats-new"])

        assert result.exit_code == 0
        # Should show message about no entries, not the old entry
        assert "No entries created or updated" in result.output


class TestWhatsNewLimit:
    """Test --limit option for whats-new."""

    def test_whats_new_default_limit_10(self, kb_root, index_root):
        """Default shows up to 10 entries."""
        now = datetime.now(timezone.utc)

        # Create 15 entries
        for i in range(15):
            created = now - timedelta(days=i)
            _create_entry(
                kb_root / "development" / f"entry{i:02d}.md",
                f"Entry {i:02d}",
                ["python"],
                created=created,
            )

        runner = CliRunner()
        result = runner.invoke(cli, ["whats-new", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 10

    def test_whats_new_custom_limit(self, kb_root, index_root):
        """--limit=3 shows only 3 entries."""
        now = datetime.now(timezone.utc)

        # Create 5 entries
        for i in range(5):
            created = now - timedelta(days=i)
            _create_entry(
                kb_root / "development" / f"entry{i}.md",
                f"Entry {i}",
                ["python"],
                created=created,
            )

        runner = CliRunner()
        result = runner.invoke(cli, ["whats-new", "--limit=3", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 3


class TestWhatsNewProject:
    """Test --project option for whats-new."""

    def test_whats_new_filters_by_project_path(self, kb_root, index_root):
        """--project filters by path (projects/myapp/)."""
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)

        _create_entry(
            kb_root / "projects" / "myapp" / "doc.md",
            "MyApp Doc",
            ["docs"],
            created=yesterday,
        )
        _create_entry(
            kb_root / "projects" / "otherapp" / "doc.md",
            "OtherApp Doc",
            ["docs"],
            created=yesterday,
        )
        _create_entry(
            kb_root / "development" / "unrelated.md",
            "Unrelated",
            ["python"],
            created=yesterday,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["whats-new", "--project=myapp"])

        assert result.exit_code == 0
        assert "MyApp Doc" in result.output
        assert "OtherApp Doc" not in result.output
        assert "Unrelated" not in result.output

    def test_whats_new_filters_by_source_project(self, kb_root, index_root):
        """--project filters by source_project metadata."""
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)

        _create_entry(
            kb_root / "development" / "myapp_related.md",
            "MyApp Related",
            ["docs"],
            created=yesterday,
            source_project="myapp",
        )
        _create_entry(
            kb_root / "development" / "other.md",
            "Other Doc",
            ["docs"],
            created=yesterday,
            source_project="otherapp",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["whats-new", "--project=myapp"])

        assert result.exit_code == 0
        assert "MyApp Related" in result.output
        assert "Other Doc" not in result.output

    def test_whats_new_filters_by_project_tag(self, kb_root, index_root):
        """--project filters by project tag."""
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)

        _create_entry(
            kb_root / "development" / "tagged_myapp.md",
            "Tagged MyApp",
            ["docs", "myapp"],
            created=yesterday,
        )
        _create_entry(
            kb_root / "development" / "tagged_other.md",
            "Tagged Other",
            ["docs", "otherapp"],
            created=yesterday,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["whats-new", "--project=myapp"])

        assert result.exit_code == 0
        assert "Tagged MyApp" in result.output
        assert "Tagged Other" not in result.output

    def test_whats_new_no_match_shows_message(self, kb_root, index_root):
        """When no entries match project filter, shows appropriate message."""
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)

        _create_entry(
            kb_root / "development" / "unrelated.md",
            "Unrelated",
            ["python"],
            created=yesterday,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["whats-new", "--project=nonexistent"])

        assert result.exit_code == 0
        assert "No entries for project 'nonexistent'" in result.output


class TestWhatsNewJsonOutput:
    """Test --json output for whats-new."""

    def test_whats_new_json_structure(self, kb_root, index_root):
        """JSON output is proper array with expected fields."""
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)

        _create_entry(
            kb_root / "development" / "entry.md",
            "Test Entry",
            ["python", "testing"],
            created=yesterday,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["whats-new", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 1

        entry = data[0]
        assert "path" in entry
        assert "title" in entry
        assert "tags" in entry
        assert "created" in entry
        assert "activity_type" in entry
        assert "activity_date" in entry

    def test_whats_new_json_includes_dates(self, kb_root, index_root):
        """JSON output includes activity_date for each entry."""
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)
        week_ago = now - timedelta(days=7)

        _create_entry(
            kb_root / "development" / "new.md",
            "New Entry",
            ["python"],
            created=yesterday,
        )
        _create_entry(
            kb_root / "development" / "updated.md",
            "Updated Entry",
            ["python"],
            created=week_ago,
            updated=yesterday,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["whats-new", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        for entry in data:
            assert "activity_date" in entry
            # Validate date is parseable ISO format
            datetime.fromisoformat(entry["activity_date"].replace("Z", "+00:00"))


class TestWhatsNewEdgeCases:
    """Test edge cases for whats-new."""

    def test_whats_new_empty_kb(self, kb_root, index_root):
        """Handles empty KB gracefully."""
        runner = CliRunner()
        result = runner.invoke(cli, ["whats-new"])

        assert result.exit_code == 0
        assert "No entries created or updated" in result.output

    def test_whats_new_all_old(self, kb_root, index_root):
        """Handles when all entries are older than window."""
        now = datetime.now(timezone.utc)
        month_ago = now - timedelta(days=35)

        _create_entry(
            kb_root / "development" / "old1.md",
            "Old Entry 1",
            ["python"],
            created=month_ago,
        )
        _create_entry(
            kb_root / "development" / "old2.md",
            "Old Entry 2",
            ["python"],
            created=month_ago,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["whats-new"])

        assert result.exit_code == 0
        assert "No entries created or updated" in result.output
