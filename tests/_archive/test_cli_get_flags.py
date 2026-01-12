"""Tests for mx get CLI flags (--json by path, --metadata edge cases)."""

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
    """Create a temporary KB root with nested structure."""
    root = tmp_path / "kb"
    root.mkdir()
    (root / "development").mkdir()
    (root / "development" / "python").mkdir()
    (root / "development" / "python" / "frameworks").mkdir()
    (root / "development" / "python" / "frameworks" / "web").mkdir()
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
    frontmatter = f"""title: {title}
tags:
{tags_yaml}
created: {created.isoformat()}"""
    if updated:
        frontmatter += f"\nupdated: {updated.isoformat()}"

    text = f"""---
{frontmatter}
---

{content if content else f"Content for {title}."}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class TestGetJsonByPath:
    """Test mx get PATH --json output includes all expected fields."""

    def test_get_json_by_path_includes_metadata(self, kb_root, index_root):
        """JSON output includes all metadata fields."""
        created = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        updated = datetime(2024, 7, 20, 14, 45, 0, tzinfo=timezone.utc)
        _create_entry(
            kb_root / "development" / "json-test.md",
            "JSON Test Entry",
            ["python", "testing"],
            "Test content for JSON output.",
            created=created,
            updated=updated,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["get", "development/json-test.md", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Check metadata structure
        assert "metadata" in data
        assert data["metadata"]["title"] == "JSON Test Entry"
        assert data["metadata"]["tags"] == ["python", "testing"]
        assert "2024-06-15" in data["metadata"]["created"]
        assert "2024-07-20" in data["metadata"]["updated"]

    def test_get_json_by_path_includes_content(self, kb_root, index_root):
        """JSON output includes full content."""
        _create_entry(
            kb_root / "development" / "content-test.md",
            "Content Test",
            ["docs"],
            "This is the main content.\n\nWith multiple paragraphs.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["get", "development/content-test.md", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        assert "content" in data
        assert "This is the main content." in data["content"]
        assert "With multiple paragraphs." in data["content"]

    def test_get_json_by_path_includes_links(self, kb_root, index_root):
        """JSON output includes outgoing links."""
        # Create target entries first
        _create_entry(
            kb_root / "development" / "target-one.md",
            "Target One",
            ["reference"],
        )
        _create_entry(
            kb_root / "development" / "target-two.md",
            "Target Two",
            ["reference"],
        )
        # Create entry with links
        _create_entry(
            kb_root / "development" / "linking-entry.md",
            "Linking Entry",
            ["test"],
            "Links to [[development/target-one.md|Target One]] and [[development/target-two.md|Target Two]].",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["get", "development/linking-entry.md", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        assert "links" in data
        assert isinstance(data["links"], list)
        # Links are stored without .md extension
        assert "development/target-one" in data["links"]
        assert "development/target-two" in data["links"]

    def test_get_json_by_path_includes_backlinks(self, kb_root, index_root):
        """JSON output includes backlinks from other entries."""
        # Create target entry
        _create_entry(
            kb_root / "development" / "popular-entry.md",
            "Popular Entry",
            ["reference"],
            "This entry is linked to by others.",
        )
        # Create entries that link to target
        _create_entry(
            kb_root / "development" / "linker-a.md",
            "Linker A",
            ["test"],
            "Links to [[development/popular-entry.md|Popular Entry]].",
        )
        _create_entry(
            kb_root / "development" / "linker-b.md",
            "Linker B",
            ["test"],
            "Also references [[development/popular-entry.md|Popular Entry]].",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["get", "development/popular-entry.md", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        assert "backlinks" in data
        assert isinstance(data["backlinks"], list)
        # Backlinks are stored without .md extension
        assert "development/linker-a" in data["backlinks"]
        assert "development/linker-b" in data["backlinks"]


class TestGetMetadata:
    """Test mx get --metadata edge cases."""

    def test_get_metadata_shows_links_count(self, kb_root, index_root):
        """Metadata output shows outgoing link count."""
        _create_entry(
            kb_root / "development" / "link-a.md",
            "Link Target A",
            ["ref"],
        )
        _create_entry(
            kb_root / "development" / "link-b.md",
            "Link Target B",
            ["ref"],
        )
        _create_entry(
            kb_root / "development" / "link-c.md",
            "Link Target C",
            ["ref"],
        )
        _create_entry(
            kb_root / "development" / "multi-linker.md",
            "Multi Linker",
            ["test"],
            "Links: [[development/link-a.md|A]], [[development/link-b.md|B]], [[development/link-c.md|C]].",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["get", "development/multi-linker.md", "--metadata"])

        assert result.exit_code == 0
        assert "Links:    3" in result.output

    def test_get_metadata_shows_backlinks_count(self, kb_root, index_root):
        """Metadata output shows incoming backlink count."""
        _create_entry(
            kb_root / "development" / "target-entry.md",
            "Target Entry",
            ["popular"],
            "This is a popular entry.",
        )
        for i in range(4):
            _create_entry(
                kb_root / "development" / f"linking-entry-{i}.md",
                f"Linking Entry {i}",
                ["test"],
                f"References [[development/target-entry.md|Target Entry]].",
            )

        runner = CliRunner()
        result = runner.invoke(cli, ["get", "development/target-entry.md", "--metadata"])

        assert result.exit_code == 0
        assert "Backlinks: 4" in result.output

    def test_get_metadata_shows_dates(self, kb_root, index_root):
        """Metadata output shows created and updated dates."""
        created = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        updated = datetime(2024, 11, 22, 16, 30, 0, tzinfo=timezone.utc)
        _create_entry(
            kb_root / "development" / "dated-entry.md",
            "Dated Entry",
            ["archive"],
            "Entry with explicit dates.",
            created=created,
            updated=updated,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["get", "development/dated-entry.md", "--metadata"])

        assert result.exit_code == 0
        assert "Created:" in result.output
        assert "2024-01-15" in result.output
        assert "Updated:" in result.output
        assert "2024-11-22" in result.output

    def test_get_metadata_shows_never_for_no_update(self, kb_root, index_root):
        """Metadata shows 'never' when entry has no updated date."""
        _create_entry(
            kb_root / "development" / "new-entry.md",
            "New Entry",
            ["fresh"],
            "Never been updated.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["get", "development/new-entry.md", "--metadata"])

        assert result.exit_code == 0
        assert "Updated:  never" in result.output


class TestGetEdgeCases:
    """Test mx get edge cases and error handling."""

    def test_get_unicode_path(self, kb_root, index_root):
        """Handles unicode characters in path."""
        (kb_root / "development" / "unicode-test").mkdir(parents=True)
        _create_entry(
            kb_root / "development" / "unicode-test" / "cafe-guide.md",
            "Cafe Guide",
            ["food", "travel"],
            "Guide to great coffee shops.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["get", "development/unicode-test/cafe-guide.md"])

        assert result.exit_code == 0
        assert "# Cafe Guide" in result.output

    def test_get_deeply_nested(self, kb_root, index_root):
        """Handles deeply nested paths (4+ levels)."""
        _create_entry(
            kb_root / "development" / "python" / "frameworks" / "web" / "flask-guide.md",
            "Flask Guide",
            ["python", "flask", "web"],
            "Guide to Flask web framework.",
        )

        runner = CliRunner()
        result = runner.invoke(
            cli, ["get", "development/python/frameworks/web/flask-guide.md"]
        )

        assert result.exit_code == 0
        assert "# Flask Guide" in result.output
        assert "Guide to Flask web framework." in result.output

    def test_get_nonexistent_shows_error(self, kb_root, index_root):
        """Clear error message for missing file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["get", "development/does-not-exist.md"])

        assert result.exit_code == 1
        assert "Error:" in result.output

    def test_get_deeply_nested_with_json(self, kb_root, index_root):
        """JSON output works for deeply nested paths."""
        _create_entry(
            kb_root / "development" / "python" / "frameworks" / "web" / "django-api.md",
            "Django API Guide",
            ["python", "django", "api"],
            "Building APIs with Django REST Framework.",
        )

        runner = CliRunner()
        result = runner.invoke(
            cli, ["get", "development/python/frameworks/web/django-api.md", "--json"]
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["path"] == "development/python/frameworks/web/django-api.md"
        assert data["metadata"]["title"] == "Django API Guide"

    def test_get_deeply_nested_with_metadata(self, kb_root, index_root):
        """Metadata output works for deeply nested paths."""
        _create_entry(
            kb_root / "development" / "python" / "frameworks" / "web" / "fastapi-intro.md",
            "FastAPI Introduction",
            ["python", "fastapi", "async"],
            "Introduction to FastAPI.",
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["get", "development/python/frameworks/web/fastapi-intro.md", "--metadata"],
        )

        assert result.exit_code == 0
        assert "Title:    FastAPI Introduction" in result.output
        assert "Tags:     python, fastapi, async" in result.output
