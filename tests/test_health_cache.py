"""Tests for health_cache module."""

import json
import time
from datetime import date, datetime
from pathlib import Path

import pytest

from memex.health_cache import (
    CACHE_FILENAME,
    _cache_path,
    _parse_file_metadata,
    ensure_health_cache,
    get_entry_metadata,
    load_cache,
    rebuild_health_cache,
    save_cache,
)


def _create_entry(
    path: Path,
    title: str,
    tags: list[str],
    content: str = "",
    created: date | None = None,
    updated: date | None = None,
) -> None:
    """Create a test entry file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    created_date = (created or date.today()).isoformat()
    updated_line = f"updated: {updated.isoformat()}\n" if updated else ""
    # Use empty list syntax for no tags, or proper YAML list for tags
    if tags:
        tags_yaml = "tags:\n" + "\n".join(f"  - {tag}" for tag in tags)
    else:
        tags_yaml = "tags: []"

    full_content = f"""---
title: {title}
{tags_yaml}
created: {created_date}
{updated_line}---

{content}
"""
    path.write_text(full_content)


class TestHealthCacheBasics:
    """Basic cache operations."""

    def test_load_cache_empty(self, tmp_path):
        """Load from non-existent cache returns empty dict."""
        files, mtime, parse_errors = load_cache(tmp_path)
        assert files == {}
        assert mtime == 0.0
        assert parse_errors == []

    def test_save_and_load_cache(self, tmp_path):
        """Save and load cache round-trips correctly."""
        files = {
            "test/entry": {
                "mtime": 123.456,
                "rel_path": "test/entry.md",
                "title": "Test Entry",
                "created": "2024-01-15",
                "updated": None,
                "links": ["other/entry"],
            }
        }
        parse_errors = [{"path": "broken.md", "error": "Test error"}]
        save_cache(files, 123.456, tmp_path, parse_errors)

        loaded_files, loaded_mtime, loaded_errors = load_cache(tmp_path)
        assert loaded_files == files
        assert loaded_mtime == 123.456
        assert loaded_errors == parse_errors

    def test_cache_path_creates_directory(self, tmp_path):
        """Cache path creates parent directory if needed."""
        index_root = tmp_path / "nested" / "index"
        path = _cache_path(index_root)
        assert path.parent.exists()
        assert path.name == CACHE_FILENAME


class TestParseFileMetadata:
    """Test file metadata parsing."""

    def test_parse_valid_entry(self, tmp_path):
        """Parse a valid entry file."""
        entry_path = tmp_path / "test.md"
        _create_entry(
            entry_path,
            title="Test Entry",
            tags=["tag1", "tag2"],
            content="Content with [[link/to/other]] reference.",
            created=date(2024, 1, 15),
        )

        meta, error = _parse_file_metadata(entry_path)

        assert meta is not None
        assert error is None
        assert meta["title"] == "Test Entry"
        # Full ISO 8601 timestamp format
        assert meta["created"] == "2024-01-15T00:00:00"
        assert meta["updated"] is None
        assert "link/to/other" in meta["links"]

    def test_parse_invalid_entry_returns_error(self, tmp_path):
        """Invalid entry returns None and error message."""
        entry_path = tmp_path / "invalid.md"
        entry_path.write_text("No frontmatter here")

        meta, error = _parse_file_metadata(entry_path)
        assert meta is None
        assert error is not None
        assert "frontmatter" in error.lower()


class TestRebuildHealthCache:
    """Test full cache rebuild."""

    def test_rebuild_empty_kb(self, tmp_path):
        """Rebuild with no files returns empty dict."""
        kb_root = tmp_path / "kb"
        kb_root.mkdir()
        index_root = tmp_path / "index"

        result = rebuild_health_cache(kb_root, index_root)
        assert result == {}

    def test_rebuild_with_entries(self, tmp_path):
        """Rebuild caches all entries."""
        kb_root = tmp_path / "kb"
        index_root = tmp_path / "index"

        _create_entry(
            kb_root / "test" / "entry1.md",
            title="Entry 1",
            tags=["tag1"],
            content="Links to [[test/entry2]].",
        )
        _create_entry(
            kb_root / "test" / "entry2.md",
            title="Entry 2",
            tags=["tag2"],
        )

        result = rebuild_health_cache(kb_root, index_root)

        assert len(result) == 2
        assert "test/entry1" in result
        assert "test/entry2" in result
        assert result["test/entry1"]["title"] == "Entry 1"
        assert "test/entry2" in result["test/entry1"]["links"]

    def test_rebuild_skips_underscore_files(self, tmp_path):
        """Rebuild skips files starting with underscore."""
        kb_root = tmp_path / "kb"
        index_root = tmp_path / "index"

        _create_entry(kb_root / "normal.md", title="Normal", tags=["test"])
        _create_entry(kb_root / "_hidden.md", title="Hidden", tags=["test"])

        result = rebuild_health_cache(kb_root, index_root)

        assert len(result) == 1
        assert "normal" in result
        assert "_hidden" not in result


class TestEnsureHealthCache:
    """Test cache invalidation and incremental updates."""

    def test_first_call_rebuilds_cache(self, tmp_path):
        """First call with no cache does full rebuild."""
        kb_root = tmp_path / "kb"
        index_root = tmp_path / "index"

        _create_entry(kb_root / "test.md", title="Test", tags=["test"])

        result = ensure_health_cache(kb_root, index_root)

        assert "test" in result
        # Cache file should exist
        assert (index_root / CACHE_FILENAME).exists()

    def test_unchanged_kb_uses_cache(self, tmp_path):
        """Unchanged KB returns cached data without reparsing."""
        kb_root = tmp_path / "kb"
        index_root = tmp_path / "index"

        _create_entry(kb_root / "test.md", title="Test", tags=["test"])

        # First call - builds cache
        ensure_health_cache(kb_root, index_root)

        # Modify cache file to detect if it's being read
        cache_path = index_root / CACHE_FILENAME
        cache_data = json.loads(cache_path.read_text())
        cache_data["files"]["test"]["title"] = "Modified In Cache"
        cache_path.write_text(json.dumps(cache_data))

        # Second call - should use cache (no file changes)
        result2 = ensure_health_cache(kb_root, index_root)

        # Should get the modified cached value
        assert result2["test"]["title"] == "Modified In Cache"

    def test_modified_file_triggers_reparse(self, tmp_path):
        """Modified file triggers incremental update."""
        kb_root = tmp_path / "kb"
        index_root = tmp_path / "index"

        _create_entry(kb_root / "test.md", title="Original", tags=["test"])

        # Build initial cache
        result1 = ensure_health_cache(kb_root, index_root)
        assert result1["test"]["title"] == "Original"

        # Wait a bit to ensure mtime changes
        time.sleep(0.1)

        # Modify the file
        _create_entry(kb_root / "test.md", title="Updated", tags=["new"])

        # Should detect change and reparse
        result2 = ensure_health_cache(kb_root, index_root)
        assert result2["test"]["title"] == "Updated"

    def test_new_file_triggers_update(self, tmp_path):
        """New file triggers incremental update."""
        kb_root = tmp_path / "kb"
        index_root = tmp_path / "index"

        _create_entry(kb_root / "first.md", title="First", tags=["test"])
        ensure_health_cache(kb_root, index_root)

        time.sleep(0.1)

        # Add new file
        _create_entry(kb_root / "second.md", title="Second", tags=["test"])

        result = ensure_health_cache(kb_root, index_root)
        assert len(result) == 2
        assert "second" in result

    def test_deleted_file_triggers_update(self, tmp_path):
        """Deleted file triggers cache update."""
        kb_root = tmp_path / "kb"
        index_root = tmp_path / "index"

        _create_entry(kb_root / "keep.md", title="Keep", tags=["test"])
        _create_entry(kb_root / "delete.md", title="Delete", tags=["test"])

        ensure_health_cache(kb_root, index_root)

        # Delete file
        (kb_root / "delete.md").unlink()

        result = ensure_health_cache(kb_root, index_root)
        assert len(result) == 1
        assert "keep" in result
        assert "delete" not in result

    def test_incremental_preserves_unchanged_entries(self, tmp_path):
        """Incremental update preserves unchanged entries without reparsing."""
        kb_root = tmp_path / "kb"
        index_root = tmp_path / "index"

        _create_entry(kb_root / "unchanged.md", title="Unchanged", tags=["test"])
        _create_entry(kb_root / "changed.md", title="Original", tags=["test"])

        # Build cache
        ensure_health_cache(kb_root, index_root)

        # Manually modify cached data for unchanged entry
        cache_path = index_root / CACHE_FILENAME
        cache_data = json.loads(cache_path.read_text())
        cache_data["files"]["unchanged"]["title"] = "Cache Marker"
        cache_path.write_text(json.dumps(cache_data))

        time.sleep(0.1)

        # Modify only the changed file
        _create_entry(kb_root / "changed.md", title="Modified", tags=["test"])

        result = ensure_health_cache(kb_root, index_root)

        # Unchanged entry should still have the cache marker (wasn't reparsed)
        assert result["unchanged"]["title"] == "Cache Marker"
        # Changed entry should be updated
        assert result["changed"]["title"] == "Modified"


class TestGetEntryMetadata:
    """Test the convenience wrapper function."""

    def test_converts_date_strings_to_dates(self, tmp_path):
        """Date strings are converted to date objects."""
        kb_root = tmp_path / "kb"
        index_root = tmp_path / "index"

        _create_entry(
            kb_root / "test.md",
            title="Test",
            tags=["test"],
            created=date(2024, 6, 15),
            updated=date(2024, 7, 20),
        )

        result = get_entry_metadata(kb_root, index_root)

        # Returns datetime objects now
        assert result["test"]["created"] == datetime(2024, 6, 15, 0, 0, 0)
        assert result["test"]["updated"] == datetime(2024, 7, 20, 0, 0, 0)

    def test_handles_null_dates(self, tmp_path):
        """Null dates remain as None."""
        kb_root = tmp_path / "kb"
        index_root = tmp_path / "index"

        _create_entry(kb_root / "test.md", title="Test", tags=["test"])

        result = get_entry_metadata(kb_root, index_root)

        # created is set, updated is None
        assert result["test"]["created"] is not None
        assert result["test"]["updated"] is None


class TestHealthCacheIntegration:
    """Integration tests with health function."""

    @pytest.mark.asyncio
    async def test_health_uses_cached_metadata(self, tmp_path, monkeypatch):
        """Health function uses cached metadata."""
        from memex import core

        kb_root = tmp_path / "kb"
        index_root = tmp_path / "index"

        monkeypatch.setattr(core, "get_kb_root", lambda: kb_root)
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        # Create entries
        _create_entry(
            kb_root / "dev" / "entry1.md",
            title="Entry 1",
            tags=["test"],
            content="See [[dev/entry2]].",
        )
        _create_entry(
            kb_root / "dev" / "entry2.md",
            title="Entry 2",
            tags=["test"],
        )

        # Run health
        result = await core.health()

        # Should complete without errors
        assert "summary" in result
        assert result["summary"]["total_entries"] == 2

    @pytest.mark.asyncio
    async def test_health_detects_broken_links_from_cache(self, tmp_path, monkeypatch):
        """Broken links detected using cached link data."""
        from memex import core

        kb_root = tmp_path / "kb"
        index_root = tmp_path / "index"

        monkeypatch.setattr(core, "get_kb_root", lambda: kb_root)
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        _create_entry(
            kb_root / "source.md",
            title="Source",
            tags=["test"],
            content="Link to [[nonexistent]].",
        )

        result = await core.health()

        assert len(result["broken_links"]) == 1
        assert result["broken_links"][0]["broken_link"] == "nonexistent"

    @pytest.mark.asyncio
    async def test_health_reports_parse_errors(self, tmp_path, monkeypatch):
        """Parse errors are tracked and reported in health."""
        from memex import core

        kb_root = tmp_path / "kb"
        index_root = tmp_path / "index"

        monkeypatch.setattr(core, "get_kb_root", lambda: kb_root)
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        # Create a valid entry
        _create_entry(
            kb_root / "valid.md",
            title="Valid Entry",
            tags=["test"],
        )

        # Create an invalid entry (no frontmatter)
        broken_path = kb_root / "broken.md"
        broken_path.parent.mkdir(parents=True, exist_ok=True)
        broken_path.write_text("No frontmatter here, just text.")

        result = await core.health()

        # Should report the parse error
        assert len(result["parse_errors"]) == 1
        assert result["parse_errors"][0]["path"] == "broken.md"
        assert "frontmatter" in result["parse_errors"][0]["error"].lower()

        # Should still count valid entries
        assert result["summary"]["total_entries"] == 1
        assert result["summary"]["parse_errors_count"] == 1

        # Health score should be penalized
        assert result["summary"]["health_score"] < 100


class TestParseErrorsFunction:
    """Test get_parse_errors function."""

    def test_get_parse_errors_returns_cached_errors(self, tmp_path):
        """get_parse_errors returns parse errors from cache."""
        kb_root = tmp_path / "kb"
        index_root = tmp_path / "index"

        # Create valid entry
        _create_entry(kb_root / "valid.md", title="Valid", tags=["test"])

        # Create invalid entry
        broken_path = kb_root / "broken.md"
        broken_path.parent.mkdir(parents=True, exist_ok=True)
        broken_path.write_text("No frontmatter")

        # Build cache
        rebuild_health_cache(kb_root, index_root)

        # Get parse errors
        from memex.health_cache import get_parse_errors

        errors = get_parse_errors(index_root)

        assert len(errors) == 1
        assert errors[0]["path"] == "broken.md"

    @pytest.mark.asyncio
    async def test_health_suggests_similar_targets_for_broken_links(
        self, tmp_path, monkeypatch
    ):
        """Broken links get suggestions for similar valid targets."""
        from memex import core

        kb_root = tmp_path / "kb"
        index_root = tmp_path / "index"

        monkeypatch.setattr(core, "get_kb_root", lambda: kb_root)
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        # Create an entry with a broken link that's similar to another entry
        _create_entry(
            kb_root / "source.md",
            title="Source",
            tags=["test"],
            content="Link to [[dev/guidlines]] (typo).",  # typo: guidlines
        )
        _create_entry(
            kb_root / "dev" / "guidelines.md",  # correct spelling
            title="Guidelines",
            tags=["test"],
        )

        result = await core.health()

        assert len(result["broken_links"]) == 1
        broken = result["broken_links"][0]
        assert broken["broken_link"] == "dev/guidlines"
        # Should suggest the similar target
        assert "suggestion" in broken
        assert broken["suggestion"] == "dev/guidelines"


class TestDescriptionMetadata:
    """Test description field in health cache."""

    def _create_entry_with_description(
        self,
        path: Path,
        title: str,
        tags: list[str],
        description: str | None = None,
        content: str = "",
    ) -> None:
        """Create a test entry file with optional description."""
        path.parent.mkdir(parents=True, exist_ok=True)
        created_date = date.today().isoformat()
        tags_yaml = "tags:\n" + "\n".join(f"  - {tag}" for tag in tags)
        desc_line = f"description: {description}\n" if description else ""

        full_content = f"""---
title: {title}
{desc_line}{tags_yaml}
created: {created_date}
---

{content}
"""
        path.write_text(full_content)

    def test_parse_entry_with_description(self, tmp_path):
        """Parse entry that has a description field."""
        entry_path = tmp_path / "test.md"
        self._create_entry_with_description(
            entry_path,
            title="Test Entry",
            tags=["test"],
            description="A short summary of the entry.",
            content="Full content here.",
        )

        result, error = _parse_file_metadata(entry_path)

        assert error is None
        assert result is not None
        assert result["title"] == "Test Entry"
        assert result["description"] == "A short summary of the entry."

    def test_parse_entry_without_description(self, tmp_path):
        """Parse entry that has no description field."""
        entry_path = tmp_path / "test.md"
        _create_entry(
            entry_path,
            title="Test Entry",
            tags=["test"],
            content="Content without description.",
        )

        result, error = _parse_file_metadata(entry_path)

        assert error is None
        assert result is not None
        assert result["description"] is None

    def test_get_entry_metadata_includes_description(self, tmp_path):
        """get_entry_metadata returns description field."""
        kb_root = tmp_path / "kb"
        index_root = tmp_path / "index"

        self._create_entry_with_description(
            kb_root / "with_desc.md",
            title="With Description",
            tags=["test"],
            description="Has a description",
        )
        _create_entry(
            kb_root / "no_desc.md",
            title="No Description",
            tags=["test"],
        )

        # Build cache and get metadata
        rebuild_health_cache(kb_root, index_root)
        metadata = get_entry_metadata(kb_root, index_root)

        assert "with_desc" in metadata
        assert metadata["with_desc"]["description"] == "Has a description"

        assert "no_desc" in metadata
        assert metadata["no_desc"]["description"] is None

    @pytest.mark.asyncio
    async def test_health_reports_missing_descriptions(self, tmp_path, monkeypatch):
        """Health check reports entries missing descriptions."""
        from memex import core

        kb_root = tmp_path / "kb"
        index_root = tmp_path / "index"

        monkeypatch.setattr(core, "get_kb_root", lambda: kb_root)
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        # Create entry with description
        self._create_entry_with_description(
            kb_root / "with_desc.md",
            title="With Description",
            tags=["test"],
            description="Has a description",
        )

        # Create entry without description
        _create_entry(
            kb_root / "no_desc.md",
            title="No Description",
            tags=["test"],
        )

        result = await core.health()

        # Should report the missing description
        assert "missing_descriptions" in result
        assert len(result["missing_descriptions"]) == 1
        assert result["missing_descriptions"][0]["path"] == "no_desc.md"
        assert result["missing_descriptions"][0]["title"] == "No Description"

        # Summary should include count
        assert result["summary"]["missing_descriptions_count"] == 1

    @pytest.mark.asyncio
    async def test_health_no_missing_descriptions_when_all_have_them(
        self, tmp_path, monkeypatch
    ):
        """Health check reports no issues when all entries have descriptions."""
        from memex import core

        kb_root = tmp_path / "kb"
        index_root = tmp_path / "index"

        monkeypatch.setattr(core, "get_kb_root", lambda: kb_root)
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        # Create entries all with descriptions
        self._create_entry_with_description(
            kb_root / "entry1.md",
            title="Entry 1",
            tags=["test"],
            description="Description 1",
        )
        self._create_entry_with_description(
            kb_root / "entry2.md",
            title="Entry 2",
            tags=["test"],
            description="Description 2",
        )

        result = await core.health()

        assert result["missing_descriptions"] == []
        assert result["summary"]["missing_descriptions_count"] == 0
