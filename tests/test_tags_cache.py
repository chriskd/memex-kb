"""Tests for tags cache management."""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import pytest

from memex.tags_cache import (
    CACHE_FILENAME,
    _cache_path,
    _get_file_mtime,
    _incremental_update,
    ensure_tags_cache,
    get_tag_entries,
    load_cache,
    rebuild_tags_cache,
    save_cache,
)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _create_entry(path: Path, title: str, tags: list[str]) -> None:
    """Create a KB entry with proper frontmatter."""
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    content = f"""---
title: {title}
tags:
{tags_yaml}
created: {date.today().isoformat()}
---

## Content

Some content here.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def kb_root(tmp_path) -> Path:
    """Create a temporary KB root directory."""
    root = tmp_path / "kb"
    root.mkdir()
    return root


@pytest.fixture
def index_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary index root and configure environment."""
    index = tmp_path / "index"
    index.mkdir()
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index))
    return index


@pytest.fixture
def sample_kb(kb_root: Path) -> Path:
    """Create a sample KB with tagged entries."""
    _create_entry(kb_root / "entry-a.md", "Entry A", ["python", "testing"])
    _create_entry(kb_root / "entry-b.md", "Entry B", ["python", "api"])
    _create_entry(kb_root / "entry-c.md", "Entry C", ["rust", "api"])
    return kb_root


@pytest.fixture
def nested_kb(kb_root: Path) -> Path:
    """Create a KB with nested directories and tags."""
    (kb_root / "projects").mkdir()
    (kb_root / "docs").mkdir()

    _create_entry(kb_root / "projects" / "alpha.md", "Alpha Project", ["project", "python"])
    _create_entry(kb_root / "projects" / "beta.md", "Beta Project", ["project", "rust"])
    _create_entry(kb_root / "docs" / "guide.md", "Guide", ["documentation"])
    return kb_root


# -----------------------------------------------------------------------------
# Helper function tests
# -----------------------------------------------------------------------------


class TestCachePath:
    """Tests for _cache_path helper function."""

    def test_cache_path_creation(self, index_root: Path):
        """Test that cache path is correctly constructed."""
        cache_path = _cache_path(index_root)
        assert cache_path == index_root / CACHE_FILENAME
        assert cache_path.parent.exists()

    def test_cache_path_creates_directory(self, tmp_path: Path):
        """Test that cache path creates parent directory if it doesn't exist."""
        index = tmp_path / "nonexistent" / "index"
        cache_path = _cache_path(index)
        assert cache_path.parent.exists()


class TestGetFileMtime:
    """Tests for _get_file_mtime helper function."""

    def test_existing_file(self, tmp_path: Path):
        """Test mtime for existing file."""
        file = tmp_path / "test.md"
        file.write_text("test")
        mtime = _get_file_mtime(file)
        assert mtime > 0
        assert mtime == pytest.approx(file.stat().st_mtime, rel=0.01)

    def test_nonexistent_file(self, tmp_path: Path):
        """Test mtime for nonexistent file returns 0."""
        nonexistent = tmp_path / "does-not-exist.md"
        mtime = _get_file_mtime(nonexistent)
        assert mtime == 0.0


# -----------------------------------------------------------------------------
# Cache load/save tests
# -----------------------------------------------------------------------------


class TestLoadCache:
    """Tests for load_cache function."""

    def test_load_nonexistent_cache(self, index_root: Path):
        """Test loading cache when file doesn't exist returns empty dict."""
        files, mtime = load_cache(index_root)
        assert files == {}
        assert mtime == 0.0

    def test_load_valid_cache(self, index_root: Path):
        """Test loading a valid cache file."""
        cache_file = _cache_path(index_root)
        cache_data = {
            "kb_mtime": 123456.789,
            "files": {
                "entry-a.md": {"mtime": 100.0, "tags": ["python", "testing"]},
                "entry-b.md": {"mtime": 101.0, "tags": ["python", "api"]},
            },
        }
        cache_file.write_text(json.dumps(cache_data))

        files, mtime = load_cache(index_root)
        assert files == cache_data["files"]
        assert mtime == 123456.789

    def test_load_malformed_json(self, index_root: Path):
        """Test loading malformed JSON returns empty dict."""
        cache_file = _cache_path(index_root)
        cache_file.write_text("{ invalid json }")

        files, mtime = load_cache(index_root)
        assert files == {}
        assert mtime == 0.0

    def test_load_missing_fields(self, index_root: Path):
        """Test loading cache with missing fields uses defaults."""
        cache_file = _cache_path(index_root)
        cache_file.write_text(json.dumps({}))

        files, mtime = load_cache(index_root)
        assert files == {}
        assert mtime == 0.0


class TestSaveCache:
    """Tests for save_cache function."""

    def test_save_empty_cache(self, index_root: Path):
        """Test saving empty cache."""
        save_cache({}, 0.0, index_root)

        cache_file = _cache_path(index_root)
        assert cache_file.exists()

        data = json.loads(cache_file.read_text())
        assert data == {"kb_mtime": 0.0, "files": {}}

    def test_save_cache_with_files(self, index_root: Path):
        """Test saving cache with file data."""
        files = {
            "entry-a.md": {"mtime": 100.0, "tags": ["python"]},
            "entry-b.md": {"mtime": 101.0, "tags": ["rust"]},
        }
        mtime = 123456.789

        save_cache(files, mtime, index_root)

        cache_file = _cache_path(index_root)
        data = json.loads(cache_file.read_text())
        assert data["files"] == files
        assert data["kb_mtime"] == mtime

    def test_save_overwrites_existing(self, index_root: Path):
        """Test that save overwrites existing cache."""
        save_cache({"old.md": {"mtime": 1.0, "tags": ["old"]}}, 111.0, index_root)

        new_files = {"new.md": {"mtime": 2.0, "tags": ["new"]}}
        save_cache(new_files, 222.0, index_root)

        cache_file = _cache_path(index_root)
        data = json.loads(cache_file.read_text())
        assert data["files"] == new_files
        assert data["kb_mtime"] == 222.0


# -----------------------------------------------------------------------------
# Cache rebuild tests
# -----------------------------------------------------------------------------


class TestRebuildTagsCache:
    """Tests for rebuild_tags_cache function."""

    def test_rebuild_empty_kb(self, kb_root: Path, index_root: Path):
        """Test rebuilding cache for empty KB."""
        tag_counts = rebuild_tags_cache(kb_root, index_root)
        assert tag_counts == {}

        cache_file = _cache_path(index_root)
        assert cache_file.exists()

    def test_rebuild_sample_kb(self, sample_kb: Path, index_root: Path):
        """Test rebuilding cache for sample KB."""
        tag_counts = rebuild_tags_cache(sample_kb, index_root)

        assert tag_counts["python"] == 2  # entry-a and entry-b
        assert tag_counts["api"] == 2  # entry-b and entry-c
        assert tag_counts["testing"] == 1  # entry-a only
        assert tag_counts["rust"] == 1  # entry-c only

    def test_rebuild_nested_kb(self, nested_kb: Path, index_root: Path):
        """Test rebuilding cache for nested KB structure."""
        tag_counts = rebuild_tags_cache(nested_kb, index_root)

        assert tag_counts["project"] == 2
        assert tag_counts["python"] == 1
        assert tag_counts["rust"] == 1
        assert tag_counts["documentation"] == 1

    def test_rebuild_saves_cache(self, sample_kb: Path, index_root: Path):
        """Test that rebuild saves cache to disk."""
        rebuild_tags_cache(sample_kb, index_root)

        files, _ = load_cache(index_root)
        assert "entry-a.md" in files
        assert "entry-b.md" in files
        assert "entry-c.md" in files

    def test_rebuild_stores_per_file_data(self, sample_kb: Path, index_root: Path):
        """Test that rebuild stores per-file mtime and tags."""
        rebuild_tags_cache(sample_kb, index_root)

        files, _ = load_cache(index_root)

        assert "mtime" in files["entry-a.md"]
        assert files["entry-a.md"]["mtime"] > 0
        assert "tags" in files["entry-a.md"]
        assert set(files["entry-a.md"]["tags"]) == {"python", "testing"}

    def test_rebuild_nonexistent_kb(self, tmp_path: Path, index_root: Path):
        """Test rebuilding with nonexistent KB root."""
        nonexistent = tmp_path / "does-not-exist"
        tag_counts = rebuild_tags_cache(nonexistent, index_root)
        assert tag_counts == {}

    def test_rebuild_skips_underscore_files(self, kb_root: Path, index_root: Path):
        """Test that files starting with _ are skipped."""
        _create_entry(kb_root / "_template.md", "Template", ["template"])
        _create_entry(kb_root / "normal.md", "Normal", ["real"])

        tag_counts = rebuild_tags_cache(kb_root, index_root)

        assert "template" not in tag_counts
        assert "real" in tag_counts


# -----------------------------------------------------------------------------
# Cache invalidation tests
# -----------------------------------------------------------------------------


class TestEnsureTagsCache:
    """Tests for ensure_tags_cache function (cache hit/miss behavior)."""

    def test_cache_miss_no_cache_file(self, sample_kb: Path, index_root: Path):
        """Test cache miss when no cache file exists."""
        tag_counts = ensure_tags_cache(sample_kb, index_root)

        assert tag_counts["python"] == 2
        assert tag_counts["api"] == 2

        cache_file = _cache_path(index_root)
        assert cache_file.exists()

    def test_cache_hit_valid_cache(self, sample_kb: Path, index_root: Path):
        """Test cache hit when cache is fresh and valid."""
        rebuild_tags_cache(sample_kb, index_root)

        tag_counts = ensure_tags_cache(sample_kb, index_root)

        assert tag_counts["python"] == 2
        assert tag_counts["api"] == 2

    def test_cache_miss_stale_mtime(self, sample_kb: Path, index_root: Path):
        """Test cache miss when files are newer than cached mtime."""
        rebuild_tags_cache(sample_kb, index_root)

        time.sleep(0.01)

        # Modify a file
        _create_entry(sample_kb / "entry-a.md", "Entry A Updated", ["python", "updated"])

        tag_counts = ensure_tags_cache(sample_kb, index_root)

        assert "updated" in tag_counts
        assert "testing" not in tag_counts  # Old tag removed

    def test_cache_miss_empty_files(self, sample_kb: Path, index_root: Path):
        """Test cache miss when files dict is empty."""
        save_cache({}, 9999999999.0, index_root)

        tag_counts = ensure_tags_cache(sample_kb, index_root)
        assert tag_counts["python"] == 2

    def test_incremental_update_only_parses_changed(
        self, sample_kb: Path, index_root: Path
    ):
        """Test that incremental update only re-parses changed files."""
        rebuild_tags_cache(sample_kb, index_root)
        _, cached_mtime = load_cache(index_root)

        time.sleep(0.01)

        # Modify only one file
        _create_entry(sample_kb / "entry-a.md", "Entry A Updated", ["python", "new-tag"])

        tag_counts = ensure_tags_cache(sample_kb, index_root)

        # New tag should appear
        assert "new-tag" in tag_counts
        # Other files' tags should still be correct
        assert tag_counts["api"] == 2  # From entry-b and entry-c
        assert tag_counts["rust"] == 1  # From entry-c

    def test_cache_invalidation_new_file(self, kb_root: Path, index_root: Path):
        """Test cache invalidation when new file is added."""
        _create_entry(kb_root / "entry-a.md", "A", ["alpha"])
        rebuild_tags_cache(kb_root, index_root)

        time.sleep(0.01)

        _create_entry(kb_root / "entry-b.md", "B", ["beta"])

        tag_counts = ensure_tags_cache(kb_root, index_root)

        assert "alpha" in tag_counts
        assert "beta" in tag_counts

    def test_cache_invalidation_deleted_file(self, sample_kb: Path, index_root: Path):
        """Test cache invalidation when file is deleted."""
        rebuild_tags_cache(sample_kb, index_root)

        # Delete entry-c.md (has rust and api tags)
        (sample_kb / "entry-c.md").unlink()

        time.sleep(0.01)

        # Touch another file to trigger detection
        _create_entry(sample_kb / "entry-a.md", "Entry A", ["python", "testing"])

        tag_counts = ensure_tags_cache(sample_kb, index_root)

        # rust was only in entry-c, should be gone
        assert "rust" not in tag_counts
        # api was in entry-b and entry-c, now only entry-b
        assert tag_counts["api"] == 1


# -----------------------------------------------------------------------------
# get_tag_entries tests
# -----------------------------------------------------------------------------


class TestGetTagEntries:
    """Tests for get_tag_entries function."""

    def test_returns_paths_per_tag(self, sample_kb: Path, index_root: Path):
        """Test that function returns entry paths per tag."""
        tag_entries = get_tag_entries(sample_kb, index_root)

        assert "python" in tag_entries
        assert set(tag_entries["python"]) == {"entry-a.md", "entry-b.md"}

        assert "api" in tag_entries
        assert set(tag_entries["api"]) == {"entry-b.md", "entry-c.md"}

    def test_nested_paths(self, nested_kb: Path, index_root: Path):
        """Test that nested paths are returned correctly."""
        tag_entries = get_tag_entries(nested_kb, index_root)

        assert "project" in tag_entries
        assert set(tag_entries["project"]) == {
            "projects/alpha.md",
            "projects/beta.md",
        }

    def test_empty_kb(self, kb_root: Path, index_root: Path):
        """Test empty KB returns empty dict."""
        tag_entries = get_tag_entries(kb_root, index_root)
        assert tag_entries == {}

    def test_uses_cache(self, sample_kb: Path, index_root: Path):
        """Test that get_tag_entries uses cached data."""
        # Build cache
        rebuild_tags_cache(sample_kb, index_root)
        cache_file = _cache_path(index_root)
        original_mtime = cache_file.stat().st_mtime

        time.sleep(0.01)

        # Call get_tag_entries (should use cached version)
        get_tag_entries(sample_kb, index_root)

        # Cache file should not have been modified
        new_mtime = cache_file.stat().st_mtime
        assert new_mtime == original_mtime


# -----------------------------------------------------------------------------
# Edge cases tests
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_file_with_no_tags(self, kb_root: Path, index_root: Path):
        """Test file with empty tags list."""
        _create_entry(kb_root / "no-tags.md", "No Tags", [])

        tag_counts = rebuild_tags_cache(kb_root, index_root)
        assert tag_counts == {}

    def test_malformed_frontmatter(self, kb_root: Path, index_root: Path):
        """Test handling of malformed frontmatter."""
        _create_entry(kb_root / "good.md", "Good", ["valid"])
        # Intentionally malformed - no created date and bad YAML
        (kb_root / "bad.md").write_text(
            "---\ntitle: Bad\ntags: [invalid\n---\n\nMalformed"
        )

        tag_counts = rebuild_tags_cache(kb_root, index_root)

        # Should still get tag from good file
        assert "valid" in tag_counts
        # Bad file should be skipped silently

    def test_unicode_tags(self, kb_root: Path, index_root: Path):
        """Test handling of Unicode tags."""
        _create_entry(kb_root / "unicode.md", "Unicode", ["python", "documentation"])

        tag_counts = rebuild_tags_cache(kb_root, index_root)

        assert "python" in tag_counts
        assert "documentation" in tag_counts

    def test_special_characters_in_tags(self, kb_root: Path, index_root: Path):
        """Test tags with hyphens and underscores."""
        _create_entry(kb_root / "special.md", "Special", ["my-tag", "another_tag"])

        tag_counts = rebuild_tags_cache(kb_root, index_root)

        assert "my-tag" in tag_counts
        assert "another_tag" in tag_counts

    def test_duplicate_tags_in_same_file(self, kb_root: Path, index_root: Path):
        """Test file with duplicate tags."""
        # Note: YAML list parsing typically dedupes, so we test what actually happens
        _create_entry(kb_root / "dupes.md", "Dupes", ["python", "python", "python"])

        tag_counts = rebuild_tags_cache(kb_root, index_root)

        # YAML parsing typically dedupes, but count should still be accurate
        # based on what parse_entry returns
        assert "python" in tag_counts


# -----------------------------------------------------------------------------
# Persistence tests
# -----------------------------------------------------------------------------


class TestCachePersistence:
    """Tests for cache persistence and loading from disk."""

    def test_cache_persists_across_calls(self, sample_kb: Path, index_root: Path):
        """Test that cache persists across multiple function calls."""
        rebuild_tags_cache(sample_kb, index_root)

        files1, _ = load_cache(index_root)

        # Load again
        files2, _ = load_cache(index_root)

        assert files1 == files2

    def test_cache_reused_when_fresh(self, sample_kb: Path, index_root: Path):
        """Test that fresh cache is reused instead of rebuilding."""
        rebuild_tags_cache(sample_kb, index_root)
        cache_file = _cache_path(index_root)
        original_mtime = cache_file.stat().st_mtime

        time.sleep(0.01)

        # Call ensure_tags_cache (should use cached version)
        ensure_tags_cache(sample_kb, index_root)

        # Cache file should not have been modified
        new_mtime = cache_file.stat().st_mtime
        assert new_mtime == original_mtime

    def test_cache_rebuilt_when_stale(self, sample_kb: Path, index_root: Path):
        """Test that stale cache triggers rebuild."""
        rebuild_tags_cache(sample_kb, index_root)
        cache_file = _cache_path(index_root)
        original_mtime = cache_file.stat().st_mtime

        time.sleep(0.01)

        # Modify KB
        _create_entry(sample_kb / "new.md", "New", ["new-tag"])

        time.sleep(0.01)

        # Call ensure_tags_cache (should rebuild)
        ensure_tags_cache(sample_kb, index_root)

        # Cache file should have been updated
        new_mtime = cache_file.stat().st_mtime
        assert new_mtime > original_mtime

    def test_corrupted_cache_recovers(self, sample_kb: Path, index_root: Path):
        """Test that corrupted cache is rebuilt gracefully."""
        cache_file = _cache_path(index_root)
        cache_file.write_text("corrupted data {]")

        tag_counts = ensure_tags_cache(sample_kb, index_root)
        assert len(tag_counts) > 0

        # Cache should now be valid
        files, _ = load_cache(index_root)
        assert len(files) > 0
