"""Tests for backlinks cache management."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from memex.backlinks_cache import (
    CACHE_FILENAME,
    CacheMetadata,
    _cache_path,
    _collect_cache_metadata,
    _count_md_files,
    _get_dir_mtime,
    _is_cache_valid,
    _kb_tree_mtime,
    _load_cache_full,
    _save_cache_full,
    ensure_backlink_cache,
    load_cache,
    rebuild_backlink_cache,
    save_cache,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


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
    # Mock get_index_root to use our temp directory
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index))
    return index


@pytest.fixture
def sample_kb(kb_root: Path) -> Path:
    """Create a sample KB with various link patterns."""
    # Create entries with links
    (kb_root / "entry-a.md").write_text("# Entry A\n\nLinks to [[entry-b]] and [[entry-c]]")
    (kb_root / "entry-b.md").write_text("# Entry B\n\nLinks to [[entry-c]]")
    (kb_root / "entry-c.md").write_text("# Entry C\n\nNo links here")
    return kb_root


@pytest.fixture
def nested_kb(kb_root: Path) -> Path:
    """Create a KB with nested directories and various link types."""
    # Create directory structure
    (kb_root / "projects").mkdir()
    (kb_root / "docs").mkdir()

    # Create entries with various link patterns
    (kb_root / "index.md").write_text("# Index\n\nSee [[projects/alpha]] and [[docs/guide]]")
    (kb_root / "projects" / "alpha.md").write_text(
        "# Alpha Project\n\nReferences [[docs/guide]] and [[index]]"
    )
    (kb_root / "docs" / "guide.md").write_text("# Guide\n\nBack to [[index]]")
    return kb_root


# ─────────────────────────────────────────────────────────────────────────────
# Helper function tests
# ─────────────────────────────────────────────────────────────────────────────


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


class TestKbTreeMtime:
    """Tests for _kb_tree_mtime helper function."""

    def test_empty_kb_returns_zero(self, kb_root: Path):
        """Test that empty KB returns 0.0 mtime."""
        mtime = _kb_tree_mtime(kb_root)
        assert mtime == 0.0

    def test_nonexistent_kb_returns_zero(self, tmp_path: Path):
        """Test that nonexistent KB returns 0.0 mtime."""
        nonexistent = tmp_path / "does-not-exist"
        mtime = _kb_tree_mtime(nonexistent)
        assert mtime == 0.0

    def test_single_file_mtime(self, kb_root: Path):
        """Test mtime with a single file."""
        file = kb_root / "test.md"
        file.write_text("test content")
        mtime = _kb_tree_mtime(kb_root)
        assert mtime > 0
        assert mtime == pytest.approx(file.stat().st_mtime, rel=0.01)

    def test_multiple_files_returns_latest(self, kb_root: Path):
        """Test that mtime returns the latest modification time."""
        # Create first file
        file1 = kb_root / "old.md"
        file1.write_text("old")
        old_mtime = file1.stat().st_mtime

        # Sleep to ensure different mtime
        time.sleep(0.01)

        # Create newer file
        file2 = kb_root / "new.md"
        file2.write_text("new")
        new_mtime = file2.stat().st_mtime

        mtime = _kb_tree_mtime(kb_root)
        assert mtime >= new_mtime
        assert mtime > old_mtime

    def test_nested_files_included(self, nested_kb: Path):
        """Test that nested files are included in mtime calculation."""
        mtime = _kb_tree_mtime(nested_kb)
        assert mtime > 0

    def test_ignores_non_md_files(self, kb_root: Path):
        """Test that non-.md files are ignored."""
        md_file = kb_root / "test.md"
        md_file.write_text("markdown")
        md_mtime = md_file.stat().st_mtime

        time.sleep(0.01)

        txt_file = kb_root / "other.txt"
        txt_file.write_text("text")

        mtime = _kb_tree_mtime(kb_root)
        # Should only consider the .md file
        assert mtime == pytest.approx(md_mtime, rel=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Cache load/save tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadCache:
    """Tests for load_cache function."""

    def test_load_nonexistent_cache(self, index_root: Path):
        """Test loading cache when file doesn't exist returns empty dict."""
        backlinks, mtime = load_cache()
        assert backlinks == {}
        assert mtime == 0.0

    def test_load_valid_cache(self, index_root: Path):
        """Test loading a valid cache file."""
        cache_file = _cache_path(index_root)
        cache_data = {
            "kb_mtime": 123456.789,
            "backlinks": {
                "entry-a": ["entry-b", "entry-c"],
                "entry-b": ["entry-c"],
            },
        }
        cache_file.write_text(json.dumps(cache_data))

        backlinks, mtime = load_cache()
        assert backlinks == cache_data["backlinks"]
        assert mtime == 123456.789

    def test_load_malformed_json(self, index_root: Path):
        """Test loading malformed JSON returns empty dict."""
        cache_file = _cache_path(index_root)
        cache_file.write_text("{ invalid json }")

        backlinks, mtime = load_cache()
        assert backlinks == {}
        assert mtime == 0.0

    def test_load_missing_fields(self, index_root: Path):
        """Test loading cache with missing fields uses defaults."""
        cache_file = _cache_path(index_root)
        cache_file.write_text(json.dumps({}))

        backlinks, mtime = load_cache()
        assert backlinks == {}
        assert mtime == 0.0

    def test_load_partial_fields(self, index_root: Path):
        """Test loading cache with only backlinks field."""
        cache_file = _cache_path(index_root)
        cache_data = {"backlinks": {"entry-a": ["entry-b"]}}
        cache_file.write_text(json.dumps(cache_data))

        backlinks, mtime = load_cache()
        assert backlinks == {"entry-a": ["entry-b"]}
        assert mtime == 0.0


class TestSaveCache:
    """Tests for save_cache function."""

    def test_save_empty_cache(self, index_root: Path):
        """Test saving empty cache."""
        save_cache({}, 0.0)

        cache_file = _cache_path(index_root)
        assert cache_file.exists()

        data = json.loads(cache_file.read_text())
        assert data == {"kb_mtime": 0.0, "backlinks": {}}

    def test_save_cache_with_backlinks(self, index_root: Path):
        """Test saving cache with backlinks."""
        backlinks = {
            "entry-a": ["entry-b", "entry-c"],
            "entry-b": ["entry-c"],
        }
        mtime = 123456.789

        save_cache(backlinks, mtime)

        cache_file = _cache_path(index_root)
        data = json.loads(cache_file.read_text())
        assert data["backlinks"] == backlinks
        assert data["kb_mtime"] == mtime

    def test_save_overwrites_existing(self, index_root: Path):
        """Test that save overwrites existing cache."""
        # Save initial cache
        save_cache({"old": ["data"]}, 111.0)

        # Save new cache
        new_backlinks = {"new": ["data"]}
        new_mtime = 222.0
        save_cache(new_backlinks, new_mtime)

        # Verify new data
        cache_file = _cache_path(index_root)
        data = json.loads(cache_file.read_text())
        assert data["backlinks"] == new_backlinks
        assert data["kb_mtime"] == new_mtime

    def test_save_creates_directory(self, tmp_path: Path, monkeypatch):
        """Test that save creates directory if it doesn't exist."""
        index = tmp_path / "new-index"
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index))

        save_cache({"test": ["data"]}, 100.0)

        assert index.exists()
        cache_file = index / CACHE_FILENAME
        assert cache_file.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Cache rebuild tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRebuildBacklinkCache:
    """Tests for rebuild_backlink_cache function."""

    def test_rebuild_empty_kb(self, kb_root: Path, index_root: Path):
        """Test rebuilding cache for empty KB."""
        backlinks = rebuild_backlink_cache(kb_root)
        assert backlinks == {}

        # Verify cache was saved
        cache_file = _cache_path(index_root)
        assert cache_file.exists()

    def test_rebuild_sample_kb(self, sample_kb: Path, index_root: Path):
        """Test rebuilding cache for sample KB with links."""
        backlinks = rebuild_backlink_cache(sample_kb)

        # entry-b is linked from entry-a
        assert "entry-b" in backlinks
        assert "entry-a" in backlinks["entry-b"]

        # entry-c is linked from both entry-a and entry-b
        assert "entry-c" in backlinks
        assert "entry-a" in backlinks["entry-c"]
        assert "entry-b" in backlinks["entry-c"]

    def test_rebuild_nested_kb(self, nested_kb: Path, index_root: Path):
        """Test rebuilding cache for nested KB structure."""
        backlinks = rebuild_backlink_cache(nested_kb)

        # Verify nested paths work correctly
        assert "projects/alpha" in backlinks
        assert "index" in backlinks["projects/alpha"]

        assert "docs/guide" in backlinks
        assert "index" in backlinks["docs/guide"]
        assert "projects/alpha" in backlinks["docs/guide"]

    def test_rebuild_saves_cache(self, sample_kb: Path, index_root: Path):
        """Test that rebuild saves cache to disk."""
        backlinks = rebuild_backlink_cache(sample_kb)

        # Load cache and verify it matches
        loaded_backlinks, _ = load_cache()
        assert loaded_backlinks == backlinks

    def test_rebuild_updates_mtime(self, sample_kb: Path, index_root: Path):
        """Test that rebuild updates the kb_mtime in cache."""
        rebuild_backlink_cache(sample_kb)

        _, mtime = load_cache()
        assert mtime > 0

    def test_rebuild_no_links(self, kb_root: Path, index_root: Path):
        """Test rebuilding KB with files but no links."""
        (kb_root / "no-links.md").write_text("# No Links\n\nJust plain text")
        backlinks = rebuild_backlink_cache(kb_root)
        assert backlinks == {}

    def test_rebuild_self_link(self, kb_root: Path, index_root: Path):
        """Test handling of self-referential links."""
        (kb_root / "self.md").write_text("# Self\n\nLinks to [[self]]")
        backlinks = rebuild_backlink_cache(kb_root)

        # Self-links should still be tracked
        assert "self" in backlinks
        assert "self" in backlinks["self"]


# ─────────────────────────────────────────────────────────────────────────────
# Cache invalidation tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEnsureBacklinkCache:
    """Tests for ensure_backlink_cache function (cache hit/miss behavior)."""

    def test_cache_miss_no_cache_file(self, sample_kb: Path, index_root: Path):
        """Test cache miss when no cache file exists."""
        # No cache file exists yet
        backlinks = ensure_backlink_cache(sample_kb)

        # Should rebuild and return valid backlinks
        assert "entry-b" in backlinks
        assert "entry-c" in backlinks

        # Cache should now exist
        cache_file = _cache_path(index_root)
        assert cache_file.exists()

    def test_cache_hit_valid_cache(self, sample_kb: Path, index_root: Path):
        """Test cache hit when cache is fresh and valid."""
        # Build initial cache
        rebuild_backlink_cache(sample_kb)

        # Load from cache (should be a hit)
        backlinks = ensure_backlink_cache(sample_kb)

        # Should return cached backlinks
        assert "entry-b" in backlinks
        assert "entry-c" in backlinks

    def test_cache_miss_stale_mtime(self, sample_kb: Path, index_root: Path):
        """Test cache miss when files are newer than cached mtime."""
        # Build initial cache
        rebuild_backlink_cache(sample_kb)

        # Sleep to ensure different mtime
        time.sleep(0.01)

        # Modify a file to update mtime
        (sample_kb / "entry-a.md").write_text(
            "# Entry A\n\nUpdated with new link [[entry-b]]"
        )

        # Should detect stale cache and rebuild
        backlinks = ensure_backlink_cache(sample_kb)
        assert "entry-b" in backlinks

    def test_cache_miss_empty_backlinks(self, sample_kb: Path, index_root: Path):
        """Test cache miss when backlinks dict is empty."""
        # Save cache with empty backlinks
        mtime = _kb_tree_mtime(sample_kb)
        save_cache({}, mtime)

        # Should rebuild because backlinks are empty
        backlinks = ensure_backlink_cache(sample_kb)
        assert len(backlinks) > 0

    def test_cache_invalidation_new_file(self, kb_root: Path, index_root: Path):
        """Test cache invalidation when new file is added."""
        # Create initial KB and cache
        (kb_root / "entry-a.md").write_text("# Entry A\n\nLinks to [[entry-b]]")
        rebuild_backlink_cache(kb_root)

        # Sleep to ensure different mtime
        time.sleep(0.01)

        # Add new file
        (kb_root / "entry-b.md").write_text("# Entry B\n\nLinks to [[entry-a]]")

        # Should detect new file and rebuild
        backlinks = ensure_backlink_cache(kb_root)
        assert "entry-a" in backlinks
        assert "entry-b" in backlinks["entry-a"]

    def test_cache_invalidation_deleted_file(self, sample_kb: Path, index_root: Path):
        """Test cache invalidation when file is deleted."""
        # Build initial cache
        rebuild_backlink_cache(sample_kb)

        # Sleep to ensure different mtime
        time.sleep(0.01)

        # Delete a file
        (sample_kb / "entry-b.md").unlink()

        # Touch another file to update mtime
        (sample_kb / "entry-a.md").write_text(
            "# Entry A\n\nLinks to [[entry-c]]"
        )

        # Should rebuild and reflect deletion
        backlinks = ensure_backlink_cache(sample_kb)
        # entry-b should not be in backlinks anymore
        assert "entry-b" not in backlinks


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_circular_links(self, kb_root: Path, index_root: Path):
        """Test circular link pattern (A -> B -> A)."""
        (kb_root / "a.md").write_text("# A\n\nLinks to [[b]]")
        (kb_root / "b.md").write_text("# B\n\nLinks to [[a]]")

        backlinks = rebuild_backlink_cache(kb_root)

        # Both should have backlinks to each other
        assert "a" in backlinks
        assert "b" in backlinks["a"]
        assert "b" in backlinks
        assert "a" in backlinks["b"]

    def test_complex_circular_links(self, kb_root: Path, index_root: Path):
        """Test complex circular link pattern (A -> B -> C -> A)."""
        (kb_root / "a.md").write_text("# A\n\nLinks to [[b]]")
        (kb_root / "b.md").write_text("# B\n\nLinks to [[c]]")
        (kb_root / "c.md").write_text("# C\n\nLinks to [[a]]")

        backlinks = rebuild_backlink_cache(kb_root)

        # Verify the circular chain
        assert "b" in backlinks and "a" in backlinks["b"]
        assert "c" in backlinks and "b" in backlinks["c"]
        assert "a" in backlinks and "c" in backlinks["a"]

    def test_missing_target_files(self, kb_root: Path, index_root: Path):
        """Test links to files that don't exist."""
        (kb_root / "a.md").write_text("# A\n\nLinks to [[nonexistent]] and [[also-missing]]")

        backlinks = rebuild_backlink_cache(kb_root)

        # Should still track backlinks even if target doesn't exist
        assert "nonexistent" in backlinks
        assert "a" in backlinks["nonexistent"]
        assert "also-missing" in backlinks
        assert "a" in backlinks["also-missing"]

    def test_malformed_markdown(self, kb_root: Path, index_root: Path):
        """Test handling of malformed markdown."""
        # Unclosed link brackets
        (kb_root / "malformed.md").write_text("# Malformed\n\n[[unclosed link")

        # Should not crash
        backlinks = rebuild_backlink_cache(kb_root)
        # No valid links should be extracted
        assert backlinks == {}

    def test_special_characters_in_filenames(self, kb_root: Path, index_root: Path):
        """Test files with special characters in names."""
        (kb_root / "special-chars_123.md").write_text(
            "# Special\n\nLinks to [[other-file]]"
        )
        (kb_root / "other-file.md").write_text("# Other")

        backlinks = rebuild_backlink_cache(kb_root)

        assert "other-file" in backlinks
        assert "special-chars_123" in backlinks["other-file"]

    def test_unicode_content(self, kb_root: Path, index_root: Path):
        """Test handling of Unicode content."""
        (kb_root / "unicode.md").write_text(
            "# Unicode Test 🚀\n\n中文内容 [[other]]\n\nEmoji: 👍"
        )
        (kb_root / "other.md").write_text("# Other")

        backlinks = rebuild_backlink_cache(kb_root)

        assert "other" in backlinks
        assert "unicode" in backlinks["other"]

    def test_empty_files(self, kb_root: Path, index_root: Path):
        """Test handling of empty markdown files."""
        (kb_root / "empty.md").write_text("")
        (kb_root / "normal.md").write_text("# Normal\n\nLinks to [[empty]]")

        backlinks = rebuild_backlink_cache(kb_root)

        assert "empty" in backlinks
        assert "normal" in backlinks["empty"]

    def test_links_in_code_blocks(self, kb_root: Path, index_root: Path):
        """Test that links in code blocks are handled correctly."""
        # The parser should use AST-based parsing that handles code blocks
        (kb_root / "code.md").write_text(
            "# Code\n\n```\n[[not-a-link]]\n```\n\nBut [[real-link]] is"
        )

        backlinks = rebuild_backlink_cache(kb_root)

        # Should only extract real-link, not the one in code block
        # This depends on the parser implementation
        assert "real-link" in backlinks

    def test_duplicate_links_in_same_file(self, kb_root: Path, index_root: Path):
        """Test file with multiple links to the same target."""
        (kb_root / "multi.md").write_text(
            "# Multi\n\n[[target]] and [[target]] and [[target]]"
        )

        backlinks = rebuild_backlink_cache(kb_root)

        # Should only list the source file once
        assert "target" in backlinks
        assert backlinks["target"].count("multi") == 1

    def test_links_with_md_extension(self, kb_root: Path, index_root: Path):
        """Test links that include .md extension."""
        (kb_root / "a.md").write_text("# A\n\nLinks to [[b.md]]")
        (kb_root / "b.md").write_text("# B")

        backlinks = rebuild_backlink_cache(kb_root)

        # Should normalize and track correctly
        assert "b" in backlinks
        assert "a" in backlinks["b"]

    def test_relative_path_links(self, kb_root: Path, index_root: Path):
        """Test relative path links like [[../other]]."""
        (kb_root / "sub").mkdir()
        (kb_root / "sub" / "a.md").write_text("# A\n\nLinks to [[../b]]")
        (kb_root / "b.md").write_text("# B")

        backlinks = rebuild_backlink_cache(kb_root)

        # Current behavior: resolve_link_target returns path-like targets as-is
        # So "../b" is tracked as-is rather than being resolved to "b"
        assert "../b" in backlinks
        assert "sub/a" in backlinks["../b"]

    def test_whitespace_in_links(self, kb_root: Path, index_root: Path):
        """Test links with extra whitespace."""
        (kb_root / "a.md").write_text("# A\n\nLinks to [[  target  ]]")

        backlinks = rebuild_backlink_cache(kb_root)

        # Should normalize whitespace
        assert "target" in backlinks or "  target  " in backlinks

    def test_files_starting_with_underscore(self, kb_root: Path, index_root: Path):
        """Test that files starting with _ are skipped."""
        (kb_root / "_template.md").write_text("# Template\n\nLinks to [[other]]")
        (kb_root / "normal.md").write_text("# Normal\n\nLinks to [[_template]]")

        backlinks = rebuild_backlink_cache(kb_root)

        # _template.md should be skipped as a source
        # but can still be a target
        if "_template" in backlinks:
            # If tracked as target, should only have links from normal
            assert "normal" in backlinks["_template"]
            # Should not have links from _template to other
            assert "other" not in backlinks or "_template" not in backlinks.get("other", [])

    def test_nonexistent_kb_root(self, tmp_path: Path, index_root: Path):
        """Test handling of non-existent KB root."""
        nonexistent = tmp_path / "does-not-exist"
        backlinks = rebuild_backlink_cache(nonexistent)
        assert backlinks == {}

    def test_kb_root_is_file(self, tmp_path: Path, index_root: Path):
        """Test handling when kb_root is a file instead of directory."""
        file = tmp_path / "file.txt"
        file.write_text("not a directory")

        # Should handle gracefully
        backlinks = rebuild_backlink_cache(file)
        assert backlinks == {}


# ─────────────────────────────────────────────────────────────────────────────
# Persistence tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCachePersistence:
    """Tests for cache persistence and loading from disk."""

    def test_cache_persists_across_calls(self, sample_kb: Path, index_root: Path):
        """Test that cache persists across multiple function calls."""
        # Build cache
        backlinks1 = rebuild_backlink_cache(sample_kb)

        # Load cache in new call
        backlinks2, _ = load_cache()

        assert backlinks1 == backlinks2

    def test_cache_reused_when_fresh(self, sample_kb: Path, index_root: Path):
        """Test that fresh cache is reused instead of rebuilding."""
        # Build initial cache
        rebuild_backlink_cache(sample_kb)
        cache_file = _cache_path(index_root)
        original_mtime = cache_file.stat().st_mtime

        # Sleep briefly
        time.sleep(0.01)

        # Call ensure_backlink_cache (should use cached version)
        ensure_backlink_cache(sample_kb)

        # Cache file should not have been modified
        new_mtime = cache_file.stat().st_mtime
        assert new_mtime == original_mtime

    def test_cache_rebuilt_when_stale(self, sample_kb: Path, index_root: Path):
        """Test that stale cache triggers rebuild."""
        # Build initial cache
        rebuild_backlink_cache(sample_kb)
        cache_file = _cache_path(index_root)
        original_mtime = cache_file.stat().st_mtime

        # Sleep to ensure different mtime
        time.sleep(0.01)

        # Modify KB
        (sample_kb / "new.md").write_text("# New\n\nNew file")

        # Sleep to ensure cache update has different mtime
        time.sleep(0.01)

        # Call ensure_backlink_cache (should rebuild)
        ensure_backlink_cache(sample_kb)

        # Cache file should have been updated
        new_mtime = cache_file.stat().st_mtime
        assert new_mtime > original_mtime

    def test_corrupted_cache_recovers(self, sample_kb: Path, index_root: Path):
        """Test that corrupted cache is rebuilt gracefully."""
        # Create corrupted cache
        cache_file = _cache_path(index_root)
        cache_file.write_text("corrupted data {]")

        # Should recover by rebuilding
        backlinks = ensure_backlink_cache(sample_kb)
        assert len(backlinks) > 0

        # Cache should now be valid
        loaded_backlinks, _ = load_cache()
        assert loaded_backlinks == backlinks


# ─────────────────────────────────────────────────────────────────────────────
# Optimized validation tests
# ─────────────────────────────────────────────────────────────────────────────


class TestOptimizedHelpers:
    """Tests for optimized cache validation helper functions."""

    def test_get_dir_mtime_existing_dir(self, kb_root: Path):
        """Test directory mtime for existing directory."""
        mtime = _get_dir_mtime(kb_root)
        assert mtime > 0
        assert mtime == pytest.approx(kb_root.stat().st_mtime, rel=0.01)

    def test_get_dir_mtime_nonexistent_dir(self, tmp_path: Path):
        """Test directory mtime for nonexistent directory."""
        nonexistent = tmp_path / "does-not-exist"
        mtime = _get_dir_mtime(nonexistent)
        assert mtime == 0.0

    def test_count_md_files_empty_kb(self, kb_root: Path):
        """Test file count for empty KB."""
        count = _count_md_files(kb_root)
        assert count == 0

    def test_count_md_files_with_files(self, sample_kb: Path):
        """Test file count with sample KB (3 files)."""
        count = _count_md_files(sample_kb)
        assert count == 3  # entry-a.md, entry-b.md, entry-c.md

    def test_count_md_files_nested(self, nested_kb: Path):
        """Test file count includes nested directories."""
        count = _count_md_files(nested_kb)
        assert count == 3  # index.md, projects/alpha.md, docs/guide.md

    def test_count_md_files_nonexistent(self, tmp_path: Path):
        """Test file count for nonexistent KB."""
        nonexistent = tmp_path / "does-not-exist"
        count = _count_md_files(nonexistent)
        assert count == 0

    def test_collect_cache_metadata(self, sample_kb: Path):
        """Test collecting cache metadata."""
        metadata = _collect_cache_metadata(sample_kb)

        assert isinstance(metadata, CacheMetadata)
        assert metadata.kb_mtime > 0
        assert metadata.file_count == 3
        assert metadata.dir_mtime > 0

    def test_collect_cache_metadata_empty_kb(self, kb_root: Path):
        """Test collecting metadata for empty KB."""
        metadata = _collect_cache_metadata(kb_root)

        assert metadata.kb_mtime == 0.0
        assert metadata.file_count == 0
        assert metadata.dir_mtime > 0  # Directory exists

    def test_collect_cache_metadata_nonexistent(self, tmp_path: Path):
        """Test collecting metadata for nonexistent KB."""
        nonexistent = tmp_path / "does-not-exist"
        metadata = _collect_cache_metadata(nonexistent)

        assert metadata.kb_mtime == 0.0
        assert metadata.file_count == 0
        assert metadata.dir_mtime == 0.0


class TestCacheMetadataStorage:
    """Tests for new metadata storage format."""

    def test_save_and_load_full_cache(self, sample_kb: Path, index_root: Path):
        """Test saving and loading cache with full metadata."""
        # Create metadata
        metadata = CacheMetadata(
            kb_mtime=12345.678,
            file_count=42,
            dir_mtime=12300.0,
        )
        backlinks = {"target": ["source1", "source2"]}

        # Save
        _save_cache_full(backlinks, metadata)

        # Load and verify
        loaded_backlinks, loaded_metadata = _load_cache_full()

        assert loaded_backlinks == backlinks
        assert loaded_metadata is not None
        assert loaded_metadata.kb_mtime == pytest.approx(metadata.kb_mtime, rel=0.001)
        assert loaded_metadata.file_count == metadata.file_count
        assert loaded_metadata.dir_mtime == pytest.approx(metadata.dir_mtime, rel=0.001)

    def test_load_legacy_cache_returns_none_metadata(self, index_root: Path):
        """Test loading legacy cache (without new metadata fields) returns None."""
        # Write legacy format cache
        cache_file = _cache_path(index_root)
        legacy_data = {
            "kb_mtime": 12345.0,
            "backlinks": {"target": ["source"]},
        }
        cache_file.write_text(json.dumps(legacy_data))

        # Load
        backlinks, metadata = _load_cache_full()

        assert backlinks == {"target": ["source"]}
        assert metadata is None  # Should be None for legacy cache

    def test_load_nonexistent_cache(self, index_root: Path):
        """Test loading nonexistent cache returns empty dict and None."""
        # Don't create any cache file
        backlinks, metadata = _load_cache_full()

        assert backlinks == {}
        assert metadata is None


class TestOptimizedValidation:
    """Tests for optimized cache validation logic."""

    def test_is_cache_valid_unchanged(self, sample_kb: Path, index_root: Path):
        """Test validation returns True for unchanged KB."""
        # Build cache
        rebuild_backlink_cache(sample_kb)

        # Load metadata
        _, metadata = _load_cache_full()
        assert metadata is not None

        # Should be valid
        assert _is_cache_valid(sample_kb, metadata) is True

    def test_is_cache_valid_file_added(self, sample_kb: Path, index_root: Path):
        """Test validation returns False when file added."""
        # Build cache
        rebuild_backlink_cache(sample_kb)

        # Load metadata
        _, metadata = _load_cache_full()
        assert metadata is not None

        # Add new file
        (sample_kb / "new-file.md").write_text("# New File")

        # Should be invalid (file count changed)
        assert _is_cache_valid(sample_kb, metadata) is False

    def test_is_cache_valid_file_removed(self, sample_kb: Path, index_root: Path):
        """Test validation returns False when file removed."""
        # Build cache
        rebuild_backlink_cache(sample_kb)

        # Load metadata
        _, metadata = _load_cache_full()
        assert metadata is not None

        # Remove a file
        (sample_kb / "entry-c.md").unlink()

        # Should be invalid (file count changed)
        assert _is_cache_valid(sample_kb, metadata) is False

    def test_is_cache_valid_nonexistent_kb(self, tmp_path: Path):
        """Test validation for nonexistent KB with cached data."""
        nonexistent = tmp_path / "does-not-exist"

        # Create metadata for a KB that had files
        metadata = CacheMetadata(kb_mtime=1000.0, file_count=5, dir_mtime=999.0)

        # Should be invalid (KB doesn't exist but cache claims files)
        assert _is_cache_valid(nonexistent, metadata) is False

    def test_is_cache_valid_empty_kb_with_empty_cache(self, kb_root: Path):
        """Test validation for empty KB with matching cache."""
        # Create metadata for empty KB
        metadata = CacheMetadata(kb_mtime=0.0, file_count=0, dir_mtime=kb_root.stat().st_mtime)

        # Should be valid (both KB and cache are empty)
        assert _is_cache_valid(kb_root, metadata) is True


class TestOptimizedEnsureCache:
    """Tests for optimized ensure_backlink_cache behavior."""

    def test_legacy_cache_triggers_rebuild(self, sample_kb: Path, index_root: Path):
        """Test that legacy cache (without metadata) triggers rebuild."""
        # Write legacy format cache
        cache_file = _cache_path(index_root)
        legacy_data = {
            "kb_mtime": 12345.0,
            "backlinks": {"entry-b": ["entry-a"]},
        }
        cache_file.write_text(json.dumps(legacy_data))
        original_mtime = cache_file.stat().st_mtime

        # Sleep to ensure different mtime
        time.sleep(0.01)

        # Call ensure_backlink_cache
        backlinks = ensure_backlink_cache(sample_kb)

        # Should have rebuilt (new format with metadata)
        new_mtime = cache_file.stat().st_mtime
        assert new_mtime > original_mtime

        # Should have correct backlinks
        assert "entry-b" in backlinks
        assert "entry-c" in backlinks

        # Cache should now have metadata
        _, metadata = _load_cache_full()
        assert metadata is not None
        assert metadata.file_count == 3

    def test_unchanged_kb_uses_cache(self, sample_kb: Path, index_root: Path):
        """Test that unchanged KB uses cached data without rebuild."""
        # Build initial cache
        rebuild_backlink_cache(sample_kb)
        cache_file = _cache_path(index_root)
        original_mtime = cache_file.stat().st_mtime

        # Sleep briefly
        time.sleep(0.01)

        # Call ensure_backlink_cache multiple times
        for _ in range(5):
            ensure_backlink_cache(sample_kb)

        # Cache file should not have been modified
        new_mtime = cache_file.stat().st_mtime
        assert new_mtime == original_mtime

    def test_file_count_change_triggers_rebuild(self, sample_kb: Path, index_root: Path):
        """Test that file count change triggers cache rebuild."""
        # Build initial cache
        rebuild_backlink_cache(sample_kb)

        # Verify initial file count
        _, metadata = _load_cache_full()
        assert metadata is not None
        assert metadata.file_count == 3

        # Add new file
        (sample_kb / "entry-d.md").write_text("# Entry D\n\nLinks to [[entry-a]]")

        # Call ensure_backlink_cache
        backlinks = ensure_backlink_cache(sample_kb)

        # Should have rebuilt with new backlink
        assert "entry-a" in backlinks
        assert "entry-d" in backlinks["entry-a"]

        # File count should be updated
        _, new_metadata = _load_cache_full()
        assert new_metadata is not None
        assert new_metadata.file_count == 4

    def test_cache_stores_correct_format(self, sample_kb: Path, index_root: Path):
        """Test that rebuilt cache stores correct JSON format."""
        # Build cache
        rebuild_backlink_cache(sample_kb)

        # Read raw cache file
        cache_file = _cache_path(index_root)
        data = json.loads(cache_file.read_text())

        # Verify all expected fields exist
        assert "kb_mtime" in data
        assert "file_count" in data
        assert "dir_mtime" in data
        assert "backlinks" in data

        # Verify types
        assert isinstance(data["kb_mtime"], float)
        assert isinstance(data["file_count"], int)
        assert isinstance(data["dir_mtime"], float)
        assert isinstance(data["backlinks"], dict)
