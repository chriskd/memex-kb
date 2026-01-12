"""Tests for memex cache systems: backlinks, tags, title index, and health cache.

Focuses on cache behaviors: build, query, invalidate, and consistency after updates.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from conftest import create_entry


# =============================================================================
# TestBacklinksCache - Build, Query, Invalidate
# =============================================================================


class TestBacklinksCache:
    """Tests for backlinks cache behavior."""

    def test_backlinks_detected(self, tmp_kb: Path):
        """Backlinks are detected when one entry links to another."""
        from memex.backlinks_cache import rebuild_backlink_cache

        create_entry(tmp_kb, "a.md", "Entry A", "Link to [[b]]")
        create_entry(tmp_kb, "b.md", "Entry B", "Content")

        backlinks = rebuild_backlink_cache(tmp_kb)

        assert "b" in backlinks
        assert "a" in backlinks["b"]

    def test_backlinks_empty_for_unlinked(self, tmp_kb: Path):
        """Entry with no inbound links has no backlinks."""
        from memex.backlinks_cache import rebuild_backlink_cache

        create_entry(tmp_kb, "orphan.md", "Orphan", "No links to me")
        create_entry(tmp_kb, "other.md", "Other", "No links at all")

        backlinks = rebuild_backlink_cache(tmp_kb)

        assert "orphan" not in backlinks
        assert "other" not in backlinks

    def test_backlinks_multiple_sources(self, tmp_kb: Path):
        """Entry linked from multiple sources tracks all backlinks."""
        from memex.backlinks_cache import rebuild_backlink_cache

        create_entry(tmp_kb, "a.md", "Entry A", "Link to [[target]]")
        create_entry(tmp_kb, "b.md", "Entry B", "Also links to [[target]]")
        create_entry(tmp_kb, "target.md", "Target", "I am the target")

        backlinks = rebuild_backlink_cache(tmp_kb)

        assert "target" in backlinks
        assert "a" in backlinks["target"]
        assert "b" in backlinks["target"]

    def test_backlinks_cache_invalidated_on_change(self, tmp_kb: Path, monkeypatch):
        """Cache is rebuilt when files change."""
        from memex.backlinks_cache import ensure_backlink_cache, rebuild_backlink_cache

        index_root = tmp_kb / ".kb-indices"
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        create_entry(tmp_kb, "a.md", "Entry A", "Link to [[b]]")
        create_entry(tmp_kb, "b.md", "Entry B", "Content")

        # Build initial cache
        rebuild_backlink_cache(tmp_kb)

        time.sleep(0.02)

        # Modify a file to trigger invalidation
        create_entry(tmp_kb, "c.md", "Entry C", "New link to [[b]]")

        backlinks = ensure_backlink_cache(tmp_kb)

        # Should detect the new backlink
        assert "c" in backlinks["b"]

    def test_backlinks_nested_paths(self, tmp_kb: Path):
        """Backlinks work with nested directory paths."""
        from memex.backlinks_cache import rebuild_backlink_cache

        (tmp_kb / "docs").mkdir()
        create_entry(tmp_kb, "index.md", "Index", "See [[docs/guide]]")
        create_entry(tmp_kb, "docs/guide.md", "Guide", "Back to [[index]]")

        backlinks = rebuild_backlink_cache(tmp_kb)

        assert "docs/guide" in backlinks
        assert "index" in backlinks["docs/guide"]
        assert "index" in backlinks
        assert "docs/guide" in backlinks["index"]


# =============================================================================
# TestTagsCache - Build, Query, Update
# =============================================================================


class TestTagsCache:
    """Tests for tags cache behavior."""

    def test_find_entries_by_tag(self, tmp_kb: Path, monkeypatch):
        """Tags cache correctly indexes entries by tag."""
        from memex.tags_cache import get_tag_entries

        index_root = tmp_kb / ".kb-indices"
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        create_entry(tmp_kb, "python-tips.md", "Python Tips", "Tips", ["python", "tips"])
        create_entry(tmp_kb, "rust-guide.md", "Rust Guide", "Guide", ["rust", "guide"])
        create_entry(tmp_kb, "testing.md", "Testing", "Tests", ["testing", "python"])

        tag_entries = get_tag_entries(tmp_kb, index_root)

        assert "python" in tag_entries
        assert len(tag_entries["python"]) == 2
        assert "python-tips.md" in tag_entries["python"]
        assert "testing.md" in tag_entries["python"]

    def test_tag_counts_accurate(self, tmp_kb: Path, monkeypatch):
        """Tags cache returns accurate counts per tag."""
        from memex.tags_cache import rebuild_tags_cache

        index_root = tmp_kb / ".kb-indices"
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        create_entry(tmp_kb, "a.md", "A", "Content", ["common", "unique-a"])
        create_entry(tmp_kb, "b.md", "B", "Content", ["common", "unique-b"])
        create_entry(tmp_kb, "c.md", "C", "Content", ["common"])

        counts = rebuild_tags_cache(tmp_kb, index_root)

        assert counts["common"] == 3
        assert counts["unique-a"] == 1
        assert counts["unique-b"] == 1

    def test_tags_cache_updated_on_file_change(self, tmp_kb: Path, monkeypatch):
        """Tags cache reflects file modifications."""
        from memex.tags_cache import ensure_tags_cache, rebuild_tags_cache

        index_root = tmp_kb / ".kb-indices"
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        create_entry(tmp_kb, "entry.md", "Entry", "Content", ["original"])
        rebuild_tags_cache(tmp_kb, index_root)

        time.sleep(0.02)

        # Update the entry with new tags
        create_entry(tmp_kb, "entry.md", "Entry", "Content", ["updated", "new"])

        counts = ensure_tags_cache(tmp_kb, index_root)

        assert "original" not in counts
        assert "updated" in counts
        assert "new" in counts

    def test_tags_cache_handles_deletion(self, tmp_kb: Path, monkeypatch):
        """Tags cache removes deleted file's tags."""
        from memex.tags_cache import ensure_tags_cache, rebuild_tags_cache

        index_root = tmp_kb / ".kb-indices"
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        create_entry(tmp_kb, "keep.md", "Keep", "Content", ["shared", "keep-only"])
        create_entry(tmp_kb, "delete.md", "Delete", "Content", ["shared", "delete-only"])
        rebuild_tags_cache(tmp_kb, index_root)

        # Delete one file
        (tmp_kb / "delete.md").unlink()

        counts = ensure_tags_cache(tmp_kb, index_root)

        assert "delete-only" not in counts
        assert counts["shared"] == 1
        assert counts["keep-only"] == 1

    def test_tags_cache_skips_underscore_files(self, tmp_kb: Path, monkeypatch):
        """Files starting with underscore are not indexed."""
        from memex.tags_cache import rebuild_tags_cache

        index_root = tmp_kb / ".kb-indices"
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        create_entry(tmp_kb, "_template.md", "Template", "Content", ["template"])
        create_entry(tmp_kb, "normal.md", "Normal", "Content", ["normal"])

        counts = rebuild_tags_cache(tmp_kb, index_root)

        assert "template" not in counts
        assert "normal" in counts


# =============================================================================
# TestTitleIndex - Build, Resolve, Handle Duplicates
# =============================================================================


class TestTitleIndex:
    """Tests for title index behavior."""

    def test_title_resolves_to_path(self, tmp_kb: Path):
        """Title lookup resolves to correct path."""
        from memex.parser.title_index import build_title_index, resolve_link_target

        create_entry(tmp_kb, "my-entry.md", "My Entry Title", "Content")

        index = build_title_index(tmp_kb)

        result = resolve_link_target("My Entry Title", index)
        assert result == "my-entry"

    def test_alias_resolves_to_path(self, tmp_kb: Path):
        """Alias lookup resolves to the entry's path."""
        from memex.parser.title_index import build_title_index, resolve_link_target

        entry_path = tmp_kb / "entry.md"
        entry_path.write_text("""---
title: Main Title
aliases:
  - My Alias
  - Another Alias
created: 2024-01-15
---

Content
""")

        index = build_title_index(tmp_kb)

        assert resolve_link_target("My Alias", index) == "entry"
        assert resolve_link_target("Another Alias", index) == "entry"
        assert resolve_link_target("Main Title", index) == "entry"

    def test_title_lookup_case_insensitive(self, tmp_kb: Path):
        """Title lookup is case-insensitive."""
        from memex.parser.title_index import build_title_index, resolve_link_target

        create_entry(tmp_kb, "entry.md", "UPPERCASE Title", "Content")

        index = build_title_index(tmp_kb)

        assert resolve_link_target("uppercase title", index) == "entry"
        assert resolve_link_target("UPPERCASE TITLE", index) == "entry"
        assert resolve_link_target("Uppercase Title", index) == "entry"

    def test_filename_index_for_duplicate_filenames(self, tmp_kb: Path):
        """Filename index tracks duplicate filenames across directories."""
        from memex.parser.title_index import TitleIndex, build_title_index

        (tmp_kb / "docs").mkdir()
        (tmp_kb / "projects").mkdir()
        create_entry(tmp_kb, "docs/guide.md", "Docs Guide", "Content")
        create_entry(tmp_kb, "projects/guide.md", "Projects Guide", "Content")

        index = build_title_index(tmp_kb)

        assert isinstance(index, TitleIndex)
        assert "guide" in index.filename_to_paths
        assert len(index.filename_to_paths["guide"]) == 2

    def test_path_link_returned_as_is(self, tmp_kb: Path):
        """Path-style links are returned without lookup."""
        from memex.parser.title_index import TitleIndex, resolve_link_target

        index = TitleIndex(title_to_path={}, filename_to_paths={})

        result = resolve_link_target("foo/bar", index)
        assert result == "foo/bar"

    def test_nonexistent_title_returns_none(self, tmp_kb: Path):
        """Non-existent title lookup returns None."""
        from memex.parser.title_index import build_title_index, resolve_link_target

        create_entry(tmp_kb, "entry.md", "Entry", "Content")

        index = build_title_index(tmp_kb)

        result = resolve_link_target("Nonexistent Title", index)
        assert result is None


# =============================================================================
# TestHealthCache - Build, Query, Detect Problems
# =============================================================================


class TestHealthCache:
    """Tests for health cache behavior."""

    def test_build_health_metadata(self, tmp_kb: Path, monkeypatch):
        """Health cache builds metadata for all entries."""
        from memex.health_cache import rebuild_health_cache

        index_root = tmp_kb / ".kb-indices"
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        create_entry(tmp_kb, "entry1.md", "Entry 1", "Content with [[entry2]]", ["test"])
        create_entry(tmp_kb, "entry2.md", "Entry 2", "More content", ["test"])

        cache = rebuild_health_cache(tmp_kb, index_root)

        assert "entry1" in cache
        assert "entry2" in cache
        assert cache["entry1"]["title"] == "Entry 1"
        assert "entry2" in cache["entry1"]["links"]

    def test_health_cache_tracks_links(self, tmp_kb: Path, monkeypatch):
        """Health cache extracts and stores link information."""
        from memex.health_cache import rebuild_health_cache

        index_root = tmp_kb / ".kb-indices"
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        create_entry(
            tmp_kb, "source.md", "Source", "Links to [[target-a]] and [[target-b]]", ["test"]
        )
        create_entry(tmp_kb, "target-a.md", "Target A", "Content", ["test"])
        create_entry(tmp_kb, "target-b.md", "Target B", "Content", ["test"])

        cache = rebuild_health_cache(tmp_kb, index_root)

        links = cache["source"]["links"]
        assert "target-a" in links
        assert "target-b" in links

    def test_health_cache_incremental_update(self, tmp_kb: Path, monkeypatch):
        """Health cache updates incrementally when files change."""
        from memex.health_cache import ensure_health_cache, rebuild_health_cache

        index_root = tmp_kb / ".kb-indices"
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        create_entry(tmp_kb, "entry.md", "Original Title", "Content", ["test"])
        rebuild_health_cache(tmp_kb, index_root)

        time.sleep(0.02)

        # Modify the entry
        create_entry(tmp_kb, "entry.md", "Updated Title", "New content", ["test"])

        cache = ensure_health_cache(tmp_kb, index_root)

        assert cache["entry"]["title"] == "Updated Title"

    def test_health_cache_stores_dates(self, tmp_kb: Path, monkeypatch):
        """Health cache stores created and updated dates."""
        from datetime import datetime

        from memex.health_cache import get_entry_metadata

        index_root = tmp_kb / ".kb-indices"
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        entry_path = tmp_kb / "dated.md"
        entry_path.write_text("""---
title: Dated Entry
tags: [test]
created: 2024-06-15
updated: 2024-07-20
---

Content
""")

        metadata = get_entry_metadata(tmp_kb, index_root)

        assert "dated" in metadata
        assert metadata["dated"]["created"] == datetime(2024, 6, 15, 0, 0, 0)
        assert metadata["dated"]["updated"] == datetime(2024, 7, 20, 0, 0, 0)

    def test_health_cache_handles_parse_errors(self, tmp_kb: Path, monkeypatch):
        """Health cache gracefully handles malformed files."""
        from memex.health_cache import get_parse_errors, rebuild_health_cache

        index_root = tmp_kb / ".kb-indices"
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        # Create valid entry (with required tag)
        create_entry(tmp_kb, "valid.md", "Valid", "Content", ["test"])

        # Create invalid entry (no frontmatter)
        (tmp_kb / "broken.md").write_text("No frontmatter here")

        rebuild_health_cache(tmp_kb, index_root)
        errors = get_parse_errors(index_root)

        assert len(errors) == 1
        assert errors[0]["path"] == "broken.md"

    def test_health_cache_skips_underscore_files(self, tmp_kb: Path, monkeypatch):
        """Files starting with underscore are not indexed."""
        from memex.health_cache import rebuild_health_cache

        index_root = tmp_kb / ".kb-indices"
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        create_entry(tmp_kb, "_hidden.md", "Hidden", "Content", ["test"])
        create_entry(tmp_kb, "visible.md", "Visible", "Content", ["test"])

        cache = rebuild_health_cache(tmp_kb, index_root)

        assert "_hidden" not in cache
        assert "visible" in cache


# =============================================================================
# TestCacheConsistency - Cross-cache coherence
# =============================================================================


class TestCacheConsistency:
    """Tests for consistency across different cache systems."""

    def test_empty_kb_returns_empty_caches(self, tmp_kb: Path, monkeypatch):
        """Empty KB returns empty caches without errors."""
        from memex.backlinks_cache import rebuild_backlink_cache
        from memex.health_cache import rebuild_health_cache
        from memex.parser.title_index import build_title_index
        from memex.tags_cache import rebuild_tags_cache

        index_root = tmp_kb / ".kb-indices"
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        # No entries created - KB is empty
        backlinks = rebuild_backlink_cache(tmp_kb)
        tags = rebuild_tags_cache(tmp_kb, index_root)
        health = rebuild_health_cache(tmp_kb, index_root)
        title_idx = build_title_index(tmp_kb)

        assert backlinks == {}
        assert tags == {}
        assert health == {}
        assert title_idx.title_to_path == {}

    def test_backlinks_match_forward_links(self, tmp_kb: Path, monkeypatch):
        """Backlinks are consistent with forward links in health cache."""
        from memex.backlinks_cache import rebuild_backlink_cache
        from memex.health_cache import rebuild_health_cache

        index_root = tmp_kb / ".kb-indices"
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        create_entry(tmp_kb, "source.md", "Source", "Link to [[target]]", ["test"])
        create_entry(tmp_kb, "target.md", "Target", "No links", ["test"])

        backlinks = rebuild_backlink_cache(tmp_kb)
        health = rebuild_health_cache(tmp_kb, index_root)

        # Forward link in health cache
        assert "target" in health["source"]["links"]
        # Corresponding backlink
        assert "source" in backlinks["target"]

    def test_caches_recover_from_corruption(self, tmp_kb: Path, monkeypatch):
        """Caches rebuild correctly after corruption."""
        from memex.backlinks_cache import ensure_backlink_cache
        from memex.tags_cache import ensure_tags_cache

        index_root = tmp_kb / ".kb-indices"
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        create_entry(tmp_kb, "entry.md", "Entry", "Link to [[other]]", ["tag1"])
        create_entry(tmp_kb, "other.md", "Other", "Content", ["tag2"])

        # Corrupt the cache files
        (index_root / "backlinks.json").write_text("{ corrupted json }")
        (index_root / "tags_cache.json").write_text("{ also corrupted }")

        # Should rebuild from scratch
        backlinks = ensure_backlink_cache(tmp_kb)
        tags = ensure_tags_cache(tmp_kb, index_root)

        assert "other" in backlinks
        assert "tag1" in tags
        assert "tag2" in tags
