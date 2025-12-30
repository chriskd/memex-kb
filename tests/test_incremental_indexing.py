"""Tests for incremental indexing functionality."""

import json
import time
from datetime import date
from pathlib import Path

import pytest

from memex.indexer.chroma_index import ChromaIndex
from memex.indexer.hybrid import HybridSearcher, ReindexStats
from memex.indexer.manifest import IndexManifest
from memex.indexer.whoosh_index import WhooshIndex


# =============================================================================
# IndexManifest Tests
# =============================================================================


class TestIndexManifest:
    """Test the IndexManifest class for tracking file states."""

    @pytest.fixture
    def manifest_dir(self, tmp_path) -> Path:
        """Create a temporary directory for manifest storage."""
        return tmp_path / "manifest_test"

    @pytest.fixture
    def manifest(self, manifest_dir) -> IndexManifest:
        """Create a fresh IndexManifest instance."""
        return IndexManifest(manifest_dir)

    def test_init_creates_no_file(self, manifest, manifest_dir):
        """Manifest file is not created until save is called."""
        # Just creating the manifest shouldn't create files
        assert not (manifest_dir / "index_manifest.json").exists()

    def test_update_and_get_file(self, manifest):
        """Can update and retrieve file state."""
        manifest.update_file("test.md", mtime=1000.0, size=500)

        state = manifest.get_file_state("test.md")
        assert state is not None
        assert state.mtime == 1000.0
        assert state.size == 500

    def test_get_nonexistent_file(self, manifest):
        """Getting nonexistent file returns None."""
        state = manifest.get_file_state("nonexistent.md")
        assert state is None

    def test_remove_file(self, manifest):
        """Can remove a file from the manifest."""
        manifest.update_file("test.md", mtime=1000.0, size=500)
        manifest.remove_file("test.md")

        assert manifest.get_file_state("test.md") is None

    def test_remove_nonexistent_file(self, manifest):
        """Removing nonexistent file doesn't raise error."""
        manifest.remove_file("nonexistent.md")  # Should not raise

    def test_get_all_paths(self, manifest):
        """Can get all tracked file paths."""
        manifest.update_file("a.md", mtime=1000.0, size=100)
        manifest.update_file("b.md", mtime=2000.0, size=200)
        manifest.update_file("sub/c.md", mtime=3000.0, size=300)

        paths = manifest.get_all_paths()
        assert paths == {"a.md", "b.md", "sub/c.md"}

    def test_clear(self, manifest, manifest_dir):
        """Clear removes all tracked files."""
        manifest.update_file("a.md", mtime=1000.0, size=100)
        manifest.save()
        assert (manifest_dir / "index_manifest.json").exists()

        manifest.clear()

        assert manifest.get_all_paths() == set()
        assert not (manifest_dir / "index_manifest.json").exists()

    def test_save_and_load(self, manifest_dir):
        """Manifest persists and loads correctly."""
        manifest1 = IndexManifest(manifest_dir)
        manifest1.update_file("test.md", mtime=1234.5, size=999)
        manifest1.save()

        # Create new manifest instance (simulating restart)
        manifest2 = IndexManifest(manifest_dir)
        state = manifest2.get_file_state("test.md")

        assert state is not None
        assert state.mtime == 1234.5
        assert state.size == 999

    def test_is_file_changed_new_file(self, manifest):
        """New file is detected as changed."""
        assert manifest.is_file_changed("new.md", 1000.0, 500)

    def test_is_file_changed_same_file(self, manifest):
        """Unchanged file is detected as not changed."""
        manifest.update_file("test.md", mtime=1000.0, size=500)

        assert not manifest.is_file_changed("test.md", 1000.0, 500)

    def test_is_file_changed_modified_mtime(self, manifest):
        """Modified mtime is detected."""
        manifest.update_file("test.md", mtime=1000.0, size=500)

        assert manifest.is_file_changed("test.md", 2000.0, 500)

    def test_is_file_changed_modified_size(self, manifest):
        """Modified size is detected."""
        manifest.update_file("test.md", mtime=1000.0, size=500)

        assert manifest.is_file_changed("test.md", 1000.0, 600)

    def test_handles_corrupted_manifest(self, manifest_dir):
        """Corrupted manifest file is handled gracefully."""
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "index_manifest.json"
        manifest_path.write_text("not valid json {{{", encoding="utf-8")

        manifest = IndexManifest(manifest_dir)
        # Should start fresh
        assert manifest.get_all_paths() == set()


# =============================================================================
# Incremental Reindex Tests
# =============================================================================


class TestIncrementalReindex:
    """Test incremental reindexing in HybridSearcher."""

    @pytest.fixture
    def index_dirs(self, tmp_path) -> tuple[Path, Path, Path]:
        """Create temporary directories for indices and manifest."""
        whoosh_dir = tmp_path / "whoosh"
        chroma_dir = tmp_path / "chroma"
        manifest_dir = tmp_path / "index"
        return whoosh_dir, chroma_dir, manifest_dir

    @pytest.fixture
    def kb_root(self, tmp_path) -> Path:
        """Create a temporary KB directory."""
        kb = tmp_path / "kb"
        kb.mkdir()
        return kb

    @pytest.fixture
    def hybrid_searcher(self, index_dirs) -> HybridSearcher:
        """Create a HybridSearcher with separate test indices."""
        whoosh_dir, chroma_dir, manifest_dir = index_dirs
        whoosh = WhooshIndex(index_dir=whoosh_dir)
        chroma = ChromaIndex(index_dir=chroma_dir)
        return HybridSearcher(
            whoosh_index=whoosh,
            chroma_index=chroma,
            index_dir=manifest_dir,
        )

    def _create_md_file(self, kb_root: Path, name: str, content: str) -> Path:
        """Helper to create a markdown file with proper frontmatter."""
        file_path = kb_root / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return file_path

    def test_initial_incremental_indexes_all(self, hybrid_searcher, kb_root):
        """First incremental reindex indexes all files."""
        self._create_md_file(
            kb_root,
            "test1.md",
            """---
title: Test One
tags:
  - test
created: 2024-01-01
---

Content for test one.
""",
        )
        self._create_md_file(
            kb_root,
            "test2.md",
            """---
title: Test Two
tags:
  - test
created: 2024-01-02
---

Content for test two.
""",
        )

        stats = hybrid_searcher.reindex(kb_root)

        assert isinstance(stats, ReindexStats)
        assert stats.added == 2
        assert stats.updated == 0
        assert stats.deleted == 0
        assert stats.unchanged == 0
        assert stats.total_chunks >= 2

    def test_incremental_skips_unchanged(self, hybrid_searcher, kb_root):
        """Incremental reindex skips unchanged files."""
        self._create_md_file(
            kb_root,
            "test.md",
            """---
title: Test
tags:
  - test
created: 2024-01-01
---

Content here.
""",
        )

        # First reindex
        stats1 = hybrid_searcher.reindex(kb_root)
        assert stats1.added == 1

        # Second reindex without changes
        stats2 = hybrid_searcher.reindex(kb_root)

        assert stats2.added == 0
        assert stats2.updated == 0
        assert stats2.deleted == 0
        assert stats2.unchanged == 1

    def test_incremental_detects_modified(self, hybrid_searcher, kb_root):
        """Incremental reindex detects and updates modified files."""
        file_path = self._create_md_file(
            kb_root,
            "test.md",
            """---
title: Original Title
tags:
  - test
created: 2024-01-01
---

Original content.
""",
        )

        # First reindex
        hybrid_searcher.reindex(kb_root)

        # Modify the file (ensure mtime changes)
        time.sleep(0.1)  # Ensure mtime difference
        file_path.write_text(
            """---
title: Updated Title
tags:
  - test
  - updated
created: 2024-01-01
---

Updated content with more text.
""",
            encoding="utf-8",
        )

        # Second reindex
        stats = hybrid_searcher.reindex(kb_root)

        assert stats.added == 0
        assert stats.updated == 1
        assert stats.deleted == 0
        assert stats.unchanged == 0

        # Verify updated content is searchable
        results = hybrid_searcher.search("updated content")
        assert len(results) >= 1

    def test_incremental_detects_deleted(self, hybrid_searcher, kb_root):
        """Incremental reindex detects and removes deleted files."""
        file1 = self._create_md_file(
            kb_root,
            "keep.md",
            """---
title: Keep Me
tags:
  - test
created: 2024-01-01
---

I stay.
""",
        )
        file2 = self._create_md_file(
            kb_root,
            "delete.md",
            """---
title: Delete Me
tags:
  - test
created: 2024-01-02
---

I will be deleted.
""",
        )

        # First reindex
        stats1 = hybrid_searcher.reindex(kb_root)
        assert stats1.added == 2

        # Delete one file
        file2.unlink()

        # Second reindex
        stats2 = hybrid_searcher.reindex(kb_root)

        assert stats2.added == 0
        assert stats2.updated == 0
        assert stats2.deleted == 1
        assert stats2.unchanged == 1

        # Verify deleted content is not searchable
        results = hybrid_searcher.search("deleted")
        paths = [r.path for r in results]
        assert "delete.md" not in paths

    def test_incremental_detects_new(self, hybrid_searcher, kb_root):
        """Incremental reindex detects and indexes new files."""
        self._create_md_file(
            kb_root,
            "existing.md",
            """---
title: Existing
tags:
  - test
created: 2024-01-01
---

Existing content.
""",
        )

        # First reindex
        hybrid_searcher.reindex(kb_root)

        # Add new file
        self._create_md_file(
            kb_root,
            "new.md",
            """---
title: New File
tags:
  - new
created: 2024-01-05
---

Brand new content.
""",
        )

        # Second reindex
        stats = hybrid_searcher.reindex(kb_root)

        assert stats.added == 1
        assert stats.updated == 0
        assert stats.deleted == 0
        assert stats.unchanged == 1

        # Verify new content is searchable
        results = hybrid_searcher.search("brand new")
        assert len(results) >= 1

    def test_force_reindex_clears_all(self, hybrid_searcher, kb_root):
        """Force reindex clears and rebuilds everything."""
        self._create_md_file(
            kb_root,
            "test.md",
            """---
title: Test
tags:
  - test
created: 2024-01-01
---

Content.
""",
        )

        # First incremental reindex
        hybrid_searcher.reindex(kb_root)

        # Force reindex
        count = hybrid_searcher.reindex(kb_root, force=True)

        # Force returns int, not stats
        assert isinstance(count, int)
        assert count >= 1

    def test_combined_operations(self, hybrid_searcher, kb_root):
        """Test add, modify, and delete in same reindex."""
        keep = self._create_md_file(
            kb_root,
            "keep.md",
            """---
title: Keep
tags:
  - test
created: 2024-01-01
---

Keep this.
""",
        )
        modify = self._create_md_file(
            kb_root,
            "modify.md",
            """---
title: Modify
tags:
  - test
created: 2024-01-02
---

Original.
""",
        )
        delete = self._create_md_file(
            kb_root,
            "delete.md",
            """---
title: Delete
tags:
  - test
created: 2024-01-03
---

Delete this.
""",
        )

        # First reindex
        hybrid_searcher.reindex(kb_root)

        # Make all changes
        time.sleep(0.1)
        modify.write_text(
            """---
title: Modified
tags:
  - test
created: 2024-01-02
---

Modified content.
""",
            encoding="utf-8",
        )
        delete.unlink()
        self._create_md_file(
            kb_root,
            "new.md",
            """---
title: New
tags:
  - test
created: 2024-01-04
---

New content.
""",
        )

        # Second reindex
        stats = hybrid_searcher.reindex(kb_root)

        assert stats.added == 1
        assert stats.updated == 1
        assert stats.deleted == 1
        assert stats.unchanged == 1

    def test_empty_kb_incremental(self, hybrid_searcher, kb_root):
        """Incremental reindex on empty KB returns zero stats."""
        stats = hybrid_searcher.reindex(kb_root)

        assert isinstance(stats, ReindexStats)
        assert stats.added == 0
        assert stats.updated == 0
        assert stats.deleted == 0
        assert stats.unchanged == 0
        assert stats.total_chunks == 0

    def test_subdirectory_files(self, hybrid_searcher, kb_root):
        """Handles files in subdirectories correctly."""
        self._create_md_file(
            kb_root,
            "root.md",
            """---
title: Root
tags:
  - test
created: 2024-01-01
---

Root level.
""",
        )
        self._create_md_file(
            kb_root,
            "sub/nested.md",
            """---
title: Nested
tags:
  - test
created: 2024-01-02
---

Nested content.
""",
        )
        self._create_md_file(
            kb_root,
            "sub/deep/file.md",
            """---
title: Deep
tags:
  - test
created: 2024-01-03
---

Deeply nested.
""",
        )

        stats = hybrid_searcher.reindex(kb_root)

        assert stats.added == 3
        assert stats.total_chunks >= 3

        # Verify all searchable
        results = hybrid_searcher.search("nested")
        assert len(results) >= 1

    def test_clear_resets_manifest(self, hybrid_searcher, kb_root, index_dirs):
        """Clearing the searcher also clears the manifest."""
        _, _, manifest_dir = index_dirs

        self._create_md_file(
            kb_root,
            "test.md",
            """---
title: Test
tags:
  - test
created: 2024-01-01
---

Content.
""",
        )

        # Index and verify manifest exists
        hybrid_searcher.reindex(kb_root)
        manifest_path = manifest_dir / "index_manifest.json"
        assert manifest_path.exists()

        # Clear
        hybrid_searcher.clear()

        # Manifest should be cleared
        assert not manifest_path.exists()

        # Next incremental reindex should see all as new
        stats = hybrid_searcher.reindex(kb_root)
        assert stats.added == 1
        assert stats.unchanged == 0


class TestReindexStatsDataclass:
    """Test ReindexStats dataclass."""

    def test_reindex_stats_fields(self):
        """ReindexStats has expected fields."""
        stats = ReindexStats(
            total_chunks=10,
            added=5,
            updated=3,
            deleted=1,
            unchanged=2,
        )

        assert stats.total_chunks == 10
        assert stats.added == 5
        assert stats.updated == 3
        assert stats.deleted == 1
        assert stats.unchanged == 2
