"""Tests for incremental indexing functionality."""

from pathlib import Path

import pytest

from memex.indexer.manifest import IndexManifest

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
# HybridSearcher Reindex Tests
# =============================================================================


@pytest.mark.semantic
class TestHybridSearcherReindex:
    """Test reindexing in HybridSearcher."""

    @pytest.fixture
    def index_dirs(self, tmp_path) -> tuple[Path, Path]:
        """Create temporary directories for indices."""
        whoosh_dir = tmp_path / "whoosh"
        chroma_dir = tmp_path / "chroma"
        return whoosh_dir, chroma_dir

    @pytest.fixture
    def kb_root(self, tmp_path) -> Path:
        """Create a temporary KB directory."""
        kb = tmp_path / "kb"
        kb.mkdir()
        return kb

    @pytest.fixture
    def hybrid_searcher(self, index_dirs):
        """Create a HybridSearcher with separate test indices."""
        from memex.indexer.chroma_index import ChromaIndex
        from memex.indexer.hybrid import HybridSearcher
        from memex.indexer.whoosh_index import WhooshIndex

        whoosh_dir, chroma_dir = index_dirs
        whoosh = WhooshIndex(index_dir=whoosh_dir)
        chroma = ChromaIndex(index_dir=chroma_dir)
        return HybridSearcher(
            whoosh_index=whoosh,
            chroma_index=chroma,
        )

    def _create_md_file(self, kb_root: Path, name: str, content: str) -> Path:
        """Helper to create a markdown file with proper frontmatter."""
        file_path = kb_root / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return file_path

    def test_reindex_indexes_all_files(self, hybrid_searcher, kb_root):
        """Reindex indexes all markdown files."""
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

        count = hybrid_searcher.reindex(kb_root)

        assert isinstance(count, int)
        assert count >= 2

    def test_reindex_returns_chunk_count(self, hybrid_searcher, kb_root):
        """Reindex returns the number of chunks indexed."""
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

        count = hybrid_searcher.reindex(kb_root)

        assert isinstance(count, int)
        assert count >= 1

    def test_reindex_clears_existing(self, hybrid_searcher, kb_root):
        """Reindex clears and rebuilds indices."""
        self._create_md_file(
            kb_root,
            "original.md",
            """---
title: Original
tags:
  - test
created: 2024-01-01
---

Original content.
""",
        )

        # First reindex
        hybrid_searcher.reindex(kb_root)

        # Verify original is searchable
        results = hybrid_searcher.search("original")
        assert len(results) >= 1

        # Remove original and add new file
        (kb_root / "original.md").unlink()
        self._create_md_file(
            kb_root,
            "new.md",
            """---
title: New
tags:
  - test
created: 2024-01-02
---

New content.
""",
        )

        # Second reindex
        hybrid_searcher.reindex(kb_root)

        # Original should no longer be found
        results = hybrid_searcher.search("original")
        paths = [r.path for r in results]
        assert "original.md" not in paths

        # New should be found
        results = hybrid_searcher.search("new")
        assert len(results) >= 1

    def test_reindex_empty_kb(self, hybrid_searcher, kb_root):
        """Reindex on empty KB returns zero."""
        count = hybrid_searcher.reindex(kb_root)

        assert count == 0

    def test_reindex_subdirectories(self, hybrid_searcher, kb_root):
        """Reindex handles files in subdirectories."""
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

        count = hybrid_searcher.reindex(kb_root)

        assert count >= 3

        # Verify all searchable
        results = hybrid_searcher.search("nested")
        assert len(results) >= 1

    def test_reindex_skips_invalid_files(self, hybrid_searcher, kb_root):
        """Reindex skips files with invalid frontmatter."""
        self._create_md_file(
            kb_root,
            "valid.md",
            """---
title: Valid
tags:
  - test
created: 2024-01-01
---

Valid content.
""",
        )
        self._create_md_file(
            kb_root,
            "invalid.md",
            """This file has no frontmatter at all.
Just plain text.
""",
        )

        # Should not crash
        count = hybrid_searcher.reindex(kb_root)

        # Valid file should be indexed
        assert count >= 1


class TestHybridSearcherClear:
    """Test clearing the HybridSearcher."""

    @pytest.fixture
    def index_dirs(self, tmp_path) -> tuple[Path, Path]:
        """Create temporary directories for indices."""
        whoosh_dir = tmp_path / "whoosh"
        chroma_dir = tmp_path / "chroma"
        return whoosh_dir, chroma_dir

    @pytest.fixture
    def kb_root(self, tmp_path) -> Path:
        """Create a temporary KB directory."""
        kb = tmp_path / "kb"
        kb.mkdir()
        return kb

    @pytest.fixture
    def hybrid_searcher(self, index_dirs):
        """Create a HybridSearcher with separate test indices."""
        from memex.indexer.chroma_index import ChromaIndex
        from memex.indexer.hybrid import HybridSearcher
        from memex.indexer.whoosh_index import WhooshIndex

        whoosh_dir, chroma_dir = index_dirs
        whoosh = WhooshIndex(index_dir=whoosh_dir)
        chroma = ChromaIndex(index_dir=chroma_dir)
        return HybridSearcher(
            whoosh_index=whoosh,
            chroma_index=chroma,
        )

    def _create_md_file(self, kb_root: Path, name: str, content: str) -> Path:
        """Helper to create a markdown file."""
        file_path = kb_root / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return file_path

    @pytest.mark.semantic
    def test_clear_removes_all_indexed_content(self, hybrid_searcher, kb_root):
        """Clear removes all indexed documents."""
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

        hybrid_searcher.reindex(kb_root)

        # Verify indexed
        results = hybrid_searcher.search("content")
        assert len(results) >= 1

        # Clear
        hybrid_searcher.clear()

        # Should be empty now
        status = hybrid_searcher.status()
        assert status.whoosh_docs == 0
        assert status.chroma_docs == 0
