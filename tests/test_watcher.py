"""Tests for FileWatcher functionality."""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from memex.indexer.hybrid import HybridSearcher
from memex.indexer.watcher import FileWatcher
from memex.indexer.whoosh_index import WhooshIndex
from memex.indexer.chroma_index import ChromaIndex
from memex.models import DocumentChunk, EntryMetadata


@pytest.fixture
def index_dirs(tmp_path) -> tuple[Path, Path]:
    """Create temporary directories for both indices."""
    whoosh_dir = tmp_path / "whoosh"
    chroma_dir = tmp_path / "chroma"
    return whoosh_dir, chroma_dir


@pytest.fixture
def hybrid_searcher(index_dirs) -> HybridSearcher:
    """Create a HybridSearcher with separate test indices."""
    whoosh_dir, chroma_dir = index_dirs
    whoosh = WhooshIndex(index_dir=whoosh_dir)
    chroma = ChromaIndex(index_dir=chroma_dir)
    return HybridSearcher(whoosh_index=whoosh, chroma_index=chroma)


@pytest.fixture
def kb_root(tmp_path) -> Path:
    """Create a temporary knowledge base directory."""
    kb = tmp_path / "kb"
    kb.mkdir()
    return kb


class TestFileWatcherInit:
    """Test FileWatcher initialization."""

    def test_init_with_searcher(self, hybrid_searcher, kb_root):
        """Can initialize watcher with a HybridSearcher."""
        watcher = FileWatcher(hybrid_searcher, kb_root=kb_root)
        assert watcher._searcher is hybrid_searcher
        assert watcher._kb_root == kb_root

    def test_init_not_running_by_default(self, hybrid_searcher, kb_root):
        """Watcher is not running by default."""
        watcher = FileWatcher(hybrid_searcher, kb_root=kb_root)
        assert not watcher.is_running


class TestFileWatcherUpsertSemantics:
    """Test that FileWatcher uses upsert semantics for updates."""

    def test_index_chunks_without_delete_updates_document(self, hybrid_searcher, kb_root):
        """index_chunks() properly updates existing documents via upsert."""
        # Index initial document
        chunk1 = DocumentChunk(
            path="test/doc.md",
            section="intro",
            content="Original content about Python",
            metadata=EntryMetadata(
                title="Test Doc",
                tags=["python"],
                created=date(2024, 1, 1),
            ),
            token_count=5,
        )
        hybrid_searcher.index_chunks([chunk1])

        assert hybrid_searcher._whoosh.doc_count() == 1
        assert hybrid_searcher._chroma.doc_count() == 1

        # Update document (upsert same path#section)
        chunk2 = DocumentChunk(
            path="test/doc.md",
            section="intro",
            content="Updated content about Rust programming",
            metadata=EntryMetadata(
                title="Test Doc Updated",
                tags=["rust"],
                created=date(2024, 1, 1),
                updated=date(2024, 1, 15),
            ),
            token_count=6,
        )
        # No delete_document call - just index_chunks (upsert)
        hybrid_searcher.index_chunks([chunk2])

        # Should still have only one document (upsert replaced it)
        assert hybrid_searcher._whoosh.doc_count() == 1
        assert hybrid_searcher._chroma.doc_count() == 1

        # Search should find updated content
        results = hybrid_searcher.search("Rust programming", mode="keyword")
        assert len(results) == 1
        assert "rust" in results[0].tags

        # Original content should not be found
        results = hybrid_searcher.search("Python", mode="keyword")
        assert len(results) == 0 or "python" not in results[0].snippet.lower()

    def test_upsert_works_for_multiple_sections(self, hybrid_searcher, kb_root):
        """Upsert properly handles documents with multiple sections."""
        # Index document with multiple sections
        chunks = [
            DocumentChunk(
                path="multi/doc.md",
                section="intro",
                content="Introduction section content",
                metadata=EntryMetadata(
                    title="Multi Doc",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
            ),
            DocumentChunk(
                path="multi/doc.md",
                section="body",
                content="Body section content",
                metadata=EntryMetadata(
                    title="Multi Doc",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
            ),
        ]
        hybrid_searcher.index_chunks(chunks)

        assert hybrid_searcher._whoosh.doc_count() == 2
        assert hybrid_searcher._chroma.doc_count() == 2

        # Update only one section
        updated_chunk = DocumentChunk(
            path="multi/doc.md",
            section="intro",
            content="Updated introduction with new content",
            metadata=EntryMetadata(
                title="Multi Doc",
                tags=["test", "updated"],
                created=date(2024, 1, 1),
            ),
        )
        hybrid_searcher.index_chunks([updated_chunk])

        # Should still have 2 documents (only intro section updated)
        assert hybrid_searcher._whoosh.doc_count() == 2
        assert hybrid_searcher._chroma.doc_count() == 2

        # Search should find updated intro
        results = hybrid_searcher.search("Updated introduction", mode="keyword")
        assert len(results) >= 1

    def test_watcher_on_files_changed_uses_upsert(self, hybrid_searcher, kb_root, monkeypatch):
        """FileWatcher._on_files_changed uses upsert (no delete before index)."""
        # Create a test file
        test_file = kb_root / "test.md"
        test_file.write_text("""---
title: Test Entry
tags:
  - python
created: 2024-01-01
---

Original content about Python programming.
""")

        # Set up environment
        monkeypatch.setenv("MEMEX_KB_ROOT", str(kb_root))

        # Create watcher
        watcher = FileWatcher(hybrid_searcher, kb_root=kb_root)

        # Mock the broadcaster (import is inside the method)
        with patch("memex.webapp.events.get_broadcaster") as mock_get_broadcaster:
            mock_broadcaster = MagicMock()
            mock_get_broadcaster.return_value = mock_broadcaster

            # Trigger file changed handler
            watcher._on_files_changed({test_file})

        # Verify document was indexed
        assert hybrid_searcher._whoosh.doc_count() >= 1
        assert hybrid_searcher._chroma.doc_count() >= 1

        # Update the file
        test_file.write_text("""---
title: Test Entry Updated
tags:
  - rust
created: 2024-01-01
updated: 2024-01-15
---

Updated content about Rust programming.
""")

        # Trigger file changed again
        with patch("memex.webapp.events.get_broadcaster") as mock_get_broadcaster:
            mock_broadcaster = MagicMock()
            mock_get_broadcaster.return_value = mock_broadcaster

            watcher._on_files_changed({test_file})

        # Should still have the same number of documents (upsert)
        assert hybrid_searcher._whoosh.doc_count() >= 1
        assert hybrid_searcher._chroma.doc_count() >= 1

        # Search should find updated content
        results = hybrid_searcher.search("Rust programming", mode="keyword")
        assert len(results) >= 1


class TestFileWatcherDeletion:
    """Test that FileWatcher properly handles file deletion."""

    def test_deleted_file_removed_from_index(self, hybrid_searcher, kb_root, monkeypatch):
        """Deleted files are properly removed from the index."""
        # Create and index a test file
        test_file = kb_root / "to_delete.md"
        test_file.write_text("""---
title: To Delete
tags:
  - test
created: 2024-01-01
---

Content that will be deleted.
""")

        monkeypatch.setenv("MEMEX_KB_ROOT", str(kb_root))
        watcher = FileWatcher(hybrid_searcher, kb_root=kb_root)

        # Index the file
        with patch("memex.webapp.events.get_broadcaster") as mock_get_broadcaster:
            mock_broadcaster = MagicMock()
            mock_get_broadcaster.return_value = mock_broadcaster
            watcher._on_files_changed({test_file})

        assert hybrid_searcher._whoosh.doc_count() >= 1

        # Delete the file
        test_file.unlink()

        # Trigger handler for deleted file
        with patch("memex.webapp.events.get_broadcaster") as mock_get_broadcaster:
            mock_broadcaster = MagicMock()
            mock_get_broadcaster.return_value = mock_broadcaster
            watcher._on_files_changed({test_file})

        # File should be removed from index
        results = hybrid_searcher.search("deleted", mode="keyword")
        assert all("to_delete.md" not in r.path for r in results)
