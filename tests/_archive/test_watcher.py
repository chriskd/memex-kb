"""Tests for FileWatcher functionality."""

import threading
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from memex.indexer import watcher as watcher_module
from memex.indexer.chroma_index import ChromaIndex
from memex.indexer.hybrid import HybridSearcher
from memex.indexer.watcher import DebouncedHandler, FileWatcher
from memex.indexer.whoosh_index import WhooshIndex
from memex.models import DocumentChunk, EntryMetadata

pytestmark = pytest.mark.semantic


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


class TestDebouncedHandler:
    """Tests for DebouncedHandler event handling and debouncing."""

    def test_ignores_directories_and_non_markdown(self):
        """Non-markdown files and directories are ignored."""
        callback = MagicMock()
        handler = DebouncedHandler(callback=callback, debounce_seconds=0.1)
        handler._schedule_callback = MagicMock()

        directory_event = SimpleNamespace(src_path="/tmp/dir", is_directory=True)
        handler.on_created(directory_event)

        text_event = SimpleNamespace(src_path="/tmp/file.txt", is_directory=False)
        handler.on_modified(text_event)

        assert handler._pending_files == set()
        handler._schedule_callback.assert_not_called()

    def test_adds_markdown_files_and_schedules_callback(self):
        """Markdown events are collected and scheduled."""
        callback = MagicMock()
        handler = DebouncedHandler(callback=callback, debounce_seconds=0.1)
        handler._schedule_callback = MagicMock()

        md_event = SimpleNamespace(src_path="/tmp/file.md", is_directory=False)
        handler.on_created(md_event)

        assert handler._pending_files == {Path("/tmp/file.md")}
        handler._schedule_callback.assert_called_once()

    def test_on_moved_tracks_src_and_dest(self):
        """Moved events collect both source and destination markdown files."""
        callback = MagicMock()
        handler = DebouncedHandler(callback=callback, debounce_seconds=0.1)
        handler._schedule_callback = MagicMock()

        move_event = SimpleNamespace(
            src_path="/tmp/old.md",
            dest_path="/tmp/new.md",
            is_directory=False,
        )
        handler.on_moved(move_event)

        assert handler._pending_files == {Path("/tmp/old.md"), Path("/tmp/new.md")}
        handler._schedule_callback.assert_called_once()

    def test_debounce_batches_events_and_cancels_previous_timer(self, monkeypatch):
        """Multiple events in the debounce window batch into one callback."""
        callback = MagicMock()
        handler = DebouncedHandler(callback=callback, debounce_seconds=0.5)

        timers = []

        class FakeTimer:
            def __init__(self, interval, function):
                self.interval = interval
                self.function = function
                self.canceled = False
                self.started = False

            def start(self):
                self.started = True

            def cancel(self):
                self.canceled = True

            def fire(self):
                if not self.canceled:
                    self.function()

        def fake_timer(interval, function):
            timer = FakeTimer(interval, function)
            timers.append(timer)
            return timer

        monkeypatch.setattr(threading, "Timer", fake_timer)

        handler.on_created(SimpleNamespace(src_path="/tmp/a.md", is_directory=False))
        handler.on_modified(SimpleNamespace(src_path="/tmp/b.md", is_directory=False))

        assert len(timers) == 2
        assert timers[0].canceled is True

        timers[1].fire()

        callback.assert_called_once()
        called_paths = callback.call_args[0][0]
        assert called_paths == {Path("/tmp/a.md"), Path("/tmp/b.md")}
        assert handler._pending_files == set()


class TestFileWatcherLifecycle:
    """Tests for FileWatcher start/stop lifecycle."""

    def test_init_uses_default_kb_root(self, tmp_path):
        """KB root defaults to config when not provided."""
        searcher = MagicMock()
        default_root = tmp_path / "kb-default"
        with patch.object(watcher_module, "get_kb_root", return_value=default_root):
            watcher = FileWatcher(searcher)
        assert watcher._kb_root == default_root

    def test_get_observer_class_uses_polling_when_forced(self, monkeypatch):
        """USE_POLLING_WATCHER forces PollingObserver usage."""
        monkeypatch.setenv("USE_POLLING_WATCHER", "1")
        with patch.object(watcher_module, "_is_in_docker", return_value=False):
            observer_class = watcher_module._get_observer_class()
        assert observer_class is watcher_module.PollingObserver

    def test_get_observer_class_defaults_to_observer(self, monkeypatch):
        """Observer is used when not in Docker and polling not enabled."""
        monkeypatch.delenv("USE_POLLING_WATCHER", raising=False)
        with patch.object(watcher_module, "_is_in_docker", return_value=False):
            observer_class = watcher_module._get_observer_class()
        assert observer_class is watcher_module.Observer

    def test_start_and_stop_manage_observer(self, tmp_path):
        """Starting schedules the observer; stopping cleans up."""
        searcher = MagicMock()
        kb_root = tmp_path / "kb"
        kb_root.mkdir()

        class FakeObserver:
            def __init__(self):
                self.scheduled = None
                self.started = False
                self.stopped = False
                self.joined = None

            def schedule(self, handler, path, recursive=False):
                self.scheduled = (handler, path, recursive)

            def start(self):
                self.started = True

            def stop(self):
                self.stopped = True

            def join(self, timeout=None):
                self.joined = timeout

        with patch.object(watcher_module, "_get_observer_class", return_value=FakeObserver):
            watcher = FileWatcher(searcher, kb_root=kb_root, debounce_seconds=0.2)
            watcher.start()

        assert watcher.is_running is True
        assert watcher._observer is not None
        observer = watcher._observer
        handler, path, recursive = observer.scheduled
        assert isinstance(handler, DebouncedHandler)
        assert handler._debounce_seconds == 0.2
        assert path == str(kb_root)
        assert recursive is True
        assert observer.started is True

        watcher.stop()

        assert watcher.is_running is False
        assert watcher._observer is None
        assert observer.stopped is True
        assert observer.joined == 5.0

    def test_start_skips_when_kb_root_missing(self, tmp_path):
        """Missing KB root prevents observer setup."""
        searcher = MagicMock()
        kb_root = tmp_path / "missing"

        with patch.object(watcher_module, "_get_observer_class") as get_observer:
            watcher = FileWatcher(searcher, kb_root=kb_root)
            watcher.start()

        assert watcher.is_running is False
        assert watcher._observer is None
        get_observer.assert_not_called()
