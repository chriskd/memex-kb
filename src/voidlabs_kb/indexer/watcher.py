"""File watcher for automatic re-indexing of changed markdown files."""

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from ..config import get_kb_root

if TYPE_CHECKING:
    from .hybrid import HybridSearcher

logger = logging.getLogger(__name__)


class DebouncedHandler(FileSystemEventHandler):
    """File system event handler with debouncing."""

    def __init__(
        self,
        callback: Callable[[set[Path]], None],
        debounce_seconds: float = 5.0,
    ):
        """Initialize the debounced handler.

        Args:
            callback: Function to call with changed files after debounce.
            debounce_seconds: Debounce window in seconds.
        """
        super().__init__()
        self._callback = callback
        self._debounce_seconds = debounce_seconds
        self._pending_files: set[Path] = set()
        self._timer: asyncio.TimerHandle | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create the event loop."""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
        return self._loop

    def _schedule_callback(self) -> None:
        """Schedule the callback after debounce period."""
        if self._timer is not None:
            self._timer.cancel()

        loop = self._get_loop()

        def fire():
            if self._pending_files:
                files = self._pending_files.copy()
                self._pending_files.clear()
                self._callback(files)

        self._timer = loop.call_later(self._debounce_seconds, fire)

    def _handle_event(self, event: FileSystemEvent) -> None:
        """Handle a file system event."""
        if event.is_directory:
            return

        src_path = Path(event.src_path)

        # Only handle markdown files
        if src_path.suffix.lower() != ".md":
            return

        self._pending_files.add(src_path)
        self._schedule_callback()

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation."""
        self._handle_event(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification."""
        self._handle_event(event)

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion."""
        self._handle_event(event)

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move/rename."""
        if event.is_directory:
            return

        src_path = Path(event.src_path)
        dest_path = Path(event.dest_path) if hasattr(event, "dest_path") else None

        if src_path.suffix.lower() == ".md":
            self._pending_files.add(src_path)

        if dest_path and dest_path.suffix.lower() == ".md":
            self._pending_files.add(dest_path)

        if self._pending_files:
            self._schedule_callback()


class FileWatcher:
    """Watch knowledge base directory for file changes and trigger re-indexing."""

    def __init__(
        self,
        searcher: "HybridSearcher",
        kb_root: Path | None = None,
        debounce_seconds: float = 5.0,
    ):
        """Initialize the file watcher.

        Args:
            searcher: HybridSearcher instance to update on changes.
            kb_root: Knowledge base root directory. Uses config default if None.
            debounce_seconds: Debounce window for batching updates.
        """
        self._searcher = searcher
        self._kb_root = kb_root or get_kb_root()
        self._debounce_seconds = debounce_seconds
        self._observer: Observer | None = None
        self._running = False

    def _on_files_changed(self, files: set[Path]) -> None:
        """Handle changed files after debounce.

        Args:
            files: Set of changed file paths.
        """
        logger.info(f"Re-indexing {len(files)} changed files")

        # Import here to avoid circular imports
        from ..models import DocumentChunk
        from ..parser import parse_entry

        for file_path in files:
            try:
                # Check if file was deleted
                if not file_path.exists():
                    relative_path = file_path.relative_to(self._kb_root)
                    self._searcher.delete_document(str(relative_path))
                    logger.debug(f"Removed from index: {relative_path}")
                    continue

                # Parse and re-index the file
                relative_path = file_path.relative_to(self._kb_root)
                content = file_path.read_text()
                entry = parse_entry(content, str(relative_path))

                if entry:
                    chunk = DocumentChunk(
                        path=str(relative_path),
                        section=None,
                        content=entry.content,
                        metadata=entry.metadata,
                    )
                    # Delete old chunks first, then add new one
                    self._searcher.delete_document(str(relative_path))
                    self._searcher.index_document(chunk)
                    logger.debug(f"Re-indexed: {relative_path}")

            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")

    def start(self) -> None:
        """Start watching for file changes."""
        if self._running:
            return

        if not self._kb_root.exists():
            logger.warning(f"KB root does not exist: {self._kb_root}")
            return

        self._observer = Observer()
        handler = DebouncedHandler(
            callback=self._on_files_changed,
            debounce_seconds=self._debounce_seconds,
        )
        self._observer.schedule(handler, str(self._kb_root), recursive=True)
        self._observer.start()
        self._running = True
        logger.info(f"Started watching: {self._kb_root}")

    def stop(self) -> None:
        """Stop watching for file changes."""
        if not self._running or self._observer is None:
            return

        self._observer.stop()
        self._observer.join(timeout=5.0)
        self._observer = None
        self._running = False
        logger.info("Stopped file watcher")

    @property
    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running

    def __enter__(self) -> "FileWatcher":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
