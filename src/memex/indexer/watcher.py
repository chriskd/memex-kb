"""File watcher for automatic re-indexing of changed markdown files."""

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from ..backlinks_cache import rebuild_backlink_cache
from ..config import get_kb_root


def _is_in_docker() -> bool:
    """Detect if running inside a Docker container."""
    # Check for .dockerenv file
    if Path("/.dockerenv").exists():
        return True
    # Check cgroup for docker
    try:
        with open("/proc/1/cgroup", "r") as f:
            return "docker" in f.read()
    except (FileNotFoundError, PermissionError):
        pass
    return False


def _get_observer_class():
    """Get the appropriate observer class for the environment.

    Uses PollingObserver in Docker (inotify doesn't work across bind mounts)
    or when USE_POLLING_WATCHER=1 is set.
    """
    use_polling = os.environ.get("USE_POLLING_WATCHER", "").lower() in ("1", "true", "yes")
    if use_polling or _is_in_docker():
        logger.info("Using PollingObserver for file watching (Docker or polling mode)")
        return PollingObserver
    return Observer

if TYPE_CHECKING:
    from .hybrid import HybridSearcher

logger = logging.getLogger(__name__)


class DebouncedHandler(FileSystemEventHandler):
    """File system event handler with debouncing.

    Uses threading.Timer for debouncing since watchdog runs in a separate thread
    without an asyncio event loop.
    """

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
        import threading

        super().__init__()
        self._callback = callback
        self._debounce_seconds = debounce_seconds
        self._pending_files: set[Path] = set()
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def _schedule_callback(self) -> None:
        """Schedule the callback after debounce period."""
        import threading

        with self._lock:
            if self._timer is not None:
                self._timer.cancel()

            def fire():
                with self._lock:
                    if self._pending_files:
                        files = self._pending_files.copy()
                        self._pending_files.clear()
                        self._callback(files)

            self._timer = threading.Timer(self._debounce_seconds, fire)
            self._timer.daemon = True
            self._timer.start()

    def _handle_event(self, event: FileSystemEvent) -> None:
        """Handle a file system event."""
        if event.is_directory:
            return

        src_path = Path(event.src_path)

        # Only handle markdown files
        if src_path.suffix.lower() != ".md":
            return

        with self._lock:
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
        print(f"[LiveReload] Detected {len(files)} changed file(s)")
        logger.info(f"Re-indexing {len(files)} changed files")

        # Import here to avoid circular imports
        from ..parser import parse_entry
        from ..webapp.events import Event, EventType, get_broadcaster

        broadcaster = get_broadcaster()
        refresh_backlinks = False
        changed_paths: list[str] = []

        for file_path in files:
            try:
                # Check if file was deleted
                if not file_path.exists():
                    relative_path = str(file_path.relative_to(self._kb_root))
                    self._searcher.delete_document(relative_path)
                    logger.debug(f"Removed from index: {relative_path}")
                    refresh_backlinks = True
                    changed_paths.append(relative_path)

                    # Broadcast delete event
                    broadcaster.broadcast_sync(
                        Event(
                            type=EventType.FILE_DELETED,
                            data={"path": relative_path},
                        )
                    )
                    continue

                relative_path = str(file_path.relative_to(self._kb_root))
                _, _, chunks = parse_entry(file_path)
                if not chunks:
                    continue

                normalized_chunks = [
                    chunk.model_copy(update={"path": relative_path}) for chunk in chunks
                ]

                # Update index
                self._searcher.delete_document(relative_path)
                self._searcher.index_chunks(normalized_chunks)
                logger.debug(f"Re-indexed: {relative_path}")
                refresh_backlinks = True
                changed_paths.append(relative_path)

                # Broadcast change event
                broadcaster.broadcast_sync(
                    Event(
                        type=EventType.FILE_CHANGED,
                        data={"path": relative_path},
                    )
                )

            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")

        if refresh_backlinks:
            rebuild_backlink_cache(self._kb_root)

        # Broadcast completion event
        if changed_paths:
            broadcaster.broadcast_sync(
                Event(
                    type=EventType.REINDEX_COMPLETE,
                    data={"paths": changed_paths, "count": len(changed_paths)},
                )
            )

    def start(self) -> None:
        """Start watching for file changes."""
        if self._running:
            return

        if not self._kb_root.exists():
            logger.warning(f"KB root does not exist: {self._kb_root}")
            return

        observer_class = _get_observer_class()
        self._observer = observer_class()
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
