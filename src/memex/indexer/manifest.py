"""Manifest for tracking indexed file mtimes to enable incremental reindexing."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

MANIFEST_FILENAME = "index_manifest.json"


@dataclass
class FileState:
    """Tracked state of an indexed file."""

    mtime: float  # Modification time as timestamp
    size: int  # File size in bytes


class IndexManifest:
    """Tracks file modification times to support incremental indexing.

    The manifest stores mtime and size for each indexed file, allowing
    the indexer to detect which files have changed since the last index.
    """

    def __init__(self, index_dir: Path):
        """Initialize the manifest.

        Args:
            index_dir: Directory where the manifest file is stored.
        """
        self._index_dir = index_dir
        self._manifest_path = index_dir / MANIFEST_FILENAME
        self._files: dict[str, FileState] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Load manifest from disk if not already loaded."""
        if self._loaded:
            return

        self._index_dir.mkdir(parents=True, exist_ok=True)

        if self._manifest_path.exists():
            try:
                data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
                self._files = {
                    path: FileState(mtime=entry["mtime"], size=entry["size"])
                    for path, entry in data.get("files", {}).items()
                }
                log.debug("Loaded manifest with %d files", len(self._files))
            except (json.JSONDecodeError, KeyError) as e:
                log.warning("Corrupted manifest, starting fresh: %s", e)
                self._files = {}
        else:
            self._files = {}

        self._loaded = True

    def save(self) -> None:
        """Persist manifest to disk."""
        self._index_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "files": {
                path: {"mtime": state.mtime, "size": state.size}
                for path, state in self._files.items()
            },
        }
        self._manifest_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        log.debug("Saved manifest with %d files", len(self._files))

    def get_file_state(self, relative_path: str) -> FileState | None:
        """Get the stored state for a file.

        Args:
            relative_path: Path relative to KB root.

        Returns:
            FileState if tracked, None otherwise.
        """
        self._ensure_loaded()
        return self._files.get(relative_path)

    def update_file(self, relative_path: str, mtime: float, size: int) -> None:
        """Update the stored state for a file.

        Args:
            relative_path: Path relative to KB root.
            mtime: Modification time as timestamp.
            size: File size in bytes.
        """
        self._ensure_loaded()
        self._files[relative_path] = FileState(mtime=mtime, size=size)

    def remove_file(self, relative_path: str) -> None:
        """Remove a file from the manifest.

        Args:
            relative_path: Path relative to KB root.
        """
        self._ensure_loaded()
        self._files.pop(relative_path, None)

    def get_all_paths(self) -> set[str]:
        """Get all tracked file paths.

        Returns:
            Set of relative paths currently in the manifest.
        """
        self._ensure_loaded()
        return set(self._files.keys())

    def clear(self) -> None:
        """Clear all tracked files."""
        self._files = {}
        self._loaded = True
        if self._manifest_path.exists():
            self._manifest_path.unlink()

    def is_file_changed(self, relative_path: str, current_mtime: float, current_size: int) -> bool:
        """Check if a file has changed since last index.

        Args:
            relative_path: Path relative to KB root.
            current_mtime: Current modification time.
            current_size: Current file size.

        Returns:
            True if file is new or modified, False if unchanged.
        """
        self._ensure_loaded()
        state = self._files.get(relative_path)
        if state is None:
            return True  # New file
        # Check both mtime and size for robustness
        return state.mtime != current_mtime or state.size != current_size
