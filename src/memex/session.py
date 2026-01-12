"""Session context persistence for memex.

Provides global, explicit-lifetime session state for filtering and boosting
KB searches. Unlike `.kbcontext` (per-directory, static), session context is
global, dynamic, and persists until explicitly cleared.

Session context supports:
- Tags: Filter searches to entries with these tags
- Project: Boost entries from this project in results
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import get_index_root

SESSION_FILENAME = "session_context.json"
SCHEMA_VERSION = 1


@dataclass
class SessionContext:
    """Active session context for filtering and boosting searches.

    Attributes:
        tags: Tags to filter search results (intersection).
        project: Project name to boost in search results.
    """

    tags: list[str] = field(default_factory=list)
    project: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionContext:
        """Create SessionContext from parsed JSON dict."""
        return cls(
            tags=data.get("tags", []),
            project=data.get("project"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "tags": self.tags,
            "project": self.project,
        }

    def is_empty(self) -> bool:
        """Check if session has no active context."""
        return not self.tags and self.project is None

    def merge_tags(self, cli_tags: list[str] | None) -> list[str] | None:
        """Merge session tags with CLI-provided tags (union).

        Args:
            cli_tags: Tags provided via command line.

        Returns:
            Merged tag list, or None if both sources are empty.
        """
        if not cli_tags and not self.tags:
            return None
        return list(set(self.tags) | set(cli_tags or []))


def _session_path(index_root: Path | None = None) -> Path:
    """Get path to session_context.json, creating directory if needed."""
    root = index_root or get_index_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / SESSION_FILENAME


def load_session(index_root: Path | None = None) -> SessionContext:
    """Load session context from disk.

    Returns:
        SessionContext (empty if no session file exists or on error).
    """
    path = _session_path(index_root)
    if not path.exists():
        return SessionContext()

    try:
        payload: dict[str, Any] = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return SessionContext()

    # Handle schema migration if needed
    version = payload.get("schema_version", 1)
    if version != SCHEMA_VERSION:
        return SessionContext()  # Reset on schema change

    return SessionContext.from_dict(payload.get("session", {}))


def save_session(
    session: SessionContext,
    index_root: Path | None = None,
) -> None:
    """Save session context to disk.

    Uses atomic write pattern (write to temp, then rename).

    Args:
        session: The session context to save.
        index_root: Optional override for index storage location.
    """
    path = _session_path(index_root)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "session": session.to_dict(),
    }

    # Atomic write
    dir_path = path.parent
    with tempfile.NamedTemporaryFile(
        mode="w", dir=dir_path, delete=False, suffix=".tmp"
    ) as f:
        json.dump(payload, f, indent=2)
        temp_path = Path(f.name)

    temp_path.rename(path)


def clear_session(index_root: Path | None = None) -> bool:
    """Clear the session context.

    Args:
        index_root: Optional override for index storage location.

    Returns:
        True if session existed and was cleared, False if no session.
    """
    path = _session_path(index_root)
    if not path.exists():
        return False

    # Check if there was an active session
    existing = load_session(index_root)
    had_session = not existing.is_empty()

    # Write empty session (preserves file for debugging)
    save_session(SessionContext(), index_root)
    return had_session


def get_session(index_root: Path | None = None) -> SessionContext:
    """Get current session context (convenience wrapper).

    Args:
        index_root: Optional override for index storage location.

    Returns:
        Current SessionContext (may be empty).
    """
    return load_session(index_root)
