"""Persistent health audit cache management.

Caches per-file metadata (title, created, updated, links) with mtime-based
invalidation to avoid parsing every file on each health audit.

This enables O(n) incremental updates instead of O(n) full parses, where only
changed files need to be re-parsed.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .config import get_index_root
from .parser import ParseError, extract_links, parse_entry

log = logging.getLogger(__name__)

CACHE_FILENAME = "health_cache.json"


def _cache_path(index_root: Path | None = None) -> Path:
    """Get path to health_cache.json, creating directory if needed."""
    root = index_root or get_index_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / CACHE_FILENAME


def load_cache(
    index_root: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], float, list[dict[str, str]]]:
    """Load health metadata cache from disk.

    Returns:
        Tuple of (file_cache, kb_mtime, parse_errors) where:
        - file_cache maps relative paths (without .md) to metadata dicts
        - kb_mtime is the knowledge base modification time
        - parse_errors is a list of {path, error} dicts for files that failed to parse
    """
    path = _cache_path(index_root)
    if not path.exists():
        return {}, 0.0, []

    try:
        payload: dict[str, Any] = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}, 0.0, []

    return (
        payload.get("files", {}),
        float(payload.get("kb_mtime", 0.0)),
        payload.get("parse_errors", []),
    )


def save_cache(
    files: dict[str, dict[str, Any]],
    kb_mtime: float,
    index_root: Path | None = None,
    parse_errors: list[dict[str, str]] | None = None,
) -> None:
    """Save health metadata cache to disk."""
    path = _cache_path(index_root)
    payload: dict[str, Any] = {"kb_mtime": kb_mtime, "files": files}
    if parse_errors:
        payload["parse_errors"] = parse_errors
    path.write_text(json.dumps(payload, indent=2))


def _get_file_mtime(file_path: Path) -> float:
    """Get file modification time, returning 0.0 on error."""
    try:
        return file_path.stat().st_mtime
    except OSError:
        return 0.0


def _datetime_to_str(d: date | datetime | None) -> str | None:
    """Convert datetime to ISO string for JSON serialization."""
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.replace(microsecond=0).isoformat()
    return d.isoformat()  # date object


def _str_to_datetime(s: str | None) -> datetime | None:
    """Convert ISO string back to datetime.

    Handles both date-only and full datetime strings for backwards compatibility.
    """
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        # Fall back to date-only parsing
        try:
            d = date.fromisoformat(s)
            return datetime(d.year, d.month, d.day, 0, 0, 0)
        except ValueError:
            return None


def _parse_file_metadata(md_file: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a single file and extract health-relevant metadata.

    Returns:
        Tuple of (metadata_dict, error_message).
        On success: (dict with title/description/created/updated/links, None)
        On error: (None, error message string)
    """
    try:
        metadata, content, _ = parse_entry(md_file)
        links = extract_links(content)
        return {
            "title": metadata.title,
            "description": metadata.description,
            "created": _datetime_to_str(metadata.created),
            "updated": _datetime_to_str(metadata.updated),
            "links": links,
        }, None
    except ParseError as e:
        log.warning("Parse error in %s: %s", md_file, e.message)
        return None, e.message


def rebuild_health_cache(
    kb_root: Path, index_root: Path | None = None
) -> dict[str, dict[str, Any]]:
    """Full rebuild of health metadata cache by scanning all files.

    Args:
        kb_root: Path to the knowledge base root.
        index_root: Optional override for cache storage location.

    Returns:
        Dict mapping path_key (relative path without .md) to metadata dict.
    """
    if not kb_root.exists():
        save_cache({}, 0.0, index_root)
        return {}

    files_cache: dict[str, dict[str, Any]] = {}
    parse_errors: list[dict[str, str]] = []
    latest_mtime = 0.0

    for md_file in kb_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue

        rel_path = str(md_file.relative_to(kb_root))
        path_key = rel_path[:-3] if rel_path.endswith(".md") else rel_path
        file_mtime = _get_file_mtime(md_file)
        latest_mtime = max(latest_mtime, file_mtime)

        file_meta, error = _parse_file_metadata(md_file)
        if file_meta is None:
            if error:
                parse_errors.append({"path": rel_path, "error": error})
            continue

        files_cache[path_key] = {
            "mtime": file_mtime,
            "rel_path": rel_path,
            **file_meta,
        }

    save_cache(files_cache, latest_mtime, index_root, parse_errors)
    return files_cache


def _incremental_update(
    kb_root: Path,
    cached_files: dict[str, dict[str, Any]],
    cached_parse_errors: list[dict[str, str]],
    index_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Incrementally update cache by checking individual file mtimes.

    Only re-parses files that have changed since last cache.

    Args:
        kb_root: Path to the knowledge base root.
        cached_files: Previously cached file data.
        cached_parse_errors: Previously cached parse errors.
        index_root: Optional override for cache storage location.

    Returns:
        Updated dict mapping path_key to metadata dict.
    """
    updated_files: dict[str, dict[str, Any]] = {}
    parse_errors: list[dict[str, str]] = []
    latest_mtime = 0.0

    # Build set of previously errored paths for quick lookup
    prev_error_paths = {e["path"] for e in cached_parse_errors}

    for md_file in kb_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue

        rel_path = str(md_file.relative_to(kb_root))
        path_key = rel_path[:-3] if rel_path.endswith(".md") else rel_path
        file_mtime = _get_file_mtime(md_file)
        latest_mtime = max(latest_mtime, file_mtime)

        # Check if file is in cache and unchanged
        cached = cached_files.get(path_key)
        if cached and cached.get("mtime", 0) >= file_mtime:
            # Use cached metadata
            updated_files[path_key] = cached
        elif rel_path in prev_error_paths and cached_files.get(path_key) is None:
            # File previously had error and hasn't changed - keep error
            for err in cached_parse_errors:
                if err["path"] == rel_path:
                    parse_errors.append(err)
                    break
        else:
            # Re-parse file (new, changed, or previously errored but now modified)
            file_meta, error = _parse_file_metadata(md_file)
            if file_meta is None:
                if error:
                    parse_errors.append({"path": rel_path, "error": error})
                continue

            updated_files[path_key] = {
                "mtime": file_mtime,
                "rel_path": rel_path,
                **file_meta,
            }

    save_cache(updated_files, latest_mtime, index_root, parse_errors)
    return updated_files


def ensure_health_cache(
    kb_root: Path, index_root: Path | None = None
) -> dict[str, dict[str, Any]]:
    """Get health metadata for all entries, using cache when valid.

    Uses incremental update strategy:
    1. If no cache exists, full rebuild
    2. If any file is newer than cache mtime, do incremental update
    3. Otherwise, return cached data directly

    Args:
        kb_root: Path to the knowledge base root.
        index_root: Optional override for cache storage location.

    Returns:
        Dict mapping path_key (relative path without .md) to metadata dict
        containing: rel_path, title, created, updated, links, mtime.
    """
    if not kb_root.exists():
        return {}

    cached_files, cached_mtime, cached_parse_errors = load_cache(index_root)

    # Check if we need any update
    if not cached_files and not cached_parse_errors:
        return rebuild_health_cache(kb_root, index_root)

    # Check for any file newer than cached mtime
    needs_update = False
    for md_file in kb_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue
        file_mtime = _get_file_mtime(md_file)
        if file_mtime > cached_mtime:
            needs_update = True
            break

    # Also check for deleted files
    if not needs_update:
        current_paths = {
            str(f.relative_to(kb_root))[:-3]  # Remove .md extension
            for f in kb_root.rglob("*.md")
            if not f.name.startswith("_")
        }
        if set(cached_files.keys()) != current_paths:
            needs_update = True

    if needs_update:
        return _incremental_update(kb_root, cached_files, cached_parse_errors, index_root)

    return cached_files


def get_entry_metadata(
    kb_root: Path, index_root: Path | None = None
) -> dict[str, dict[str, Any]]:
    """Get entry metadata suitable for health checks.

    Convenience wrapper that ensures cache is up to date and converts
    datetime strings back to datetime objects.

    Args:
        kb_root: Path to the knowledge base root.
        index_root: Optional override for cache storage location.

    Returns:
        Dict mapping path_key to metadata with datetime objects (not strings).
    """
    cached = ensure_health_cache(kb_root, index_root)

    # Convert datetime strings to datetime objects
    result: dict[str, dict[str, Any]] = {}
    for path_key, meta in cached.items():
        result[path_key] = {
            "path": meta.get("rel_path", f"{path_key}.md"),
            "title": meta.get("title", ""),
            "description": meta.get("description"),
            "created": _str_to_datetime(meta.get("created")),
            "updated": _str_to_datetime(meta.get("updated")),
            "links": meta.get("links", []),
        }

    return result


def get_parse_errors(index_root: Path | None = None) -> list[dict[str, str]]:
    """Get parse errors from the health cache.

    Args:
        index_root: Optional override for cache storage location.

    Returns:
        List of {path, error} dicts for files that failed to parse.
    """
    _, _, parse_errors = load_cache(index_root)
    return parse_errors
