"""Persistent health audit cache management.

Caches per-file metadata (title, created, updated, links) with mtime-based
invalidation to avoid parsing every file on each health audit.

This enables O(n) incremental updates instead of O(n) full parses, where only
changed files need to be re-parsed.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from .config import get_index_root
from .parser import ParseError, extract_links, parse_entry

CACHE_FILENAME = "health_cache.json"


def _cache_path(index_root: Path | None = None) -> Path:
    """Get path to health_cache.json, creating directory if needed."""
    root = index_root or get_index_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / CACHE_FILENAME


def load_cache(index_root: Path | None = None) -> tuple[dict[str, dict[str, Any]], float]:
    """Load health metadata cache from disk.

    Returns:
        Tuple of (file_cache, kb_mtime) where file_cache maps relative paths
        (without .md) to metadata dicts containing:
        - mtime: file modification time
        - title: entry title
        - created: ISO date string or None
        - updated: ISO date string or None
        - links: list of outgoing link targets
    """
    path = _cache_path(index_root)
    if not path.exists():
        return {}, 0.0

    try:
        payload: dict[str, Any] = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}, 0.0

    return payload.get("files", {}), float(payload.get("kb_mtime", 0.0))


def save_cache(
    files: dict[str, dict[str, Any]],
    kb_mtime: float,
    index_root: Path | None = None,
) -> None:
    """Save health metadata cache to disk."""
    path = _cache_path(index_root)
    payload = {"kb_mtime": kb_mtime, "files": files}
    path.write_text(json.dumps(payload, indent=2))


def _get_file_mtime(file_path: Path) -> float:
    """Get file modification time, returning 0.0 on error."""
    try:
        return file_path.stat().st_mtime
    except OSError:
        return 0.0


def _date_to_str(d: date | None) -> str | None:
    """Convert date to ISO string for JSON serialization."""
    return d.isoformat() if d else None


def _str_to_date(s: str | None) -> date | None:
    """Convert ISO string back to date."""
    return date.fromisoformat(s) if s else None


def _parse_file_metadata(md_file: Path) -> dict[str, Any] | None:
    """Parse a single file and extract health-relevant metadata.

    Returns:
        Dict with title, created, updated, links, or None on parse error.
    """
    try:
        metadata, content, _ = parse_entry(md_file)
        links = extract_links(content)
        return {
            "title": metadata.title,
            "created": _date_to_str(metadata.created),
            "updated": _date_to_str(metadata.updated),
            "links": links,
        }
    except ParseError:
        return None


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
    latest_mtime = 0.0

    for md_file in kb_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue

        rel_path = str(md_file.relative_to(kb_root))
        path_key = rel_path[:-3] if rel_path.endswith(".md") else rel_path
        file_mtime = _get_file_mtime(md_file)
        latest_mtime = max(latest_mtime, file_mtime)

        file_meta = _parse_file_metadata(md_file)
        if file_meta is None:
            continue

        files_cache[path_key] = {
            "mtime": file_mtime,
            "rel_path": rel_path,
            **file_meta,
        }

    save_cache(files_cache, latest_mtime, index_root)
    return files_cache


def _incremental_update(
    kb_root: Path,
    cached_files: dict[str, dict[str, Any]],
    index_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Incrementally update cache by checking individual file mtimes.

    Only re-parses files that have changed since last cache.

    Args:
        kb_root: Path to the knowledge base root.
        cached_files: Previously cached file data.
        index_root: Optional override for cache storage location.

    Returns:
        Updated dict mapping path_key to metadata dict.
    """
    updated_files: dict[str, dict[str, Any]] = {}
    latest_mtime = 0.0

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
        else:
            # Re-parse file
            file_meta = _parse_file_metadata(md_file)
            if file_meta is None:
                continue

            updated_files[path_key] = {
                "mtime": file_mtime,
                "rel_path": rel_path,
                **file_meta,
            }

    save_cache(updated_files, latest_mtime, index_root)
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

    cached_files, cached_mtime = load_cache(index_root)

    # Check if we need any update
    if not cached_files:
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
        return _incremental_update(kb_root, cached_files, index_root)

    return cached_files


def get_entry_metadata(
    kb_root: Path, index_root: Path | None = None
) -> dict[str, dict[str, Any]]:
    """Get entry metadata suitable for health checks.

    Convenience wrapper that ensures cache is up to date and converts
    date strings back to date objects.

    Args:
        kb_root: Path to the knowledge base root.
        index_root: Optional override for cache storage location.

    Returns:
        Dict mapping path_key to metadata with date objects (not strings).
    """
    cached = ensure_health_cache(kb_root, index_root)

    # Convert date strings to date objects
    result: dict[str, dict[str, Any]] = {}
    for path_key, meta in cached.items():
        result[path_key] = {
            "path": meta.get("rel_path", f"{path_key}.md"),
            "title": meta.get("title", ""),
            "created": _str_to_date(meta.get("created")),
            "updated": _str_to_date(meta.get("updated")),
            "links": meta.get("links", []),
        }

    return result
