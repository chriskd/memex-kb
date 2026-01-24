"""Persistent tag taxonomy cache management.

Caches per-file tag counts with mtime-based invalidation to avoid full KB scans
when computing tag taxonomy for suggestions and tag listing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import get_index_root
from .parser import ParseError, parse_entry

log = logging.getLogger(__name__)

CACHE_FILENAME = "tags_cache.json"


def _cache_path(index_root: Path | None = None) -> Path:
    """Get path to tags_cache.json, creating directory if needed."""
    root = index_root or get_index_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / CACHE_FILENAME


def load_cache(index_root: Path | None = None) -> tuple[dict[str, dict[str, Any]], float]:
    """Load tag cache from disk.

    Returns:
        Tuple of (file_cache, kb_mtime) where file_cache maps relative paths
        to {mtime, tags} dicts.
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
    """Save tag cache to disk."""
    path = _cache_path(index_root)
    payload = {"kb_mtime": kb_mtime, "files": files}
    path.write_text(json.dumps(payload, indent=2))


def _get_file_mtime(file_path: Path) -> float:
    """Get file modification time, returning 0.0 on error."""
    try:
        return file_path.stat().st_mtime
    except OSError:
        return 0.0


def rebuild_tags_cache(kb_root: Path, index_root: Path | None = None) -> dict[str, int]:
    """Full rebuild of tag cache by scanning all files.

    Args:
        kb_root: Path to the knowledge base root.
        index_root: Optional override for cache storage location.

    Returns:
        Dict mapping tag name to usage count.
    """
    if not kb_root.exists():
        save_cache({}, 0.0, index_root)
        return {}

    files_cache: dict[str, dict[str, Any]] = {}
    tag_counts: dict[str, int] = {}
    latest_mtime = 0.0

    for md_file in kb_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue

        rel_path = str(md_file.relative_to(kb_root))
        file_mtime = _get_file_mtime(md_file)
        latest_mtime = max(latest_mtime, file_mtime)

        try:
            metadata, _, _ = parse_entry(md_file)
            tags = list(metadata.tags)
        except ParseError as e:
            log.warning("Parse error in %s: %s", md_file, e.message)
            continue

        files_cache[rel_path] = {
            "mtime": file_mtime,
            "tags": tags,
        }

        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    save_cache(files_cache, latest_mtime, index_root)
    return tag_counts


def _incremental_update(
    kb_root: Path,
    cached_files: dict[str, dict[str, Any]],
    index_root: Path | None = None,
) -> dict[str, int]:
    """Incrementally update cache by checking individual file mtimes.

    Only re-parses files that have changed since last cache.

    Args:
        kb_root: Path to the knowledge base root.
        cached_files: Previously cached file data.
        index_root: Optional override for cache storage location.

    Returns:
        Dict mapping tag name to usage count.
    """
    tag_counts: dict[str, int] = {}
    updated_files: dict[str, dict[str, Any]] = {}
    latest_mtime = 0.0

    # Track which cached files still exist
    current_files: set[str] = set()

    for md_file in kb_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue

        rel_path = str(md_file.relative_to(kb_root))
        current_files.add(rel_path)
        file_mtime = _get_file_mtime(md_file)
        latest_mtime = max(latest_mtime, file_mtime)

        # Check if file is in cache and unchanged
        cached = cached_files.get(rel_path)
        if cached and cached.get("mtime", 0) >= file_mtime:
            # Use cached tags
            tags = cached.get("tags", [])
            updated_files[rel_path] = cached
        else:
            # Re-parse file
            try:
                metadata, _, _ = parse_entry(md_file)
                tags = list(metadata.tags)
            except ParseError as e:
                log.warning("Parse error in %s: %s", md_file, e.message)
                continue

            updated_files[rel_path] = {
                "mtime": file_mtime,
                "tags": tags,
            }

        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    save_cache(updated_files, latest_mtime, index_root)
    return tag_counts


def ensure_tags_cache(kb_root: Path, index_root: Path | None = None) -> dict[str, int]:
    """Get tag taxonomy, using cache when valid.

    Uses incremental update strategy:
    1. If no cache exists, full rebuild
    2. If any file is newer than cache mtime, do incremental update
    3. Otherwise, aggregate from cached data

    Args:
        kb_root: Path to the knowledge base root.
        index_root: Optional override for cache storage location.

    Returns:
        Dict mapping tag name to usage count.
    """
    if not kb_root.exists():
        return {}

    cached_files, cached_mtime = load_cache(index_root)

    # Check if we need any update
    if not cached_files:
        return rebuild_tags_cache(kb_root, index_root)

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
            str(f.relative_to(kb_root))
            for f in kb_root.rglob("*.md")
            if not f.name.startswith("_")
        }
        if set(cached_files.keys()) != current_paths:
            needs_update = True

    if needs_update:
        return _incremental_update(kb_root, cached_files, index_root)

    # Cache is fresh - aggregate from cached data
    tag_counts: dict[str, int] = {}
    for file_data in cached_files.values():
        for tag in file_data.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    return tag_counts


def get_tag_entries(kb_root: Path, index_root: Path | None = None) -> dict[str, list[str]]:
    """Get all tags with their entry paths.

    Similar to ensure_tags_cache but returns entry paths per tag
    instead of just counts. Uses the same caching mechanism.

    Args:
        kb_root: Path to the knowledge base root.
        index_root: Optional override for cache storage location.

    Returns:
        Dict mapping tag name to list of entry paths.
    """
    if not kb_root.exists():
        return {}

    # Ensure cache is up to date (this updates the cache if needed)
    ensure_tags_cache(kb_root, index_root)

    # Load fresh cache
    cached_files, _ = load_cache(index_root)

    tag_entries: dict[str, list[str]] = {}
    for rel_path, file_data in cached_files.items():
        for tag in file_data.get("tags", []):
            if tag not in tag_entries:
                tag_entries[tag] = []
            tag_entries[tag].append(rel_path)

    return tag_entries
