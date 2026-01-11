"""Persistent backlink cache management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import get_index_root
from .parser import resolve_backlinks

CACHE_FILENAME = "backlinks.json"


def _cache_path(index_root: Path | None = None) -> Path:
    root = index_root or get_index_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / CACHE_FILENAME


def _kb_tree_mtime(kb_root: Path) -> float:
    latest = 0.0
    if not kb_root.exists():
        return latest

    for md_file in kb_root.rglob("*.md"):
        try:
            latest = max(latest, md_file.stat().st_mtime)
        except OSError:
            continue

    return latest


def load_cache() -> tuple[dict[str, list[str]], float]:
    path = _cache_path()
    if not path.exists():
        return {}, 0.0

    try:
        payload: dict[str, Any] = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}, 0.0

    return payload.get("backlinks", {}), float(payload.get("kb_mtime", 0.0))


def save_cache(backlinks: dict[str, list[str]], kb_mtime: float) -> None:
    path = _cache_path()
    payload = {"kb_mtime": kb_mtime, "backlinks": backlinks}
    path.write_text(json.dumps(payload, indent=2))


def rebuild_backlink_cache(kb_root: Path) -> dict[str, list[str]]:
    backlinks = resolve_backlinks(kb_root)
    kb_mtime = _kb_tree_mtime(kb_root)
    save_cache(backlinks, kb_mtime)
    return backlinks


def ensure_backlink_cache(kb_root: Path) -> dict[str, list[str]]:
    backlinks, cached_mtime = load_cache()
    current_mtime = _kb_tree_mtime(kb_root)

    if current_mtime > cached_mtime or not backlinks:
        return rebuild_backlink_cache(kb_root)

    return backlinks
