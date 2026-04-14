"""Timestamp helpers shared across recency and repair flows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter

_DATETIME_ADAPTER = TypeAdapter(datetime)


@dataclass(frozen=True)
class FilesystemTimestamps:
    """Best-effort filesystem timestamps for an entry."""

    created_source: str
    created: datetime
    updated_source: str
    updated: datetime


@dataclass(frozen=True)
class RecencySnapshot:
    """Best-effort metadata used for recent-entry views."""

    title: str
    tags: list[str]
    source_project: str | None
    created: datetime | None
    updated: datetime | None
    effective_created: datetime
    effective_updated: datetime


def ensure_aware(dt: datetime | None) -> datetime | None:
    """Return a timezone-aware datetime, assuming UTC for naive values."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def coerce_datetime(value: Any) -> datetime | None:
    """Parse a date-like scalar into a datetime when possible."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_aware(value)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=UTC)
    try:
        return ensure_aware(_DATETIME_ADAPTER.validate_python(value))
    except Exception:
        return None


def format_timestamp_for_storage(value: datetime | None) -> str | None:
    """Normalize a datetime to the frontmatter storage format."""
    if value is None:
        return None
    normalized = ensure_aware(value)
    if normalized is None:
        return None
    return normalized.astimezone(UTC).replace(microsecond=0).isoformat()


def get_filesystem_timestamps(path: Path) -> FilesystemTimestamps:
    """Return best-effort created/updated datetimes from the filesystem."""
    stat = path.stat()
    if hasattr(stat, "st_birthtime"):
        created_source = "birthtime"
        created_dt = datetime.fromtimestamp(stat.st_birthtime, UTC)
    else:
        created_source = "ctime"
        created_dt = datetime.fromtimestamp(stat.st_ctime, UTC)
    updated_dt = datetime.fromtimestamp(stat.st_mtime, UTC)
    return FilesystemTimestamps(
        created_source=created_source,
        created=created_dt.replace(microsecond=0),
        updated_source="mtime",
        updated=updated_dt.replace(microsecond=0),
    )


def extract_frontmatter(content: str) -> tuple[str, str | None, str]:
    """Return leading prefix, frontmatter block, and body content."""
    lines = content.splitlines(keepends=True)
    start = 0
    while start < len(lines) and lines[start].strip() == "":
        start += 1

    if start >= len(lines) or lines[start].strip() != "---":
        return "", None, content

    end = start + 1
    while end < len(lines):
        if lines[end].strip() in {"---", "..."}:
            prefix = "".join(lines[:start])
            frontmatter_block = "".join(lines[start : end + 1])
            body = "".join(lines[end + 1 :])
            return prefix, frontmatter_block, body
        end += 1

    raise ValueError("Unterminated YAML frontmatter")


def parse_frontmatter_mapping(frontmatter_block: str) -> dict[str, Any]:
    """Parse a YAML frontmatter block into a mapping."""
    lines = frontmatter_block.splitlines()
    if len(lines) < 2:
        raise ValueError("Invalid frontmatter block")

    inner = "\n".join(lines[1:-1])
    try:
        data = yaml.safe_load(inner) if inner.strip() else {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse frontmatter YAML: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Frontmatter must be a YAML mapping")
    return data


def derive_entry_title(path: Path, raw_content: str) -> str:
    """Best-effort title for a partially valid entry."""
    for line in raw_content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or path.stem.replace("-", " ").title()
    return path.stem.replace("-", " ").replace("_", " ").title()


def read_recency_snapshot(path: Path) -> RecencySnapshot | None:
    """Read enough metadata to support recent-entry views.

    This is intentionally more forgiving than full entry parsing so recent views can
    still work when timestamps are missing and agents edited files directly.
    """
    raw_content = path.read_text(encoding="utf-8")
    _prefix, frontmatter_block, body = extract_frontmatter(raw_content)

    title = derive_entry_title(path, body if frontmatter_block is not None else raw_content)
    tags: list[str] = []
    source_project: str | None = None
    created: datetime | None = None
    updated: datetime | None = None

    if frontmatter_block is not None:
        try:
            metadata = parse_frontmatter_mapping(frontmatter_block)
        except ValueError:
            return None

        if isinstance(metadata.get("title"), str) and metadata["title"].strip():
            title = metadata["title"].strip()
        raw_tags = metadata.get("tags")
        if isinstance(raw_tags, list):
            tags = [str(tag) for tag in raw_tags if str(tag).strip()]
        elif isinstance(raw_tags, str) and raw_tags.strip():
            tags = [raw_tags.strip()]
        if isinstance(metadata.get("source_project"), str):
            source_project = metadata["source_project"]
        created = coerce_datetime(metadata.get("created"))
        updated = coerce_datetime(metadata.get("updated"))

    fs = get_filesystem_timestamps(path)
    effective_created = created or fs.created
    if updated is None:
        effective_updated = fs.updated
    else:
        effective_updated = max(updated, fs.updated)

    return RecencySnapshot(
        title=title,
        tags=tags,
        source_project=source_project,
        created=created,
        updated=updated,
        effective_created=effective_created,
        effective_updated=effective_updated,
    )
