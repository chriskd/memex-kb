"""Helpers for `mx doctor` audits and repairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .config import get_kb_roots_for_indexing
from .timestamps import (
    coerce_datetime,
    extract_frontmatter,
    format_timestamp_for_storage,
    get_filesystem_timestamps as load_filesystem_timestamps,
    parse_frontmatter_mapping,
)


@dataclass
class TimestampFieldAudit:
    """Audit state for a single frontmatter timestamp field."""

    status: str
    before: str | None
    after: str | None
    source: str | None
    changed: bool
    would_change: bool


@dataclass
class TimestampFileAudit:
    """Audit result for one markdown entry."""

    path: str
    scope: str | None
    created: TimestampFieldAudit
    updated: TimestampFieldAudit
    changed: bool
    would_change: bool


@dataclass
class TimestampError:
    """File-level error encountered during audit/fix."""

    path: str
    scope: str | None
    error: str


class TimestampDoctorError(Exception):
    """Raised when a file cannot be audited or updated safely."""


def audit_frontmatter_timestamps(
    *,
    fix: bool,
    dry_run: bool,
    force: bool,
    scope: str | None,
    limit: int | None,
) -> dict[str, Any]:
    """Audit and optionally fix frontmatter created/updated timestamps."""

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    checked = 0
    changed = 0
    would_change = 0
    candidates = 0
    remaining = limit

    for scope_label, kb_root in get_kb_roots_for_indexing(scope=scope):
        if remaining is not None and remaining <= 0:
            break
        if not kb_root.exists():
            continue

        md_files = sorted(p for p in kb_root.rglob("*.md") if not p.name.startswith("_"))
        if remaining is not None:
            md_files = md_files[:remaining]

        for md_file in md_files:
            checked += 1
            rel_path = str(md_file.relative_to(kb_root))
            display_path = f"@{scope_label}/{rel_path}" if scope_label else rel_path

            try:
                file_result = _audit_timestamp_file(
                    md_file,
                    display_path=display_path,
                    scope=scope_label,
                    fix=fix,
                    dry_run=dry_run,
                    force=force,
                )
            except TimestampDoctorError as exc:
                errors.append(
                    asdict(
                        TimestampError(
                            path=display_path,
                            scope=scope_label,
                            error=str(exc),
                        )
                    )
                )
                continue
            if file_result is None:
                continue

            if file_result["changed"]:
                changed += 1
            if file_result["would_change"]:
                would_change += 1
            if file_result["changed"] or file_result["would_change"]:
                candidates += 1
                results.append(file_result)

        if remaining is not None:
            remaining -= len(md_files)

    return {
        "summary": {
            "checked": checked,
            "candidates": candidates,
            "changed": changed,
            "would_change": would_change,
            "skipped": max(checked - candidates - len(errors), 0),
            "errors": len(errors),
        },
        "entries": results,
        "errors": errors,
    }


def _audit_timestamp_file(
    path: Path,
    *,
    display_path: str,
    scope: str | None,
    fix: bool,
    dry_run: bool,
    force: bool,
) -> dict[str, Any] | None:
    raw_content = path.read_text(encoding="utf-8")
    try:
        prefix, frontmatter_block, body = extract_frontmatter(raw_content)
    except ValueError as exc:
        raise TimestampDoctorError(str(exc)) from exc
    if frontmatter_block is None:
        return None

    try:
        metadata = parse_frontmatter_mapping(frontmatter_block)
    except ValueError as exc:
        raise TimestampDoctorError(str(exc)) from exc
    created_before = _format_frontmatter_scalar(metadata.get("created"))
    updated_before = _format_frontmatter_scalar(metadata.get("updated"))

    created_parsed = coerce_datetime(metadata.get("created"))
    updated_parsed = coerce_datetime(metadata.get("updated"))

    created_source, created_fs, updated_source, updated_fs = _get_filesystem_timestamps(path)

    created_status = _timestamp_status(created_parsed, created_before)
    updated_status = _updated_timestamp_status(updated_parsed, updated_before, updated_fs)

    created_after = created_fs if force or created_status in {"missing", "invalid"} else None
    updated_after = updated_fs if force or updated_status in {"missing", "invalid", "stale"} else None

    created_changed = created_after is not None and created_after != created_before
    updated_changed = updated_after is not None and updated_after != updated_before
    would_change = created_changed or updated_changed
    did_change = False

    if fix and not dry_run and would_change:
        updated_frontmatter = _apply_timestamp_updates(
            frontmatter_block,
            created=created_after if created_changed else None,
            updated=updated_after if updated_changed else None,
        )
        path.write_text(prefix + updated_frontmatter + body, encoding="utf-8")
        did_change = True

    file_audit = TimestampFileAudit(
        path=display_path,
        scope=scope,
        created=TimestampFieldAudit(
            status=created_status,
            before=created_before,
            after=created_after if created_changed else None,
            source=created_source if created_changed else None,
            changed=did_change and created_changed,
            would_change=(not did_change) and created_changed,
        ),
        updated=TimestampFieldAudit(
            status=updated_status,
            before=updated_before,
            after=updated_after if updated_changed else None,
            source=updated_source if updated_changed else None,
            changed=did_change and updated_changed,
            would_change=(not did_change) and updated_changed,
        ),
        changed=did_change,
        would_change=(not did_change) and would_change,
    )
    return asdict(file_audit)


def _format_frontmatter_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _timestamp_status(parsed: datetime | None, original: str | None) -> str:
    if parsed is not None:
        return "valid"
    if original is None:
        return "missing"
    return "invalid"

def _updated_timestamp_status(parsed: datetime | None, original: str | None, filesystem: str | None) -> str:
    status = _timestamp_status(parsed, original)
    if status != "valid":
        return status
    if format_timestamp_for_storage(parsed) != filesystem:
        return "stale"
    return "valid"


def _get_filesystem_timestamps(path: Path) -> tuple[str, str, str, str]:
    fs_timestamps = load_filesystem_timestamps(path)
    created = format_timestamp_for_storage(fs_timestamps.created)
    updated = format_timestamp_for_storage(fs_timestamps.updated)
    assert created is not None
    assert updated is not None
    return fs_timestamps.created_source, created, fs_timestamps.updated_source, updated


def _apply_timestamp_updates(
    frontmatter_block: str,
    *,
    created: str | None,
    updated: str | None,
) -> str:
    lines = frontmatter_block.splitlines(keepends=True)
    if not lines:
        raise TimestampDoctorError("Invalid frontmatter block")

    newline = "\n"
    for line in lines:
        if line.endswith("\r\n"):
            newline = "\r\n"
            break

    if created is not None:
        created_idx = _find_top_level_key(lines, "created")
        if created_idx is None:
            updated_idx = _find_top_level_key(lines, "updated")
            insert_at = updated_idx if updated_idx is not None else len(lines) - 1
            lines.insert(insert_at, f"created: {created}{newline}")
        else:
            lines[created_idx] = _replace_scalar_line(lines[created_idx], "created", created)

    if updated is not None:
        updated_idx = _find_top_level_key(lines, "updated")
        if updated_idx is None:
            created_idx = _find_top_level_key(lines, "created")
            insert_at = created_idx + 1 if created_idx is not None else len(lines) - 1
            lines.insert(insert_at, f"updated: {updated}{newline}")
        else:
            lines[updated_idx] = _replace_scalar_line(lines[updated_idx], "updated", updated)

    return "".join(lines)


def _find_top_level_key(lines: list[str], key: str) -> int | None:
    prefix = f"{key}:"
    for idx, line in enumerate(lines):
        if line.startswith(prefix):
            return idx
    return None


def _replace_scalar_line(line: str, key: str, value: str) -> str:
    if line.endswith("\r\n"):
        ending = "\r\n"
    elif line.endswith("\n"):
        ending = "\n"
    else:
        ending = ""
    return f"{key}: {value}{ending}"
