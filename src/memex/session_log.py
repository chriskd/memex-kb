"""Agent-authored session handoff entries for Memex.

This module deliberately stores handoffs as ordinary KB entries. The portable
contract is the CLI and Markdown shape, not a harness-specific transcript file.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from .config import (
    ConfigurationError,
    get_kb_root,
    get_kb_root_by_scope,
    get_project_kb_root,
    get_user_kb_root,
    parse_scoped_path,
    resolve_scoped_path,
)
from .core import (
    add_entry,
    get_actor_identity,
    get_current_project,
    get_git_branch,
    get_llm_model,
    slugify,
    update_entry,
)
from .parser import ParseError, parse_entry
from .timestamps import ensure_aware

SESSION_DIRECTORY = "sessions"
SESSION_TAG = "session"
HANDOFF_TAG = "handoff"
OPEN_TAG = "session-open"
CLOSED_TAG = "session-closed"
HOOK_CONTEXT_COMMAND = "mx sessions hook context"


@dataclass
class SessionRecord:
    """A summarized session handoff entry."""

    path: str
    title: str
    project: str
    harness: str | None
    transcript: str | None
    status: Literal["open", "closed", "unknown"]
    summary: str
    updated: str | None
    tags: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "title": self.title,
            "project": self.project,
            "harness": self.harness,
            "transcript": self.transcript,
            "status": self.status,
            "summary": self.summary,
            "updated": self.updated,
            "tags": self.tags,
        }


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _format_ts(value: datetime | None = None) -> str:
    return (value or _now()).isoformat()


def _tag_slug(value: str) -> str:
    slug = slugify(value)
    return slug or "unknown"


def _project_tag(project: str) -> str:
    return f"project-{_tag_slug(project)}"


def _harness_tag(harness: str) -> str:
    return f"harness-{_tag_slug(harness)}"


def detect_harness() -> str | None:
    """Best-effort harness name from environment variables."""
    explicit = os.environ.get("MEMEX_HARNESS")
    if explicit:
        return explicit

    # These checks are intentionally conservative. Unknown is better than a bad label.
    if os.environ.get("CLAUDECODE") or os.environ.get("CLAUDE_CODE"):
        return "claude"
    if os.environ.get("CODEX_HOME") or os.environ.get("CODEX_SANDBOX"):
        return "codex"
    return None


def _current_project(project: str | None = None) -> str:
    return project or get_current_project() or Path.cwd().name


def _display_path(scope: str | None, relative_path: str) -> str:
    return f"@{scope}/{relative_path}" if scope else relative_path


def _roots_for_scope(scope: str | None) -> list[tuple[str | None, Path]]:
    if scope:
        return [(scope, get_kb_root_by_scope(scope))]

    project_kb = get_project_kb_root()
    user_kb = get_user_kb_root()
    if project_kb and user_kb and project_kb != user_kb:
        return [("project", project_kb), ("user", user_kb)]
    if project_kb:
        return [(None, project_kb)]
    if user_kb:
        return [(None, user_kb)]

    # Raise the canonical configuration guidance.
    return [(None, get_kb_root())]


def _section_text(content: str, heading: str) -> str | None:
    pattern = re.compile(rf"^## {re.escape(heading)}\s*$", re.MULTILINE)
    match = pattern.search(content)
    if not match:
        return None

    start = match.end()
    next_heading = re.search(r"^##\s+", content[start:], re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(content)
    value = content[start:end].strip()
    return value or None


def _section_first_value(content: str, heading: str) -> str | None:
    section = _section_text(content, heading)
    if not section:
        return None
    for line in section.splitlines():
        cleaned = line.strip().removeprefix("-").strip().strip("`")
        if cleaned and cleaned.lower() != "none recorded.":
            return cleaned
    return None


def _replace_h2_section(content: str, heading: str, replacement: str) -> str:
    heading_line = f"## {heading}"
    pattern = re.compile(rf"^## {re.escape(heading)}\s*$", re.MULTILINE)
    match = pattern.search(content)
    section = f"{heading_line}\n\n{replacement.strip()}\n"
    if not match:
        return content.rstrip() + "\n\n" + section

    start = match.start()
    after_heading = match.end()
    next_heading = re.search(r"^##\s+", content[after_heading:], re.MULTILINE)
    end = after_heading + next_heading.start() if next_heading else len(content)
    return content[:start].rstrip() + "\n\n" + section + "\n" + content[end:].lstrip()


def _append_to_h2_section(content: str, heading: str, addition: str) -> str:
    pattern = re.compile(rf"^## {re.escape(heading)}\s*$", re.MULTILINE)
    match = pattern.search(content)
    if not match:
        return content.rstrip() + f"\n\n## {heading}\n\n{addition.strip()}\n"

    start = match.end()
    next_heading = re.search(r"^##\s+", content[start:], re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(content)
    section_body = content[start:end].rstrip()
    inserted = section_body + "\n\n" + addition.strip() + "\n\n"
    return content[:start] + inserted + content[end:].lstrip()


def _status_from_tags(tags: list[str]) -> Literal["open", "closed", "unknown"]:
    if CLOSED_TAG in tags:
        return "closed"
    if OPEN_TAG in tags:
        return "open"
    return "unknown"


def _updated_iso(created: object, updated: object) -> str | None:
    dt = ensure_aware(updated if isinstance(updated, datetime) else None)
    if dt is None:
        dt = ensure_aware(created if isinstance(created, datetime) else None)
    return dt.isoformat() if dt else None


def _record_from_entry(
    *,
    scope: str | None,
    kb_root: Path,
    path: Path,
) -> SessionRecord | None:
    try:
        metadata, content, _ = parse_entry(path)
    except (ParseError, OSError, UnicodeDecodeError):
        return None

    if SESSION_TAG not in metadata.tags or HANDOFF_TAG not in metadata.tags:
        return None

    relative = str(path.relative_to(kb_root))
    project = "unknown"
    for tag in metadata.tags:
        if tag.startswith("project-"):
            project = tag.removeprefix("project-")
            break

    harness = None
    for tag in metadata.tags:
        if tag.startswith("harness-"):
            harness = tag.removeprefix("harness-")
            break

    summary = (
        _section_text(content, "Final Summary")
        or _section_text(content, "Current Summary")
        or _section_text(content, "Goal")
        or ""
    )
    summary = " ".join(summary.split())
    if len(summary) > 240:
        summary = summary[:237].rsplit(" ", 1)[0] + "..."

    return SessionRecord(
        path=_display_path(scope, relative),
        title=metadata.title,
        project=project,
        harness=harness,
        transcript=_section_first_value(content, "Transcript Reference"),
        status=_status_from_tags(list(metadata.tags)),
        summary=summary,
        updated=_updated_iso(metadata.created, metadata.updated),
        tags=list(metadata.tags),
    )


def _session_tags(
    *,
    project: str,
    harness: str | None,
    status: Literal["open", "closed"] = "open",
) -> list[str]:
    tags = [SESSION_TAG, HANDOFF_TAG, _project_tag(project)]
    tags.append(OPEN_TAG if status == "open" else CLOSED_TAG)
    if harness:
        tags.append(_harness_tag(harness))
    return tags


def _session_title(goal: str, project: str, started: datetime) -> str:
    compact_goal = " ".join(goal.strip().split())
    if len(compact_goal) > 72:
        compact_goal = compact_goal[:69].rsplit(" ", 1)[0] + "..."
    return f"Session Handoff - {project} - {started:%Y%m%d-%H%M%S} - {compact_goal}"


def _format_list(items: list[str] | None, empty: str = "None recorded.") -> str:
    if not items:
        return f"- {empty}"
    return "\n".join(f"- `{item}`" for item in items)


def _format_reference(value: str | None, empty: str = "None recorded.") -> str:
    if not value:
        return f"- {empty}"
    return f"- `{value}`"


def _progress_entry(
    *,
    kind: Literal["start", "update", "finish"],
    summary: str,
    files: list[str] | None = None,
    tests: list[str] | None = None,
    next_steps: str | None = None,
    transcript: str | None = None,
    when: datetime | None = None,
) -> str:
    label = {"start": "start", "update": "update", "finish": "finish"}[kind]
    lines = [f"### {_format_ts(when)} {label}", "", summary.strip()]
    if files:
        lines.extend(["", "Files:", *_format_list(files).splitlines()])
    if tests:
        lines.extend(["", "Verification:", *_format_list(tests).splitlines()])
    if transcript:
        lines.extend(["", "Transcript:", _format_reference(transcript)])
    if next_steps:
        lines.extend(["", "Next:", next_steps.strip()])
    return "\n".join(lines).rstrip()


def _initial_content(
    *,
    title: str,
    goal: str,
    summary: str,
    project: str,
    harness: str | None,
    transcript: str | None,
    cwd: Path,
    started: datetime,
) -> str:
    branch = get_git_branch() or "(none)"
    model = get_llm_model() or "(unknown)"
    actor = get_actor_identity() or "(unknown)"
    harness_value = harness or "(unspecified)"
    progress = _progress_entry(kind="start", summary=summary, when=started)
    return f"""# {title}

## Snapshot

- Project: `{project}`
- CWD: `{cwd}`
- Branch: `{branch}`
- Harness: `{harness_value}`
- Model: `{model}`
- Actor: `{actor}`
- Started: `{_format_ts(started)}`

## Transcript Reference

{_format_reference(transcript)}

## Goal

{goal.strip()}

## Current Summary

{summary.strip()}

## Progress Log

{progress}

## Decisions

- None recorded.

## Files Touched

- None recorded.

## Verification

- Not run yet.

## Blockers

- None recorded.

## Next Steps

- Continue from the goal above.
"""


def _read_session(path: str) -> tuple[Path, str, list[str]]:
    actual_path = resolve_scoped_path(path)
    metadata, content, _ = parse_entry(actual_path)
    if SESSION_TAG not in metadata.tags or HANDOFF_TAG not in metadata.tags:
        raise ValueError(f"Entry is not a session handoff: {path}")
    return actual_path, content, list(metadata.tags)


def _latest_path(
    *,
    project: str | None = None,
    scope: str | None = None,
    status: Literal["open", "closed", "any"] = "open",
) -> str:
    records = list_recent_sessions(project=project, scope=scope, limit=50)
    if status != "any":
        records = [record for record in records if record.status == status]
    if not records:
        label = status if status != "any" else "matching"
        raise ValueError(f"No {label} session handoff found for this project")
    return records[0].path


def _with_status_tags(tags: list[str], status: Literal["open", "closed"]) -> list[str]:
    cleaned = [tag for tag in tags if tag not in {OPEN_TAG, CLOSED_TAG}]
    cleaned.append(OPEN_TAG if status == "open" else CLOSED_TAG)
    return list(dict.fromkeys(cleaned))


async def start_session(
    *,
    goal: str,
    summary: str | None = None,
    harness: str | None = None,
    transcript: str | None = None,
    project: str | None = None,
    scope: str | None = None,
    cwd: Path | None = None,
) -> SessionRecord:
    """Create a new session handoff entry."""
    clean_goal = goal.strip()
    if not clean_goal:
        raise ValueError("Session goal cannot be empty")

    started = _now()
    project_name = _current_project(project)
    harness_name = harness or detect_harness()
    initial_summary = (summary or clean_goal).strip()
    title = _session_title(clean_goal, project_name, started)
    content = _initial_content(
        title=title,
        goal=clean_goal,
        summary=initial_summary,
        project=project_name,
        harness=harness_name,
        transcript=transcript,
        cwd=(cwd or Path.cwd()).resolve(),
        started=started,
    )

    result = await add_entry(
        title=title,
        content=content,
        tags=_session_tags(project=project_name, harness=harness_name),
        category=SESSION_DIRECTORY,
        scope=scope,
    )
    path = str(result["path"])
    if scope and not path.startswith("@"):
        path = _display_path(scope, path)

    record = list_recent_sessions(project=project_name, scope=scope, limit=1)
    if record and record[0].path == path:
        return record[0]

    return SessionRecord(
        path=path,
        title=title,
        project=_tag_slug(project_name),
        harness=_tag_slug(harness_name) if harness_name else None,
        transcript=transcript,
        status="open",
        summary=initial_summary,
        updated=_format_ts(started),
        tags=_session_tags(project=project_name, harness=harness_name),
    )


async def append_session(
    *,
    path: str | None,
    latest: bool = False,
    summary: str,
    files: list[str] | None = None,
    tests: list[str] | None = None,
    next_steps: str | None = None,
    transcript: str | None = None,
    project: str | None = None,
    scope: str | None = None,
) -> SessionRecord:
    """Append an agent-authored progress update to a session handoff."""
    clean_summary = summary.strip()
    if not clean_summary:
        raise ValueError("Session summary cannot be empty")
    if latest:
        path = _latest_path(project=project, scope=scope, status="open")
    if not path:
        raise ValueError("Provide a session path or use --latest")

    _, content, tags = _read_session(path)
    new_content = _replace_h2_section(content, "Current Summary", clean_summary)
    new_content = _append_to_h2_section(
        new_content,
        "Progress Log",
        _progress_entry(
            kind="update",
            summary=clean_summary,
            files=files,
            tests=tests,
            next_steps=next_steps,
            transcript=transcript,
        ),
    )
    if transcript:
        new_content = _replace_h2_section(
            new_content,
            "Transcript Reference",
            _format_reference(transcript),
        )
    if next_steps:
        new_content = _replace_h2_section(new_content, "Next Steps", next_steps)

    await update_entry(path=path, content=new_content, tags=tags)
    return _record_for_path(path)


async def finish_session(
    *,
    path: str | None,
    latest: bool = False,
    summary: str,
    files: list[str] | None = None,
    tests: list[str] | None = None,
    next_steps: str | None = None,
    transcript: str | None = None,
    project: str | None = None,
    scope: str | None = None,
) -> SessionRecord:
    """Mark a session handoff closed and append its final summary."""
    clean_summary = summary.strip()
    if not clean_summary:
        raise ValueError("Session summary cannot be empty")
    if latest:
        path = _latest_path(project=project, scope=scope, status="open")
    if not path:
        raise ValueError("Provide a session path or use --latest")

    _, content, tags = _read_session(path)
    closed_tags = _with_status_tags(tags, "closed")
    new_content = _replace_h2_section(content, "Current Summary", clean_summary)
    new_content = _replace_h2_section(new_content, "Final Summary", clean_summary)
    new_content = _append_to_h2_section(
        new_content,
        "Progress Log",
        _progress_entry(
            kind="finish",
            summary=clean_summary,
            files=files,
            tests=tests,
            next_steps=next_steps,
            transcript=transcript,
        ),
    )
    if transcript:
        new_content = _replace_h2_section(
            new_content,
            "Transcript Reference",
            _format_reference(transcript),
        )
    if next_steps:
        new_content = _replace_h2_section(new_content, "Next Steps", next_steps)

    await update_entry(path=path, content=new_content, tags=closed_tags)
    return _record_for_path(path)


def _record_for_path(path: str) -> SessionRecord:
    scope, relative = parse_scoped_path(path)
    kb_root = get_kb_root_by_scope(scope) if scope else get_kb_root()
    actual = kb_root / relative
    record = _record_from_entry(scope=scope, kb_root=kb_root, path=actual)
    if not record:
        raise ValueError(f"Entry is not a session handoff: {path}")
    return record


def list_recent_sessions(
    *,
    project: str | None = None,
    scope: str | None = None,
    limit: int = 5,
) -> list[SessionRecord]:
    """List recent session handoff entries for the current project."""
    project_name = _current_project(project)
    wanted_project_tag = _project_tag(project_name)
    records: list[SessionRecord] = []

    try:
        roots = _roots_for_scope(scope)
    except ConfigurationError:
        return []

    for scope_label, kb_root in roots:
        session_dir = kb_root / SESSION_DIRECTORY
        if not session_dir.exists():
            continue

        for md_file in session_dir.rglob("*.md"):
            if md_file.name.startswith("_"):
                continue
            record = _record_from_entry(scope=scope_label, kb_root=kb_root, path=md_file)
            if not record:
                continue
            if wanted_project_tag not in record.tags:
                continue
            records.append(record)

    def sort_key(record: SessionRecord) -> str:
        return record.updated or ""

    records.sort(key=sort_key, reverse=True)
    return records[:limit]


def build_agent_instructions(harness: str) -> str:
    """Markdown instructions for agents using session handoffs."""
    name = "Claude Code" if harness == "claude" else "Codex" if harness == "codex" else harness
    lifecycle = (
        "Install a project-local `SessionStart` hook with matcher `startup|resume` to run "
        f"`{HOOK_CONTEXT_COMMAND}`; this loads recent handoffs without writing anything and stays "
        "silent when no KB is configured."
    )
    cadence = ""
    if harness in {"claude", "codex"}:
        cadence = (
            f"\n\n{name} note: `UserPromptSubmit` and `Stop` do not support per-N matchers, so "
            "they fire every turn. For periodic session-log reminders, point `UserPromptSubmit` "
            f"at `{HOOK_CONTEXT_COMMAND} --turns 5 --reminder`; it counts prompts per "
            "`session_id`/`cwd` and exits silently except on the Nth prompt."
        )
    return f"""## Memex Session Handoffs ({name})

Use Memex to leave portable handoffs for other agents and harnesses.

- {lifecycle}
- When starting substantial work, run `mx sessions start --goal "..." --harness {harness}`.
- During work, append useful progress with `mx sessions append --latest --summary "..."`.
- Before stopping or compacting, run `mx sessions finish --latest --summary "..."`.
- Keep summaries concise: what changed, files touched, tests run, blockers, and next steps.
- Do not store full transcripts, secrets, or raw private conversation logs.
{cadence}
"""


def build_hook_snippet(harness: str, *, turns: int | None = None) -> str:
    """Return a print-safe hook/config snippet for a harness."""
    if harness not in {"claude", "codex"}:
        raise ValueError(f"Unknown harness: {harness}")

    hooks: dict[str, list[dict]] = {
        "SessionStart": [
            {
                "matcher": "startup|resume",
                "hooks": [
                    {
                        "type": "command",
                        "command": HOOK_CONTEXT_COMMAND,
                        "statusMessage": "Loading Memex handoffs",
                    }
                ],
            }
        ]
    }
    if turns is not None:
        hooks["UserPromptSubmit"] = [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{HOOK_CONTEXT_COMMAND} --turns {turns} --reminder",
                    }
                ]
            }
        ]
    return json.dumps({"hooks": hooks}, indent=2) + "\n"
