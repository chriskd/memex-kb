from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import (
    ConfigurationError,
    get_kb_root,
    get_project_kb_root,
    get_user_kb_root,
)
from .context import KBContext, get_kb_context, matches_glob
from .parser import ParseError, parse_entry

CACHE_DIR = Path("/tmp")
CACHE_TTL_SECONDS = 3600  # 1 hour
DEFAULT_MAX_ENTRIES = 4
DEFAULT_RECENT_DAYS = 14
DEFAULT_RECENT_LIMIT = 5
CACHE_VERSION = 5
HOOK_CONTEXT_COMMAND = "mx sessions hook context"
LEGACY_SESSION_CONTEXT_COMMAND = "mx session-context"
SESSION_START_MATCHER = "startup|resume"
SESSION_CONTEXT_STATUS_MESSAGE = "Loading Memex handoffs"
SESSION_REMINDER_HEADING = "## Memex Session Log Reminder"
HOOK_SESSION_LOG_HEADING = "## Memex Session Log"
HOOK_STATE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60


@dataclass
class SessionContextResult:
    project: str
    entries: list[dict]
    content: str
    recent_entries: list[dict] = field(default_factory=list)
    recent_sessions: list[dict] = field(default_factory=list)
    cached: bool = False


def _stable_cache_token(value: str) -> str:
    # Keep filenames short and stable without leaking full paths.
    return re.sub(r"[^\w\-]", "_", value)[-80:]


def _get_cache_path(
    project_name: str,
    *,
    kb_root: Path,
    max_entries: int,
    recent_limit: int,
    recent_days: int,
) -> Path:
    safe_name = _stable_cache_token(project_name)
    safe_kb = _stable_cache_token(str(kb_root.resolve()))
    cache_name = (
        f"memex-context-{safe_name}-{safe_kb}-"
        f"{max_entries}-{recent_limit}-{recent_days}-v{CACHE_VERSION}.json"
    )
    return CACHE_DIR / cache_name


def _load_cached_context(
    project_name: str,
    *,
    kb_root: Path,
    max_entries: int,
    recent_limit: int,
    recent_days: int,
) -> SessionContextResult | None:
    cache_path = _get_cache_path(
        project_name,
        kb_root=kb_root,
        max_entries=max_entries,
        recent_limit=recent_limit,
        recent_days=recent_days,
    )
    if not cache_path.exists():
        return None

    try:
        data = json.loads(cache_path.read_text())
        if time.time() - data.get("timestamp", 0) < CACHE_TTL_SECONDS:
            return SessionContextResult(
                project=data.get("project", project_name),
                entries=data.get("entries", []),
                recent_entries=data.get("recent_entries", []),
                recent_sessions=data.get("recent_sessions", []),
                content=data.get("content", ""),
                cached=True,
            )
    except (json.JSONDecodeError, OSError):
        return None

    return None


def _get_git_remote(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        return None
    return None


def _extract_project_name(remote_url: str | None, cwd: Path) -> str:
    if remote_url:
        ssh_match = re.search(r":([^/]+/[^/]+?)(?:\.git)?$", remote_url)
        if ssh_match:
            return ssh_match.group(1).split("/")[-1]

        https_match = re.search(r"/([^/]+?)(?:\.git)?$", remote_url)
        if https_match:
            return https_match.group(1)

    return cwd.name


def _resolve_project_name(cwd: Path, context: KBContext | None) -> str:
    if context:
        project_name = context.get_project_name()
        if project_name:
            return project_name

    remote_url = _get_git_remote(cwd)
    return _extract_project_name(remote_url, cwd)


def _get_project_tokens(project_name: str, cwd: Path) -> set[str]:
    tokens = set()

    for part in re.split(r"[-_]", project_name.lower()):
        if len(part) > 2:
            tokens.add(part)

    if (cwd / "pyproject.toml").exists() or (cwd / "requirements.txt").exists():
        tokens.update({"python", "uv", "pip"})

    if (cwd / "Dockerfile").exists():
        tokens.update({"docker", "deployment"})

    if (cwd / "package.json").exists():
        tokens.update({"node", "npm", "javascript", "typescript"})

    if (cwd / "Cargo.toml").exists():
        tokens.update({"rust", "cargo"})

    if (cwd / ".devcontainer").exists():
        tokens.add("devcontainer")

    return tokens


def _extract_summary(content: str, max_length: int = 150) -> str:
    lines: list[str] = []
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
            if len(" ".join(lines)) > max_length:
                break

    summary = " ".join(lines)
    if len(summary) > max_length:
        summary = summary[:max_length].rsplit(" ", 1)[0] + "..."
    return summary


def _scan_kb_entries(kb_root: Path) -> list[dict]:
    entries: list[dict] = []

    for md_file in kb_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue

        try:
            metadata, content, _ = parse_entry(md_file)
            relative_path = md_file.relative_to(kb_root)
            tags_str = " ".join(metadata.tags) if metadata.tags else ""

            entries.append(
                {
                    "path": str(relative_path),
                    "title": metadata.title,
                    "tags": tags_str,
                    "summary": _extract_summary(content),
                    "category": relative_path.parent.name if relative_path.parent.name else "root",
                }
            )
        except (OSError, UnicodeDecodeError, ParseError):
            continue

    return entries


def _score_entry(
    entry: dict, project_name: str, project_tokens: set[str], context: KBContext | None
) -> float:
    score = 0.0
    entry_text = f"{entry['title']} {entry['tags']} {entry['category']}".lower()

    if project_name.lower() in entry_text:
        score += 10.0

    for token in project_tokens:
        if token in entry_text:
            score += 2.0

    if context:
        boost_paths = context.get_all_boost_paths()
        if boost_paths:
            for pattern in boost_paths:
                if matches_glob(entry["path"], pattern):
                    score += 4.0
                    break
        if context.primary and matches_glob(entry["path"], context.primary):
            score += 2.0

    category_weights = {
        "devops": 1.5,
        "development": 1.5,
        "patterns": 1.2,
        "architecture": 1.0,
        "infrastructure": 1.0,
    }
    score *= category_weights.get(entry["category"], 1.0)

    return score


def _select_relevant_entries(
    entries: list[dict],
    project_name: str,
    project_tokens: set[str],
    context: KBContext | None,
    max_entries: int,
) -> list[dict]:
    if not entries:
        return []

    scored = [
        (entry, _score_entry(entry, project_name, project_tokens, context)) for entry in entries
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [entry for entry, score in scored[:max_entries] if score > 0]


def _resolve_recent_scope() -> str | None:
    project_kb = get_project_kb_root()
    if project_kb and project_kb.exists():
        return "project"

    user_kb = get_user_kb_root()
    if user_kb and user_kb.exists():
        return "user"

    return None


def _get_recent_entries(scope: str | None, *, days: int, limit: int) -> list[dict]:
    from .core import whats_new as core_whats_new

    try:
        return asyncio.run(core_whats_new(days=days, limit=limit, scope=scope))
    except Exception:
        return []


def _format_output(
    entries: list[dict],
    project_name: str,
    *,
    kb_root: Path,
    scope: str | None,
    context: KBContext | None,
    recent_entries: list[dict],
    recent_sessions: list[dict],
) -> str:
    import importlib.util

    project_kb = get_project_kb_root()
    user_kb = get_user_kb_root()

    lines = [
        "## Memex Knowledge Base",
        "",
        "Search before creating new content. Add discoveries for future agents.",
        "",
        "**Context snapshot:**",
    ]
    context_parts = [f"write_kb={kb_root}"]
    if scope:
        context_parts.append(f"scope={scope}")
    if project_kb and project_kb != kb_root:
        context_parts.append(f"project_kb={project_kb}")
    if user_kb and user_kb != kb_root:
        context_parts.append(f"user_kb={user_kb}")
    if context and context.source_file:
        context_parts.append(f"context={context.source_file}")
    if context and context.primary:
        context_parts.append(f"primary={context.primary}")
    if context and context.default_tags:
        context_parts.append(f"default_tags={','.join(context.default_tags)}")
    lines.append(" | ".join(context_parts))
    lines.append("")

    if importlib.util.find_spec("whoosh") is None:
        lines.append(
            "Note: `mx search` requires optional dependencies. Run `mx doctor` for install hints."
        )
        lines.append("")

    lines.extend(
        [
            "**5-minute flow:**",
            "If you don't have a KB yet: `mx init` (project) or `mx init --user` (personal).",
            "1) `mx info` - active KB paths + categories",
            "2) `mx context show` - .kbconfig (primary category + default tags)",
            "3) `mx add --title=\"...\" --tags=\"...\" --category=... "
            "--content=\"...\"` - create entry",
            "   (omit --category if `.kbconfig` sets `primary`; otherwise defaults "
            "to KB root (.) with a warning)",
            "4) `mx categories` - discover available categories (directories)",
            "5) `mx list --limit=5` - confirm entry path",
            "   Optional: `mx search \"query\"` - verify indexing",
            "6) `mx get path/to/entry.md` - read entry",
            "7) `mx health` - audit (orphans = entries with no incoming links; normal early)",
            "",
            "**Required frontmatter:**",
            "- `title`",
            "- `tags`",
            "README.md inside the KB should include frontmatter or be excluded "
            "(prefix \"_\" or move it).",
            "Quick fix: add frontmatter or rename it to _README.md.",
            "",
            "**Quick reference:**",
            "| Action | Command |",
            "|--------|---------|",
            "| Inspect scope | `mx info` |",
            "| Inspect .kbconfig | `mx context show` |",
            "| Search | `mx search <query>` |",
            "| Browse | `mx categories`, `mx list`, or `mx tree` |",
            "| Read entry | `mx get <path>` |",
            "| Add new | `mx add --title=... --tags=... --category=... --content=...` |",
            "",
        ]
    )

    if entries:
        lines.append(f"**Relevant entries for {project_name}:**")
        lines.append("")
        for entry in entries:
            tags = f" [{entry['tags']}]" if entry["tags"] else ""
            lines.append(f"- **{entry['title']}** (`{entry['path']}`){tags}")
        lines.append("")

    if recent_entries:
        lines.append("**Recent Entries:**")
        lines.append("")
        for entry in recent_entries:
            activity = "NEW" if entry.get("activity_type") == "created" else "UPD"
            date_str = str(entry.get("activity_date", ""))
            path = entry.get("path", "")
            title = entry.get("title", "Untitled")
            lines.append(f"- {date_str} {activity} `{path}` — {title}")
        lines.append("")

    if recent_sessions:
        lines.append("**Recent Session Handoffs:**")
        lines.append("")
        for session in recent_sessions:
            status = str(session.get("status", "unknown")).upper()
            date_str = str(session.get("updated", ""))
            path = session.get("path", "")
            summary = session.get("summary", "")
            lines.append(f"- {date_str} {status} `{path}` - {summary}")
        lines.append("")

    lines.append("**Next Commands:**")
    lines.append("")
    if scope:
        lines.append(f"- `mx whats-new --scope={scope} --days=7`")
    else:
        lines.append("- `mx whats-new --days=7`")
    lines.append("- `mx tags`")
    lines.append("")

    return "\n".join(lines)


def build_session_context(
    *,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    recent_limit: int = DEFAULT_RECENT_LIMIT,
    recent_days: int = DEFAULT_RECENT_DAYS,
) -> SessionContextResult | None:
    cwd = Path.cwd()
    context = get_kb_context()
    project_name = _resolve_project_name(cwd, context)

    try:
        kb_root = get_kb_root()
    except ConfigurationError:
        return None

    if not kb_root.exists():
        return None

    scope = _resolve_recent_scope()
    try:
        from .session_log import list_recent_sessions

        recent_sessions = [
            record.to_dict()
            for record in list_recent_sessions(project=project_name, scope=scope, limit=3)
        ]
    except Exception:
        recent_sessions = []

    cached = _load_cached_context(
        project_name,
        kb_root=kb_root,
        max_entries=max_entries,
        recent_limit=recent_limit,
        recent_days=recent_days,
    )
    if cached:
        cached.recent_sessions = recent_sessions
        cached.content = _format_output(
            cached.entries,
            cached.project,
            kb_root=kb_root,
            scope=scope,
            context=context,
            recent_entries=cached.recent_entries,
            recent_sessions=recent_sessions,
        )
        return cached

    project_tokens = _get_project_tokens(project_name, cwd)
    entries = _scan_kb_entries(kb_root)
    relevant = _select_relevant_entries(entries, project_name, project_tokens, context, max_entries)
    recent_entries = _get_recent_entries(scope, days=recent_days, limit=recent_limit)
    content = _format_output(
        relevant,
        project_name,
        kb_root=kb_root,
        scope=scope,
        context=context,
        recent_entries=recent_entries,
        recent_sessions=recent_sessions,
    )

    result = SessionContextResult(
        project=project_name,
        entries=relevant,
        recent_entries=recent_entries,
        recent_sessions=recent_sessions,
        content=content,
        cached=False,
    )
    cache_path = _get_cache_path(
        result.project,
        kb_root=kb_root,
        max_entries=max_entries,
        recent_limit=recent_limit,
        recent_days=recent_days,
    )
    payload = {
        "timestamp": time.time(),
        "project": result.project,
        "entries": result.entries,
        "recent_entries": result.recent_entries,
        "recent_sessions": result.recent_sessions,
        "content": result.content,
    }
    try:
        cache_path.write_text(json.dumps(payload))
    except OSError:
        pass
    return result


def _has_hook_kb() -> bool:
    try:
        kb_root = get_kb_root()
    except ConfigurationError:
        return False
    return kb_root.exists()


def _get_recent_sessions_for_hook(project_name: str, scope: str | None, limit: int) -> list[dict]:
    try:
        from .session_log import list_recent_sessions

        return [
            record.to_dict()
            for record in list_recent_sessions(project=project_name, scope=scope, limit=limit)
        ]
    except Exception:
        return []


def _format_hook_session_log(project_name: str, recent_sessions: list[dict]) -> str:
    lines = [HOOK_SESSION_LOG_HEADING, ""]
    if not recent_sessions:
        lines.append(
            f"Memex session log is available for `{project_name}`; no recent working sessions yet."
        )
        return "\n".join(lines)

    lines.append(f"Recent working sessions for `{project_name}`:")
    lines.append("")
    for session in recent_sessions:
        status = str(session.get("status", "unknown")).upper()
        date_str = str(session.get("updated", ""))
        path = session.get("path", "")
        summary = session.get("summary", "")
        lines.append(f"- {date_str} {status} `{path}` - {summary}")
    return "\n".join(lines)


def build_hook_session_log_output(*, recent_limit: int = DEFAULT_RECENT_LIMIT) -> str | None:
    """Return the short session-start hook payload, or None when no KB exists."""
    if not _has_hook_kb():
        return None

    cwd = Path.cwd()
    context = get_kb_context()
    project_name = _resolve_project_name(cwd, context)
    scope = _resolve_recent_scope()
    recent_sessions = _get_recent_sessions_for_hook(project_name, scope, recent_limit)
    return _format_hook_session_log(project_name, recent_sessions)


def _parse_hook_payload(stdin_text: str | None) -> dict:
    if not stdin_text:
        return {}
    try:
        payload = json.loads(stdin_text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _hook_state_path() -> Path:
    explicit = os.environ.get("MEMEX_SESSION_HOOK_STATE")
    if explicit:
        return Path(explicit).expanduser()

    state_home = os.environ.get("XDG_STATE_HOME")
    state_root = Path(state_home).expanduser() if state_home else Path.home() / ".local" / "state"
    return state_root / "memex" / "session-hook-context.json"


def _hook_state_key(payload: dict) -> str:
    session_id = (
        payload.get("session_id")
        or payload.get("sessionId")
        or payload.get("transcript_path")
        or payload.get("transcriptPath")
        or ""
    )
    event = (
        payload.get("hook_event_name")
        or payload.get("hookEventName")
        or payload.get("event")
        or ""
    )
    cwd = payload.get("cwd") or str(Path.cwd())
    raw = json.dumps([cwd, session_id, event], sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _hook_transcript_reference(payload: dict) -> str | None:
    value = (
        payload.get("transcript_path")
        or payload.get("transcriptPath")
        or payload.get("session_path")
        or payload.get("sessionPath")
    )
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _load_hook_state(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_hook_state(path: Path, state: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        tmp_path.write_text(json.dumps(state, sort_keys=True))
        tmp_path.replace(path)
    except OSError:
        pass


def _is_recent_hook_state_value(value: object, cutoff: float) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        updated = float(value.get("updated", 0) or 0)
    except (TypeError, ValueError):
        return False
    return updated >= cutoff


def _should_emit_for_turn(payload: dict, turns: int | None) -> bool:
    if turns is None or turns <= 1:
        return True

    state_path = _hook_state_path()
    now = time.time()
    state = _load_hook_state(state_path)
    sessions = state.get("sessions")
    if not isinstance(sessions, dict):
        sessions = {}

    cutoff = now - HOOK_STATE_MAX_AGE_SECONDS
    sessions = {
        key: value
        for key, value in sessions.items()
        if _is_recent_hook_state_value(value, cutoff)
    }

    key = _hook_state_key(payload)
    current = sessions.get(key)
    try:
        count = int(current.get("count", 0)) if isinstance(current, dict) else 0
    except (TypeError, ValueError):
        count = 0
    count += 1
    sessions[key] = {"count": count, "updated": now}
    state["sessions"] = sessions
    _write_hook_state(state_path, state)
    return count % turns == 0


def build_hook_context_output(
    *,
    turns: int | None = None,
    stdin_text: str | None = None,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    recent_limit: int = DEFAULT_RECENT_LIMIT,
    reminder: bool = False,
) -> str | None:
    """Return hook-safe output, or None when hooks should stay silent."""
    if not _has_hook_kb():
        return None

    payload = _parse_hook_payload(stdin_text)
    if not _should_emit_for_turn(payload, turns):
        return None

    if reminder:
        interval = f"last {turns} prompt turns" if turns and turns > 1 else "recent prompt turns"
        transcript = _hook_transcript_reference(payload)
        transcript_flag = f" --transcript {shlex.quote(transcript)}" if transcript else ""
        transcript_lines = (
            [
                "",
                f"Transcript reference from hook payload: `{transcript}`",
                "Include it with `--transcript` if appending a handoff update.",
            ]
            if transcript
            else []
        )
        return "\n".join(
            [
                SESSION_REMINDER_HEADING,
                "",
                f"If meaningful work happened in the {interval}, append only that delta to "
                "the active Memex session handoff.",
                "",
                "Include only continuation-relevant files with `--files`; add `--tests` "
                "and `--next` when useful.",
                *transcript_lines,
                "",
                'Run: `mx sessions append --latest --summary "..." --files path/to/file '
                f'--tests "..." --next "..."{transcript_flag}`',
                "",
                "Keep it compact: what changed, why it matters, files touched, verification, "
                "blockers, and next steps. Do not re-summarize the whole session, dump diffs, "
                "or store secrets/raw transcript.",
            ]
        )

    return build_hook_session_log_output(recent_limit=recent_limit)


def find_git_root(start_dir: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
            cwd=start_dir,
        )
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        return None


def default_settings_path(start_dir: Path) -> Path:
    repo_root = find_git_root(start_dir) or start_dir
    return repo_root / ".claude" / "settings.local.json"


def default_codex_hooks_path(start_dir: Path) -> Path:
    repo_root = find_git_root(start_dir) or start_dir
    return repo_root / ".codex" / "hooks.json"


def _load_settings(settings_path: Path) -> dict:
    if not settings_path.exists():
        return {}

    data = json.loads(settings_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Hook settings must be a JSON object")
    return data


def _remove_setup_remote(hook: dict) -> bool:
    command = hook.get("command")
    return isinstance(command, str) and "setup-remote.sh" in command


def _is_memex_context_command(hook: dict) -> bool:
    command = hook.get("command")
    if not isinstance(command, str):
        return False
    return command == LEGACY_SESSION_CONTEXT_COMMAND or command.startswith(HOOK_CONTEXT_COMMAND)


def _session_context_hook(command: str, *, status_message: str | None) -> dict[str, str]:
    hook = {"type": "command", "command": command}
    if status_message:
        hook["statusMessage"] = status_message
    return hook


def _ensure_hook_event(
    settings: dict,
    event: str,
    *,
    command: str,
    matcher: str | None = None,
    status_message: str | None = None,
) -> None:
    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("Hook settings 'hooks' must be a JSON object")

    event_entries = hooks.setdefault(event, [])
    if not isinstance(event_entries, list):
        raise ValueError(f"Hook settings 'hooks.{event}' must be a list")

    target_entry: dict | None = None
    cleaned_entries: list[object] = []
    for entry in event_entries:
        if not isinstance(entry, dict):
            cleaned_entries.append(entry)
            continue

        if "hooks" not in entry:
            if _remove_setup_remote(entry) or _is_memex_context_command(entry):
                continue
            cleaned_entries.append(entry)
            continue

        hooks_list = entry.get("hooks")
        if not isinstance(hooks_list, list):
            hooks_list = []
            entry["hooks"] = hooks_list

        hooks_list[:] = [
            hook
            for hook in hooks_list
            if not (
                isinstance(hook, dict)
                and (_remove_setup_remote(hook) or _is_memex_context_command(hook))
            )
        ]

        if (matcher is None and "matcher" not in entry) or entry.get("matcher") == matcher:
            target_entry = entry
        cleaned_entries.append(entry)

    event_entries[:] = cleaned_entries
    if target_entry is None:
        target_entry = {"hooks": []}
        if matcher is not None:
            target_entry["matcher"] = matcher
        event_entries.append(target_entry)

    hooks_list = target_entry.get("hooks")
    if not isinstance(hooks_list, list):
        hooks_list = []
        target_entry["hooks"] = hooks_list
    hooks_list.append(_session_context_hook(command, status_message=status_message))


def _remove_memex_context_event(settings: dict, event: str) -> None:
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return

    event_entries = hooks.get(event)
    if not isinstance(event_entries, list):
        return

    cleaned_entries: list[object] = []
    for entry in event_entries:
        if not isinstance(entry, dict):
            cleaned_entries.append(entry)
            continue

        if "hooks" not in entry:
            if not _is_memex_context_command(entry):
                cleaned_entries.append(entry)
            continue

        hooks_list = entry.get("hooks")
        if isinstance(hooks_list, list):
            entry["hooks"] = [
                hook
                for hook in hooks_list
                if not (isinstance(hook, dict) and _is_memex_context_command(hook))
            ]
        if entry.get("hooks") or len(entry) > 1:
            cleaned_entries.append(entry)

    if cleaned_entries:
        hooks[event] = cleaned_entries
    else:
        hooks.pop(event, None)


def _ensure_session_start(settings: dict, command: str) -> None:
    _ensure_hook_event(
        settings,
        "SessionStart",
        command=command,
        matcher=SESSION_START_MATCHER,
        status_message=SESSION_CONTEXT_STATUS_MESSAGE,
    )


def _ensure_periodic_prompt_context(settings: dict, turns: int | None) -> None:
    if turns is None:
        _remove_memex_context_event(settings, "UserPromptSubmit")
        return

    _ensure_hook_event(
        settings,
        "UserPromptSubmit",
        command=f"{HOOK_CONTEXT_COMMAND} --turns {turns} --reminder",
    )


def install_session_hook(
    settings_path: Path,
    *,
    command: str = HOOK_CONTEXT_COMMAND,
    turns: int | None = None,
) -> Path:
    settings_path = settings_path.expanduser()
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    settings = _load_settings(settings_path)
    _ensure_session_start(settings, command)
    _ensure_periodic_prompt_context(settings, turns)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    return settings_path


def install_codex_hook(
    hooks_path: Path,
    *,
    command: str = HOOK_CONTEXT_COMMAND,
    turns: int | None = None,
) -> Path:
    hooks_path = hooks_path.expanduser()
    hooks_path.parent.mkdir(parents=True, exist_ok=True)

    settings = _load_settings(hooks_path)
    _ensure_session_start(settings, command)
    _ensure_periodic_prompt_context(settings, turns)
    hooks_path.write_text(json.dumps(settings, indent=2) + "\n")
    return hooks_path
