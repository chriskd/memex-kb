from __future__ import annotations

import asyncio
import json
import re
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
CACHE_VERSION = 4


@dataclass
class SessionContextResult:
    project: str
    entries: list[dict]
    content: str
    recent_entries: list[dict] = field(default_factory=list)
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
    return (
        CACHE_DIR
        / f"memex-context-{safe_name}-{safe_kb}-{max_entries}-{recent_limit}-{recent_days}-v{CACHE_VERSION}.json"
    )


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
            "3) `mx add --title=\"...\" --tags=\"...\" --category=... --content=\"...\"` - create entry",
            "   (omit --category if `.kbconfig` sets `primary`; otherwise defaults to KB root (.) with a warning)",
            "4) `mx list --limit=5` - confirm entry path",
            "   Optional: `mx search \"query\"` - verify indexing",
            "5) `mx get path/to/entry.md` - read entry",
            "6) `mx health` - audit (orphans = entries with no incoming links; normal early)",
            "",
            "**Required frontmatter:**",
            "- `title`",
            "- `tags`",
            "README.md inside the KB should include frontmatter or be excluded (prefix \"_\" or move it).",
            "Quick fix: add frontmatter or rename it to _README.md.",
            "",
            "**Quick reference:**",
            "| Action | Command |",
            "|--------|---------|",
            "| Inspect scope | `mx info` |",
            "| Inspect .kbconfig | `mx context show` |",
            "| Search | `mx search <query>` |",
            "| Browse | `mx list` or `mx tree` |",
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

    cached = _load_cached_context(
        project_name,
        kb_root=kb_root,
        max_entries=max_entries,
        recent_limit=recent_limit,
        recent_days=recent_days,
    )
    if cached:
        return cached

    project_tokens = _get_project_tokens(project_name, cwd)
    entries = _scan_kb_entries(kb_root)
    relevant = _select_relevant_entries(entries, project_name, project_tokens, context, max_entries)
    scope = _resolve_recent_scope()
    recent_entries = _get_recent_entries(scope, days=recent_days, limit=recent_limit)
    content = _format_output(
        relevant,
        project_name,
        kb_root=kb_root,
        scope=scope,
        context=context,
        recent_entries=recent_entries,
    )

    result = SessionContextResult(
        project=project_name,
        entries=relevant,
        recent_entries=recent_entries,
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
        "content": result.content,
    }
    try:
        cache_path.write_text(json.dumps(payload))
    except OSError:
        pass
    return result


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
    return repo_root / ".claude" / "settings.json"


def _load_settings(settings_path: Path) -> dict:
    if not settings_path.exists():
        return {}

    data = json.loads(settings_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Claude settings must be a JSON object")
    return data


def _remove_setup_remote(hook: dict) -> bool:
    command = hook.get("command")
    return isinstance(command, str) and "setup-remote.sh" in command


def _ensure_session_start(settings: dict, command: str) -> None:
    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("Claude settings 'hooks' must be a JSON object")

    session_start = hooks.setdefault("SessionStart", [])
    if not isinstance(session_start, list):
        raise ValueError("Claude settings 'hooks.SessionStart' must be a list")

    use_nested = any(isinstance(entry, dict) and "hooks" in entry for entry in session_start)

    if use_nested:
        if not session_start:
            session_start.append({"hooks": []})

        for entry in session_start:
            if not isinstance(entry, dict):
                continue
            hooks_list = entry.get("hooks")
            if not isinstance(hooks_list, list):
                hooks_list = []
                entry["hooks"] = hooks_list

            hooks_list[:] = [
                hook
                for hook in hooks_list
                if not (isinstance(hook, dict) and _remove_setup_remote(hook))
            ]

            if not any(
                isinstance(hook, dict)
                and hook.get("type") == "command"
                and hook.get("command") == command
                for hook in hooks_list
            ):
                hooks_list.append({"type": "command", "command": command})
    else:
        session_start[:] = [
            entry
            for entry in session_start
            if not (isinstance(entry, dict) and _remove_setup_remote(entry))
        ]

        if not any(
            isinstance(entry, dict) and entry.get("command") == command for entry in session_start
        ):
            session_start.append({"command": command})


def install_session_hook(settings_path: Path, *, command: str = "mx session-context") -> Path:
    settings_path = settings_path.expanduser()
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    settings = _load_settings(settings_path)
    _ensure_session_start(settings, command)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    return settings_path
