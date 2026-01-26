from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigurationError, get_kb_root
from .context import get_kb_context
from .parser import ParseError, parse_entry

CACHE_DIR = Path("/tmp")
CACHE_TTL_SECONDS = 3600  # 1 hour
DEFAULT_MAX_ENTRIES = 4


@dataclass
class SessionContextResult:
    project: str
    entries: list[dict]
    content: str
    cached: bool = False


def _get_cache_path(project_name: str, max_entries: int) -> Path:
    safe_name = re.sub(r"[^\w\-]", "_", project_name)
    return CACHE_DIR / f"memex-context-{safe_name}-{max_entries}.json"


def _load_cached_context(project_name: str, max_entries: int) -> SessionContextResult | None:
    cache_path = _get_cache_path(project_name, max_entries)
    if not cache_path.exists():
        return None

    try:
        data = json.loads(cache_path.read_text())
        if time.time() - data.get("timestamp", 0) < CACHE_TTL_SECONDS:
            return SessionContextResult(
                project=data.get("project", project_name),
                entries=data.get("entries", []),
                content=data.get("content", ""),
                cached=True,
            )
    except (json.JSONDecodeError, OSError):
        return None

    return None


def _save_cached_context(result: SessionContextResult, max_entries: int) -> None:
    cache_path = _get_cache_path(result.project, max_entries)
    payload = {
        "timestamp": time.time(),
        "project": result.project,
        "entries": result.entries,
        "content": result.content,
    }
    try:
        cache_path.write_text(json.dumps(payload))
    except OSError:
        pass


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


def _resolve_project_name(cwd: Path) -> str:
    context = get_kb_context()
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


def _score_entry(entry: dict, project_name: str, project_tokens: set[str]) -> float:
    score = 0.0
    entry_text = f"{entry['title']} {entry['tags']} {entry['category']}".lower()

    if project_name.lower() in entry_text:
        score += 10.0

    for token in project_tokens:
        if token in entry_text:
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
    entries: list[dict], project_name: str, project_tokens: set[str], max_entries: int
) -> list[dict]:
    if not entries:
        return []

    scored = [(entry, _score_entry(entry, project_name, project_tokens)) for entry in entries]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [entry for entry, score in scored[:max_entries] if score > 0]


def _format_output(entries: list[dict], project_name: str) -> str:
    lines = [
        "## Memex Knowledge Base",
        "",
        "You have access to a knowledge base with documentation, patterns,",
        "and operational guides. **Search before creating** to avoid duplicates.",
        "",
        "**Quick Reference:**",
        "| Action | Command |",
        "|--------|---------|",
        "| Search | `mx search <query>` |",
        "| Browse | `mx list` or `mx tree` |",
        "| Read entry | `mx get <path>` |",
        "| Add new | `mx add --title=... --tags=... --content=...` |",
        "",
    ]

    if entries:
        lines.append(f"**Relevant entries for {project_name}:**")
        lines.append("")
        for entry in entries:
            tags = f" [{entry['tags']}]" if entry["tags"] else ""
            lines.append(f"- **{entry['title']}** (`{entry['path']}`){tags}")
        lines.append("")

    return "\n".join(lines)


def build_session_context(*, max_entries: int = DEFAULT_MAX_ENTRIES) -> SessionContextResult | None:
    cwd = Path.cwd()
    project_name = _resolve_project_name(cwd)

    cached = _load_cached_context(project_name, max_entries)
    if cached:
        return cached

    try:
        kb_root = get_kb_root()
    except ConfigurationError:
        return None

    if not kb_root.exists():
        return None

    project_tokens = _get_project_tokens(project_name, cwd)
    entries = _scan_kb_entries(kb_root)
    relevant = _select_relevant_entries(entries, project_name, project_tokens, max_entries)
    content = _format_output(relevant, project_name)

    result = SessionContextResult(
        project=project_name,
        entries=relevant,
        content=content,
        cached=False,
    )
    _save_cached_context(result, max_entries)
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


def default_hook_path(start_dir: Path) -> Path:
    repo_root = find_git_root(start_dir) or start_dir
    return repo_root / "hooks" / "session-context.sh"


def install_session_hook(target_path: Path) -> Path:
    target_path = target_path.expanduser()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    script = """#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"

if [ -n "$REPO_ROOT" ]; then
  cd "$REPO_ROOT"
else
  cd "$SCRIPT_DIR"
fi

exec mx session-context
"""

    target_path.write_text(script)
    target_path.chmod(0o755)
    return target_path
