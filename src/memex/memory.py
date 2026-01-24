"""Agent memory subsystem for memex.

Provides automatic session memory capture and injection for AI coding assistants.
Features:
- Automatic capture via hooks (Stop/PreCompact)
- Automatic injection at session start
- Per-day session files with retention
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict


class InitResult(TypedDict):
    """Result type for init_memory function."""
    success: bool
    actions: list[str]
    warnings: list[str]


class DisableResult(TypedDict):
    """Result type for disable_memory function."""
    success: bool
    actions: list[str]

# Default configuration
DEFAULT_SESSION_DIR = "sessions"
DEFAULT_RETENTION_DAYS = 30


def get_project_path() -> Path:
    """Get the project directory path."""
    if project_dir := os.environ.get("CLAUDE_PROJECT_DIR"):
        return Path(project_dir)
    return Path.cwd()


def get_project_name(project_path: Path) -> str:
    """Extract project name from path or git remote."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project_path,
        )
        if result.returncode == 0:
            remote_url = result.stdout.strip()
            match = re.search(r"/([^/]+?)(?:\.git)?$", remote_url)
            if match:
                return match.group(1)
    except (subprocess.TimeoutExpired, OSError):
        pass
    return project_path.name


def get_memory_config(project_path: Path | None = None) -> dict[str, Any]:
    """Get memory configuration from .kbconfig.

    Returns:
        Dict with session_dir, retention_days, and enabled status.
    """
    if project_path is None:
        project_path = get_project_path()

    config = {
        "session_dir": DEFAULT_SESSION_DIR,
        "retention_days": DEFAULT_RETENTION_DAYS,
        "enabled": False,
        "kb_path": None,
    }

    kbconfig_path = project_path / ".kbconfig"
    if not kbconfig_path.exists():
        return config

    try:
        import yaml
        with open(kbconfig_path) as f:
            data = yaml.safe_load(f) or {}

        config["kb_path"] = data.get("kb_path")
        config["session_dir"] = data.get("session_dir", DEFAULT_SESSION_DIR)
        config["retention_days"] = data.get("session_retention_days", DEFAULT_RETENTION_DAYS)
        # Memory is enabled if session_dir is configured or hooks exist
        config["enabled"] = "session_dir" in data

    except Exception:
        pass

    return config


def get_session_file_path(kb_root: Path, session_dir: str, date: datetime | None = None) -> Path:
    """Get the path for today's session file.

    Args:
        kb_root: KB root directory
        session_dir: Session directory name (e.g., "sessions")
        date: Date for the file (defaults to today)

    Returns:
        Path like kb/sessions/2026-01-12.md
    """
    if date is None:
        date = datetime.now()

    date_str = date.strftime("%Y-%m-%d")
    return kb_root / session_dir / f"{date_str}.md"


def ensure_session_dir(kb_root: Path, session_dir: str) -> Path:
    """Ensure the session directory exists.

    Returns:
        Path to the session directory
    """
    session_path = kb_root / session_dir
    session_path.mkdir(parents=True, exist_ok=True)
    return session_path


def add_memory(
    message: str,
    kb_root: Path,
    session_dir: str = DEFAULT_SESSION_DIR,
    tags: list[str] | None = None,
    timestamp: bool = True,
) -> dict[str, Any]:
    """Add a manual memory note to today's session file.

    Args:
        message: The memory content to add
        kb_root: KB root directory
        session_dir: Session directory name
        tags: Optional tags to include
        timestamp: Whether to add a timestamp

    Returns:
        Dict with path and status
    """
    session_file = get_session_file_path(kb_root, session_dir)
    ensure_session_dir(kb_root, session_dir)

    # Build the content to append
    lines = []

    if timestamp:
        now = datetime.now(UTC)
        lines.append(f"\n## {now.strftime('%Y-%m-%d %H:%M')} UTC\n")

    lines.append(message)

    if tags:
        lines.append(f"\nTags: {', '.join(tags)}")

    lines.append("\n")
    content = "\n".join(lines)

    # Create file with frontmatter if it doesn't exist
    if not session_file.exists():
        date_str = datetime.now().strftime("%Y-%m-%d")
        frontmatter = f"""---
title: Session Log {date_str}
tags: [sessions, memory]
created: {datetime.now(UTC).isoformat()}
---

# Session Log - {date_str}

"""
        session_file.write_text(frontmatter + content)
    else:
        # Append to existing file
        with open(session_file, "a") as f:
            f.write(content)

    return {
        "path": str(session_file.relative_to(kb_root)),
        "message": "Memory added",
    }


def get_hooks_config() -> dict[str, Any]:
    """Get the hooks that should be installed for memory to work."""
    return {
        "SessionStart": [
            {
                "matcher": "",
                "hooks": [
                    {
                        "type": "command",
                        "command": "mx memory inject",
                        "timeout": 30000,
                    }
                ],
            }
        ],
        "Stop": [
            {
                "matcher": "",
                "hooks": [
                    {
                        "type": "command",
                        "command": "mx memory capture",
                        "timeout": 60000,
                    }
                ],
            }
        ],
        "PreCompact": [
            {
                "matcher": "",
                "hooks": [
                    {
                        "type": "command",
                        "command": "mx memory capture --event=precompact",
                        "timeout": 60000,
                    }
                ],
            }
        ],
    }


def init_memory(
    project_path: Path | None = None,
    user_scope: bool = False,
    session_dir: str = DEFAULT_SESSION_DIR,
    retention_days: int = DEFAULT_RETENTION_DAYS,
) -> InitResult:
    """Initialize memory for a project or user.

    Args:
        project_path: Project directory (defaults to cwd)
        user_scope: If True, install hooks user-wide
        session_dir: Directory name for session files
        retention_days: Days to retain session files

    Returns:
        Dict with status and paths modified
    """
    if project_path is None:
        project_path = get_project_path()

    result: InitResult = {
        "success": True,
        "actions": [],
        "warnings": [],
    }

    # 1. Update .kbconfig with session settings
    kbconfig_path = project_path / ".kbconfig"
    if kbconfig_path.exists():
        try:
            import yaml
            with open(kbconfig_path) as f:
                config = yaml.safe_load(f) or {}

            config["session_dir"] = session_dir
            config["session_retention_days"] = retention_days

            with open(kbconfig_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            result["actions"].append(f"Updated {kbconfig_path}")
        except Exception as e:
            result["warnings"].append(f"Could not update .kbconfig: {e}")
    else:
        result["warnings"].append("No .kbconfig found - run 'mx init' first")

    # 2. Create session directory in KB
    kb_path = None
    if kbconfig_path.exists():
        try:
            import yaml
            with open(kbconfig_path) as f:
                config = yaml.safe_load(f) or {}
            kb_path = config.get("kb_path")
        except Exception:
            pass

    if kb_path:
        kb_root = project_path / kb_path
        session_path = ensure_session_dir(kb_root, session_dir)
        result["actions"].append(f"Created {session_path}")

    # 3. Install hooks
    hooks_config = get_hooks_config()

    if user_scope:
        settings_path = Path.home() / ".claude" / "settings.json"
    else:
        settings_path = project_path / ".claude" / "settings.local.json"

    settings_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing settings
    settings = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError:
            pass

    # Merge hooks
    if "hooks" not in settings:
        settings["hooks"] = {}

    for event, event_hooks in hooks_config.items():
        if event not in settings["hooks"]:
            settings["hooks"][event] = []

        # Check if memory hooks already installed
        existing_commands = {
            h.get("hooks", [{}])[0].get("command", "")
            for h in settings["hooks"][event]
        }

        for hook in event_hooks:
            hook_cmd = hook.get("hooks", [{}])[0].get("command", "")
            if hook_cmd not in existing_commands:
                settings["hooks"][event].append(hook)

    # Write settings
    settings_path.write_text(json.dumps(settings, indent=2))
    result["actions"].append(f"Installed hooks in {settings_path}")

    # 4. Check for LLM API key (either Anthropic or OpenRouter)
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENROUTER_API_KEY"):
        result["warnings"].append(
            "No LLM API key set - memory capture requires ANTHROPIC_API_KEY or OPENROUTER_API_KEY"
        )

    return result


def disable_memory(
    project_path: Path | None = None,
    user_scope: bool = False,
) -> DisableResult:
    """Disable memory by removing hooks.

    Args:
        project_path: Project directory (defaults to cwd)
        user_scope: If True, remove from user-wide settings

    Returns:
        Dict with status
    """
    if project_path is None:
        project_path = get_project_path()

    result: DisableResult = {
        "success": True,
        "actions": [],
    }

    if user_scope:
        settings_path = Path.home() / ".claude" / "settings.json"
    else:
        settings_path = project_path / ".claude" / "settings.local.json"

    if not settings_path.exists():
        result["actions"].append("No settings file found - nothing to disable")
        return result

    try:
        settings = json.loads(settings_path.read_text())
    except json.JSONDecodeError:
        result["actions"].append("Could not parse settings file")
        return result

    if "hooks" not in settings:
        result["actions"].append("No hooks configured - nothing to disable")
        return result

    # Remove memory hooks
    memory_commands = {"mx memory inject", "mx memory capture"}

    for event in list(settings["hooks"].keys()):
        original_count = len(settings["hooks"][event])
        settings["hooks"][event] = [
            h for h in settings["hooks"][event]
            if not any(
                cmd in h.get("hooks", [{}])[0].get("command", "")
                for cmd in memory_commands
            )
        ]
        removed = original_count - len(settings["hooks"][event])
        if removed:
            result["actions"].append(f"Removed {removed} hook(s) from {event}")

        # Clean up empty event lists
        if not settings["hooks"][event]:
            del settings["hooks"][event]

    # Clean up empty hooks dict
    if not settings["hooks"]:
        del settings["hooks"]

    settings_path.write_text(json.dumps(settings, indent=2))
    result["actions"].append(f"Updated {settings_path}")

    return result


def get_memory_status(project_path: Path | None = None) -> dict[str, Any]:
    """Get the current memory configuration status.

    Returns:
        Dict with enabled status, config, and hook status
    """
    if project_path is None:
        project_path = get_project_path()

    config = get_memory_config(project_path)

    # Check if hooks are installed
    hooks_installed = {
        "project": False,
        "user": False,
    }

    for scope, path in [
        ("project", project_path / ".claude" / "settings.local.json"),
        ("user", Path.home() / ".claude" / "settings.json"),
    ]:
        if path.exists():
            try:
                settings = json.loads(path.read_text())
                hooks = settings.get("hooks", {})
                # Check if any memory hooks exist
                for event_hooks in hooks.values():
                    for h in event_hooks:
                        cmd = h.get("hooks", [{}])[0].get("command", "")
                        if "mx memory" in cmd:
                            hooks_installed[scope] = True
                            break
            except Exception:
                pass

    return {
        "config": config,
        "hooks_installed": hooks_installed,
        "api_key_set": bool(
            os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        ),
        "project_path": str(project_path),
    }


def inject_memory(
    project_path: Path | None = None,
    max_days: int = 7,
    max_tokens: int = 1000,
) -> dict[str, Any]:
    """Generate memory context for injection at session start.

    Reads recent session files and formats them for injection as a system reminder.
    Called automatically by the SessionStart hook.

    Args:
        project_path: Project directory (defaults to cwd or CLAUDE_PROJECT_DIR)
        max_days: Maximum days of history to include
        max_tokens: Approximate token budget (chars / 4)

    Returns:
        Dict with 'output' (formatted text) and 'sessions' (count)
    """
    if project_path is None:
        project_path = get_project_path()

    config = get_memory_config(project_path)
    if not config["kb_path"]:
        return {"output": "", "sessions": 0, "error": "No kb_path configured"}

    kb_root = project_path / config["kb_path"]
    session_dir = kb_root / config["session_dir"]

    if not session_dir.exists():
        return {"output": "", "sessions": 0}

    # Find recent session files (sorted by date, newest first)
    session_files = sorted(session_dir.glob("*.md"), reverse=True)

    if not session_files:
        return {"output": "", "sessions": 0}

    # Parse and collect entries
    entries = []
    char_budget = max_tokens * 4  # Rough chars-to-tokens

    for session_file in session_files[:max_days]:
        try:
            content = session_file.read_text()
            # Extract date from filename
            date_str = session_file.stem  # e.g., "2026-01-12"

            # Skip frontmatter, get body
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    body = parts[2].strip()
                else:
                    body = content
            else:
                body = content

            # Extract session entries (## timestamps)
            for section in body.split("\n## ")[1:]:  # Skip header
                lines = section.strip().split("\n")
                if not lines:
                    continue

                timestamp_line = lines[0].strip()
                entry_content = "\n".join(lines[1:]).strip()

                if entry_content:
                    entries.append({
                        "date": date_str,
                        "timestamp": timestamp_line,
                        "content": entry_content,
                    })

        except Exception:
            continue

    if not entries:
        return {"output": "", "sessions": 0}

    # Format output within budget
    project_name = get_project_name(project_path)
    output_lines = [f"## Recent Memory ({project_name})", ""]

    current_chars = sum(len(line) for line in output_lines)

    for entry in entries[:10]:  # Cap at 10 entries
        # Format: **2026-01-12 10:30 UTC** (timestamp already includes date)
        entry_header = f"**{entry['timestamp']}**"
        entry_text = entry["content"]

        # Truncate long entries
        if len(entry_text) > 500:
            entry_text = entry_text[:500] + "..."

        entry_block = f"{entry_header}\n{entry_text}\n"

        if current_chars + len(entry_block) > char_budget:
            break

        output_lines.append(entry_block)
        current_chars += len(entry_block)

    output = "\n".join(output_lines).strip()

    return {
        "output": output,
        "sessions": len(entries),
    }


def capture_memory(
    project_path: Path | None = None,
    event: str = "manual",
) -> dict[str, Any]:
    """Capture session memory by summarizing the current conversation.

    Reads the conversation from Claude's project directory, calls an LLM
    to extract observations, and writes to today's session file.

    Supports both Anthropic (direct API) and OpenRouter providers.

    Args:
        project_path: Project directory (defaults to CLAUDE_PROJECT_DIR)
        event: Event type (stop, precompact, manual)

    Returns:
        Dict with status and path
    """
    from .config import get_llm_config
    from .llm_providers import (
        LLMProviderError,
        get_sync_client,
        make_completion_sync,
    )

    if project_path is None:
        project_path = get_project_path()

    config = get_memory_config(project_path)
    if not config["kb_path"]:
        return {"error": "No kb_path configured", "captured": False}

    kb_root = project_path / config["kb_path"]
    session_dir = config["session_dir"]

    # Find the conversation file
    claude_project_dir = os.environ.get("CLAUDE_PROJECT_DIR")
    if not claude_project_dir:
        return {"error": "CLAUDE_PROJECT_DIR not set", "captured": False}

    # Claude stores conversations in ~/.claude/projects/<encoded-path>/
    # The path is encoded with dashes replacing slashes
    home = Path.home()
    encoded_path = claude_project_dir.replace("/", "-")
    if encoded_path.startswith("-"):
        encoded_path = encoded_path[1:]  # Remove leading dash

    conversations_dir = home / ".claude" / "projects" / encoded_path

    if not conversations_dir.exists():
        return {"error": f"Conversations dir not found: {conversations_dir}", "captured": False}

    # Find most recent conversation file
    conv_files = sorted(conversations_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not conv_files:
        return {"error": "No conversation files found", "captured": False}

    latest_conv = conv_files[0]

    # Read conversation (last N lines to stay within limits)
    try:
        lines = latest_conv.read_text().strip().split("\n")
        # Take last 100 messages max
        recent_lines = lines[-100:]
        messages = []
        for line in recent_lines:
            try:
                msg = json.loads(line)
                if msg.get("type") == "user" or msg.get("type") == "assistant":
                    content = msg.get("message", {}).get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            c.get("text", "") for c in content if c.get("type") == "text"
                        )
                    if content:
                        messages.append(f"{msg['type']}: {content[:500]}")
            except json.JSONDecodeError:
                continue

        if not messages:
            return {"error": "No messages found in conversation", "captured": False}

        conversation_text = "\n\n".join(messages[-20:])  # Last 20 messages

    except Exception as e:
        return {"error": f"Failed to read conversation: {e}", "captured": False}

    # Get LLM client and config
    try:
        client, provider = get_sync_client()
        llm_config = get_llm_config()
        model = llm_config.get_model("memory_capture")
    except LLMProviderError as e:
        return {"error": str(e), "captured": False}

    try:
        prompt = f"""Summarize this coding session in 2-3 sentences, then list key observations.

Use these categories for observations:
- [learned] - New knowledge or insights discovered
- [decision] - Choices made and reasoning
- [pattern] - Recurring approaches or conventions
- [issue] - Problems encountered
- [todo] - Follow-up work identified

Format:
<summary>
Brief 2-3 sentence summary of what was accomplished.
</summary>

<observations>
- [category] observation text
- [category] observation text
</observations>

Conversation:
{conversation_text}"""

        result_text = make_completion_sync(
            client=client,
            provider=provider,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )

        # Parse response
        summary = ""
        observations = []

        if "<summary>" in result_text and "</summary>" in result_text:
            summary = result_text.split("<summary>")[1].split("</summary>")[0].strip()

        if "<observations>" in result_text and "</observations>" in result_text:
            obs_text = result_text.split("<observations>")[1].split("</observations>")[0].strip()
            observations = [line.strip() for line in obs_text.split("\n") if line.strip().startswith("-")]

        # Format the memory entry
        memory_content = summary
        if observations:
            memory_content += "\n\n### Observations\n" + "\n".join(observations)

        # Add to session file
        result = add_memory(
            message=memory_content,
            kb_root=kb_root,
            session_dir=session_dir,
            timestamp=True,
        )

        return {
            "captured": True,
            "path": result["path"],
            "summary": summary,
            "observations": len(observations),
            "event": event,
        }

    except Exception as e:
        return {"error": f"Failed to summarize: {e}", "captured": False}
