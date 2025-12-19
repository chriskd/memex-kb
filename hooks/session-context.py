#!/usr/bin/env python3
"""SessionStart hook that injects relevant KB context for voidlabs projects.

This script is invoked when a Claude Code session starts. It searches the
knowledge base for entries relevant to the current project and outputs
context as markdown.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Cache configuration
CACHE_DIR = Path("/tmp")
CACHE_TTL_SECONDS = 3600  # 1 hour


def get_cache_path(project_name: str) -> Path:
    """Get cache file path for a project."""
    safe_name = re.sub(r"[^\w\-]", "_", project_name)
    return CACHE_DIR / f"voidlabs-kb-context-{safe_name}.json"


def load_cached_context(project_name: str) -> str | None:
    """Load cached context if still valid."""
    cache_path = get_cache_path(project_name)
    if not cache_path.exists():
        return None

    try:
        data = json.loads(cache_path.read_text())
        if time.time() - data.get("timestamp", 0) < CACHE_TTL_SECONDS:
            return data.get("context")
    except (json.JSONDecodeError, OSError):
        pass

    return None


def save_cached_context(project_name: str, context: str) -> None:
    """Save context to cache."""
    cache_path = get_cache_path(project_name)
    try:
        cache_path.write_text(
            json.dumps({"timestamp": time.time(), "context": context})
        )
    except OSError:
        pass  # Ignore cache write errors


def get_git_remote() -> str | None:
    """Get git remote origin URL."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def extract_project_name(remote_url: str | None, cwd: Path) -> str:
    """Extract project name from git remote or directory."""
    if remote_url:
        # Handle SSH format: git@github.com:user/repo.git
        ssh_match = re.search(r":([^/]+/[^/]+?)(?:\.git)?$", remote_url)
        if ssh_match:
            return ssh_match.group(1).split("/")[-1]

        # Handle HTTPS format: https://github.com/user/repo.git
        https_match = re.search(r"/([^/]+?)(?:\.git)?$", remote_url)
        if https_match:
            return https_match.group(1)

    # Fallback to directory name
    return cwd.name


def get_kb_root() -> Path | None:
    """Get the KB root directory."""
    # Check environment variable first
    if kb_root := os.environ.get("KB_ROOT"):
        path = Path(kb_root)
        if path.exists():
            return path

    # Find plugin root from environment
    if plugin_root := os.environ.get("CLAUDE_PLUGIN_ROOT"):
        kb_path = Path(plugin_root) / "kb"
        if kb_path.exists():
            return kb_path

    # Try to find relative to this script
    script_dir = Path(__file__).parent.parent
    kb_path = script_dir / "kb"
    if kb_path.exists():
        return kb_path

    return None


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown content."""
    if not content.startswith("---"):
        return {}, content

    # Find end of frontmatter
    end_match = re.search(r"\n---\n", content[3:])
    if not end_match:
        return {}, content

    frontmatter_text = content[4 : end_match.start() + 3]
    body = content[end_match.end() + 4 :]

    # Simple YAML parsing for frontmatter
    metadata: dict = {}
    for line in frontmatter_text.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value.startswith("[") or value.startswith("-"):
                continue  # Skip complex values for simplicity
            if value:
                metadata[key] = value

    return metadata, body


def extract_summary(content: str, max_length: int = 150) -> str:
    """Extract first meaningful paragraph as summary."""
    # Remove headers
    lines = []
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


def scan_kb_entries(kb_root: Path) -> list[dict]:
    """Scan KB for all entries with metadata."""
    entries = []

    for md_file in kb_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue  # Skip index files

        try:
            content = md_file.read_text()
            metadata, body = parse_frontmatter(content)

            relative_path = md_file.relative_to(kb_root)
            title = metadata.get("title", md_file.stem.replace("-", " ").title())
            tags = metadata.get("tags", "")

            entries.append(
                {
                    "path": str(relative_path),
                    "title": title,
                    "tags": tags,
                    "summary": extract_summary(body),
                    "category": relative_path.parent.name if relative_path.parent.name else "root",
                }
            )
        except (OSError, UnicodeDecodeError):
            continue

    return entries


def score_entry(entry: dict, project_name: str, project_tokens: set[str]) -> float:
    """Score an entry based on relevance to project."""
    score = 0.0
    entry_text = f"{entry['title']} {entry['tags']} {entry['category']}".lower()

    # Exact project name match (high value)
    if project_name.lower() in entry_text:
        score += 10.0

    # Token matching
    for token in project_tokens:
        if token in entry_text:
            score += 2.0

    # Category bonuses for common needs
    category_weights = {
        "devops": 1.5,
        "development": 1.5,
        "patterns": 1.2,
        "architecture": 1.0,
        "infrastructure": 1.0,
    }
    score *= category_weights.get(entry["category"], 1.0)

    return score


def get_project_tokens(project_name: str, cwd: Path) -> set[str]:
    """Extract relevant tokens from project context."""
    tokens = set()

    # Tokens from project name
    for part in re.split(r"[-_]", project_name.lower()):
        if len(part) > 2:
            tokens.add(part)

    # Check for common project indicators
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


def select_relevant_entries(
    entries: list[dict], project_name: str, project_tokens: set[str], max_entries: int = 4
) -> list[dict]:
    """Select most relevant entries for the project."""
    if not entries:
        return []

    # Score all entries
    scored = [(entry, score_entry(entry, project_name, project_tokens)) for entry in entries]

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Take top entries with non-zero scores
    return [entry for entry, score in scored[:max_entries] if score > 0]


def format_output(entries: list[dict], project_name: str) -> str:
    """Format entries as markdown output."""
    if not entries:
        return ""

    lines = [
        "## Voidlabs Knowledge Base",
        "",
        f"Relevant organizational knowledge for **{project_name}**:",
        "",
    ]

    for entry in entries:
        lines.append(f"**{entry['title']}** ({entry['path']})")
        if entry["summary"]:
            lines.append(entry["summary"])
        lines.append("")

    lines.append('Use `/kb search <query>` to find more organizational knowledge.')

    return "\n".join(lines)


def main() -> None:
    """Main entry point."""
    cwd = Path.cwd()

    # Get project context
    remote_url = get_git_remote()
    project_name = extract_project_name(remote_url, cwd)

    # Check cache first
    cached = load_cached_context(project_name)
    if cached:
        print(cached)
        return

    # Find KB root
    kb_root = get_kb_root()
    if not kb_root:
        return  # Silent exit if KB not found

    # Get project tokens for matching
    project_tokens = get_project_tokens(project_name, cwd)

    # Scan and select entries
    entries = scan_kb_entries(kb_root)
    relevant = select_relevant_entries(entries, project_name, project_tokens)

    # Format and output
    output = format_output(relevant, project_name)
    if output:
        # Cache the result
        save_cached_context(project_name, output)
        print(output)


if __name__ == "__main__":
    main()
