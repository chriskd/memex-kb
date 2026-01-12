"""Client for interacting with beads (bd) CLI.

Shells out to the bd CLI for read-only operations. Designed to work
with arbitrary beads projects, not just the KB root's own .beads.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class BeadsProject:
    """Represents a beads project location."""

    path: Path  # Path to project root (containing .beads/)
    db_path: Path  # Path to .beads/beads.db


def find_beads_db(project_path: str | Path) -> BeadsProject | None:
    """Find .beads/beads.db within a project path.

    Args:
        project_path: Path to project root OR .beads directory

    Returns:
        BeadsProject if found, None otherwise
    """
    path = Path(project_path).expanduser().resolve()

    # Handle both "/project" and "/project/.beads" forms
    if path.name == ".beads":
        beads_dir = path
        project_root = path.parent
    else:
        beads_dir = path / ".beads"
        project_root = path

    db_path = beads_dir / "beads.db"
    if db_path.exists():
        return BeadsProject(path=project_root, db_path=db_path)

    return None


def run_bd_command(
    db_path: Path,
    command: list[str],
    *,
    timeout: float = 10.0,
) -> dict | list | None:
    """Run a bd command and parse JSON output.

    Args:
        db_path: Path to .beads/beads.db
        command: Command args after 'bd --db <path> --json'
        timeout: Command timeout in seconds

    Returns:
        Parsed JSON output, or None on error
    """
    full_cmd = [
        "bd",
        "--db",
        str(db_path),
        "--json",
        "--readonly",  # Safety: never modify from webapp
        *command,
    ]

    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            log.warning("bd command failed: %s stderr=%s", full_cmd, result.stderr)
            return None

        if not result.stdout.strip():
            return []

        return json.loads(result.stdout)

    except subprocess.TimeoutExpired:
        log.error("bd command timed out: %s", full_cmd)
        return None
    except json.JSONDecodeError as e:
        log.error("Failed to parse bd output: %s", e)
        return None
    except FileNotFoundError:
        log.warning("bd CLI not found in PATH")
        return None


def list_issues(db_path: Path) -> list[dict]:
    """List all issues in a beads project."""
    result = run_bd_command(db_path, ["list"])
    return result if isinstance(result, list) else []


def show_issue(db_path: Path, issue_id: str) -> dict | None:
    """Get detailed info for a single issue."""
    result = run_bd_command(db_path, ["show", issue_id])
    # bd show returns a single object in JSON mode
    if isinstance(result, dict):
        return result
    # Handle list with one item (older bd versions)
    if isinstance(result, list) and len(result) > 0:
        return result[0]
    return None


def get_status(db_path: Path) -> dict | None:
    """Get project status/statistics."""
    result = run_bd_command(db_path, ["status"])
    return result if isinstance(result, dict) else None


def get_comments(db_path: Path, issue_id: str) -> list[dict]:
    """Get comments for an issue."""
    result = run_bd_command(db_path, ["comments", issue_id])
    return result if isinstance(result, list) else []
