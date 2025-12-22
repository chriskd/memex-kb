"""Beads issue tracker client for KB integration."""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

from .config import get_kb_root
from .models import BeadsComment, BeadsIssue, BeadsKanbanColumn, BeadsKanbanData

# Registry file name
REGISTRY_FILE = ".beads-registry.yaml"


def load_beads_registry() -> dict[str, Path]:
    """Load the beads registry mapping prefixes to project paths.

    Registry file format (.beads-registry.yaml):
        # Maps issue ID prefixes to project directories
        dv: /srv/fast/code/docviewer
        kb: .
        cp: /srv/fast/code/claude-plugins

    Searches for registry in:
        1. KB root directory (kb/)
        2. KB project root (parent of kb/ where .beads/ typically lives)

    Returns:
        Dict mapping prefix to absolute Path.
    """
    kb_root = get_kb_root()

    # Search locations for registry file
    search_paths = [
        kb_root / REGISTRY_FILE,
        kb_root.parent / REGISTRY_FILE,  # Project root
    ]

    registry_path = None
    base_path = kb_root  # For resolving relative paths

    for path in search_paths:
        if path.exists():
            registry_path = path
            base_path = path.parent
            break

    if not registry_path:
        return {}

    registry: dict[str, Path] = {}
    for line in registry_path.read_text().splitlines():
        line = line.strip()
        # Skip comments and empty lines
        if not line or line.startswith("#"):
            continue

        # Parse "prefix: path" format
        if ":" in line:
            prefix, path_str = line.split(":", 1)
            prefix = prefix.strip()
            path_str = path_str.strip()

            # Resolve relative paths from registry file location
            path = Path(path_str)
            if not path.is_absolute():
                path = (base_path / path).resolve()

            registry[prefix] = path

    return registry


def parse_issue_prefix(issue_id: str) -> str | None:
    """Extract prefix from issue ID (e.g., 'dv-45' -> 'dv', 'epstein-2h0' -> 'epstein').

    Handles both numeric (dv-45) and alphanumeric (epstein-2h0) suffixes.
    Also handles hyphenated prefixes (voidlabs-kb-1a2 -> voidlabs-kb).
    """
    # Find the last hyphen followed by alphanumeric (the issue number)
    # This handles both "dv-45" and "voidlabs-kb-1a2"
    match = re.match(r"^(.+)-[a-zA-Z0-9]+$", issue_id)
    if match:
        return match.group(1)
    return None


# Cache for project-specific clients
_project_clients: dict[str, "BeadsClient"] = {}


def get_client_for_issue(issue_id: str) -> "BeadsClient | None":
    """Get a BeadsClient for the project that owns an issue.

    Uses the registry to find the project path from the issue prefix.
    """
    prefix = parse_issue_prefix(issue_id)
    if not prefix:
        return None

    # Check cache
    if prefix in _project_clients:
        return _project_clients[prefix]

    # Look up in registry
    registry = load_beads_registry()
    if prefix not in registry:
        return None

    project_path = registry[prefix]
    if not (project_path / ".beads").exists():
        return None

    client = BeadsClient(beads_root=project_path)
    _project_clients[prefix] = client
    return client


class BeadsClient:
    """Client for accessing beads issue data.

    Provides access to beads issues via the `bd` CLI (preferred) or direct
    JSONL parsing (fallback). Auto-discovers the beads project location.
    """

    def __init__(self, beads_root: Path | None = None):
        """Initialize beads client.

        Args:
            beads_root: Path containing .beads/ directory.
                        If None, auto-discovers from KB root or parent.
        """
        self.beads_root = beads_root or self._discover_beads_root()
        self._bd_available: bool | None = None

    def _discover_beads_root(self) -> Path | None:
        """Discover .beads/ directory location.

        Search order:
        1. KB root directory
        2. KB root parent (for monorepo layouts)
        3. Current working directory
        """
        kb_root = get_kb_root()

        # Check KB root
        if (kb_root / ".beads").exists():
            return kb_root

        # Check parent (common for plugins in a larger project)
        if (kb_root.parent / ".beads").exists():
            return kb_root.parent

        # Check cwd
        cwd = Path.cwd()
        if (cwd / ".beads").exists():
            return cwd

        return None

    def _check_bd_available(self) -> bool:
        """Check if bd CLI is available."""
        if self._bd_available is not None:
            return self._bd_available

        try:
            result = subprocess.run(
                ["bd", "--version"],
                capture_output=True,
                timeout=5,
            )
            self._bd_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._bd_available = False

        return self._bd_available

    @property
    def is_available(self) -> bool:
        """Check if beads is available for this KB."""
        if self.beads_root is None:
            return False
        return self._check_bd_available() or (
            self.beads_root / ".beads/issues.jsonl"
        ).exists()

    def get_project_name(self) -> str | None:
        """Get the beads project name/prefix."""
        if not self.beads_root:
            return None

        # Try bd info first (faster, uses daemon)
        if self._check_bd_available():
            try:
                result = subprocess.run(
                    ["bd", "info", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=self.beads_root,
                )
                if result.returncode == 0:
                    info = json.loads(result.stdout)
                    # Config structure: {"config": {"issue_prefix": "..."}}
                    prefix = info.get("config", {}).get("issue_prefix")
                    if prefix:
                        return prefix
            except (subprocess.TimeoutExpired, json.JSONDecodeError):
                pass

        # Fallback: extract from issue IDs
        issues = self.list_issues(limit=1)
        if issues:
            # ID format: "project-xxx" -> extract "project"
            parts = issues[0].id.rsplit("-", 1)
            if len(parts) == 2:
                return parts[0]

        # Fallback: use directory name
        return self.beads_root.name if self.beads_root else None

    def list_issues(
        self,
        status: Literal["open", "in_progress", "closed", "all"] = "all",
        limit: int = 100,
        priority: int | None = None,
    ) -> list[BeadsIssue]:
        """List issues with optional filters.

        Args:
            status: Filter by status. "all" returns all statuses.
            limit: Maximum issues to return.
            priority: Filter by priority (0-4).

        Returns:
            List of BeadsIssue objects.
        """
        if self._check_bd_available() and self.beads_root:
            return self._list_via_bd(status, limit, priority)
        elif self.beads_root:
            return self._list_via_jsonl(status, limit, priority)
        return []

    def _list_via_bd(
        self,
        status: str,
        limit: int,
        priority: int | None,
    ) -> list[BeadsIssue]:
        """List issues using bd CLI."""
        cmd = ["bd", "list", "--json"]

        if status != "all":
            cmd.extend(["--status", status])

        if priority is not None:
            cmd.extend(["--priority", str(priority)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.beads_root,
            )

            if result.returncode != 0:
                return []

            issues_data = json.loads(result.stdout)
            return [self._parse_issue(d) for d in issues_data[:limit]]
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            return []

    def _list_via_jsonl(
        self,
        status: str,
        limit: int,
        priority: int | None,
    ) -> list[BeadsIssue]:
        """List issues by parsing JSONL directly."""
        if not self.beads_root:
            return []

        jsonl_path = self.beads_root / ".beads/issues.jsonl"
        if not jsonl_path.exists():
            return []

        issues = []
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)

                # Apply filters
                if status != "all" and data.get("status") != status:
                    continue
                if priority is not None and data.get("priority") != priority:
                    continue

                issues.append(self._parse_issue(data))

                if len(issues) >= limit:
                    break
            except json.JSONDecodeError:
                continue

        return issues

    def _parse_issue(self, data: dict) -> BeadsIssue:
        """Parse raw issue data into BeadsIssue model."""
        return BeadsIssue(
            id=data["id"],
            title=data["title"],
            description=data.get("description"),
            status=data.get("status", "open"),
            priority=data.get("priority", 2),
            issue_type=data.get("issue_type", "task"),
            created_at=self._parse_datetime(data.get("created_at", "")),
            updated_at=self._parse_datetime(data.get("updated_at", "")),
            closed_at=self._parse_datetime(data.get("closed_at"))
            if data.get("closed_at")
            else None,
            close_reason=data.get("close_reason"),
            dependency_count=len(data.get("dependencies", [])),
            dependent_count=0,  # Would need reverse lookup
        )

    def _parse_datetime(self, dt_str: str) -> datetime:
        """Parse datetime string from beads format."""
        if not dt_str:
            return datetime.now()

        # Handle various formats
        # "2025-12-19T20:59:25.432530369-05:00" -> strip nanoseconds
        dt_str = dt_str.replace("Z", "+00:00")

        # Strip nanoseconds if present (Python doesn't handle 9-digit precision)
        if "." in dt_str:
            base, frac_and_tz = dt_str.split(".", 1)
            # Find timezone offset (+ or -)
            for sep in ["+", "-"]:
                if sep in frac_and_tz[1:]:  # Skip first char (could be negative)
                    idx = frac_and_tz.index(sep, 1)
                    frac = frac_and_tz[:idx][:6]  # Keep only 6 digits
                    tz = frac_and_tz[idx:]
                    dt_str = f"{base}.{frac}{tz}"
                    break

        try:
            return datetime.fromisoformat(dt_str)
        except ValueError:
            return datetime.now()

    def get_issue(self, issue_id: str) -> BeadsIssue | None:
        """Get a specific issue by ID."""
        if self._check_bd_available() and self.beads_root:
            try:
                result = subprocess.run(
                    ["bd", "show", issue_id, "--json"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=self.beads_root,
                )
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    # bd show returns a single object or list with one item
                    if isinstance(data, list) and data:
                        return self._parse_issue(data[0])
                    elif isinstance(data, dict):
                        return self._parse_issue(data)
            except (subprocess.TimeoutExpired, json.JSONDecodeError):
                pass

        # Fallback: scan JSONL
        for issue in self._list_via_jsonl("all", 10000, None):
            if issue.id == issue_id:
                return issue
        return None

    def get_comments(self, issue_id: str) -> list[BeadsComment]:
        """Get comments for an issue.

        Args:
            issue_id: The issue ID to get comments for.

        Returns:
            List of BeadsComment objects, ordered by creation date.
        """
        if not self.beads_root:
            return []

        # Try bd CLI first (uses daemon, faster if available)
        if self._check_bd_available():
            comments = self._get_comments_via_bd(issue_id)
            if comments is not None:
                return comments

        # Fallback: parse directly from SQLite database
        return self._get_comments_via_db(issue_id)

    def _get_comments_via_bd(self, issue_id: str) -> list[BeadsComment] | None:
        """Get comments using bd CLI.

        Returns:
            List of comments, or None if CLI failed (to trigger fallback).
        """
        try:
            result = subprocess.run(
                ["bd", "comments", issue_id, "--json"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.beads_root,
            )
            if result.returncode != 0:
                return None

            comments_data = json.loads(result.stdout)
            if not comments_data:
                return []

            return [
                BeadsComment(
                    id=str(c.get("id", "")),
                    issue_id=issue_id,
                    content=c.get("text", c.get("content", "")),
                    author=c.get("author", "unknown"),
                    created_at=self._parse_datetime(c.get("created_at", "")),
                )
                for c in comments_data
            ]
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            return None

    def _get_comments_via_db(self, issue_id: str) -> list[BeadsComment]:
        """Get comments by parsing SQLite database directly.

        This fallback method works when bd CLI isn't available (e.g., in Docker).
        """
        if not self.beads_root:
            return []

        db_path = self.beads_root / ".beads/beads.db"
        if not db_path.exists():
            return []

        try:
            # Use immutable=1 for read-only access without needing journal files
            # This is required when the database is on a read-only mount (e.g., Docker)
            conn = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT id, issue_id, author, text, created_at
                FROM comments
                WHERE issue_id = ?
                ORDER BY created_at ASC
                """,
                (issue_id,),
            )
            rows = cursor.fetchall()
            conn.close()

            return [
                BeadsComment(
                    id=str(row["id"]),
                    issue_id=row["issue_id"],
                    content=row["text"],
                    author=row["author"],
                    created_at=self._parse_datetime(row["created_at"]),
                )
                for row in rows
            ]
        except sqlite3.Error:
            return []

    def get_issues_by_ids(self, issue_ids: list[str]) -> list[BeadsIssue]:
        """Get multiple issues by ID."""
        issues = []
        for issue_id in issue_ids:
            issue = self.get_issue(issue_id)
            if issue:
                issues.append(issue)
        return issues

    def get_kanban_data(self, project: str | None = None) -> BeadsKanbanData | None:
        """Get issues grouped by status for kanban display.

        Args:
            project: Filter to a specific project prefix. If None, uses current project.

        Returns:
            BeadsKanbanData with issues grouped into columns.
        """
        if not self.is_available:
            return None

        project_name = project or self.get_project_name()
        if not project_name:
            return None

        all_issues = self.list_issues(status="all", limit=1000)

        # Filter by project if specified
        if project:
            all_issues = [i for i in all_issues if i.id.startswith(f"{project}-")]

        # Group by status
        columns: dict[str, list[BeadsIssue]] = {
            "open": [],
            "blocked": [],
            "in_progress": [],
            "closed": [],
        }

        for issue in all_issues:
            if issue.status in columns:
                columns[issue.status].append(issue)

        # Sort each column by priority (lower = higher priority), then by created date
        for status in columns:
            columns[status].sort(key=lambda i: (i.priority, i.created_at))

        status_labels = {
            "open": "Open",
            "blocked": "Blocked",
            "in_progress": "In Progress",
            "closed": "Closed",
        }

        return BeadsKanbanData(
            project=project_name,
            columns=[
                BeadsKanbanColumn(
                    status=status,
                    label=status_labels[status],
                    issues=columns[status],
                )
                for status in ["open", "blocked", "in_progress", "closed"]
            ],
            total_issues=len(all_issues),
        )


# Singleton instance for KB's own beads
_beads_client: BeadsClient | None = None


def get_beads_client() -> BeadsClient:
    """Get or create the BeadsClient singleton for KB's own beads."""
    global _beads_client
    if _beads_client is None:
        _beads_client = BeadsClient()
    return _beads_client


# =============================================================================
# Registry-aware cross-project functions
# =============================================================================


def resolve_issue(issue_id: str) -> BeadsIssue | None:
    """Resolve an issue from any registered project.

    Uses the registry to find the correct project based on issue prefix.
    Falls back to the default KB client if not found in registry.

    Args:
        issue_id: Issue ID (e.g., "dv-45", "kb-12").

    Returns:
        BeadsIssue if found, None otherwise.
    """
    # Try registry-based lookup first
    client = get_client_for_issue(issue_id)
    if client:
        return client.get_issue(issue_id)

    # Fall back to default client
    default_client = get_beads_client()
    if default_client.is_available:
        return default_client.get_issue(issue_id)

    return None


def resolve_issues(issue_ids: list[str]) -> list[BeadsIssue]:
    """Resolve multiple issues across projects.

    Groups issues by prefix for efficient batch lookups.

    Args:
        issue_ids: List of issue IDs.

    Returns:
        List of found BeadsIssue objects (missing issues omitted).
    """
    # Group by prefix for efficiency
    by_prefix: dict[str | None, list[str]] = {}
    for issue_id in issue_ids:
        prefix = parse_issue_prefix(issue_id)
        if prefix not in by_prefix:
            by_prefix[prefix] = []
        by_prefix[prefix].append(issue_id)

    results: list[BeadsIssue] = []

    for prefix, ids in by_prefix.items():
        if prefix is None:
            # Invalid format, try default client
            default_client = get_beads_client()
            for issue_id in ids:
                issue = default_client.get_issue(issue_id)
                if issue:
                    results.append(issue)
        else:
            # Use registry or fall back to default
            client = get_client_for_issue(ids[0])
            if client is None:
                client = get_beads_client()

            if client and client.is_available:
                for issue_id in ids:
                    issue = client.get_issue(issue_id)
                    if issue:
                        results.append(issue)

    return results


def get_kanban_for_project(prefix: str) -> BeadsKanbanData | None:
    """Get kanban data for a specific registered project.

    Args:
        prefix: Project prefix (e.g., "dv", "kb").

    Returns:
        BeadsKanbanData if project found and accessible, None otherwise.
    """
    registry = load_beads_registry()
    if prefix not in registry:
        # Try default client if prefix matches its project
        default_client = get_beads_client()
        if default_client.is_available:
            project_name = default_client.get_project_name()
            if project_name == prefix:
                return default_client.get_kanban_data()
        return None

    project_path = registry[prefix]
    if not (project_path / ".beads").exists():
        return None

    client = BeadsClient(beads_root=project_path)
    return client.get_kanban_data()


def list_registered_projects() -> list[dict]:
    """List all registered projects with availability status.

    Returns:
        List of dicts with 'prefix', 'path', 'available' keys.
    """
    registry = load_beads_registry()
    projects = []

    for prefix, path in registry.items():
        available = (path / ".beads").exists()
        projects.append({
            "prefix": prefix,
            "path": str(path),
            "available": available,
        })

    return projects
