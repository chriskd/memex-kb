"""Beads issue tracker client for KB integration."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Literal

from .config import get_kb_root
from .models import BeadsIssue, BeadsKanbanColumn, BeadsKanbanData


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
                for status in ["open", "in_progress", "closed"]
            ],
            total_issues=len(all_issues),
        )


# Singleton instance
_beads_client: BeadsClient | None = None


def get_beads_client() -> BeadsClient:
    """Get or create the BeadsClient singleton."""
    global _beads_client
    if _beads_client is None:
        _beads_client = BeadsClient()
    return _beads_client
