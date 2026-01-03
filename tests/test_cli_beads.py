"""Tests for beads CLI commands (mx beads ...).

Tests use click.testing.CliRunner and mock beads_client functions to isolate CLI logic.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from memex.cli import (
    _format_priority,
    _get_beads_db_or_fail,
    _load_beads_registry,
    _parse_issue_id,
    cli,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_beads_project():
    """Mock BeadsProject."""
    project = MagicMock()
    project.path = Path("/test/project")
    project.db_path = Path("/test/project/.beads/beads.db")
    return project


@pytest.fixture
def mock_issues():
    """Mock list of beads issues."""
    return [
        {
            "id": "test-1",
            "title": "First issue",
            "description": "Description of first issue",
            "status": "open",
            "priority": 1,
            "issue_type": "task",
            "created_at": "2025-01-01T10:00:00",
            "created_by": "user",
        },
        {
            "id": "test-2",
            "title": "Second issue",
            "description": "Description of second issue",
            "status": "in_progress",
            "priority": 2,
            "issue_type": "bug",
            "created_at": "2025-01-02T10:00:00",
            "created_by": "user",
        },
        {
            "id": "test-3",
            "title": "Third issue",
            "description": "Done",
            "status": "closed",
            "priority": 3,
            "issue_type": "feature",
            "created_at": "2025-01-03T10:00:00",
            "created_by": "user",
        },
    ]


@pytest.fixture
def mock_registry(tmp_path, monkeypatch):
    """Create a mock registry file and set KB root."""
    kb_root = tmp_path / "kb"
    kb_root.mkdir()

    # Create registry
    registry_content = """
test: /test/project
other: /other/project
"""
    (kb_root / ".beads-registry.yaml").write_text(registry_content)

    monkeypatch.setenv("MEMEX_KB_ROOT", str(kb_root))
    return kb_root


# ─────────────────────────────────────────────────────────────────────────────
# Helper Function Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFormatPriority:
    """Tests for _format_priority helper."""

    def test_format_priority_critical(self):
        assert _format_priority(0) == "critical"

    def test_format_priority_high(self):
        assert _format_priority(1) == "high"

    def test_format_priority_medium(self):
        assert _format_priority(2) == "medium"

    def test_format_priority_low(self):
        assert _format_priority(3) == "low"

    def test_format_priority_backlog(self):
        assert _format_priority(4) == "backlog"

    def test_format_priority_none(self):
        assert _format_priority(None) == "medium"

    def test_format_priority_unknown(self):
        assert _format_priority(99) == "medium"


class TestLoadBeadsRegistry:
    """Tests for _load_beads_registry helper."""

    def test_load_registry_missing_file(self, tmp_path, monkeypatch):
        """Returns empty dict when registry file doesn't exist."""
        monkeypatch.setenv("MEMEX_KB_ROOT", str(tmp_path))
        registry = _load_beads_registry()
        assert registry == {}

    def test_load_registry_valid(self, mock_registry):
        """Loads and parses valid registry."""
        registry = _load_beads_registry()
        assert "test" in registry
        assert "other" in registry
        assert registry["test"] == Path("/test/project")

    def test_load_registry_relative_path(self, tmp_path, monkeypatch):
        """Resolves relative paths relative to KB root."""
        kb_root = tmp_path / "kb"
        kb_root.mkdir()

        registry_content = "local: ."
        (kb_root / ".beads-registry.yaml").write_text(registry_content)
        monkeypatch.setenv("MEMEX_KB_ROOT", str(kb_root))

        registry = _load_beads_registry()
        assert registry["local"] == kb_root


class TestParseIssueId:
    """Tests for _parse_issue_id helper."""

    def test_parse_full_issue_id(self, mock_registry):
        """Parses issue ID with known prefix."""
        prefix, full_id = _parse_issue_id("test-123", None)
        assert prefix == "test"
        assert full_id == "test-123"

    def test_parse_with_explicit_project(self, mock_registry):
        """Uses explicit project when provided."""
        prefix, full_id = _parse_issue_id("456", "test")
        assert prefix == "test"
        assert full_id == "test-456"

    def test_parse_unknown_prefix_without_project_fails(self, mock_registry):
        """Fails when prefix unknown and no project specified."""
        import click
        with pytest.raises(click.ClickException) as exc_info:
            _parse_issue_id("unknown-123", None)
        assert "Cannot determine project" in str(exc_info.value)


class TestGetBeadsDbOrFail:
    """Tests for _get_beads_db_or_fail helper."""

    def test_path_not_exists(self, tmp_path):
        """Fails when project path doesn't exist."""
        import click
        with pytest.raises(click.ClickException) as exc_info:
            _get_beads_db_or_fail(tmp_path / "nonexistent", "test")
        assert "does not exist" in str(exc_info.value)

    @patch("memex.beads_client.find_beads_db")
    def test_no_beads_db(self, mock_find, tmp_path):
        """Fails when beads.db not found."""
        import click
        mock_find.return_value = None
        with pytest.raises(click.ClickException) as exc_info:
            _get_beads_db_or_fail(tmp_path, "test")
        assert "No beads database found" in str(exc_info.value)

    @patch("memex.beads_client.find_beads_db")
    def test_success(self, mock_find, tmp_path, mock_beads_project):
        """Returns beads project when found."""
        mock_find.return_value = mock_beads_project
        result = _get_beads_db_or_fail(tmp_path, "test")
        assert result == mock_beads_project


# ─────────────────────────────────────────────────────────────────────────────
# Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBeadsHelp:
    """Tests for beads command group help."""

    def test_beads_help(self, runner):
        """Shows help message."""
        result = runner.invoke(cli, ["beads", "--help"])
        assert result.exit_code == 0
        assert "Browse beads issue tracking" in result.output
        assert "list" in result.output
        assert "show" in result.output
        assert "kanban" in result.output


class TestBeadsList:
    """Tests for 'mx beads list' command."""

    @patch("memex.cli._load_beads_registry")
    @patch("memex.beads_client.find_beads_db")
    @patch("memex.beads_client.list_issues")
    def test_list_basic(self, mock_list, mock_find, mock_registry_fn,
                        runner, mock_beads_project, mock_issues, tmp_path, monkeypatch):
        """Lists issues in table format."""
        monkeypatch.setenv("MEMEX_KB_ROOT", str(tmp_path))
        # Use tmp_path so the path exists
        mock_registry_fn.return_value = {"test": tmp_path}
        mock_find.return_value = mock_beads_project
        mock_list.return_value = mock_issues

        result = runner.invoke(cli, ["beads", "list", "-p", "test"])

        assert result.exit_code == 0
        assert "test-1" in result.output
        assert "First issue" in result.output

    @patch("memex.cli._load_beads_registry")
    @patch("memex.beads_client.find_beads_db")
    @patch("memex.beads_client.list_issues")
    def test_list_with_status_filter(
        self, mock_list, mock_find, mock_registry_fn, runner,
        mock_beads_project, mock_issues, tmp_path, monkeypatch,
    ):
        """Filters issues by status."""
        monkeypatch.setenv("MEMEX_KB_ROOT", str(tmp_path))
        mock_registry_fn.return_value = {"test": tmp_path}
        mock_find.return_value = mock_beads_project
        mock_list.return_value = mock_issues

        result = runner.invoke(cli, ["beads", "list", "-p", "test", "--status", "open"])

        assert result.exit_code == 0
        assert "test-1" in result.output
        assert "test-2" not in result.output  # in_progress
        assert "test-3" not in result.output  # closed

    @patch("memex.cli._load_beads_registry")
    @patch("memex.beads_client.find_beads_db")
    @patch("memex.beads_client.list_issues")
    def test_list_json_output(self, mock_list, mock_find, mock_registry_fn,
                              runner, mock_beads_project, mock_issues, tmp_path, monkeypatch):
        """Outputs JSON when --json flag is set."""
        monkeypatch.setenv("MEMEX_KB_ROOT", str(tmp_path))
        mock_registry_fn.return_value = {"test": tmp_path}
        mock_find.return_value = mock_beads_project
        mock_list.return_value = mock_issues

        result = runner.invoke(cli, ["beads", "list", "-p", "test", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 3
        assert data[0]["id"] == "test-1"

    @patch("memex.cli._load_beads_registry")
    def test_list_unknown_project(self, mock_registry_fn, runner, tmp_path, monkeypatch):
        """Fails with unknown project."""
        monkeypatch.setenv("MEMEX_KB_ROOT", str(tmp_path))
        mock_registry_fn.return_value = {"known": tmp_path}

        result = runner.invoke(cli, ["beads", "list", "-p", "unknown"])

        assert result.exit_code != 0
        assert "Unknown project" in result.output


class TestBeadsShow:
    """Tests for 'mx beads show' command."""

    @patch("memex.cli._load_beads_registry")
    @patch("memex.beads_client.find_beads_db")
    @patch("memex.beads_client.show_issue")
    @patch("memex.beads_client.get_comments")
    def test_show_with_comments(self, mock_comments, mock_show, mock_find, mock_registry_fn,
                                runner, mock_beads_project, tmp_path, monkeypatch):
        """Shows issue details with comments."""
        monkeypatch.setenv("MEMEX_KB_ROOT", str(tmp_path))
        mock_registry_fn.return_value = {"test": tmp_path}
        mock_find.return_value = mock_beads_project
        mock_show.return_value = {
            "id": "test-1",
            "title": "Test Issue",
            "description": "Test description",
            "status": "open",
            "priority": 1,
            "issue_type": "task",
            "created_at": "2025-01-01T10:00:00",
            "created_by": "user",
        }
        mock_comments.return_value = [
            {"author": "alice", "content": "Comment 1", "created_at": "2025-01-02"},
        ]

        result = runner.invoke(cli, ["beads", "show", "test-1"])

        assert result.exit_code == 0
        assert "Test Issue" in result.output
        assert "Test description" in result.output
        assert "Comment 1" in result.output

    @patch("memex.cli._load_beads_registry")
    @patch("memex.beads_client.find_beads_db")
    @patch("memex.beads_client.show_issue")
    def test_show_issue_not_found(self, mock_show, mock_find, mock_registry_fn,
                                  runner, mock_beads_project, tmp_path, monkeypatch):
        """Fails when issue not found."""
        monkeypatch.setenv("MEMEX_KB_ROOT", str(tmp_path))
        mock_registry_fn.return_value = {"test": tmp_path}
        mock_find.return_value = mock_beads_project
        mock_show.return_value = None

        result = runner.invoke(cli, ["beads", "show", "test-999"])

        assert result.exit_code != 0
        assert "Issue not found" in result.output


class TestBeadsKanban:
    """Tests for 'mx beads kanban' command."""

    @patch("memex.cli._load_beads_registry")
    @patch("memex.beads_client.find_beads_db")
    @patch("memex.beads_client.list_issues")
    def test_kanban_groups_by_status(
        self, mock_list, mock_find, mock_registry_fn, runner,
        mock_beads_project, mock_issues, tmp_path, monkeypatch,
    ):
        """Groups issues by status."""
        monkeypatch.setenv("MEMEX_KB_ROOT", str(tmp_path))
        mock_registry_fn.return_value = {"test": tmp_path}
        mock_find.return_value = mock_beads_project
        mock_list.return_value = mock_issues

        result = runner.invoke(cli, ["beads", "kanban", "-p", "test"])

        assert result.exit_code == 0
        assert "OPEN" in result.output
        assert "IN PROGRESS" in result.output
        assert "CLOSED" in result.output

    @patch("memex.cli._load_beads_registry")
    @patch("memex.beads_client.find_beads_db")
    @patch("memex.beads_client.list_issues")
    def test_kanban_json_output(self, mock_list, mock_find, mock_registry_fn,
                                runner, mock_beads_project, mock_issues, tmp_path, monkeypatch):
        """Outputs JSON when --json flag is set."""
        monkeypatch.setenv("MEMEX_KB_ROOT", str(tmp_path))
        mock_registry_fn.return_value = {"test": tmp_path}
        mock_find.return_value = mock_beads_project
        mock_list.return_value = mock_issues

        result = runner.invoke(cli, ["beads", "kanban", "-p", "test", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "columns" in data
        assert len(data["columns"]) == 3  # open, in_progress, closed


class TestBeadsStatus:
    """Tests for 'mx beads status' command."""

    @patch("memex.cli._load_beads_registry")
    @patch("memex.beads_client.find_beads_db")
    @patch("memex.beads_client.list_issues")
    def test_status_shows_counts(self, mock_list, mock_find, mock_registry_fn,
                                 runner, mock_beads_project, mock_issues, tmp_path, monkeypatch):
        """Shows issue counts by status, priority, and type."""
        monkeypatch.setenv("MEMEX_KB_ROOT", str(tmp_path))
        mock_registry_fn.return_value = {"test": tmp_path}
        mock_find.return_value = mock_beads_project
        mock_list.return_value = mock_issues

        result = runner.invoke(cli, ["beads", "status", "-p", "test"])

        assert result.exit_code == 0
        assert "By Status:" in result.output
        assert "By Priority:" in result.output
        assert "By Type:" in result.output


class TestBeadsProjects:
    """Tests for 'mx beads projects' command."""

    @patch("memex.cli._load_beads_registry")
    @patch("memex.beads_client.find_beads_db")
    def test_projects_lists_registry(self, mock_find, mock_registry_fn,
                                     runner, mock_beads_project, tmp_path, monkeypatch):
        """Lists all registered projects."""
        monkeypatch.setenv("MEMEX_KB_ROOT", str(tmp_path))
        # Create subdirs so paths exist
        (tmp_path / "project1").mkdir()
        (tmp_path / "project2").mkdir()
        mock_registry_fn.return_value = {
            "project1": tmp_path / "project1",
            "project2": tmp_path / "project2",
        }
        mock_find.side_effect = lambda p: mock_beads_project if "project1" in str(p) else None

        result = runner.invoke(cli, ["beads", "projects"])

        assert result.exit_code == 0
        assert "project1" in result.output
        assert "project2" in result.output

    @patch("memex.cli._load_beads_registry")
    def test_projects_empty_registry(self, mock_registry_fn, runner, tmp_path, monkeypatch):
        """Shows message when no projects registered."""
        monkeypatch.setenv("MEMEX_KB_ROOT", str(tmp_path))
        mock_registry_fn.return_value = {}

        result = runner.invoke(cli, ["beads", "projects"])

        assert result.exit_code == 0
        assert "No projects registered" in result.output
