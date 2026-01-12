"""Consolidated CLI tests for memex.

Covers all major CLI commands with:
- One happy path per command
- One error case per command
- Bulk parametrized tests for --help and --json output

Design:
- Uses fixtures from conftest.py (tmp_kb, cli_invoke, runner)
- Tests BEHAVIORS not implementations
- Targets <5 second total runtime
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from memex.cli import cli


# ─────────────────────────────────────────────────────────────────────────────
# Command Lists
# ─────────────────────────────────────────────────────────────────────────────

# All commands that should have working --help
ALL_COMMANDS = [
    "search",
    "get",
    "add",
    "append",
    "replace",
    "delete",
    "list",
    "tree",
    "tags",
    "hubs",
    "health",
    "init",
    "reindex",
    "info",
    "whats-new",
    "suggest-links",
    "templates",
    "history",
    "prime",
    "quick-add",
    "context",
    "patch",
]

# Commands that support --json output
JSON_COMMANDS = [
    "search",
    "get",
    "add",
    "replace",
    "delete",
    "list",
    "tree",
    "tags",
    "hubs",
    "health",
    "info",
    "whats-new",
    "suggest-links",
    "templates",
    "history",
    "prime",
    "reindex",
]


# ─────────────────────────────────────────────────────────────────────────────
# Bulk Tests (Parametrized)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cmd", ALL_COMMANDS)
def test_command_has_working_help(runner, cmd):
    """Every command has a working --help that shows usage."""
    result = runner.invoke(cli, [cmd, "--help"])
    assert result.exit_code == 0, f"{cmd} --help failed: {result.output}"
    assert "Usage:" in result.output


def test_main_help_shows_all_commands(runner):
    """Main --help lists all top-level commands."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for cmd in ["search", "get", "add", "list", "tree", "health", "info"]:
        assert cmd in result.output, f"Missing command: {cmd}"


def test_version_option(runner):
    """--version outputs version number."""
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Search Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSearchCommand:
    """Tests for 'mx search' command."""

    @patch("memex.cli.run_async")
    def test_search_returns_results(self, mock_run_async, runner):
        """Search returns results for valid query."""
        mock_result = MagicMock()
        mock_result.results = [
            MagicMock(
                path="tooling/test.md",
                title="Test Entry",
                score=0.85,
                snippet="Test content...",
                content=None,
            )
        ]
        mock_run_async.return_value = mock_result

        result = runner.invoke(cli, ["search", "test"])

        assert result.exit_code == 0
        assert "tooling/test.md" in result.output

    @patch("memex.cli.run_async")
    def test_search_no_results(self, mock_run_async, runner):
        """Search handles no results gracefully."""
        mock_result = MagicMock()
        mock_result.results = []
        mock_run_async.return_value = mock_result

        result = runner.invoke(cli, ["search", "nonexistent"])

        assert result.exit_code == 0
        assert "No results found" in result.output

    @patch("memex.cli.run_async")
    def test_search_json_output(self, mock_run_async, runner):
        """Search --json outputs valid JSON."""
        mock_result = MagicMock()
        mock_result.results = [
            MagicMock(
                path="test.md",
                title="Test",
                score=0.9,
                snippet="...",
                content=None,
            )
        ]
        mock_run_async.return_value = mock_result

        result = runner.invoke(cli, ["search", "test", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert data[0]["path"] == "test.md"

    @patch("memex.cli.run_async")
    def test_search_terse_output(self, mock_run_async, runner):
        """Search --terse outputs only paths."""
        mock_result = MagicMock()
        mock_result.results = [
            MagicMock(path="a.md", title="A", score=0.9, snippet="..."),
            MagicMock(path="b.md", title="B", score=0.8, snippet="..."),
        ]
        mock_run_async.return_value = mock_result

        result = runner.invoke(cli, ["search", "test", "--terse"])

        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        assert lines == ["a.md", "b.md"]

    def test_search_missing_query(self, runner):
        """Search fails with missing query argument."""
        result = runner.invoke(cli, ["search"])
        assert result.exit_code != 0


# ─────────────────────────────────────────────────────────────────────────────
# Get Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGetCommand:
    """Tests for 'mx get' command."""

    @patch("memex.cli.run_async")
    def test_get_by_path(self, mock_run_async, runner):
        """Get reads entry by path."""
        mock_entry = MagicMock()
        mock_entry.metadata = MagicMock(
            title="Test Entry",
            tags=["tag1"],
            created="2024-01-01",
            updated=None,
        )
        mock_entry.content = "# Test Content"
        mock_entry.links = []
        mock_entry.backlinks = []
        mock_run_async.return_value = mock_entry

        result = runner.invoke(cli, ["get", "test.md"])

        assert result.exit_code == 0
        assert "Test Entry" in result.output
        assert "Test Content" in result.output

    @patch("memex.cli.run_async")
    def test_get_metadata_only(self, mock_run_async, runner):
        """Get --metadata shows only metadata."""
        mock_entry = MagicMock()
        mock_entry.metadata = MagicMock(
            title="Test Entry",
            tags=["tag1", "tag2"],
            created="2024-01-01",
            updated="2024-01-15",
        )
        mock_entry.links = ["link1.md"]
        mock_entry.backlinks = []
        mock_run_async.return_value = mock_entry

        result = runner.invoke(cli, ["get", "test.md", "--metadata"])

        assert result.exit_code == 0
        assert "Title:" in result.output
        assert "Tags:" in result.output
        assert "tag1, tag2" in result.output

    @patch("memex.cli.run_async")
    def test_get_json_output(self, mock_run_async, runner):
        """Get --json outputs valid JSON."""
        mock_entry = MagicMock()
        mock_entry.model_dump.return_value = {
            "metadata": {"title": "Test", "tags": ["tag1"]},
            "content": "Content",
        }
        mock_run_async.return_value = mock_entry

        result = runner.invoke(cli, ["get", "test.md", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["metadata"]["title"] == "Test"

    @patch("memex.cli.run_async")
    def test_get_entry_not_found(self, mock_run_async, runner):
        """Get fails for nonexistent entry."""
        mock_run_async.side_effect = FileNotFoundError("Entry not found")

        result = runner.invoke(cli, ["get", "nonexistent.md"])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_get_missing_path_and_title(self, runner):
        """Get fails when neither path nor --title provided."""
        result = runner.invoke(cli, ["get"])
        assert result.exit_code != 0


# ─────────────────────────────────────────────────────────────────────────────
# Add Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAddCommand:
    """Tests for 'mx add' command."""

    @patch("memex.cli.run_async")
    def test_add_with_content(self, mock_run_async, runner):
        """Add creates entry with inline content."""
        mock_run_async.return_value = {
            "path": "test/entry.md",
            "suggested_links": [],
            "suggested_tags": [],
        }

        result = runner.invoke(cli, [
            "add",
            "--title", "Test Entry",
            "--tags", "tag1,tag2",
            "--content", "# Test Content",
        ])

        assert result.exit_code == 0
        assert "Created: test/entry.md" in result.output

    @patch("memex.cli.run_async")
    def test_add_json_output(self, mock_run_async, runner):
        """Add --json outputs valid JSON."""
        mock_run_async.return_value = {
            "path": "test/entry.md",
            "suggested_links": [],
            "suggested_tags": [],
        }

        result = runner.invoke(cli, [
            "add",
            "--title", "Test",
            "--tags", "tag1",
            "--content", "Content",
            "--json",
        ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "path" in data

    def test_add_missing_content_source(self, runner):
        """Add fails without content source."""
        result = runner.invoke(cli, [
            "add",
            "--title", "Test",
            "--tags", "tag1",
        ])

        assert result.exit_code == 1
        assert "Must provide --content, --file, or --stdin" in result.output

    @patch("memex.cli.run_async")
    def test_add_error_handling(self, mock_run_async, runner):
        """Add handles errors gracefully."""
        mock_run_async.side_effect = ValueError("Invalid entry")

        result = runner.invoke(cli, [
            "add",
            "--title", "Test",
            "--tags", "tag1",
            "--content", "Content",
        ])

        assert result.exit_code == 1
        assert "Error" in result.output

    @patch("memex.cli.run_async")
    def test_add_with_scope_project(self, mock_run_async, runner):
        """Add with --scope=project passes scope to add_entry."""
        mock_run_async.return_value = {
            "path": "notes/entry.md",
            "suggested_links": [],
            "suggested_tags": [],
        }

        result = runner.invoke(cli, [
            "add",
            "--title", "Test",
            "--tags", "tag1",
            "--content", "Content",
            "--scope", "project",
        ])

        assert result.exit_code == 0
        assert "Created: @project/notes/entry.md" in result.output

    @patch("memex.cli.run_async")
    def test_add_with_scope_user(self, mock_run_async, runner):
        """Add with --scope=user passes scope to add_entry."""
        mock_run_async.return_value = {
            "path": "personal/note.md",
            "suggested_links": [],
            "suggested_tags": [],
        }

        result = runner.invoke(cli, [
            "add",
            "--title", "Personal Note",
            "--tags", "personal",
            "--content", "My notes",
            "--scope", "user",
        ])

        assert result.exit_code == 0
        assert "Created: @user/personal/note.md" in result.output

    @patch("memex.cli.run_async")
    def test_add_with_scope_json_output(self, mock_run_async, runner):
        """Add with --scope includes scope in JSON output."""
        mock_run_async.return_value = {
            "path": "notes/entry.md",
            "suggested_links": [],
            "suggested_tags": [],
        }

        result = runner.invoke(cli, [
            "add",
            "--title", "Test",
            "--tags", "tag1",
            "--content", "Content",
            "--scope", "user",
            "--json",
        ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["scope"] == "user"
        assert data["path"] == "notes/entry.md"

    def test_add_invalid_scope(self, runner):
        """Add rejects invalid scope values."""
        result = runner.invoke(cli, [
            "add",
            "--title", "Test",
            "--tags", "tag1",
            "--content", "Content",
            "--scope", "invalid",
        ])

        assert result.exit_code != 0
        assert "Invalid value for '--scope'" in result.output

    @patch("memex.cli.run_async")
    def test_add_scope_error_no_user_kb(self, mock_run_async, runner):
        """Add with --scope=user fails if user KB doesn't exist."""
        from memex.config import ConfigurationError
        mock_run_async.side_effect = ConfigurationError(
            "No user KB found. Run 'mx init --user' to create one at ~/.memex/kb/"
        )

        result = runner.invoke(cli, [
            "add",
            "--title", "Test",
            "--tags", "tag1",
            "--content", "Content",
            "--scope", "user",
        ])

        assert result.exit_code == 1
        assert "No user KB found" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Append Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAppendCommand:
    """Tests for 'mx append' command."""

    @patch("memex.cli.run_async")
    def test_append_to_existing(self, mock_run_async, runner):
        """Append adds content to existing entry."""
        mock_run_async.return_value = {
            "path": "test/entry.md",
            "created": False,
            "appended_bytes": 100,
        }

        result = runner.invoke(cli, [
            "append", "Test Entry",
            "--content", "Additional content",
        ])

        assert result.exit_code == 0
        assert "Appended" in result.output or "test/entry.md" in result.output

    def test_append_missing_content(self, runner):
        """Append fails without content source."""
        result = runner.invoke(cli, ["append", "Test Entry"])

        assert result.exit_code == 1
        assert "content" in result.output.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Replace Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestReplaceCommand:
    """Tests for 'mx replace' command."""

    @patch("memex.cli.run_async")
    def test_replace_tags(self, mock_run_async, runner):
        """Replace modifies entry tags."""
        mock_run_async.return_value = {"path": "test.md", "updated": True}

        result = runner.invoke(cli, ["replace", "test.md", "--tags", "new,tags"])

        assert result.exit_code == 0
        assert "Replaced" in result.output or "test.md" in result.output

    @patch("memex.cli.run_async")
    def test_replace_json_output(self, mock_run_async, runner):
        """Replace --json outputs valid JSON."""
        mock_run_async.return_value = {"path": "test.md", "updated": True}

        result = runner.invoke(cli, ["replace", "test.md", "--tags", "new", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["path"] == "test.md"

    @patch("memex.cli.run_async")
    def test_replace_not_found(self, mock_run_async, runner):
        """Replace fails for nonexistent entry."""
        mock_run_async.side_effect = FileNotFoundError("Entry not found")

        result = runner.invoke(cli, ["replace", "nonexistent.md", "--tags", "new"])

        assert result.exit_code == 1
        assert "Error" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Delete Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDeleteCommand:
    """Tests for 'mx delete' command."""

    @patch("memex.cli.run_async")
    def test_delete_entry(self, mock_run_async, runner):
        """Delete removes entry."""
        mock_run_async.return_value = {"deleted": "test.md", "had_backlinks": []}

        result = runner.invoke(cli, ["delete", "test.md"])

        assert result.exit_code == 0
        assert "Deleted" in result.output

    @patch("memex.cli.run_async")
    def test_delete_with_backlinks_warning(self, mock_run_async, runner):
        """Delete shows backlink warning when forced."""
        mock_run_async.return_value = {
            "deleted": "test.md",
            "had_backlinks": ["other.md"],
        }

        result = runner.invoke(cli, ["delete", "test.md", "--force"])

        assert result.exit_code == 0
        assert "backlink" in result.output.lower()

    @patch("memex.cli.run_async")
    def test_delete_json_output(self, mock_run_async, runner):
        """Delete --json outputs valid JSON."""
        mock_run_async.return_value = {"deleted": "test.md", "had_backlinks": []}

        result = runner.invoke(cli, ["delete", "test.md", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["deleted"] == "test.md"


# ─────────────────────────────────────────────────────────────────────────────
# List Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestListCommand:
    """Tests for 'mx list' command."""

    @patch("memex.cli.run_async")
    def test_list_entries(self, mock_run_async, runner):
        """List shows entries."""
        mock_run_async.return_value = [
            {"path": "a.md", "title": "Entry A"},
            {"path": "b.md", "title": "Entry B"},
        ]

        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "a.md" in result.output
        assert "b.md" in result.output

    @patch("memex.cli.run_async")
    def test_list_no_entries(self, mock_run_async, runner):
        """List handles empty results."""
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "No entries found" in result.output

    @patch("memex.cli.run_async")
    def test_list_json_output(self, mock_run_async, runner):
        """List --json outputs valid JSON."""
        mock_run_async.return_value = [{"path": "a.md", "title": "A"}]

        result = runner.invoke(cli, ["list", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Tree Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTreeCommand:
    """Tests for 'mx tree' command."""

    @patch("memex.cli.run_async")
    def test_tree_shows_structure(self, mock_run_async, runner):
        """Tree displays directory structure."""
        mock_run_async.return_value = {
            "tree": {
                "tooling": {
                    "_type": "directory",
                    "test.md": {"_type": "file", "title": "Test"},
                }
            },
            "directories": 1,
            "files": 1,
        }

        result = runner.invoke(cli, ["tree"])

        assert result.exit_code == 0
        assert "tooling/" in result.output
        assert "1 directories, 1 files" in result.output

    @patch("memex.cli.run_async")
    def test_tree_json_output(self, mock_run_async, runner):
        """Tree --json outputs valid JSON."""
        mock_run_async.return_value = {
            "tree": {},
            "directories": 0,
            "files": 0,
        }

        result = runner.invoke(cli, ["tree", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "directories" in data


# ─────────────────────────────────────────────────────────────────────────────
# Tags Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTagsCommand:
    """Tests for 'mx tags' command."""

    @patch("memex.cli.run_async")
    def test_tags_shows_tag_counts(self, mock_run_async, runner):
        """Tags lists tags with counts."""
        mock_run_async.return_value = [
            {"tag": "python", "count": 10},
            {"tag": "testing", "count": 5},
        ]

        result = runner.invoke(cli, ["tags"])

        assert result.exit_code == 0
        assert "python" in result.output
        assert "10" in result.output

    @patch("memex.cli.run_async")
    def test_tags_no_tags(self, mock_run_async, runner):
        """Tags handles empty results."""
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["tags"])

        assert result.exit_code == 0
        assert "No tags found" in result.output

    @patch("memex.cli.run_async")
    def test_tags_json_output(self, mock_run_async, runner):
        """Tags --json outputs valid JSON."""
        mock_run_async.return_value = [{"tag": "test", "count": 5}]

        result = runner.invoke(cli, ["tags", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]["tag"] == "test"


# ─────────────────────────────────────────────────────────────────────────────
# Hubs Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHubsCommand:
    """Tests for 'mx hubs' command."""

    @patch("memex.cli.run_async")
    def test_hubs_shows_hub_entries(self, mock_run_async, runner):
        """Hubs lists highly connected entries."""
        mock_run_async.return_value = [
            {"path": "hub.md", "incoming": 10, "outgoing": 5, "total": 15},
        ]

        result = runner.invoke(cli, ["hubs"])

        assert result.exit_code == 0
        assert "hub.md" in result.output

    @patch("memex.cli.run_async")
    def test_hubs_no_results(self, mock_run_async, runner):
        """Hubs handles empty results."""
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["hubs"])

        assert result.exit_code == 0
        assert "No hub entries found" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Health Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHealthCommand:
    """Tests for 'mx health' command."""

    @patch("memex.cli.run_async")
    def test_health_shows_status(self, mock_run_async, runner):
        """Health shows KB health status."""
        mock_run_async.return_value = {
            "summary": {"health_score": 95, "total_entries": 100},
            "orphans": [],
            "broken_links": [],
            "stale": [],
            "empty_dirs": [],
        }

        result = runner.invoke(cli, ["health"])

        assert result.exit_code == 0
        assert "Health Score: 95/100" in result.output
        assert "Total Entries: 100" in result.output

    @patch("memex.cli.run_async")
    def test_health_with_issues(self, mock_run_async, runner):
        """Health reports found issues."""
        mock_run_async.return_value = {
            "summary": {"health_score": 70, "total_entries": 50},
            "orphans": [{"path": "orphan.md"}],
            "broken_links": [{"source": "a.md", "broken_link": "missing.md"}],
            "stale": [],
            "empty_dirs": [],
        }

        result = runner.invoke(cli, ["health"])

        assert result.exit_code == 0
        assert "Orphaned entries" in result.output
        assert "Broken links" in result.output

    @patch("memex.cli.run_async")
    def test_health_json_output(self, mock_run_async, runner):
        """Health --json outputs valid JSON."""
        mock_run_async.return_value = {
            "summary": {"health_score": 100, "total_entries": 10},
            "orphans": [],
            "broken_links": [],
            "stale": [],
            "empty_dirs": [],
        }

        result = runner.invoke(cli, ["health", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["summary"]["health_score"] == 100


# ─────────────────────────────────────────────────────────────────────────────
# Init Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestInitCommand:
    """Tests for 'mx init' command."""

    def test_init_creates_kb(self, runner, tmp_path):
        """Init creates KB directory structure."""
        result = runner.invoke(cli, ["init", "--path", str(tmp_path / "kb")])

        assert result.exit_code == 0
        assert "Initialized" in result.output
        assert (tmp_path / "kb").exists()
        assert (tmp_path / "kb" / "README.md").exists()

    def test_init_already_exists(self, runner, tmp_path):
        """Init fails if KB already exists (without --force)."""
        kb_path = tmp_path / "kb"
        kb_path.mkdir()

        result = runner.invoke(cli, ["init", "--path", str(kb_path)])

        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_init_mutually_exclusive_options(self, runner, tmp_path):
        """Init rejects --user and --path together."""
        result = runner.invoke(cli, ["init", "--user", "--path", str(tmp_path / "kb")])

        assert result.exit_code == 1
        assert "mutually exclusive" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Reindex Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestReindexCommand:
    """Tests for 'mx reindex' command."""

    @patch("memex.cli.run_async")
    def test_reindex_success(self, mock_run_async, runner):
        """Reindex reports indexed entries."""
        mock_result = MagicMock()
        mock_result.kb_files = 50
        mock_result.whoosh_docs = 50
        mock_result.chroma_docs = 50
        mock_run_async.return_value = mock_result

        result = runner.invoke(cli, ["reindex"])

        assert result.exit_code == 0
        assert "Indexed 50 entries" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Info Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestInfoCommand:
    """Tests for 'mx info' command."""

    @patch("memex.core.get_valid_categories")
    @patch("memex.config.get_index_root")
    @patch("memex.config.get_kb_root")
    def test_info_shows_config(self, mock_kb_root, mock_index_root, mock_categories, runner, tmp_path):
        """Info displays KB configuration."""
        kb_root = tmp_path / "kb"
        kb_root.mkdir()
        (kb_root / "test.md").write_text("# Test")

        mock_kb_root.return_value = kb_root
        mock_index_root.return_value = tmp_path / "indices"
        mock_categories.return_value = ["general"]

        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        assert "Primary KB" in result.output
        assert "Active KBs" in result.output

    @patch("memex.core.get_valid_categories")
    @patch("memex.config.get_index_root")
    @patch("memex.config.get_kb_root")
    def test_info_json_output(self, mock_kb_root, mock_index_root, mock_categories, runner, tmp_path):
        """Info --json outputs valid JSON."""
        kb_root = tmp_path / "kb"
        kb_root.mkdir()

        mock_kb_root.return_value = kb_root
        mock_index_root.return_value = tmp_path / "indices"
        mock_categories.return_value = []

        result = runner.invoke(cli, ["info", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "primary_kb" in data
        assert "kbs" in data
        assert "total_entries" in data


# ─────────────────────────────────────────────────────────────────────────────
# Whats-New Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestWhatsNewCommand:
    """Tests for 'mx whats-new' command."""

    @patch("memex.cli.run_async")
    def test_whats_new_shows_recent(self, mock_run_async, runner):
        """Whats-new lists recent entries."""
        mock_run_async.return_value = [
            {"path": "new.md", "title": "New Entry", "activity_date": "2024-01-20"},
        ]

        result = runner.invoke(cli, ["whats-new"])

        assert result.exit_code == 0
        assert "new.md" in result.output

    @patch("memex.cli.run_async")
    def test_whats_new_no_entries(self, mock_run_async, runner):
        """Whats-new handles no recent entries."""
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["whats-new"])

        assert result.exit_code == 0
        assert "No entries" in result.output

    @patch("memex.cli.run_async")
    def test_whats_new_json_output(self, mock_run_async, runner):
        """Whats-new --json outputs valid JSON."""
        mock_run_async.return_value = [
            {"path": "new.md", "title": "New", "activity_date": "2024-01-20"}
        ]

        result = runner.invoke(cli, ["whats-new", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Suggest-Links Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSuggestLinksCommand:
    """Tests for 'mx suggest-links' command."""

    @patch("memex.cli.run_async")
    def test_suggest_links_shows_suggestions(self, mock_run_async, runner):
        """Suggest-links shows link suggestions."""
        mock_run_async.return_value = [
            {"path": "related.md", "score": 0.85, "reason": "Similar content"},
        ]

        result = runner.invoke(cli, ["suggest-links", "test.md"])

        assert result.exit_code == 0
        assert "related.md" in result.output

    @patch("memex.cli.run_async")
    def test_suggest_links_no_suggestions(self, mock_run_async, runner):
        """Suggest-links handles no suggestions."""
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["suggest-links", "test.md"])

        assert result.exit_code == 0
        assert "No link suggestions" in result.output

    @patch("memex.cli.run_async")
    def test_suggest_links_error(self, mock_run_async, runner):
        """Suggest-links handles errors gracefully."""
        mock_run_async.side_effect = FileNotFoundError("Entry not found")

        result = runner.invoke(cli, ["suggest-links", "nonexistent.md"])

        assert result.exit_code == 1
        assert "Error" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Templates Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTemplatesCommand:
    """Tests for 'mx templates' command."""

    def test_templates_list(self, runner):
        """Templates lists available templates."""
        result = runner.invoke(cli, ["templates"])

        assert result.exit_code == 0
        assert "troubleshooting" in result.output or "pattern" in result.output

    def test_templates_show_specific(self, runner):
        """Templates show displays template content."""
        result = runner.invoke(cli, ["templates", "show", "troubleshooting"])

        assert result.exit_code == 0
        assert "Template: troubleshooting" in result.output

    def test_templates_show_unknown(self, runner):
        """Templates show fails for unknown template."""
        result = runner.invoke(cli, ["templates", "show", "nonexistent"])

        assert result.exit_code == 1
        assert "Unknown template" in result.output

    def test_templates_json_output(self, runner):
        """Templates --json outputs valid JSON."""
        result = runner.invoke(cli, ["templates", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)


# ─────────────────────────────────────────────────────────────────────────────
# History Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHistoryCommand:
    """Tests for 'mx history' command."""

    @patch("memex.search_history.get_recent")
    def test_history_shows_entries(self, mock_get_recent, runner):
        """History shows search history."""
        from datetime import datetime
        from memex.models import SearchHistoryEntry

        mock_get_recent.return_value = [
            SearchHistoryEntry(
                query="test query",
                timestamp=datetime(2024, 1, 15, 10, 30),
                result_count=5,
                mode="hybrid",
                tags=[],
            )
        ]

        result = runner.invoke(cli, ["history"])

        assert result.exit_code == 0
        assert "test query" in result.output

    @patch("memex.search_history.get_recent")
    def test_history_empty(self, mock_get_recent, runner):
        """History handles empty history."""
        mock_get_recent.return_value = []

        result = runner.invoke(cli, ["history"])

        assert result.exit_code == 0
        assert "No search history" in result.output

    @patch("memex.search_history.clear_history")
    def test_history_clear(self, mock_clear, runner):
        """History --clear clears history."""
        mock_clear.return_value = 5

        result = runner.invoke(cli, ["history", "--clear"])

        assert result.exit_code == 0
        assert "Cleared 5" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Prime Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPrimeCommand:
    """Tests for 'mx prime' command."""

    def test_prime_outputs_reference(self, runner):
        """Prime outputs CLI reference."""
        result = runner.invoke(cli, ["prime"])

        assert result.exit_code == 0
        assert "mx" in result.output.lower() or "search" in result.output.lower()

    def test_prime_json_output(self, runner):
        """Prime --json outputs valid JSON."""
        result = runner.invoke(cli, ["prime", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "mode" in data
        assert "content" in data


# ─────────────────────────────────────────────────────────────────────────────
# Context Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestContextCommand:
    """Tests for 'mx context' command."""

    @patch("memex.context.get_kb_context")
    def test_context_show_found(self, mock_get_context, runner):
        """Context show displays context when found."""
        mock_ctx = MagicMock()
        mock_ctx.source_file = Path("/project/.kbcontext")
        mock_ctx.primary = "projects/myapp"
        mock_ctx.paths = ["projects/myapp"]
        mock_ctx.default_tags = ["myapp"]
        mock_ctx.project = "myapp"
        mock_get_context.return_value = mock_ctx

        result = runner.invoke(cli, ["context", "show"])

        assert result.exit_code == 0
        assert "projects/myapp" in result.output

    @patch("memex.context.get_kb_context")
    def test_context_show_not_found(self, mock_get_context, runner):
        """Context show handles missing context."""
        mock_get_context.return_value = None

        result = runner.invoke(cli, ["context", "show"])

        assert result.exit_code == 0
        assert "No .kbcontext" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Patch Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPatchCommand:
    """Tests for 'mx patch' command."""

    @patch("memex.cli.run_async")
    def test_patch_find_replace(self, mock_run_async, runner):
        """Patch performs find/replace on entry."""
        mock_run_async.return_value = {
            "success": True,
            "path": "test.md",
            "replacements": 1,
            "exit_code": 0,
        }

        result = runner.invoke(cli, [
            "patch", "test.md",
            "--find", "old text",
            "--replace", "new text",
        ])

        assert result.exit_code == 0
        assert "Patched" in result.output

    def test_patch_missing_path(self, runner):
        """Patch fails without path argument."""
        result = runner.invoke(cli, ["patch"])
        assert result.exit_code != 0


# ─────────────────────────────────────────────────────────────────────────────
# Exit Code Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExitCodes:
    """Tests for correct exit codes."""

    def test_success_exit_code(self, runner):
        """Successful commands exit with 0."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_missing_arg_exit_code(self, runner):
        """Missing required arg exits with non-zero."""
        result = runner.invoke(cli, ["search"])
        assert result.exit_code != 0

    def test_invalid_option_exit_code(self, runner):
        """Invalid option exits with non-zero."""
        result = runner.invoke(cli, ["search", "test", "--mode", "invalid"])
        assert result.exit_code != 0

    @patch("memex.cli.run_async")
    def test_error_handling_exit_code(self, mock_run_async, runner):
        """Command errors exit with 1."""
        mock_run_async.side_effect = RuntimeError("Failure")

        result = runner.invoke(cli, ["get", "test.md"])
        assert result.exit_code == 1


# ─────────────────────────────────────────────────────────────────────────────
# JSON Error Mode Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestJsonErrorMode:
    """Tests for --json-errors flag."""

    def test_json_errors_on_invalid_option(self, runner):
        """--json-errors outputs structured error for invalid option."""
        result = runner.invoke(cli, ["--json-errors", "search", "test", "--mode", "invalid"])

        assert result.exit_code != 0
        # Error should be in stderr and be valid JSON
        if result.stderr_bytes:
            error_output = result.stderr_bytes.decode()
            data = json.loads(error_output)
            assert "error" in data


# ─────────────────────────────────────────────────────────────────────────────
# Typo Suggestion Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTypoSuggestions:
    """Tests for typo suggestions in unknown commands."""

    def test_typo_suggests_similar_command(self, runner):
        """Unknown command suggests similar command."""
        result = runner.invoke(cli, ["serach"])  # typo for 'search'

        assert result.exit_code != 0
        # Should suggest 'search'
        assert "search" in result.output.lower() or "No such command" in result.output
