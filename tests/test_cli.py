"""Tests for CLI commands (cli.py).

Tests use click.testing.CliRunner and mock core functions to isolate CLI logic.
Tests cover command-line argument parsing, option handling, output formatting,
and error handling for all CLI commands.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from memex.cli import _normalize_error_message, cli, format_table, format_tree

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_search_result():
    """Mock search result with typical structure."""
    result = MagicMock()
    result.results = [
        MagicMock(
            path="tooling/beads.md",
            title="Beads Issue Tracker",
            score=0.95,
            snippet="Issue tracking tool...",
        ),
        MagicMock(
            path="tooling/git.md",
            title="Git Workflow",
            score=0.75,
            snippet="Version control...",
        ),
    ]
    return result


@pytest.fixture
def mock_entry():
    """Mock KB entry with typical structure."""
    entry = MagicMock()
    entry.metadata = MagicMock(
        title="Test Entry",
        tags=["tag1", "tag2"],
        created="2024-01-15",
        updated="2024-01-20",
    )
    entry.content = "# Test Entry\n\nContent here."
    entry.links = ["link1.md", "link2.md"]
    entry.backlinks = ["backlink1.md"]
    entry.model_dump = MagicMock(return_value={
        "metadata": {
            "title": "Test Entry",
            "tags": ["tag1", "tag2"],
            "created": "2024-01-15",
            "updated": "2024-01-20",
        },
        "content": "# Test Entry\n\nContent here.",
        "links": ["link1.md", "link2.md"],
        "backlinks": ["backlink1.md"],
    })
    return entry


@pytest.fixture
def mock_add_result():
    """Mock add entry result."""
    result = MagicMock()
    result.created = True
    result.path = "tooling/new-entry.md"
    result.suggested_links = [
        MagicMock(path="related.md", score=0.85),
    ]
    result.suggested_tags = [
        MagicMock(tag="automation", reason="frequent term"),
    ]
    result.model_dump = MagicMock(return_value={
        "created": True,
        "path": "tooling/new-entry.md",
        "suggested_links": [{"path": "related.md", "score": 0.85}],
        "suggested_tags": [{"tag": "automation", "reason": "frequent term"}],
    })
    return result


@pytest.fixture
def mock_add_preview():
    """Mock add entry preview result."""
    result = MagicMock()
    result.path = "tooling/preview-entry.md"
    result.absolute_path = "/kb/tooling/preview-entry.md"
    result.frontmatter = (
        "---\n" "title: Preview\n" "tags:\n" "  - foo\n" "created: 2026-01-03\n" "---\n\n"
    )
    result.content = "# Preview\n\nContent."
    result.warning = None
    result.potential_duplicates = []
    result.model_dump = MagicMock(return_value={
        "path": "tooling/preview-entry.md",
        "absolute_path": "/kb/tooling/preview-entry.md",
        "frontmatter": result.frontmatter,
        "content": result.content,
        "warning": None,
        "potential_duplicates": [],
    })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Utility Function Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFormatTable:
    """Tests for table formatting utility."""

    def test_empty_table(self):
        """Test formatting empty table returns empty string."""
        result = format_table([], ["col1", "col2"])
        assert result == ""

    def test_basic_table(self):
        """Test basic table formatting."""
        rows = [
            {"name": "Alice", "age": "30"},
            {"name": "Bob", "age": "25"},
        ]
        result = format_table(rows, ["name", "age"])

        assert "NAME" in result
        assert "AGE" in result
        assert "Alice" in result
        assert "Bob" in result
        assert "30" in result
        assert "25" in result

    def test_truncation(self):
        """Test long values are truncated."""
        rows = [{"text": "a" * 100}]
        result = format_table(rows, ["text"], max_widths={"text": 20})

        assert "..." in result
        assert len(result.split("\n")[2]) <= 50  # Header + separator + row

    def test_missing_column_values(self):
        """Test handling of missing values in rows."""
        rows = [{"name": "Alice"}, {"age": "25"}]
        result = format_table(rows, ["name", "age"])

        assert "Alice" in result
        assert "25" in result


class TestFormatTree:
    """Tests for tree formatting utility."""

    def test_empty_tree(self):
        """Test formatting empty tree."""
        result = format_tree({})
        assert result == ""

    def test_single_file(self):
        """Test formatting single file."""
        tree = {"file.md": {"_type": "file", "title": "My File"}}
        result = format_tree(tree)

        assert "file.md" in result
        assert "My File" in result

    def test_directory_with_files(self):
        """Test formatting directory structure."""
        tree = {
            "dir": {
                "_type": "directory",
                "file1.md": {"_type": "file", "title": "File 1"},
                "file2.md": {"_type": "file", "title": "File 2"},
            }
        }
        result = format_tree(tree)

        assert "dir/" in result
        assert "file1.md" in result
        assert "file2.md" in result

    def test_nested_structure(self):
        """Test formatting nested directories."""
        tree = {
            "parent": {
                "_type": "directory",
                "child": {
                    "_type": "directory",
                    "file.md": {"_type": "file", "title": "Nested File"},
                },
            }
        }
        result = format_tree(tree)

        assert "parent/" in result
        assert "child/" in result
        assert "file.md" in result


class TestNormalizeErrorMessage:
    """Tests for CLI error normalization."""

    def test_rewrites_category_directory_hint(self):
        """Category/directory guidance uses CLI flag name."""
        message = (
            "Either 'category' or 'directory' must be provided. "
            "Existing categories: tooling, devops"
        )
        normalized = _normalize_error_message(message)

        assert "Either --category must be provided" in normalized

    def test_rewrites_force_true_to_flag(self):
        """force=True becomes --force."""
        message = "Use force=True to delete anyway, or update linking entries first."
        normalized = _normalize_error_message(message)

        assert "force=True" not in normalized
        assert "--force" in normalized

    def test_rewrites_rmdir_guidance(self):
        """Removes references to missing CLI commands."""
        message = "Path is not a file: docs. Use rmdir for directories."
        normalized = _normalize_error_message(message)

        assert "rmdir" not in normalized
        assert "Delete entries inside" in normalized


# ─────────────────────────────────────────────────────────────────────────────
# Search Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSearchCommand:
    """Tests for 'mx search' command."""

    @patch("memex.cli.run_async")
    def test_basic_search(self, mock_run_async, runner, mock_search_result):
        """Test basic search with query only."""
        mock_run_async.return_value = mock_search_result

        result = runner.invoke(cli, ["search", "deployment"])

        assert result.exit_code == 0
        mock_run_async.assert_called_once()
        assert "tooling/beads.md" in result.output
        assert "Beads Issue Tracker" in result.output

    @patch("memex.cli.run_async")
    def test_search_with_tags(self, mock_run_async, runner, mock_search_result):
        """Test search with tag filtering."""
        mock_run_async.return_value = mock_search_result

        result = runner.invoke(cli, ["search", "docker", "--tags", "infrastructure,devops"])

        assert result.exit_code == 0
        # Verify tags were split correctly - coroutine was called with tags parameter
        _ = mock_run_async.call_args[0][0]

    @patch("memex.cli.run_async")
    def test_search_with_mode(self, mock_run_async, runner, mock_search_result):
        """Test search with different search modes."""
        mock_run_async.return_value = mock_search_result

        result = runner.invoke(cli, ["search", "api", "--mode", "semantic"])

        assert result.exit_code == 0

    @patch("memex.cli.run_async")
    def test_search_with_limit(self, mock_run_async, runner, mock_search_result):
        """Test search with result limit."""
        mock_run_async.return_value = mock_search_result

        result = runner.invoke(cli, ["search", "test", "--limit", "5"])

        assert result.exit_code == 0

    @patch("memex.cli.run_async")
    def test_search_json_output(self, mock_run_async, runner, mock_search_result):
        """Test search with JSON output."""
        mock_run_async.return_value = mock_search_result

        result = runner.invoke(cli, ["search", "test", "--json"])

        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert isinstance(output_data, list)
        assert len(output_data) == 2
        assert output_data[0]["path"] == "tooling/beads.md"

    @patch("memex.cli.run_async")
    def test_search_no_results(self, mock_run_async, runner):
        """Test search with no results."""
        mock_result = MagicMock()
        mock_result.results = []
        mock_run_async.return_value = mock_result

        result = runner.invoke(cli, ["search", "nonexistent"])

        assert result.exit_code == 0
        assert "No results found" in result.output

    @patch("memex.cli.run_async")
    def test_search_with_content_flag(self, mock_run_async, runner, mock_search_result):
        """Test search with content inclusion flag."""
        mock_run_async.return_value = mock_search_result

        result = runner.invoke(cli, ["search", "test", "--content"])

        assert result.exit_code == 0


# ─────────────────────────────────────────────────────────────────────────────
# Get Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGetCommand:
    """Tests for 'mx get' command."""

    @patch("memex.cli.run_async")
    def test_get_basic(self, mock_run_async, runner, mock_entry):
        """Test basic get command."""
        mock_run_async.return_value = mock_entry

        result = runner.invoke(cli, ["get", "tooling/beads.md"])

        assert result.exit_code == 0
        assert "Test Entry" in result.output
        assert "Content here" in result.output

    @patch("memex.cli.run_async")
    def test_get_json_output(self, mock_run_async, runner, mock_entry):
        """Test get with JSON output."""
        mock_run_async.return_value = mock_entry

        result = runner.invoke(cli, ["get", "tooling/beads.md", "--json"])

        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert output_data["metadata"]["title"] == "Test Entry"
        assert "tag1" in output_data["metadata"]["tags"]

    @patch("memex.cli.run_async")
    def test_get_metadata_only(self, mock_run_async, runner, mock_entry):
        """Test get with metadata flag."""
        mock_run_async.return_value = mock_entry

        result = runner.invoke(cli, ["get", "tooling/beads.md", "--metadata"])

        assert result.exit_code == 0
        assert "Title:" in result.output
        assert "Tags:" in result.output
        assert "Test Entry" in result.output
        assert "tag1, tag2" in result.output

    @patch("memex.cli.run_async")
    def test_get_error_handling(self, mock_run_async, runner):
        """Test get command error handling."""
        mock_run_async.side_effect = Exception("Entry not found")

        result = runner.invoke(cli, ["get", "nonexistent.md"])

        assert result.exit_code == 1
        assert "Error" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Add Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAddCommand:
    """Tests for 'mx add' command."""

    @patch("memex.cli.run_async")
    def test_add_with_content(self, mock_run_async, runner, mock_add_result):
        """Test add command with inline content."""
        mock_run_async.return_value = mock_add_result

        result = runner.invoke(cli, [
            "add",
            "--title", "My Entry",
            "--tags", "foo,bar",
            "--content", "# My Content",
        ])

        assert result.exit_code == 0
        assert "Created: tooling/new-entry.md" in result.output

    @patch("memex.cli.run_async")
    def test_add_with_file(self, mock_run_async, runner, mock_add_result, tmp_path):
        """Test add command with file input."""
        content_file = tmp_path / "content.md"
        content_file.write_text("# File Content")

        mock_run_async.return_value = mock_add_result

        result = runner.invoke(cli, [
            "add",
            "--title", "My Entry",
            "--tags", "foo,bar",
            "--file", str(content_file),
        ])

        assert result.exit_code == 0
        assert "Created" in result.output

    @patch("memex.cli.run_async")
    def test_add_with_stdin(self, mock_run_async, runner, mock_add_result):
        """Test add command with stdin input."""
        mock_run_async.return_value = mock_add_result

        result = runner.invoke(cli, [
            "add",
            "--title", "My Entry",
            "--tags", "foo,bar",
            "--stdin",
        ], input="# Content from stdin\n")

        assert result.exit_code == 0
        assert "Created" in result.output

    @patch("memex.cli.run_async")
    def test_add_dry_run_outputs_preview(self, mock_run_async, runner, mock_add_preview):
        """Test add command dry-run preview output."""
        mock_run_async.return_value = mock_add_preview

        result = runner.invoke(cli, [
            "add",
            "--title", "Preview Entry",
            "--tags", "foo",
            "--content", "# Preview\n\nContent.",
            "--dry-run",
        ])

        assert result.exit_code == 0
        assert "Would create: /kb/tooling/preview-entry.md" in result.output
        assert "title: Preview" in result.output
        assert "# Preview" in result.output
        assert "No duplicates detected." in result.output

    @patch("memex.cli.run_async")
    def test_add_dry_run_json_output(self, mock_run_async, runner, mock_add_preview):
        """Test add command dry-run JSON output."""
        mock_run_async.return_value = mock_add_preview

        result = runner.invoke(cli, [
            "add",
            "--title", "Preview Entry",
            "--tags", "foo",
            "--content", "# Preview\n\nContent.",
            "--dry-run",
            "--json",
        ])

        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert output_data["absolute_path"] == "/kb/tooling/preview-entry.md"
        assert "frontmatter" in output_data
        assert "content" in output_data

    def test_add_no_content_source(self, runner):
        """Test add command fails without content source."""
        result = runner.invoke(cli, [
            "add",
            "--title", "My Entry",
            "--tags", "foo,bar",
        ])

        assert result.exit_code == 1
        assert "Must provide --content, --file, or --stdin" in result.output

    def test_add_help_includes_required_and_common_issues(self, runner):
        """Help text highlights required flags and common issues."""
        result = runner.invoke(cli, ["add", "--help"])

        assert result.exit_code == 0
        assert "Required:" in result.output
        assert "--category" in result.output
        assert "Common issues:" in result.output
        assert "--force" in result.output

    @patch("memex.cli.run_async")
    def test_add_with_category(self, mock_run_async, runner, mock_add_result):
        """Test add command with category."""
        mock_run_async.return_value = mock_add_result

        result = runner.invoke(cli, [
            "add",
            "--title", "My Entry",
            "--tags", "foo",
            "--category", "projects",
            "--content", "Content",
        ])

        assert result.exit_code == 0

    @patch("memex.cli.run_async")
    def test_add_json_output(self, mock_run_async, runner, mock_add_result):
        """Test add command with JSON output."""
        mock_run_async.return_value = mock_add_result

        result = runner.invoke(cli, [
            "add",
            "--title", "My Entry",
            "--tags", "foo",
            "--content", "Content",
            "--json",
        ])

        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert "path" in str(output_data) or "created" in str(output_data)

    @patch("memex.cli.run_async")
    def test_add_with_suggestions(self, mock_run_async, runner, mock_add_result):
        """Test add command shows suggestions."""
        mock_run_async.return_value = mock_add_result

        result = runner.invoke(cli, [
            "add",
            "--title", "My Entry",
            "--tags", "foo",
            "--content", "Content",
        ])

        assert result.exit_code == 0
        assert "Suggested links:" in result.output
        assert "Suggested tags:" in result.output

    @patch("memex.cli.run_async")
    def test_add_duplicate_warning(self, mock_run_async, runner):
        """Test add command with duplicate warning."""
        mock_result = MagicMock()
        mock_result.created = False
        mock_result.warning = "Potential duplicate found"
        mock_result.potential_duplicates = [
            MagicMock(path="existing.md", score=0.95),
        ]
        mock_run_async.return_value = mock_result

        result = runner.invoke(cli, [
            "add",
            "--title", "My Entry",
            "--tags", "foo",
            "--content", "Content",
        ])

        assert result.exit_code == 0
        assert "Warning:" in result.output
        assert "Potential duplicates:" in result.output

    @patch("memex.cli.run_async")
    def test_add_error_handling(self, mock_run_async, runner):
        """Test add command error handling."""
        mock_run_async.side_effect = Exception("Failed to add entry")

        with patch("memex.core.add_entry", new=lambda *args, **kwargs: None):
            result = runner.invoke(cli, [
                "add",
                "--title", "My Entry",
                "--tags", "foo",
                "--content", "Content",
            ])

        assert result.exit_code == 1
        assert "Error" in result.output

    @patch("memex.cli.run_async")
    def test_add_missing_category_guidance(self, mock_run_async, runner):
        """Missing category errors include suggestions and examples."""
        mock_run_async.side_effect = Exception(
            "Either 'category' or 'directory' must be provided "
            "(no .kbcontext file found with 'primary' field). "
            "Existing categories: tooling, devops"
        )

        with patch("memex.core.add_entry", new=lambda *args, **kwargs: None), patch(
            "memex.core.get_valid_categories",
            return_value=["tooling", "devops"],
        ):
            result = runner.invoke(cli, [
                "add",
                "--title", "My Entry",
                "--tags", "tooling,foo",
                "--content", "Content",
            ])

        assert result.exit_code == 1
        assert "Error: --category required." in result.output
        assert "Suggested: --category=tooling" in result.output
        assert "Available categories: tooling, devops" in result.output
        assert "mx context init" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Quick-Add Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestQuickAddCommand:
    """Tests for 'mx quick-add' command."""

    @patch("memex.core.get_valid_categories")
    @patch("memex.config.get_kb_root")
    def test_quick_add_json_mode(self, mock_kb_root, mock_get_categories, runner, tmp_path):
        """Test quick-add in JSON mode (non-interactive)."""
        mock_get_categories.return_value = ["projects", "tooling"]
        mock_kb_root.return_value = tmp_path

        result = runner.invoke(cli, [
            "quick-add",
            "--content", "# My Title\n\nContent here",
            "--json",
        ])

        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert "title" in output_data
        assert "tags" in output_data
        assert "category" in output_data

    @patch("memex.cli.run_async")
    @patch("memex.core.get_valid_categories")
    @patch("memex.config.get_kb_root")
    def test_quick_add_with_overrides(
        self, mock_kb_root, mock_get_categories, mock_run_async,
        runner, tmp_path, mock_add_result,
    ):
        """Test quick-add with manual overrides."""
        mock_get_categories.return_value = ["projects", "tooling"]
        mock_kb_root.return_value = tmp_path
        mock_run_async.return_value = mock_add_result

        result = runner.invoke(cli, [
            "quick-add",
            "--content", "Content",
            "--title", "Custom Title",
            "--tags", "custom,tags",
            "--category", "projects",
            "--confirm",
        ])

        assert result.exit_code == 0

    def test_quick_add_empty_content(self, runner):
        """Test quick-add with empty content."""
        result = runner.invoke(cli, [
            "quick-add",
            "--content", "   ",
            "--json",
        ])

        assert result.exit_code == 1
        assert "Content is empty" in result.output

    def test_quick_add_no_content_source(self, runner):
        """Test quick-add without content source."""
        result = runner.invoke(cli, ["quick-add"])

        assert result.exit_code == 1
        assert "Must provide --content, --file, or --stdin" in result.output

    @patch("memex.cli.run_async")
    @patch("memex.core.get_valid_categories")
    @patch("memex.config.get_kb_root")
    def test_quick_add_stdin(
        self, mock_kb_root, mock_get_categories, mock_run_async,
        runner, tmp_path, mock_add_result,
    ):
        """Test quick-add with stdin input."""
        mock_get_categories.return_value = ["projects"]
        mock_kb_root.return_value = tmp_path
        mock_run_async.return_value = mock_add_result

        result = runner.invoke(cli, [
            "quick-add",
            "--stdin",
            "--confirm",
        ], input="# Title from stdin\n\nContent")

        assert result.exit_code == 0


# ─────────────────────────────────────────────────────────────────────────────
# Info Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestInfoCommand:
    """Tests for 'mx info' and 'mx config' commands."""

    @patch("memex.core.get_valid_categories")
    @patch("memex.config.get_index_root")
    @patch("memex.config.get_kb_root")
    def test_info_basic(
        self, mock_get_kb_root, mock_get_index_root, mock_get_categories, runner, tmp_path,
    ):
        """Test basic info output."""
        kb_root = tmp_path / "kb"
        index_root = tmp_path / "indices"
        kb_root.mkdir()
        index_root.mkdir()
        (kb_root / "entry.md").write_text("# Entry")
        (kb_root / "docs").mkdir()
        (kb_root / "docs" / "second.md").write_text("# Second")

        mock_get_kb_root.return_value = kb_root
        mock_get_index_root.return_value = index_root
        mock_get_categories.return_value = ["docs", "notes"]

        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        lines = [line for line in result.output.splitlines() if ":" in line]
        values = {line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip() for line in lines}
        assert values["KB Root"] == str(kb_root)
        assert values["Index Root"] == str(index_root)
        assert values["Entries"] == "2"
        assert values["Categories"] == "docs, notes"

    @patch("memex.core.get_valid_categories")
    @patch("memex.config.get_index_root")
    @patch("memex.config.get_kb_root")
    def test_info_json_output(
        self, mock_get_kb_root, mock_get_index_root, mock_get_categories, runner, tmp_path,
    ):
        """Test info JSON output."""
        kb_root = tmp_path / "kb"
        index_root = tmp_path / "indices"
        kb_root.mkdir()
        index_root.mkdir()
        (kb_root / "entry.md").write_text("# Entry")

        mock_get_kb_root.return_value = kb_root
        mock_get_index_root.return_value = index_root
        mock_get_categories.return_value = ["docs"]

        result = runner.invoke(cli, ["info", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["kb_root"] == str(kb_root)
        assert payload["index_root"] == str(index_root)
        assert payload["categories"] == ["docs"]
        assert payload["entry_count"] == 1

    @patch("memex.core.get_valid_categories")
    @patch("memex.config.get_index_root")
    @patch("memex.config.get_kb_root")
    def test_config_alias(
        self, mock_get_kb_root, mock_get_index_root, mock_get_categories, runner, tmp_path,
    ):
        """Test config alias delegates to info."""
        kb_root = tmp_path / "kb"
        index_root = tmp_path / "indices"
        kb_root.mkdir()
        index_root.mkdir()

        mock_get_kb_root.return_value = kb_root
        mock_get_index_root.return_value = index_root
        mock_get_categories.return_value = []

        result = runner.invoke(cli, ["config", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["kb_root"] == str(kb_root)
        assert payload["index_root"] == str(index_root)


# ─────────────────────────────────────────────────────────────────────────────
# Tree Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTreeCommand:
    """Tests for 'mx tree' command."""

    @patch("memex.cli.run_async")
    def test_tree_basic(self, mock_run_async, runner):
        """Test basic tree command."""
        mock_run_async.return_value = {
            "tree": {
                "tooling": {
                    "_type": "directory",
                    "beads.md": {"_type": "file", "title": "Beads"},
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
    def test_tree_with_path(self, mock_run_async, runner):
        """Test tree with specific path."""
        mock_run_async.return_value = {
            "tree": {},
            "directories": 0,
            "files": 0,
        }

        result = runner.invoke(cli, ["tree", "tooling"])

        assert result.exit_code == 0

    @patch("memex.cli.run_async")
    def test_tree_with_depth(self, mock_run_async, runner):
        """Test tree with depth limit."""
        mock_run_async.return_value = {
            "tree": {},
            "directories": 0,
            "files": 0,
        }

        result = runner.invoke(cli, ["tree", "--depth", "2"])

        assert result.exit_code == 0

    @patch("memex.cli.run_async")
    def test_tree_json_output(self, mock_run_async, runner):
        """Test tree with JSON output."""
        tree_data = {
            "tree": {},
            "directories": 1,
            "files": 2,
        }
        mock_run_async.return_value = tree_data

        result = runner.invoke(cli, ["tree", "--json"])

        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert output_data["directories"] == 1
        assert output_data["files"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# List Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestListCommand:
    """Tests for 'mx list' command."""

    @patch("memex.cli.run_async")
    def test_list_basic(self, mock_run_async, runner):
        """Test basic list command."""
        mock_run_async.return_value = [
            {"path": "tooling/beads.md", "title": "Beads"},
            {"path": "tooling/git.md", "title": "Git"},
        ]

        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "tooling/beads.md" in result.output
        assert "Beads" in result.output

    @patch("memex.cli.run_async")
    def test_list_with_tag(self, mock_run_async, runner):
        """Test list with tag filter."""
        mock_run_async.return_value = [
            {"path": "tooling/beads.md", "title": "Beads"},
        ]

        result = runner.invoke(cli, ["list", "--tag", "tooling"])

        assert result.exit_code == 0

    @patch("memex.cli.run_async")
    def test_list_with_category(self, mock_run_async, runner):
        """Test list with category filter."""
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["list", "--category", "infrastructure"])

        assert result.exit_code == 0

    @patch("memex.cli.run_async")
    def test_list_with_limit(self, mock_run_async, runner):
        """Test list with limit."""
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["list", "--limit", "5"])

        assert result.exit_code == 0

    @patch("memex.cli.run_async")
    def test_list_no_entries(self, mock_run_async, runner):
        """Test list with no entries."""
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "No entries found" in result.output

    @patch("memex.cli.run_async")
    def test_list_json_output(self, mock_run_async, runner):
        """Test list with JSON output."""
        entries = [{"path": "test.md", "title": "Test"}]
        mock_run_async.return_value = entries

        result = runner.invoke(cli, ["list", "--json"])

        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert len(output_data) == 1
        assert output_data[0]["path"] == "test.md"


# ─────────────────────────────────────────────────────────────────────────────
# Health Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHealthCommand:
    """Tests for 'mx health' command."""

    @patch("memex.cli.run_async")
    def test_health_basic(self, mock_run_async, runner):
        """Test basic health command."""
        mock_run_async.return_value = {
            "summary": {
                "health_score": 95,
                "total_entries": 100,
            },
            "orphans": [],
            "broken_links": [],
            "stale": [],
            "empty_dirs": [],
        }

        result = runner.invoke(cli, ["health"])

        assert result.exit_code == 0
        assert "Health Score: 95/100" in result.output
        assert "Total Entries: 100" in result.output
        assert "No orphaned entries" in result.output

    @patch("memex.cli.run_async")
    def test_health_with_issues(self, mock_run_async, runner):
        """Test health command with issues found."""
        mock_run_async.return_value = {
            "summary": {
                "health_score": 70,
                "total_entries": 50,
            },
            "orphans": [{"path": "orphan.md"}],
            "broken_links": [{"source": "foo.md", "broken_link": "missing.md"}],
            "stale": [{"path": "old.md"}],
            "empty_dirs": ["empty_dir"],
        }

        result = runner.invoke(cli, ["health"])

        assert result.exit_code == 0
        assert "Orphaned entries (1)" in result.output
        assert "Broken links (1)" in result.output
        assert "Stale entries (1)" in result.output
        assert "Empty directories (1)" in result.output

    @patch("memex.cli.run_async")
    def test_health_json_output(self, mock_run_async, runner):
        """Test health with JSON output."""
        health_data = {
            "summary": {"health_score": 100, "total_entries": 10},
            "orphans": [],
            "broken_links": [],
            "stale": [],
            "empty_dirs": [],
        }
        mock_run_async.return_value = health_data

        result = runner.invoke(cli, ["health", "--json"])

        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert output_data["summary"]["health_score"] == 100


# ─────────────────────────────────────────────────────────────────────────────
# Whats-New Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestWhatsNewCommand:
    """Tests for 'mx whats-new' command."""

    @patch("memex.cli.run_async")
    def test_whats_new_basic(self, mock_run_async, runner):
        """Test basic whats-new command."""
        mock_run_async.return_value = [
            {"path": "new.md", "title": "New Entry", "activity_date": "2024-01-20"},
        ]

        result = runner.invoke(cli, ["whats-new"])

        assert result.exit_code == 0
        assert "new.md" in result.output
        assert "New Entry" in result.output

    @patch("memex.cli.run_async")
    def test_whats_new_with_days(self, mock_run_async, runner):
        """Test whats-new with days option."""
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["whats-new", "--days", "7"])

        assert result.exit_code == 0

    @patch("memex.cli.run_async")
    def test_whats_new_with_limit(self, mock_run_async, runner):
        """Test whats-new with limit option."""
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["whats-new", "--limit", "5"])

        assert result.exit_code == 0

    @patch("memex.cli.run_async")
    def test_whats_new_with_project(self, mock_run_async, runner):
        """Test whats-new with project filter."""
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["whats-new", "--project", "myapp"])

        assert result.exit_code == 0
        assert "No entries for project 'myapp'" in result.output

    @patch("memex.cli.run_async")
    def test_whats_new_no_results(self, mock_run_async, runner):
        """Test whats-new with no results."""
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["whats-new"])

        assert result.exit_code == 0
        assert "No entries created or updated" in result.output

    @patch("memex.cli.run_async")
    def test_whats_new_json_output(self, mock_run_async, runner):
        """Test whats-new with JSON output."""
        entries = [{"path": "new.md", "title": "New", "activity_date": "2024-01-20"}]
        mock_run_async.return_value = entries

        result = runner.invoke(cli, ["whats-new", "--json"])

        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert len(output_data) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Tags Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTagsCommand:
    """Tests for 'mx tags' command."""

    @patch("memex.cli.run_async")
    def test_tags_basic(self, mock_run_async, runner):
        """Test basic tags command."""
        mock_run_async.return_value = [
            {"tag": "tooling", "count": 15},
            {"tag": "infrastructure", "count": 10},
        ]

        result = runner.invoke(cli, ["tags"])

        assert result.exit_code == 0
        assert "tooling: 15" in result.output
        assert "infrastructure: 10" in result.output

    @patch("memex.cli.run_async")
    def test_tags_with_min_count(self, mock_run_async, runner):
        """Test tags with min-count filter."""
        mock_run_async.return_value = [
            {"tag": "popular", "count": 5},
        ]

        result = runner.invoke(cli, ["tags", "--min-count", "3"])

        assert result.exit_code == 0

    @patch("memex.cli.run_async")
    def test_tags_no_results(self, mock_run_async, runner):
        """Test tags with no results."""
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["tags"])

        assert result.exit_code == 0
        assert "No tags found" in result.output

    @patch("memex.cli.run_async")
    def test_tags_json_output(self, mock_run_async, runner):
        """Test tags with JSON output."""
        tags = [{"tag": "test", "count": 5}]
        mock_run_async.return_value = tags

        result = runner.invoke(cli, ["tags", "--json"])

        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert len(output_data) == 1
        assert output_data[0]["tag"] == "test"


# ─────────────────────────────────────────────────────────────────────────────
# Hubs Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHubsCommand:
    """Tests for 'mx hubs' command."""

    @patch("memex.cli.run_async")
    def test_hubs_basic(self, mock_run_async, runner):
        """Test basic hubs command."""
        mock_run_async.return_value = [
            {"path": "hub.md", "incoming": 10, "outgoing": 5, "total": 15},
        ]

        result = runner.invoke(cli, ["hubs"])

        assert result.exit_code == 0
        assert "hub.md" in result.output

    @patch("memex.cli.run_async")
    def test_hubs_with_limit(self, mock_run_async, runner):
        """Test hubs with limit option."""
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["hubs", "--limit", "5"])

        assert result.exit_code == 0

    @patch("memex.cli.run_async")
    def test_hubs_no_results(self, mock_run_async, runner):
        """Test hubs with no results."""
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["hubs"])

        assert result.exit_code == 0
        assert "No hub entries found" in result.output

    @patch("memex.cli.run_async")
    def test_hubs_json_output(self, mock_run_async, runner):
        """Test hubs with JSON output."""
        hubs = [{"path": "hub.md", "incoming": 10, "outgoing": 5, "total": 15}]
        mock_run_async.return_value = hubs

        result = runner.invoke(cli, ["hubs", "--json"])

        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert len(output_data) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Suggest-Links Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSuggestLinksCommand:
    """Tests for 'mx suggest-links' command."""

    @patch("memex.cli.run_async")
    def test_suggest_links_basic(self, mock_run_async, runner):
        """Test basic suggest-links command."""
        mock_run_async.return_value = [
            {"path": "related.md", "score": 0.85, "reason": "Similar content"},
        ]

        result = runner.invoke(cli, ["suggest-links", "entry.md"])

        assert result.exit_code == 0
        assert "related.md" in result.output
        assert "0.85" in result.output

    @patch("memex.cli.run_async")
    def test_suggest_links_with_limit(self, mock_run_async, runner):
        """Test suggest-links with limit option."""
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["suggest-links", "entry.md", "--limit", "3"])

        assert result.exit_code == 0

    @patch("memex.cli.run_async")
    def test_suggest_links_no_results(self, mock_run_async, runner):
        """Test suggest-links with no results."""
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["suggest-links", "entry.md"])

        assert result.exit_code == 0
        assert "No link suggestions found" in result.output

    @patch("memex.cli.run_async")
    def test_suggest_links_error(self, mock_run_async, runner):
        """Test suggest-links error handling."""
        mock_run_async.side_effect = Exception("Entry not found")

        result = runner.invoke(cli, ["suggest-links", "missing.md"])

        assert result.exit_code == 1
        assert "Error" in result.output

    @patch("memex.cli.run_async")
    def test_suggest_links_json_output(self, mock_run_async, runner):
        """Test suggest-links with JSON output."""
        links = [{"path": "related.md", "score": 0.85, "reason": "Similar"}]
        mock_run_async.return_value = links

        result = runner.invoke(cli, ["suggest-links", "entry.md", "--json"])

        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert len(output_data) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Update Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestUpdateCommand:
    """Tests for 'mx update' command."""

    @patch("memex.cli.run_async")
    def test_update_tags(self, mock_run_async, runner):
        """Test update command with tags."""
        mock_run_async.return_value = {"path": "entry.md"}

        result = runner.invoke(cli, ["update", "entry.md", "--tags", "new,tags"])

        assert result.exit_code == 0
        assert "Updated: entry.md" in result.output

    @patch("memex.cli.run_async")
    def test_update_content(self, mock_run_async, runner):
        """Test update command with content."""
        mock_run_async.return_value = {"path": "entry.md"}

        result = runner.invoke(cli, ["update", "entry.md", "--content", "New content"])

        assert result.exit_code == 0

    @patch("memex.cli.run_async")
    def test_update_from_file(self, mock_run_async, runner, tmp_path):
        """Test update command with file."""
        content_file = tmp_path / "new_content.md"
        content_file.write_text("Updated content")

        mock_run_async.return_value = {"path": "entry.md"}

        result = runner.invoke(cli, ["update", "entry.md", "--file", str(content_file)])

        assert result.exit_code == 0

    @patch("memex.cli.run_async")
    def test_update_error(self, mock_run_async, runner):
        """Test update command error handling."""
        mock_run_async.side_effect = Exception("Update failed")

        result = runner.invoke(cli, ["update", "entry.md", "--tags", "new"])

        assert result.exit_code == 1
        assert "Error" in result.output

    @patch("memex.cli.run_async")
    def test_update_json_output(self, mock_run_async, runner):
        """Test update command with JSON output."""
        mock_run_async.return_value = {"path": "entry.md", "updated": True}

        result = runner.invoke(cli, ["update", "entry.md", "--tags", "new", "--json"])

        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert output_data["path"] == "entry.md"


# ─────────────────────────────────────────────────────────────────────────────
# Delete Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDeleteCommand:
    """Tests for 'mx delete' command."""

    @patch("memex.cli.run_async")
    def test_delete_basic(self, mock_run_async, runner):
        """Test basic delete command."""
        mock_run_async.return_value = {"deleted": "entry.md", "had_backlinks": []}

        result = runner.invoke(cli, ["delete", "entry.md"])

        assert result.exit_code == 0
        assert "Deleted: entry.md" in result.output

    @patch("memex.cli.run_async")
    def test_delete_with_backlinks(self, mock_run_async, runner):
        """Test delete with backlinks warning."""
        mock_run_async.return_value = {
            "deleted": "entry.md",
            "had_backlinks": ["other.md"],
        }

        result = runner.invoke(cli, ["delete", "entry.md", "--force"])

        assert result.exit_code == 0
        assert "Warning: Entry had 1 backlinks" in result.output

    @patch("memex.cli.run_async")
    def test_delete_error(self, mock_run_async, runner):
        """Test delete command error handling."""
        mock_run_async.side_effect = Exception("Delete failed")

        result = runner.invoke(cli, ["delete", "entry.md"])

        assert result.exit_code == 1
        assert "Error" in result.output

    @patch("memex.cli.run_async")
    def test_delete_json_output(self, mock_run_async, runner):
        """Test delete command with JSON output."""
        mock_run_async.return_value = {"deleted": "entry.md", "had_backlinks": []}

        result = runner.invoke(cli, ["delete", "entry.md", "--json"])

        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert output_data["deleted"] == "entry.md"


# ─────────────────────────────────────────────────────────────────────────────
# Reindex Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestReindexCommand:
    """Tests for 'mx reindex' command."""

    @patch("memex.cli.run_async")
    def test_reindex_basic(self, mock_run_async, runner):
        """Test basic reindex command."""
        mock_result = MagicMock()
        mock_result.kb_files = 100
        mock_result.whoosh_docs = 100
        mock_result.chroma_docs = 100
        mock_run_async.return_value = mock_result

        result = runner.invoke(cli, ["reindex"])

        assert result.exit_code == 0
        assert "Indexed 100 entries" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Context Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestContextCommand:
    """Tests for 'mx context' subcommands."""

    @patch("memex.context.get_kb_context")
    def test_context_show_basic(self, mock_get_context, runner):
        """Test context show command."""
        mock_ctx = MagicMock()
        mock_ctx.source_file = Path("/project/.kbcontext")
        mock_ctx.primary = "projects/myapp"
        mock_ctx.paths = ["projects/myapp", "tooling"]
        mock_ctx.default_tags = ["myapp"]
        mock_ctx.project = "myapp"
        mock_get_context.return_value = mock_ctx

        result = runner.invoke(cli, ["context", "show"])

        assert result.exit_code == 0
        assert "projects/myapp" in result.output
        assert "myapp" in result.output

    @patch("memex.context.get_kb_context")
    def test_context_show_not_found(self, mock_get_context, runner):
        """Test context show when no context file exists."""
        mock_get_context.return_value = None

        result = runner.invoke(cli, ["context", "show"])

        assert result.exit_code == 0
        assert "No .kbcontext file found" in result.output

    @patch("memex.context.get_kb_context")
    def test_context_show_json(self, mock_get_context, runner):
        """Test context show with JSON output."""
        mock_ctx = MagicMock()
        mock_ctx.source_file = Path("/project/.kbcontext")
        mock_ctx.primary = "projects/myapp"
        mock_ctx.paths = ["projects/myapp"]
        mock_ctx.default_tags = ["myapp"]
        mock_ctx.project = "myapp"
        mock_get_context.return_value = mock_ctx

        result = runner.invoke(cli, ["context", "show", "--json"])

        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert output_data["found"] is True
        assert output_data["primary"] == "projects/myapp"

    def test_context_init_basic(self, runner, tmp_path, monkeypatch):
        """Test context init command."""
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(cli, ["context", "init", "--project", "myapp"])

        assert result.exit_code == 0
        assert "Created .kbcontext" in result.output
        assert (tmp_path / ".kbcontext").exists()

    def test_context_init_exists_no_force(self, runner, tmp_path, monkeypatch):
        """Test context init when file already exists."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".kbcontext").write_text("existing")

        result = runner.invoke(cli, ["context", "init"])

        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_context_init_with_force(self, runner, tmp_path, monkeypatch):
        """Test context init with force flag."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".kbcontext").write_text("existing")

        result = runner.invoke(cli, ["context", "init", "--force"])

        assert result.exit_code == 0

    @patch("memex.context.get_kb_context")
    @patch("memex.config.get_kb_root")
    @patch("memex.context.validate_context")
    def test_context_validate_success(
        self, mock_validate, mock_kb_root, mock_get_context, runner, tmp_path,
    ):
        """Test context validate with no warnings."""
        mock_ctx = MagicMock()
        mock_ctx.source_file = Path("/project/.kbcontext")
        mock_get_context.return_value = mock_ctx
        mock_kb_root.return_value = tmp_path
        mock_validate.return_value = []

        result = runner.invoke(cli, ["context", "validate"])

        assert result.exit_code == 0
        assert "All paths are valid" in result.output

    @patch("memex.context.get_kb_context")
    @patch("memex.config.get_kb_root")
    @patch("memex.context.validate_context")
    def test_context_validate_with_warnings(
        self, mock_validate, mock_kb_root, mock_get_context, runner, tmp_path,
    ):
        """Test context validate with warnings."""
        mock_ctx = MagicMock()
        mock_ctx.source_file = Path("/project/.kbcontext")
        mock_get_context.return_value = mock_ctx
        mock_kb_root.return_value = tmp_path
        mock_validate.return_value = ["Path 'foo' does not exist"]

        result = runner.invoke(cli, ["context", "validate"])

        assert result.exit_code == 0
        assert "Warnings:" in result.output
        assert "foo" in result.output

    @patch("memex.context.get_kb_context")
    def test_context_validate_no_file(self, mock_get_context, runner):
        """Test context validate when no context file exists."""
        mock_get_context.return_value = None

        result = runner.invoke(cli, ["context", "validate"])

        assert result.exit_code == 1
        assert "No .kbcontext file found" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Prime Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPrimeCommand:
    """Tests for 'mx prime' command."""

    @patch("memex.cli._detect_mcp_mode")
    @patch("memex.cli._detect_current_project")
    @patch("memex.cli._get_recent_project_entries")
    def test_prime_basic(self, mock_recent, mock_project, mock_mcp, runner):
        """Test basic prime command."""
        mock_mcp.return_value = False
        mock_project.return_value = None
        mock_recent.return_value = []

        result = runner.invoke(cli, ["prime"])

        assert result.exit_code == 0
        assert "Memex Knowledge Base" in result.output

    @patch("memex.cli._detect_mcp_mode")
    @patch("memex.cli._detect_current_project")
    @patch("memex.cli._get_recent_project_entries")
    def test_prime_mcp_mode(self, mock_recent, mock_project, mock_mcp, runner):
        """Test prime in MCP mode."""
        mock_mcp.return_value = True
        mock_project.return_value = None
        mock_recent.return_value = []

        result = runner.invoke(cli, ["prime"])

        assert result.exit_code == 0
        assert "Memex KB Active" in result.output

    @patch("memex.cli._detect_mcp_mode")
    @patch("memex.cli._detect_current_project")
    @patch("memex.cli._get_recent_project_entries")
    def test_prime_with_project(self, mock_recent, mock_project, mock_mcp, runner):
        """Test prime with project context."""
        mock_mcp.return_value = False
        mock_project.return_value = "myapp"
        mock_recent.return_value = [
            {
                "path": "projects/myapp/doc.md",
                "title": "Recent Doc",
                "activity_type": "created",
                "activity_date": "2024-01-20",
            }
        ]

        result = runner.invoke(cli, ["prime"])

        assert result.exit_code == 0
        assert "Recent KB Updates" in result.output

    @patch("memex.cli._detect_mcp_mode")
    @patch("memex.cli._detect_current_project")
    @patch("memex.cli._get_recent_project_entries")
    def test_prime_force_full(self, mock_recent, mock_project, mock_mcp, runner):
        """Test prime with --full flag."""
        mock_project.return_value = None
        mock_recent.return_value = []

        result = runner.invoke(cli, ["prime", "--full"])

        assert result.exit_code == 0
        assert "Memex Knowledge Base" in result.output

    @patch("memex.cli._detect_mcp_mode")
    @patch("memex.cli._detect_current_project")
    @patch("memex.cli._get_recent_project_entries")
    def test_prime_json_output(self, mock_recent, mock_project, mock_mcp, runner):
        """Test prime with JSON output."""
        mock_mcp.return_value = False
        mock_project.return_value = "myapp"
        mock_recent.return_value = []

        result = runner.invoke(cli, ["prime", "--json"])

        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert "mode" in output_data
        assert "content" in output_data


# ─────────────────────────────────────────────────────────────────────────────
# History Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHistoryCommand:
    """Tests for 'mx history' command."""

    @patch("memex.search_history.get_recent")
    def test_history_empty(self, mock_get_recent, runner):
        """Test history with no entries."""
        mock_get_recent.return_value = []

        result = runner.invoke(cli, ["history"])

        assert result.exit_code == 0
        assert "No search history" in result.output

    @patch("memex.search_history.get_recent")
    def test_history_shows_entries(self, mock_get_recent, runner):
        """Test history displays entries correctly."""
        from datetime import datetime

        from memex.models import SearchHistoryEntry

        mock_get_recent.return_value = [
            SearchHistoryEntry(
                query="deployment",
                timestamp=datetime(2024, 1, 15, 10, 30),
                result_count=5,
                mode="hybrid",
                tags=[],
            ),
            SearchHistoryEntry(
                query="docker",
                timestamp=datetime(2024, 1, 15, 10, 25),
                result_count=3,
                mode="semantic",
                tags=["infrastructure"],
            ),
        ]

        result = runner.invoke(cli, ["history"])

        assert result.exit_code == 0
        assert "deployment" in result.output
        assert "docker" in result.output
        assert "5 results" in result.output
        assert "infrastructure" in result.output

    @patch("memex.search_history.get_recent")
    def test_history_json_output(self, mock_get_recent, runner):
        """Test history with JSON output."""
        from datetime import datetime

        from memex.models import SearchHistoryEntry

        mock_get_recent.return_value = [
            SearchHistoryEntry(
                query="test",
                timestamp=datetime(2024, 1, 15, 10, 30),
                result_count=2,
                mode="hybrid",
                tags=["tag1"],
            ),
        ]

        result = runner.invoke(cli, ["history", "--json"])

        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert len(output_data) == 1
        assert output_data[0]["query"] == "test"
        assert output_data[0]["position"] == 1

    @patch("memex.search_history.clear_history")
    def test_history_clear(self, mock_clear, runner):
        """Test history clear."""
        mock_clear.return_value = 5

        result = runner.invoke(cli, ["history", "--clear"])

        assert result.exit_code == 0
        assert "Cleared 5 search history entries" in result.output
        mock_clear.assert_called_once()

    @patch("memex.search_history.record_search")
    @patch("memex.search_history.get_by_index")
    @patch("memex.cli.run_async")
    def test_history_rerun(self, mock_run_async, mock_get_by_index, mock_record, runner):
        """Test history rerun."""
        from datetime import datetime

        from memex.models import SearchHistoryEntry

        mock_get_by_index.return_value = SearchHistoryEntry(
            query="deployment",
            timestamp=datetime(2024, 1, 15, 10, 30),
            result_count=5,
            mode="hybrid",
            tags=[],
        )

        mock_result = MagicMock()
        mock_result.results = [
            MagicMock(path="test.md", title="Test", score=0.9, snippet="..."),
        ]
        mock_run_async.return_value = mock_result

        result = runner.invoke(cli, ["history", "--rerun", "1"])

        assert result.exit_code == 0
        assert "Re-running: deployment" in result.output
        mock_get_by_index.assert_called_once_with(1)

    @patch("memex.search_history.get_by_index")
    def test_history_rerun_invalid_index(self, mock_get_by_index, runner):
        """Test history rerun with invalid index."""
        mock_get_by_index.return_value = None

        result = runner.invoke(cli, ["history", "--rerun", "999"])

        assert result.exit_code == 1
        assert "No search at position 999" in result.output

    @patch("memex.search_history.get_recent")
    def test_history_with_limit(self, mock_get_recent, runner):
        """Test history with limit option."""
        mock_get_recent.return_value = []

        result = runner.invoke(cli, ["history", "-n", "5"])

        assert result.exit_code == 0
        mock_get_recent.assert_called_once_with(limit=5)


# ─────────────────────────────────────────────────────────────────────────────
# Edge Cases and Error Handling
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCasesAndErrors:
    """Tests for edge cases and error handling."""

    def test_missing_required_argument(self, runner):
        """Test error when required argument is missing."""
        result = runner.invoke(cli, ["search"])

        assert result.exit_code != 0

    def test_invalid_option_value(self, runner):
        """Test error with invalid option value."""
        result = runner.invoke(cli, ["search", "test", "--mode", "invalid"])

        assert result.exit_code != 0

    def test_help_text(self, runner):
        """Test help text is displayed."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "mx: Token-efficient CLI" in result.output
        assert "info" in result.output

    def test_command_help(self, runner):
        """Test individual command help."""
        result = runner.invoke(cli, ["search", "--help"])

        assert result.exit_code == 0
        assert "Search the knowledge base" in result.output

    def test_version_option(self, runner):
        """Test version option."""
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output
