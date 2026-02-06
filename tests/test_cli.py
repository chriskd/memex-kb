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

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from memex import __version__ as MEMEX_VERSION
from memex.cli import cli


class CoroutineClosingMock(MagicMock):
    """MagicMock that closes coroutine args to avoid un-awaited warnings."""

    last_coro_locals: dict[str, object] | None = None

    def __call__(self, *args, **kwargs):
        if args and inspect.iscoroutine(args[0]):
            frame = args[0].cr_frame
            if frame is not None:
                self.last_coro_locals = frame.f_locals.copy()
            try:
                args[0].close()
            except RuntimeError:
                # Closed or invalid state; keep test behavior intact.
                pass
        return super().__call__(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Command Lists
# ─────────────────────────────────────────────────────────────────────────────

# All commands that should have working --help
ALL_COMMANDS = [
    "help",
    "doctor",
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
    "relations-lint",
    "relations-add",
    "relations-remove",
]

# Commands that support --json output
JSON_COMMANDS = [
    "doctor",
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
    "relations-lint",
    "relations-add",
    "relations-remove",
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


def test_help_command_prints_main_help(runner):
    """`mx help` is a common expectation; it should print the main help."""
    result = runner.invoke(cli, ["help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Commands:" in result.output


def test_help_command_prints_subcommand_help(runner):
    """`mx help search` should print search help."""
    result = runner.invoke(cli, ["help", "search"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Search the knowledge base" in result.output


def test_version_option(runner):
    """--version outputs version number."""
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert MEMEX_VERSION in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Search Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSearchCommand:
    """Tests for 'mx search' command."""

    @patch("memex.config.get_kb_root")
    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_search_returns_results(self, mock_run_async, mock_get_kb_root, runner, tmp_path):
        """Search returns results for valid query."""
        mock_get_kb_root.return_value = tmp_path
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

    def test_search_missing_dependency_is_friendly(self, runner, tmp_kb):
        """Missing optional deps should not print a traceback; it should give an install hint."""
        from memex.errors import MemexError

        with patch("memex.core.get_searcher") as mock_get_searcher:
            mock_get_searcher.side_effect = MemexError.dependency_missing(
                feature="semantic search",
                missing=["chromadb", "sentence-transformers"],
                suggestion="uv tool install 'memex-kb[search]'",
            )

            result = runner.invoke(cli, ["search", "test"])
            assert result.exit_code == 1
            assert "Traceback" not in result.output
            assert "Optional dependency missing" in result.output
            assert "Hint:" in result.output

    def test_search_missing_dependency_json_errors(self, runner, tmp_kb):
        """--json-errors should include dependency details for missing deps."""
        from memex.errors import MemexError

        with patch("memex.core.get_searcher") as mock_get_searcher:
            mock_get_searcher.side_effect = MemexError.dependency_missing(
                feature="semantic search",
                missing=["chromadb", "sentence-transformers"],
                suggestion="uv tool install 'memex-kb[search]'",
            )

            result = runner.invoke(cli, ["--json-errors", "search", "test"])
            assert result.exit_code == 1
            assert result.stderr_bytes, "Expected JSON error in stderr"
            data = json.loads(result.stderr_bytes.decode())
            assert data["error"] == "DEPENDENCY_MISSING"
            assert data["details"]["feature"] == "semantic search"
            assert "chromadb" in data["details"]["missing"]


# ─────────────────────────────────────────────────────────────────────────────
# Relations Edit Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRelationsEditCommands:
    """Tests for relations add/remove helpers."""

    def test_relations_add_updates_frontmatter(self, runner: CliRunner, tmp_kb: Path):
        from memex.parser import parse_entry

        entry_path = tmp_kb / "a.md"
        entry_path.write_text(
            """---
title: Entry A
tags: [test]
created: 2024-01-15
---

Content A
"""
        )
        (tmp_kb / "b.md").write_text(
            """---
title: Entry B
tags: [test]
created: 2024-01-15
---

Content B
"""
        )

        result = runner.invoke(
            cli,
            ["relations-add", "a.md", "--relation", "b.md=depends_on"],
            env={"MEMEX_USER_KB_ROOT": str(tmp_kb)},
        )

        assert result.exit_code == 0
        metadata, _, _ = parse_entry(entry_path)
        assert len(metadata.relations) == 1
        assert metadata.relations[0].path == "b.md"
        assert metadata.relations[0].type == "depends_on"

    def test_relations_remove_updates_frontmatter(self, runner: CliRunner, tmp_kb: Path):
        from memex.parser import parse_entry

        entry_path = tmp_kb / "a.md"
        entry_path.write_text(
            """---
title: Entry A
tags: [test]
created: 2024-01-15
relations:
  - path: b.md
    type: depends_on
---

Content A
"""
        )
        (tmp_kb / "b.md").write_text(
            """---
title: Entry B
tags: [test]
created: 2024-01-15
---

Content B
"""
        )

        result = runner.invoke(
            cli,
            ["relations-remove", "a.md", "--relation", "b.md=depends_on"],
            env={"MEMEX_USER_KB_ROOT": str(tmp_kb)},
        )

        assert result.exit_code == 0
        metadata, _, _ = parse_entry(entry_path)
        assert metadata.relations == []

    @patch("memex.config.get_kb_root")
    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_search_no_results(self, mock_run_async, mock_get_kb_root, runner, tmp_path):
        """Search handles no results gracefully."""
        mock_get_kb_root.return_value = tmp_path
        mock_result = MagicMock()
        mock_result.results = []
        mock_run_async.return_value = mock_result

        result = runner.invoke(cli, ["search", "nonexistent"])

        assert result.exit_code == 0
        assert "No results found" in result.output

    @patch("memex.config.get_kb_root")
    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_search_json_output(self, mock_run_async, mock_get_kb_root, runner, tmp_path):
        """Search --json outputs valid JSON."""
        mock_get_kb_root.return_value = tmp_path
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

    @patch("memex.config.get_kb_root")
    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_search_terse_output(self, mock_run_async, mock_get_kb_root, runner, tmp_path):
        """Search --terse outputs only paths."""
        mock_get_kb_root.return_value = tmp_path
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

    def test_search_no_kb_configured(self, runner):
        """Search shows friendly error when no KB configured."""
        from memex.config import ConfigurationError

        with patch("memex.config.get_kb_root") as mock_get_kb_root:
            mock_get_kb_root.side_effect = ConfigurationError(
                "No knowledge base found. Options:\n"
                "  1. Run 'mx init' to create a project KB at ./kb/\n"
                "  2. Run 'mx init --user' to create a personal KB at ~/.memex/kb/"
            )

            result = runner.invoke(cli, ["search", "test"])

            assert result.exit_code == 1
            assert "No knowledge base found" in result.output

    @patch("memex.config.get_kb_root")
    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    @patch("memex.search_history.record_search")
    def test_search_records_history(
        self,
        mock_record,
        mock_run_async,
        mock_get_kb_root,
        runner,
        tmp_path,
    ):
        """Search records query in history."""
        mock_get_kb_root.return_value = tmp_path
        mock_result = MagicMock()
        mock_result.results = [MagicMock(path="test.md", title="Test", score=0.9, snippet="...")]
        mock_run_async.return_value = mock_result

        result = runner.invoke(
            cli,
            ["search", "my query", "--tags=infra,docker", "--mode=semantic"],
        )

        assert result.exit_code == 0
        mock_record.assert_called_once_with(
            query="my query",
            result_count=1,
            mode="semantic",
            tags=["infra", "docker"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Get Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGetCommand:
    """Tests for 'mx get' command."""

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
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

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
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

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
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

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
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

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_add_with_content(self, mock_run_async, runner):
        """Add creates entry with inline content."""
        mock_run_async.return_value = {
            "path": "test/entry.md",
            "suggested_links": [],
            "suggested_tags": [],
        }

        result = runner.invoke(
            cli,
            [
                "add",
                "--title",
                "Test Entry",
                "--tags",
                "tag1,tag2",
                "--content",
                "# Test Content",
            ],
        )

        assert result.exit_code == 0
        assert "Created: test/entry.md" in result.output

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_add_json_output(self, mock_run_async, runner):
        """Add --json outputs valid JSON."""
        mock_run_async.return_value = {
            "path": "test/entry.md",
            "suggested_links": [],
            "suggested_tags": [],
        }

        result = runner.invoke(
            cli,
            [
                "add",
                "--title",
                "Test",
                "--tags",
                "tag1",
                "--content",
                "Content",
                "--json",
            ],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "path" in data

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_add_warns_on_frontmatter_file(self, mock_run_async, runner, tmp_path):
        """Add warns when --file input already has frontmatter."""
        input_file = tmp_path / "input.md"
        input_file.write_text(
            "---\n"
            "title: Existing\n"
            "tags: [test]\n"
            "---\n"
            "\n"
            "# Existing Content\n"
        )

        mock_run_async.return_value = {
            "path": "test/entry.md",
            "suggested_links": [],
            "suggested_tags": [],
        }

        result = runner.invoke(
            cli,
            [
                "add",
                "--title",
                "Test",
                "--tags",
                "tag1",
                "--file",
                str(input_file),
            ],
        )

        assert result.exit_code == 0
        combined_output = result.output + getattr(result, "stderr", "")
        assert "Warning: input file already has YAML frontmatter" in combined_output

    def test_add_missing_content_source(self, runner):
        """Add fails without content source."""
        result = runner.invoke(
            cli,
            [
                "add",
                "--title",
                "Test",
                "--tags",
                "tag1",
            ],
        )

        assert result.exit_code == 1
        assert "Must provide --content, --file, or --stdin" in result.output

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_add_error_handling(self, mock_run_async, runner):
        """Add handles errors gracefully."""
        mock_run_async.side_effect = ValueError("Invalid entry")

        result = runner.invoke(
            cli,
            [
                "add",
                "--title",
                "Test",
                "--tags",
                "tag1",
                "--content",
                "Content",
            ],
        )

        assert result.exit_code == 1
        assert "Error" in result.output

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_add_with_scope_project(self, mock_run_async, runner):
        """Add with --scope=project passes scope to add_entry."""
        mock_run_async.return_value = {
            "path": "notes/entry.md",
            "suggested_links": [],
            "suggested_tags": [],
        }

        result = runner.invoke(
            cli,
            [
                "add",
                "--title",
                "Test",
                "--tags",
                "tag1",
                "--content",
                "Content",
                "--scope",
                "project",
            ],
        )

        assert result.exit_code == 0
        assert "Created: @project/notes/entry.md" in result.output

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_add_with_scope_user(self, mock_run_async, runner):
        """Add with --scope=user passes scope to add_entry."""
        mock_run_async.return_value = {
            "path": "personal/note.md",
            "suggested_links": [],
            "suggested_tags": [],
        }

        result = runner.invoke(
            cli,
            [
                "add",
                "--title",
                "Personal Note",
                "--tags",
                "personal",
                "--content",
                "My notes",
                "--scope",
                "user",
            ],
        )

        assert result.exit_code == 0
        assert "Created: @user/personal/note.md" in result.output

    @patch("memex.context.get_kb_context")
    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_add_warns_on_missing_category_defaults_to_root(
        self,
        mock_run_async,
        mock_get_context,
        runner,
    ):
        """Add warns when category is omitted and no primary is set."""
        mock_get_context.return_value = None
        mock_run_async.return_value = {
            "path": "root-entry.md",
            "suggested_links": [],
            "suggested_tags": [],
        }

        result = runner.invoke(
            cli,
            [
                "add",
                "--title",
                "Root Entry",
                "--tags",
                "tag1",
                "--content",
                "Content",
            ],
        )

        assert result.exit_code == 0
        combined_output = result.output + getattr(result, "stderr", "")
        assert "defaulting to KB root" in combined_output

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_add_with_scope_json_output(self, mock_run_async, runner):
        """Add with --scope includes scope in JSON output."""
        mock_run_async.return_value = {
            "path": "notes/entry.md",
            "suggested_links": [],
            "suggested_tags": [],
        }

        result = runner.invoke(
            cli,
            [
                "add",
                "--title",
                "Test",
                "--tags",
                "tag1",
                "--content",
                "Content",
                "--scope",
                "user",
                "--json",
            ],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["scope"] == "user"
        assert data["path"] == "notes/entry.md"

    def test_add_invalid_scope(self, runner):
        """Add rejects invalid scope values."""
        result = runner.invoke(
            cli,
            [
                "add",
                "--title",
                "Test",
                "--tags",
                "tag1",
                "--content",
                "Content",
                "--scope",
                "invalid",
            ],
        )

        assert result.exit_code != 0
        assert "Invalid value for '--scope'" in result.output

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_add_scope_error_no_user_kb(self, mock_run_async, runner):
        """Add with --scope=user fails if user KB doesn't exist."""
        from memex.config import ConfigurationError

        mock_run_async.side_effect = ConfigurationError(
            "No user KB found. Run 'mx init --user' to create one at ~/.memex/kb/"
        )

        result = runner.invoke(
            cli,
            [
                "add",
                "--title",
                "Test",
                "--tags",
                "tag1",
                "--content",
                "Content",
                "--scope",
                "user",
            ],
        )

        assert result.exit_code == 1
        assert "No user KB found" in result.output

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_add_decodes_escape_sequences(self, mock_run_async, runner):
        """Add decodes \\n, \\t, \\\\ escape sequences in --content."""
        mock_run_async.return_value = {
            "path": "test/entry.md",
            "suggested_links": [],
            "suggested_tags": [],
        }

        result = runner.invoke(
            cli,
            [
                "add",
                "--title",
                "Test Entry",
                "--tags",
                "tag1",
                "--content",
                r"Line 1\nLine 2\tTabbed\\Backslash",
            ],
        )

        assert result.exit_code == 0
        assert mock_run_async.called

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_add_with_semantic_links(self, mock_run_async, runner):
        """Add with --semantic-links passes parsed links to add_entry."""
        mock_run_async.return_value = {
            "path": "test/entry.md",
            "suggested_links": [],
            "suggested_tags": [],
        }

        result = runner.invoke(
            cli,
            [
                "add",
                "--title",
                "Test Entry",
                "--tags",
                "tag1",
                "--content",
                "Content",
                "--semantic-links",
                '[{"path": "ref/other.md", "score": 0.8, "reason": "related"}]',
            ],
        )

        assert result.exit_code == 0
        assert mock_run_async.called
        assert "Created" in result.output

    def test_add_semantic_links_invalid_json(self, runner):
        """Add fails with helpful error for invalid JSON."""
        result = runner.invoke(
            cli,
            [
                "add",
                "--title",
                "Test",
                "--tags",
                "tag1",
                "--content",
                "Content",
                "--semantic-links",
                "not valid json",
            ],
        )

        assert result.exit_code == 1
        assert "not valid JSON" in result.output

    def test_add_semantic_links_not_array(self, runner):
        """Add fails when --semantic-links is not an array."""
        result = runner.invoke(
            cli,
            [
                "add",
                "--title",
                "Test",
                "--tags",
                "tag1",
                "--content",
                "Content",
                "--semantic-links",
                '{"path": "foo.md", "score": 0.5, "reason": "test"}',
            ],
        )

        assert result.exit_code == 1
        assert "must be a JSON array" in result.output

    def test_add_semantic_links_missing_fields(self, runner):
        """Add fails when semantic link object is missing required fields."""
        result = runner.invoke(
            cli,
            [
                "add",
                "--title",
                "Test",
                "--tags",
                "tag1",
                "--content",
                "Content",
                "--semantic-links",
                '[{"path": "foo.md"}]',
            ],
        )

        assert result.exit_code == 1
        assert "missing required fields" in result.output
        assert "score" in result.output
        assert "reason" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Escape Sequence Decoding Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEscapeSequenceDecoding:
    """Tests for escape sequence handling in --content option."""

    def test_decode_escape_sequences_newline(self):
        """decode_escape_sequences converts \\n to actual newline."""
        from memex.cli import decode_escape_sequences

        result = decode_escape_sequences(r"Line 1\nLine 2")
        assert result == "Line 1\nLine 2"

    def test_decode_escape_sequences_tab(self):
        """decode_escape_sequences converts \\t to actual tab."""
        from memex.cli import decode_escape_sequences

        result = decode_escape_sequences(r"Col1\tCol2")
        assert result == "Col1\tCol2"

    def test_decode_escape_sequences_backslash(self):
        """decode_escape_sequences converts \\\\ to single backslash."""
        from memex.cli import decode_escape_sequences

        result = decode_escape_sequences(r"path\\to\\file")
        assert result == "path\\to\\file"

    def test_decode_escape_sequences_mixed(self):
        """decode_escape_sequences handles multiple escape types."""
        from memex.cli import decode_escape_sequences

        result = decode_escape_sequences(r"Line 1\n\tIndented\\escaped")
        assert result == "Line 1\n\tIndented\\escaped"

    def test_decode_escape_sequences_no_escapes(self):
        """decode_escape_sequences preserves text without escapes."""
        from memex.cli import decode_escape_sequences

        result = decode_escape_sequences("Plain text content")
        assert result == "Plain text content"


# ─────────────────────────────────────────────────────────────────────────────
# Append Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAppendCommand:
    """Tests for 'mx append' command."""

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_append_to_existing(self, mock_run_async, runner):
        """Append adds content to existing entry."""
        mock_run_async.return_value = {
            "path": "test/entry.md",
            "created": False,
            "appended_bytes": 100,
        }

        result = runner.invoke(
            cli,
            [
                "append",
                "Test Entry",
                "--content",
                "Additional content",
            ],
        )

        assert result.exit_code == 0
        assert "Appended" in result.output or "test/entry.md" in result.output

    def test_append_missing_content(self, runner):
        """Append fails without content source."""
        result = runner.invoke(cli, ["append", "Test Entry"])

        assert result.exit_code == 1
        assert "content" in result.output.lower()

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_append_decodes_escape_sequences(self, mock_run_async, runner):
        """Append decodes \\n, \\t, \\\\ in --content."""
        mock_run_async.return_value = {
            "path": "test/entry.md",
            "action": "appended",
            "suggested_links": [],
        }

        result = runner.invoke(
            cli,
            [
                "append",
                "Test Entry",
                "--content",
                r"Line 1\nLine 2\tTabbed\\Backslash",
            ],
        )

        assert result.exit_code == 0
        # Verify the content passed to append_entry has decoded escapes
        assert mock_run_async.last_coro_locals is not None
        assert mock_run_async.last_coro_locals["content"] == "Line 1\nLine 2\tTabbed\\Backslash"
        assert mock_run_async.called


# ─────────────────────────────────────────────────────────────────────────────
# Replace Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestReplaceCommand:
    """Tests for 'mx replace' command."""

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_replace_tags(self, mock_run_async, runner):
        """Replace modifies entry tags."""
        mock_run_async.return_value = {"path": "test.md", "updated": True}

        result = runner.invoke(cli, ["replace", "test.md", "--tags", "new,tags"])

        assert result.exit_code == 0
        assert "Replaced" in result.output or "test.md" in result.output

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_replace_json_output(self, mock_run_async, runner):
        """Replace --json outputs valid JSON."""
        mock_run_async.return_value = {"path": "test.md", "updated": True}

        result = runner.invoke(cli, ["replace", "test.md", "--tags", "new", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["path"] == "test.md"

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_replace_not_found(self, mock_run_async, runner):
        """Replace fails for nonexistent entry."""
        mock_run_async.side_effect = FileNotFoundError("Entry not found")

        result = runner.invoke(cli, ["replace", "nonexistent.md", "--tags", "new"])

        assert result.exit_code == 1
        assert "Error" in result.output

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_replace_decodes_escape_sequences(self, mock_run_async, runner):
        """Replace decodes \\n, \\t, \\\\ escape sequences in --content."""
        mock_run_async.return_value = {"path": "test.md", "updated": True}

        result = runner.invoke(
            cli,
            [
                "replace",
                "test.md",
                "--content",
                r"New content\nwith newline",
            ],
        )

        assert result.exit_code == 0
        assert mock_run_async.called

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_replace_with_semantic_links(self, mock_run_async, runner):
        """Replace with --semantic-links passes parsed links to update_entry."""
        mock_run_async.return_value = {"path": "test.md", "updated": True}

        result = runner.invoke(
            cli,
            [
                "replace",
                "test.md",
                "--semantic-links",
                '[{"path": "ref/related.md", "score": 0.9, "reason": "manual"}]',
            ],
        )

        assert result.exit_code == 0
        assert mock_run_async.called
        assert "Replaced" in result.output

    def test_replace_semantic_links_invalid_json(self, runner):
        """Replace fails with helpful error for invalid JSON."""
        result = runner.invoke(
            cli,
            [
                "replace",
                "test.md",
                "--semantic-links",
                "not valid json",
            ],
        )

        assert result.exit_code == 1
        assert "not valid JSON" in result.output

    def test_replace_semantic_links_not_array(self, runner):
        """Replace fails when --semantic-links is not an array."""
        result = runner.invoke(
            cli,
            [
                "replace",
                "test.md",
                "--semantic-links",
                '{"path": "foo.md", "score": 0.5, "reason": "test"}',
            ],
        )

        assert result.exit_code == 1
        assert "must be a JSON array" in result.output

    def test_replace_semantic_links_missing_fields(self, runner):
        """Replace fails when semantic link object is missing required fields."""
        result = runner.invoke(
            cli,
            [
                "replace",
                "test.md",
                "--semantic-links",
                '[{"path": "foo.md", "score": 0.5}]',
            ],
        )

        assert result.exit_code == 1
        assert "missing required fields" in result.output
        assert "reason" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Delete Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDeleteCommand:
    """Tests for 'mx delete' command."""

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_delete_entry(self, mock_run_async, runner):
        """Delete removes entry."""
        mock_run_async.return_value = {"deleted": "test.md", "had_backlinks": []}

        result = runner.invoke(cli, ["delete", "test.md"])

        assert result.exit_code == 0
        assert "Deleted" in result.output

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_delete_with_backlinks_warning(self, mock_run_async, runner):
        """Delete shows backlink warning when forced."""
        mock_run_async.return_value = {
            "deleted": "test.md",
            "had_backlinks": ["other.md"],
        }

        result = runner.invoke(cli, ["delete", "test.md", "--force"])

        assert result.exit_code == 0
        assert "backlink" in result.output.lower()

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
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

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
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

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_list_no_entries(self, mock_run_async, runner):
        """List handles empty results."""
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "No entries found" in result.output

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
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

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
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

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
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

    @patch("memex.config.get_kb_root")
    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_tags_shows_tag_counts(self, mock_run_async, mock_get_kb_root, runner, tmp_path):
        """Tags lists tags with counts."""
        mock_get_kb_root.return_value = tmp_path
        mock_run_async.return_value = [
            {"tag": "python", "count": 10},
            {"tag": "testing", "count": 5},
        ]

        result = runner.invoke(cli, ["tags"])

        assert result.exit_code == 0
        assert "python" in result.output
        assert "10" in result.output

    @patch("memex.config.get_kb_root")
    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_tags_no_tags(self, mock_run_async, mock_get_kb_root, runner, tmp_path):
        """Tags handles empty results."""
        mock_get_kb_root.return_value = tmp_path
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["tags"])

        assert result.exit_code == 0
        assert "No tags found" in result.output

    @patch("memex.config.get_kb_root")
    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_tags_json_output(self, mock_run_async, mock_get_kb_root, runner, tmp_path):
        """Tags --json outputs valid JSON."""
        mock_get_kb_root.return_value = tmp_path
        mock_run_async.return_value = [{"tag": "test", "count": 5}]

        result = runner.invoke(cli, ["tags", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]["tag"] == "test"

    def test_tags_no_kb_configured(self, runner):
        """Tags shows friendly error when no KB configured."""
        from memex.config import ConfigurationError

        with patch("memex.config.get_kb_root") as mock_get_kb_root:
            mock_get_kb_root.side_effect = ConfigurationError(
                "No knowledge base found. Options:\n"
                "  1. Run 'mx init' to create a project KB at ./kb/\n"
                "  2. Run 'mx init --user' to create a personal KB at ~/.memex/kb/"
            )

            result = runner.invoke(cli, ["tags"])

            assert result.exit_code == 1
            assert "No knowledge base found" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Relations Lint Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRelationsLintCommand:
    """Tests for 'mx relations-lint' command."""

    @patch("memex.config.get_kb_root")
    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_relations_lint_reports_issues(
        self, mock_run_async, mock_get_kb_root, runner, tmp_path
    ):
        """relations-lint prints warnings for unknown/inconsistent types."""
        mock_get_kb_root.return_value = tmp_path
        mock_run_async.return_value = {
            "summary": {
                "entries_scanned": 2,
                "relations_scanned": 2,
                "unknown_count": 1,
                "inconsistent_count": 1,
                "missing_count": 0,
            },
            "canonical_types": {"depends_on": "desc", "related": "desc"},
            "issues": [
                {
                    "issue": "unknown",
                    "path": "a.md",
                    "scope": None,
                    "target": "b.md",
                    "type": "mystery",
                    "suggestion": None,
                },
                {
                    "issue": "inconsistent",
                    "path": "c.md",
                    "scope": None,
                    "target": "d.md",
                    "type": "Depends-On",
                    "suggestion": "depends_on",
                },
            ],
        }

        result = runner.invoke(cli, ["relations-lint"])

        assert result.exit_code == 0
        assert "Relations Type Lint" in result.output
        assert "Unknown types: 1" in result.output
        assert "Inconsistent types: 1" in result.output
        assert "use depends_on" in result.output

    @patch("memex.config.get_kb_root")
    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_relations_lint_strict_exits_nonzero(
        self, mock_run_async, mock_get_kb_root, runner, tmp_path
    ):
        """relations-lint --strict exits with code 1 when issues exist."""
        mock_get_kb_root.return_value = tmp_path
        mock_run_async.return_value = {
            "summary": {
                "entries_scanned": 1,
                "relations_scanned": 1,
                "unknown_count": 1,
                "inconsistent_count": 0,
                "missing_count": 0,
            },
            "canonical_types": {},
            "issues": [
                {
                    "issue": "unknown",
                    "path": "a.md",
                    "scope": None,
                    "target": "b.md",
                    "type": "mystery",
                    "suggestion": None,
                }
            ],
        }

        result = runner.invoke(cli, ["relations-lint", "--strict"])

        assert result.exit_code == 1

    @patch("memex.config.get_kb_root")
    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_relations_lint_json_output(self, mock_run_async, mock_get_kb_root, runner, tmp_path):
        """relations-lint --json outputs valid JSON."""
        mock_get_kb_root.return_value = tmp_path
        mock_run_async.return_value = {
            "summary": {
                "entries_scanned": 1,
                "relations_scanned": 0,
                "unknown_count": 0,
                "inconsistent_count": 0,
                "missing_count": 0,
            },
            "canonical_types": {"depends_on": "desc"},
            "issues": [],
        }

        result = runner.invoke(cli, ["relations-lint", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["summary"]["relations_scanned"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Hubs Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHubsCommand:
    """Tests for 'mx hubs' command."""

    @patch("memex.config.get_kb_root")
    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_hubs_shows_hub_entries(self, mock_run_async, mock_get_kb_root, runner, tmp_path):
        """Hubs lists highly connected entries."""
        mock_get_kb_root.return_value = tmp_path
        mock_run_async.return_value = [
            {"path": "hub.md", "incoming": 10, "outgoing": 5, "total": 15},
        ]

        result = runner.invoke(cli, ["hubs"])

        assert result.exit_code == 0
        assert "hub.md" in result.output

    @patch("memex.config.get_kb_root")
    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_hubs_no_results(self, mock_run_async, mock_get_kb_root, runner, tmp_path):
        """Hubs handles empty results."""
        mock_get_kb_root.return_value = tmp_path
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["hubs"])

        assert result.exit_code == 0
        assert "No hub entries found" in result.output

    def test_hubs_no_kb_configured(self, runner):
        """Hubs shows friendly error when no KB configured."""
        from memex.config import ConfigurationError

        with patch("memex.config.get_kb_root") as mock_get_kb_root:
            mock_get_kb_root.side_effect = ConfigurationError(
                "No knowledge base found. Options:\n"
                "  1. Run 'mx init' to create a project KB at ./kb/\n"
                "  2. Run 'mx init --user' to create a personal KB at ~/.memex/kb/"
            )

            result = runner.invoke(cli, ["hubs"])

            assert result.exit_code == 1
            assert "No knowledge base found" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Health Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHealthCommand:
    """Tests for 'mx health' command."""

    @patch("memex.config.get_kb_root")
    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_health_shows_status(self, mock_run_async, mock_get_kb_root, runner, tmp_path):
        """Health shows KB health status."""
        mock_get_kb_root.return_value = tmp_path
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

    @patch("memex.config.get_kb_root")
    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_health_with_issues(self, mock_run_async, mock_get_kb_root, runner, tmp_path):
        """Health reports found issues."""
        mock_get_kb_root.return_value = tmp_path
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

    @patch("memex.config.get_kb_root")
    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_health_json_output(self, mock_run_async, mock_get_kb_root, runner, tmp_path):
        """Health --json outputs valid JSON."""
        mock_get_kb_root.return_value = tmp_path
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

    def test_health_no_kb_configured(self, runner):
        """Health shows friendly error when no KB configured."""
        from memex.config import ConfigurationError

        with patch("memex.config.get_kb_root") as mock_get_kb_root:
            mock_get_kb_root.side_effect = ConfigurationError(
                "No knowledge base found. Options:\n"
                "  1. Run 'mx init' to create a project KB at ./kb/\n"
                "  2. Run 'mx init --user' to create a personal KB at ~/.memex/kb/"
            )

            result = runner.invoke(cli, ["health"])

            assert result.exit_code == 1
            assert "No knowledge base found" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Init Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestInitCommand:
    """Tests for 'mx init' command."""

    def test_init_creates_kb(self, runner, tmp_path, monkeypatch):
        """Init creates KB directory structure."""
        # Change to tmp_path so .kbconfig is written there, not project root
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(cli, ["init", "--path", str(tmp_path / "kb")])

        assert result.exit_code == 0
        assert "Initialized" in result.output
        assert "--local" not in result.output
        assert (tmp_path / "kb").exists()
        assert (tmp_path / "kb" / "README.md").exists()
        # Verify .kbconfig was written to tmp_path, not project root
        assert (tmp_path / ".kbconfig").exists()

        from memex.parser import parse_entry

        metadata, content, _chunks = parse_entry(tmp_path / "kb" / "README.md")
        assert metadata.title == "Project Knowledge Base"
        assert "project" in metadata.tags
        assert content.lstrip().startswith("# Project Knowledge Base")

    def test_init_already_exists(self, runner, tmp_path, monkeypatch):
        """Init fails if KB already exists (without --force)."""
        monkeypatch.chdir(tmp_path)
        kb_path = tmp_path / "kb"
        kb_path.mkdir()

        result = runner.invoke(cli, ["init", "--path", str(kb_path)])

        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_init_mutually_exclusive_options(self, runner, tmp_path, monkeypatch):
        """Init rejects --user and --path together."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(cli, ["init", "--user", "--path", str(tmp_path / "kb")])

        assert result.exit_code == 1
        assert "mutually exclusive" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Reindex Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestReindexCommand:
    """Tests for 'mx reindex' command."""

    @patch("memex.config.get_kb_root")
    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_reindex_success(self, mock_run_async, mock_get_kb_root, runner, tmp_path):
        """Reindex reports indexed entries."""
        mock_get_kb_root.return_value = tmp_path
        mock_result = MagicMock()
        mock_result.kb_files = 50
        mock_result.whoosh_docs = 50
        mock_result.chroma_docs = 50
        mock_run_async.return_value = mock_result

        result = runner.invoke(cli, ["reindex"])

        assert result.exit_code == 0
        assert "Indexed 50 entries" in result.output

    def test_reindex_no_kb_configured(self, runner):
        """Reindex shows friendly error when no KB configured."""
        from memex.config import ConfigurationError

        with patch("memex.config.get_kb_root") as mock_get_kb_root:
            mock_get_kb_root.side_effect = ConfigurationError(
                "No knowledge base found. Options:\n"
                "  1. Run 'mx init' to create a project KB at ./kb/\n"
                "  2. Run 'mx init --user' to create a personal KB at ~/.memex/kb/"
            )

            result = runner.invoke(cli, ["reindex"])

            assert result.exit_code == 1
            assert "No knowledge base found" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Info Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestInfoCommand:
    """Tests for 'mx info' command."""

    @patch("memex.core.get_valid_categories")
    @patch("memex.config.get_index_root")
    @patch("memex.config.get_kb_root")
    def test_info_shows_config(
        self,
        mock_kb_root,
        mock_index_root,
        mock_categories,
        runner,
        tmp_path,
    ):
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
    def test_info_json_output(
        self,
        mock_kb_root,
        mock_index_root,
        mock_categories,
        runner,
        tmp_path,
    ):
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

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_whats_new_shows_recent(self, mock_run_async, runner):
        """Whats-new lists recent entries."""
        mock_run_async.return_value = [
            {"path": "new.md", "title": "New Entry", "activity_date": "2024-01-20"},
        ]

        result = runner.invoke(cli, ["whats-new"])

        assert result.exit_code == 0
        assert "new.md" in result.output

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_whats_new_no_entries(self, mock_run_async, runner):
        """Whats-new handles no recent entries."""
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["whats-new"])

        assert result.exit_code == 0
        assert "No entries" in result.output

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
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

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_suggest_links_shows_suggestions(self, mock_run_async, runner):
        """Suggest-links shows link suggestions."""
        mock_run_async.return_value = [
            {"path": "related.md", "score": 0.85, "reason": "Similar content"},
        ]

        result = runner.invoke(cli, ["suggest-links", "test.md"])

        assert result.exit_code == 0
        assert "related.md" in result.output

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_suggest_links_no_suggestions(self, mock_run_async, runner):
        """Suggest-links handles no suggestions."""
        mock_run_async.return_value = []

        result = runner.invoke(cli, ["suggest-links", "test.md"])

        assert result.exit_code == 0
        assert "No link suggestions" in result.output

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
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
        mock_ctx.source_file = Path("/project/.kbconfig")
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
        assert "No .kbconfig" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Patch Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPatchCommand:
    """Tests for 'mx patch' command."""

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
    def test_patch_find_replace(self, mock_run_async, runner):
        """Patch performs find/replace on entry."""
        mock_run_async.return_value = {
            "success": True,
            "path": "test.md",
            "replacements": 1,
            "exit_code": 0,
        }

        result = runner.invoke(
            cli,
            [
                "patch",
                "test.md",
                "--find",
                "old text",
                "--replace",
                "new text",
            ],
        )

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

    @patch("memex.cli.run_async", new_callable=CoroutineClosingMock)
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

    def test_json_errors_on_configuration_error_search(self, runner):
        """--json-errors outputs JSON for ConfigurationError in search command."""
        from memex.config import ConfigurationError

        with patch("memex.config.get_kb_root") as mock_get_kb_root:
            mock_get_kb_root.side_effect = ConfigurationError("No knowledge base found.")

            result = runner.invoke(cli, ["--json-errors", "search", "test"])

            assert result.exit_code != 0
            # Should output JSON to stderr
            if result.stderr_bytes:
                error_output = result.stderr_bytes.decode()
                data = json.loads(error_output)
                assert "error" in data or "code" in data
                assert "No knowledge base found" in error_output

    def test_json_errors_on_configuration_error_health(self, runner):
        """--json-errors outputs JSON for ConfigurationError in health command."""
        from memex.config import ConfigurationError

        with patch("memex.config.get_kb_root") as mock_get_kb_root:
            mock_get_kb_root.side_effect = ConfigurationError("No knowledge base found.")

            result = runner.invoke(cli, ["--json-errors", "health"])

            assert result.exit_code != 0
            if result.stderr_bytes:
                error_output = result.stderr_bytes.decode()
                data = json.loads(error_output)
                assert "error" in data or "code" in data

    def test_json_errors_on_configuration_error_tags(self, runner):
        """--json-errors outputs JSON for ConfigurationError in tags command."""
        from memex.config import ConfigurationError

        with patch("memex.config.get_kb_root") as mock_get_kb_root:
            mock_get_kb_root.side_effect = ConfigurationError("No knowledge base found.")

            result = runner.invoke(cli, ["--json-errors", "tags"])

            assert result.exit_code != 0
            if result.stderr_bytes:
                error_output = result.stderr_bytes.decode()
                data = json.loads(error_output)
                assert "error" in data or "code" in data

    def test_plain_text_error_without_json_errors_flag(self, runner):
        """Without --json-errors, ConfigurationError outputs plain text."""
        from memex.config import ConfigurationError

        with patch("memex.config.get_kb_root") as mock_get_kb_root:
            mock_get_kb_root.side_effect = ConfigurationError("No knowledge base found.")

            result = runner.invoke(cli, ["search", "test"])

            assert result.exit_code != 0
            # Should NOT be valid JSON (plain text error)
            output = result.output
            assert "Error:" in output or "No knowledge base found" in output
            # Verify it's not JSON
            try:
                json.loads(output)
                pytest.fail("Output should not be valid JSON without --json-errors")
            except json.JSONDecodeError:
                pass  # Expected - plain text is not JSON


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


# ─────────────────────────────────────────────────────────────────────────────
# Input Validation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestInputValidation:
    """Tests for CLI input validation - ensures clean errors, not tracebacks.

    These tests verify that invalid inputs produce user-friendly error messages
    rather than Python tracebacks. Covers bugs: jcg7, e4sx, wydj.
    """

    # --limit validation (bug: jcg7)

    def test_search_limit_zero_shows_error(self, runner):
        """--limit 0 should show a clean error, not a traceback."""
        result = runner.invoke(cli, ["search", "test", "--limit", "0"])
        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "0 is not in the range" in result.output

    def test_search_limit_negative_shows_error(self, runner):
        """--limit -1 should show a clean error, not a traceback."""
        result = runner.invoke(cli, ["search", "test", "--limit", "-1"])
        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "-1 is not in the range" in result.output

    def test_search_limit_positive_accepted(self, runner):
        """--limit with positive values should be accepted."""
        result = runner.invoke(cli, ["search", "test", "--limit", "1"])
        assert "is not in the range" not in result.output

    # Empty query validation (bug: terw)

    def test_search_empty_query_shows_error(self, runner):
        """Empty string query should show a clean error."""
        result = runner.invoke(cli, ["search", ""])
        assert result.exit_code != 0
        assert "Query cannot be empty" in result.output

    def test_search_whitespace_query_shows_error(self, runner):
        """Whitespace-only query should show a clean error."""
        result = runner.invoke(cli, ["search", "   "])
        assert result.exit_code != 0
        assert "Query cannot be empty" in result.output

    # --json-errors validation (bug: wydj)

    def test_empty_query_json_errors_mode(self, runner):
        """Empty query with --json-errors should return JSON error."""
        result = runner.invoke(cli, ["--json-errors", "search", ""])
        assert result.exit_code != 0
        error_data = json.loads(result.output)
        assert "error" in error_data
        assert "Query cannot be empty" in error_data["error"]["message"]

    # Invalid category validation (bug: e4sx)

    def test_list_invalid_category_no_traceback(self, runner):
        """Invalid category should not cause traceback."""
        result = runner.invoke(cli, ["list", "--category", "nonexistent"])
        # Should handle gracefully (either error or empty results), never traceback
        assert "Traceback" not in result.output


# ─────────────────────────────────────────────────────────────────────────────
# KBContext Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestKBContext:
    """Tests for KBContext dataclass and loading."""

    def test_kbconfig_has_project_kb_attribute(self):
        """KBContext has project_kb attribute."""
        from memex.context import KBContext

        ctx = KBContext()
        assert hasattr(ctx, "project_kb")
        assert ctx.project_kb is None

    def test_kbconfig_has_publish_base_url_attribute(self):
        """KBContext has publish_base_url attribute."""
        from memex.context import KBContext

        ctx = KBContext()
        assert hasattr(ctx, "publish_base_url")
        assert ctx.publish_base_url is None

    def test_kbconfig_from_dict_loads_project_kb(self):
        """KBContext.from_dict loads project_kb field."""
        from memex.context import KBContext

        data = {"project_kb": "./kb"}
        ctx = KBContext.from_dict(data)
        assert ctx.project_kb == "./kb"

    def test_kbconfig_from_dict_loads_publish_base_url(self):
        """KBContext.from_dict loads publish_base_url field."""
        from memex.context import KBContext

        data = {"publish_base_url": "/my-repo"}
        ctx = KBContext.from_dict(data)
        assert ctx.publish_base_url == "/my-repo"

    def test_load_kbconfig_as_context_loads_publish_fields(self, tmp_path):
        """_load_kbconfig_as_context loads project_kb and publish_base_url."""
        from memex.context import _load_kbconfig_as_context

        config_file = tmp_path / ".kbconfig"
        config_file.write_text("""
project_kb: ./kb
publish_base_url: /my-project
primary: docs
""")

        ctx = _load_kbconfig_as_context(config_file)
        assert ctx is not None
        assert ctx.project_kb == "./kb"
        assert ctx.publish_base_url == "/my-project"
        assert ctx.primary == "docs"


# ─────────────────────────────────────────────────────────────────────────────
# Publish Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPublishCommand:
    """Tests for mx publish command."""

    def test_publish_help_works(self, runner):
        """publish --help shows usage."""
        result = runner.invoke(cli, ["publish", "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_publish_no_kb_shows_options(self, runner, tmp_path, monkeypatch):
        """publish with no KB shows helpful error with options."""
        # Run from a directory with no KB
        from memex.context import clear_context_cache, clear_kbconfig_cache

        clear_context_cache()
        clear_kbconfig_cache()
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(
            cli,
            ["publish"],
            env={"MEMEX_USER_KB_ROOT": str(tmp_path)},
        )
        assert result.exit_code == 1
        assert "No KB found to publish" in result.output or "Error:" in result.output

    def test_publish_with_kb_root_works(self, tmp_kb_with_entries, runner):
        """publish --kb-root successfully publishes entries."""
        import tempfile

        with tempfile.TemporaryDirectory() as output_dir:
            result = runner.invoke(
                cli,
                ["publish", "--kb-root", str(tmp_kb_with_entries), "-o", output_dir],
            )
            assert result.exit_code == 0
            assert "Published" in result.output
            # Check output was created
            assert (Path(output_dir) / "index.html").exists()

    def test_publish_uses_kb_path_from_kbconfig(self, runner, tmp_path, monkeypatch):
        """publish without --kb-root uses kb_path from .kbconfig."""
        import tempfile

        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        (kb_dir / "entry.md").write_text("---\ntitle: Entry\ntags: [test]\n---\nContent")
        (tmp_path / ".kbconfig").write_text("kb_path: ./kb\n")

        monkeypatch.chdir(tmp_path)

        with tempfile.TemporaryDirectory() as output_dir:
            result = runner.invoke(
                cli,
                ["publish", "-o", output_dir],
            )
            assert result.exit_code == 0
            assert (Path(output_dir) / "index.html").exists()

    def test_setup_github_actions_dry_run(self, runner, tmp_path):
        """--setup-github-actions --dry-run shows workflow without creating file."""
        import subprocess

        # Create a git repo with a KB directory
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        (kb_dir / "test.md").write_text("---\ntitle: Test\ntags: [test]\n---\nContent")

        result = runner.invoke(
            cli,
            ["publish", "--setup-github-actions", "--dry-run", "--kb-root", str(kb_dir)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "Would create workflow at:" in result.output
        assert "publish-kb.yml" in result.output
        assert "name: Publish KB to GitHub Pages" in result.output
        assert "paths:" in result.output
        assert "'kb/**'" in result.output
        # Should not actually create the file
        assert not (tmp_path / ".github" / "workflows" / "publish-kb.yml").exists()

    def test_setup_github_actions_creates_workflow(self, runner, tmp_path):
        """--setup-github-actions creates workflow file."""
        import subprocess

        # Create a git repo with a KB directory
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        (kb_dir / "test.md").write_text("---\ntitle: Test\ntags: [test]\n---\nContent")

        result = runner.invoke(
            cli,
            ["publish", "--setup-github-actions", "--kb-root", str(kb_dir)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "Created GitHub Actions workflow:" in result.output
        assert "Next steps:" in result.output

        # Verify the file was created
        workflow_path = tmp_path / ".github" / "workflows" / "publish-kb.yml"
        assert workflow_path.exists()

        # Verify content
        content = workflow_path.read_text()
        assert "name: Publish KB to GitHub Pages" in content
        assert "mx publish --kb-root ./kb" in content
        assert "actions/deploy-pages@v4" in content

    def test_setup_github_actions_uses_explicit_base_url(self, runner, tmp_path):
        """--setup-github-actions uses explicit --base-url in workflow."""
        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        (kb_dir / "test.md").write_text("---\ntitle: Test\ntags: [test]\n---\nContent")

        result = runner.invoke(
            cli,
            [
                "publish",
                "--setup-github-actions",
                "--dry-run",
                "--kb-root",
                str(kb_dir),
                "--base-url",
                "/my-project",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "--base-url /my-project" in result.output
        # Should NOT use the GitHub variable when explicit URL is given
        assert "${GITHUB_REPOSITORY" not in result.output

    def test_setup_github_actions_auto_detects_base_url(self, runner, tmp_path):
        """--setup-github-actions uses GitHub variable for base URL by default."""
        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        (kb_dir / "test.md").write_text("---\ntitle: Test\ntags: [test]\n---\nContent")

        result = runner.invoke(
            cli,
            ["publish", "--setup-github-actions", "--dry-run", "--kb-root", str(kb_dir)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        # Should use the GitHub variable for auto-detection
        assert "${GITHUB_REPOSITORY#*/}" in result.output

    def test_setup_github_actions_fails_outside_git_repo(self, runner, tmp_path):
        """--setup-github-actions fails outside a git repository."""
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        (kb_dir / "test.md").write_text("---\ntitle: Test\ntags: [test]\n---\nContent")

        result = runner.invoke(
            cli,
            ["publish", "--setup-github-actions", "--kb-root", str(kb_dir)],
        )

        assert result.exit_code == 1
        assert "Not in a git repository" in result.output
