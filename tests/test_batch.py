"""Tests for batch command parsing and execution."""

import json

import pytest
from click.testing import CliRunner

from memex.batch import (
    BatchParseError,
    ParsedCommand,
    parse_batch_command,
    validate_command,
)
from memex.cli import cli
from memex.errors import ErrorCode


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_kb(tmp_path, monkeypatch):
    """Create a temporary knowledge base for integration tests."""
    kb_root = tmp_path / "kb"
    kb_root.mkdir()

    # Create directories
    (kb_root / "tooling").mkdir()
    (kb_root / "projects").mkdir()

    # Create a sample entry for testing get/update/delete
    sample_entry = """---
title: Test Entry
tags:
  - test
  - sample
created: 2025-01-01
---

# Test Entry

This is a test entry.
"""
    (kb_root / "tooling" / "test-entry.md").write_text(sample_entry)

    # Set environment
    monkeypatch.setenv("MEMEX_KB_ROOT", str(kb_root))
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(tmp_path / "index"))

    return kb_root


class TestBatchParsing:
    """Tests for batch command parsing."""

    def test_parse_add_command_with_equals(self):
        """Parse add command with = syntax."""
        cmd = parse_batch_command(
            "add --title='My Note' --tags='a,b' --category=tooling --content='Hello'"
        )
        assert cmd.operation == "add"
        assert cmd.options["title"] == "My Note"
        assert cmd.options["tags"] == "a,b"
        assert cmd.options["category"] == "tooling"
        assert cmd.options["content"] == "Hello"

    def test_parse_add_command_with_double_quotes(self):
        """Parse add command with double quotes."""
        cmd = parse_batch_command(
            'add --title="Double quotes" --tags="x,y" --content="Content here"'
        )
        assert cmd.options["title"] == "Double quotes"
        assert cmd.options["tags"] == "x,y"
        assert cmd.options["content"] == "Content here"

    def test_parse_search_command(self):
        """Parse search command with positional argument."""
        cmd = parse_batch_command("search 'api documentation'")
        assert cmd.operation == "search"
        assert cmd.args == ["api documentation"]

    def test_parse_search_with_options(self):
        """Parse search command with options."""
        cmd = parse_batch_command("search 'query' --tags='a,b' --mode=keyword --limit=5")
        assert cmd.operation == "search"
        assert cmd.args == ["query"]
        assert cmd.options["tags"] == "a,b"
        assert cmd.options["mode"] == "keyword"
        assert cmd.options["limit"] == "5"

    def test_parse_update_with_path(self):
        """Parse update command with path argument."""
        cmd = parse_batch_command("update tooling/note.md --tags='new,tags'")
        assert cmd.operation == "update"
        assert cmd.args == ["tooling/note.md"]
        assert cmd.options["tags"] == "new,tags"

    def test_parse_get_command(self):
        """Parse get command."""
        cmd = parse_batch_command("get tooling/sample.md")
        assert cmd.operation == "get"
        assert cmd.args == ["tooling/sample.md"]

    def test_parse_get_with_metadata(self):
        """Parse get command with metadata flag."""
        cmd = parse_batch_command("get tooling/sample.md --metadata")
        assert cmd.operation == "get"
        assert cmd.args == ["tooling/sample.md"]
        assert cmd.options["metadata"] is True

    def test_parse_delete_command(self):
        """Parse delete command."""
        cmd = parse_batch_command("delete old/entry.md --force")
        assert cmd.operation == "delete"
        assert cmd.args == ["old/entry.md"]
        assert cmd.options["force"] is True

    def test_parse_upsert_command(self):
        """Parse upsert command."""
        cmd = parse_batch_command(
            "upsert 'Project Notes' --content='New content' --tags='notes'"
        )
        assert cmd.operation == "upsert"
        assert cmd.args == ["Project Notes"]
        assert cmd.options["content"] == "New content"
        assert cmd.options["tags"] == "notes"

    def test_parse_short_options(self):
        """Parse commands with short option flags."""
        cmd = parse_batch_command("add -t 'Title' --tags='tag1' -c 'Content'")
        assert cmd.options["title"] == "Title"
        assert cmd.options["content"] == "Content"

    def test_parse_unknown_command(self):
        """Unknown command raises error."""
        with pytest.raises(BatchParseError) as exc_info:
            parse_batch_command("unknown_command arg")
        assert exc_info.value.code == ErrorCode.BATCH_UNKNOWN_COMMAND

    def test_parse_empty_command(self):
        """Empty command raises error."""
        with pytest.raises(BatchParseError) as exc_info:
            parse_batch_command("")
        assert exc_info.value.code == ErrorCode.BATCH_PARSE_ERROR

    def test_parse_malformed_quotes(self):
        """Malformed quotes raise error."""
        with pytest.raises(BatchParseError) as exc_info:
            parse_batch_command("add --title='unclosed quote")
        assert exc_info.value.code == ErrorCode.BATCH_PARSE_ERROR


class TestBatchValidation:
    """Tests for command validation."""

    def test_validate_add_missing_title(self):
        """Add command requires title."""
        cmd = ParsedCommand(operation="add", args=[], options={"tags": "a,b"})
        with pytest.raises(BatchParseError) as exc_info:
            validate_command(cmd)
        assert exc_info.value.code == ErrorCode.BATCH_MISSING_ARGUMENT
        assert "title" in exc_info.value.message

    def test_validate_add_missing_tags(self):
        """Add command requires tags."""
        cmd = ParsedCommand(operation="add", args=[], options={"title": "Test"})
        with pytest.raises(BatchParseError) as exc_info:
            validate_command(cmd)
        assert exc_info.value.code == ErrorCode.BATCH_MISSING_ARGUMENT
        assert "tags" in exc_info.value.message

    def test_validate_update_missing_path(self):
        """Update command requires path argument."""
        cmd = ParsedCommand(operation="update", args=[], options={"tags": "a"})
        with pytest.raises(BatchParseError) as exc_info:
            validate_command(cmd)
        assert exc_info.value.code == ErrorCode.BATCH_MISSING_ARGUMENT

    def test_validate_search_missing_query(self):
        """Search command requires query argument."""
        cmd = ParsedCommand(operation="search", args=[], options={})
        with pytest.raises(BatchParseError) as exc_info:
            validate_command(cmd)
        assert exc_info.value.code == ErrorCode.BATCH_MISSING_ARGUMENT


class TestBatchCLI:
    """Integration tests for batch CLI command."""

    def test_batch_empty_input(self, runner, temp_kb):
        """Empty input returns error."""
        result = runner.invoke(cli, ["batch"], input="")

        assert result.exit_code == 1
        assert "No commands provided" in result.output

    def test_batch_comments_ignored(self, runner, temp_kb):
        """Comments and empty lines are ignored."""
        input_text = """# This is a comment

# Another comment
get tooling/test-entry.md
"""
        result = runner.invoke(cli, ["batch"], input=input_text)

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total"] == 1

    def test_batch_single_get(self, runner, temp_kb):
        """Execute single get command."""
        result = runner.invoke(cli, ["batch"], input="get tooling/test-entry.md")

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total"] == 1
        assert data["succeeded"] == 1
        assert data["failed"] == 0
        assert data["results"][0]["success"] is True
        assert "test-entry.md" in data["results"][0]["result"]["path"]

    def test_batch_single_search(self, runner, temp_kb):
        """Execute single search command."""
        result = runner.invoke(cli, ["batch"], input="search 'test'")

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total"] == 1
        assert data["succeeded"] == 1

    def test_batch_single_add(self, runner, temp_kb):
        """Execute single add command."""
        input_text = "add --title='New Entry' --tags='test' --category=tooling --content='Hello world'"
        result = runner.invoke(cli, ["batch"], input=input_text)

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["succeeded"] == 1
        assert data["results"][0]["success"] is True
        assert data["results"][0]["result"]["created"] is True

        # Verify file was created
        assert (temp_kb / "tooling" / "new-entry.md").exists()

    def test_batch_multiple_commands(self, runner, temp_kb):
        """Execute multiple commands."""
        input_text = """add --title='Note 1' --tags='test' --category=tooling --content='One'
add --title='Note 2' --tags='test' --category=tooling --content='Two'
search 'Note'
"""
        result = runner.invoke(cli, ["batch"], input=input_text.strip())

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total"] == 3
        assert data["succeeded"] == 3
        assert data["failed"] == 0

    def test_batch_with_error_continues(self, runner, temp_kb):
        """Errors don't stop processing by default."""
        input_text = """get tooling/test-entry.md
get nonexistent.md
get tooling/test-entry.md
"""
        result = runner.invoke(cli, ["batch"], input=input_text.strip())

        # Exit code is 1 because there was a failure
        assert result.exit_code == 1

        data = json.loads(result.output)
        assert data["total"] == 3
        assert data["succeeded"] == 2
        assert data["failed"] == 1

        # First and third succeed
        assert data["results"][0]["success"] is True
        assert data["results"][2]["success"] is True
        # Second fails
        assert data["results"][1]["success"] is False
        assert "error" in data["results"][1]

    def test_batch_stop_on_error(self, runner, temp_kb):
        """Stop on error when flag is set."""
        input_text = """get tooling/test-entry.md
get nonexistent.md
get tooling/test-entry.md
"""
        result = runner.invoke(
            cli, ["batch", "--stop-on-error"], input=input_text.strip()
        )

        assert result.exit_code == 1
        data = json.loads(result.output)
        # Should stop after second command
        assert data["total"] == 2
        assert data["succeeded"] == 1
        assert data["failed"] == 1

    def test_batch_update_with_content(self, runner, temp_kb):
        """Update entry content via batch."""
        input_text = "update tooling/test-entry.md --content='Updated content' --tags='updated,modified'"
        result = runner.invoke(cli, ["batch"], input=input_text)

        assert result.exit_code == 0, f"Output: {result.output}"
        data = json.loads(result.output)
        assert data["succeeded"] == 1

    def test_batch_from_file(self, runner, temp_kb, tmp_path):
        """Read commands from file."""
        cmd_file = tmp_path / "commands.txt"
        cmd_file.write_text("get tooling/test-entry.md\nsearch 'test'")

        result = runner.invoke(cli, ["batch", "-f", str(cmd_file)])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total"] == 2
        assert data["succeeded"] == 2

    def test_batch_unknown_command_error(self, runner, temp_kb):
        """Unknown command returns error in results."""
        result = runner.invoke(cli, ["batch"], input="invalid_cmd arg")

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["failed"] == 1
        assert "BATCH_UNKNOWN_COMMAND" in str(data["results"][0]["error"])

    def test_batch_output_format(self, runner, temp_kb):
        """Verify output JSON format."""
        result = runner.invoke(cli, ["batch"], input="get tooling/test-entry.md")

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Check required fields
        assert "total" in data
        assert "succeeded" in data
        assert "failed" in data
        assert "results" in data
        assert isinstance(data["results"], list)

        # Check result entry format
        res = data["results"][0]
        assert "index" in res
        assert "command" in res
        assert "success" in res
        assert res["index"] == 0
