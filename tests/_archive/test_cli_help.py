"""Tests for CLI help - focused on catching real regressions."""

import json
import tomllib
from pathlib import Path

from click.testing import CliRunner

from memex.cli import cli

# All top-level commands (used for bulk validation)
TOP_LEVEL_COMMANDS = [
    "search", "get", "add", "update", "delete", "tree", "list",
    "whats-new", "health", "info", "config", "tags", "hubs",
    "suggest-links", "history", "reindex", "prime", "quick-add",
    "context", "schema",
]


def test_main_help_lists_all_commands():
    """Main help lists all commands without errors."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    for cmd in TOP_LEVEL_COMMANDS:
        assert cmd in result.output, f"Missing command: {cmd}"


def test_version_matches_pyproject():
    """Version output matches pyproject.toml."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        expected_version = tomllib.load(f)["project"]["version"]
    assert expected_version in result.output


def test_all_commands_have_working_help():
    """Every command's --help runs without error."""
    runner = CliRunner()
    for cmd in TOP_LEVEL_COMMANDS:
        result = runner.invoke(cli, [cmd, "--help"])
        assert result.exit_code == 0, f"{cmd} --help failed: {result.output}"
        assert "Usage:" in result.output


def test_schema_outputs_valid_json_with_all_commands():
    """Schema command outputs valid JSON containing all commands."""
    runner = CliRunner()
    result = runner.invoke(cli, ["schema"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "version" in data
    assert "commands" in data
    for cmd in TOP_LEVEL_COMMANDS:
        assert cmd in data["commands"], f"Missing {cmd} in schema"


def test_error_messages_are_helpful():
    """Error messages indicate what went wrong."""
    runner = CliRunner()

    # Missing required arg
    result = runner.invoke(cli, ["get"])
    assert result.exit_code != 0
    assert "PATH" in result.output or "Missing" in result.output

    # Invalid option value
    result = runner.invoke(cli, ["search", "test", "--mode", "invalid"])
    assert result.exit_code != 0
