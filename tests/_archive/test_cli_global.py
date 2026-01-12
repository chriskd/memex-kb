"""Tests for mx CLI global flags (--version, --help)."""

import re

import pytest
from click.testing import CliRunner

from memex.cli import cli


class TestVersion:
    """Tests for --version flag."""

    def test_version_flag_shows_version(self):
        """--version shows version number."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower() or "0." in result.output

    def test_version_format(self):
        """Version matches expected semantic versioning format."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        # Version should match semantic versioning (e.g., 0.1.0)
        assert re.search(r"\d+\.\d+\.\d+", result.output)


class TestHelp:
    """Tests for --help flag."""

    def test_help_shows_all_commands(self):
        """--help lists all available commands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        # Should list main commands
        commands = ["search", "get", "add", "list", "tree", "tags", "health"]
        for cmd in commands:
            assert cmd in result.output, f"Command '{cmd}' not found in help output"

    def test_command_help_shows_options(self):
        """Command-specific help shows all relevant options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "--help"])
        assert result.exit_code == 0
        # Should show search-specific options
        assert "--tags" in result.output or "-t" in result.output
        assert "--limit" in result.output or "-n" in result.output
        assert "--json" in result.output

    def test_help_shows_usage(self):
        """Help text includes usage pattern."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output or "usage:" in result.output.lower()
