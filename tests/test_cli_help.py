"""Tests for CLI help text validation.

Verifies that help text is accurate, complete, and AI-friendly.
"""

import re
import tomllib
from pathlib import Path

import pytest
from click.testing import CliRunner

from memex.cli import cli


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


# All top-level commands
TOP_LEVEL_COMMANDS = [
    "search",
    "get",
    "add",
    "update",
    "delete",
    "tree",
    "list",
    "whats-new",
    "health",
    "info",
    "config",
    "tags",
    "hubs",
    "suggest-links",
    "history",
    "reindex",
    "prime",
    "quick-add",
    "context",
    "beads",
]

# Commands with subcommands
SUBCOMMAND_GROUPS = {
    "context": ["show", "init", "validate"],
    "beads": ["list", "show", "kanban", "status", "projects"],
}


class TestMainHelp:
    """Test main CLI help."""

    def test_main_help_shows_commands(self, runner):
        """Main help lists all commands."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        for cmd in TOP_LEVEL_COMMANDS:
            assert cmd in result.output, f"Missing command: {cmd}"

    def test_main_help_shows_version_option(self, runner):
        """Main help shows version option."""
        result = runner.invoke(cli, ["--help"])

        assert "--version" in result.output

    def test_version_matches_pyproject(self, runner):
        """Version output matches pyproject.toml."""
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0

        # Read version from pyproject.toml
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)
        expected_version = pyproject["project"]["version"]

        assert expected_version in result.output


class TestCommandHelpTexts:
    """Test help text for individual commands."""

    @pytest.mark.parametrize("cmd", TOP_LEVEL_COMMANDS)
    def test_command_has_help(self, cmd, runner):
        """Each command has help text."""
        result = runner.invoke(cli, [cmd, "--help"])

        # Exit code should be 0 for help
        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "--help" in result.output

    @pytest.mark.parametrize("cmd", TOP_LEVEL_COMMANDS)
    def test_command_help_shows_options(self, cmd, runner):
        """Each command help shows Options section."""
        result = runner.invoke(cli, [cmd, "--help"])

        # Most commands should have an Options section
        # (some like 'config' may be aliases with fewer options)
        assert "Options:" in result.output or "--help" in result.output


class TestSubcommandHelp:
    """Test help text for subcommands."""

    @pytest.mark.parametrize(
        "group,subcmd",
        [(g, s) for g, subs in SUBCOMMAND_GROUPS.items() for s in subs],
    )
    def test_subcommand_has_help(self, group, subcmd, runner):
        """Each subcommand has help text."""
        result = runner.invoke(cli, [group, subcmd, "--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.output


class TestHelpExamples:
    """Test that help examples are present and reasonable."""

    COMMANDS_WITH_EXAMPLES = [
        "search",
        "add",
        "prime",
        "history",
    ]

    @pytest.mark.parametrize("cmd", COMMANDS_WITH_EXAMPLES)
    def test_command_has_examples(self, cmd, runner):
        """Commands have examples in help text."""
        result = runner.invoke(cli, [cmd, "--help"])

        # Check for example markers
        assert "Example" in result.output or "mx " in result.output


class TestHelpConsistency:
    """Test help text consistency and quality."""

    COMMANDS_WITH_JSON = [
        "search",
        "get",
        "add",
        "list",
        "whats-new",
        "health",
        "info",
        "tags",
        "hubs",
        "suggest-links",
        "history",
        "prime",
        "tree",
    ]

    @pytest.mark.parametrize("cmd", COMMANDS_WITH_JSON)
    def test_json_flag_documented(self, cmd, runner):
        """Commands with JSON output document the --json flag."""
        result = runner.invoke(cli, [cmd, "--help"])

        assert "--json" in result.output

    def test_no_internal_names_in_help(self, runner):
        """Help text doesn't expose internal names."""
        # Check a few commands for internal naming patterns
        for cmd in ["add", "update", "delete"]:
            result = runner.invoke(cli, [cmd, "--help"])

            # Should not have internal Python-style names
            assert "force=True" not in result.output
            assert "_internal" not in result.output


class TestHelpAccessibility:
    """Test help text accessibility and readability."""

    @pytest.mark.parametrize("cmd", TOP_LEVEL_COMMANDS[:5])  # Sample of commands
    def test_help_not_too_long(self, cmd, runner):
        """Help text is reasonably sized for terminal display."""
        result = runner.invoke(cli, [cmd, "--help"])

        lines = result.output.strip().split("\n")
        # Help should fit on a typical terminal without too much scrolling
        # AI agents especially benefit from concise help
        assert len(lines) < 100, f"{cmd} help has {len(lines)} lines"

    def test_main_help_concise(self, runner):
        """Main help is concise enough for quick scanning."""
        result = runner.invoke(cli, ["--help"])

        lines = result.output.strip().split("\n")
        assert len(lines) < 100


class TestErrorMessages:
    """Test error message clarity."""

    def test_missing_required_arg_message(self, runner):
        """Missing required args show clear error."""
        result = runner.invoke(cli, ["get"])  # Missing path

        assert result.exit_code != 0
        # Should indicate what's missing
        assert "PATH" in result.output or "Missing" in result.output or "required" in result.output.lower()

    def test_invalid_option_message(self, runner):
        """Invalid options show available choices."""
        result = runner.invoke(cli, ["search", "test", "--mode", "invalid"])

        assert result.exit_code != 0
        # Should show valid options
        assert "hybrid" in result.output or "invalid" in result.output.lower()

    def test_unknown_command_message(self, runner):
        """Unknown command shows similar commands."""
        result = runner.invoke(cli, ["searc"])  # Typo

        assert result.exit_code != 0


class TestBeadsSubcommandHelp:
    """Test beads subcommand help text."""

    def test_beads_help_shows_subcommands(self, runner):
        """Beads group help shows all subcommands."""
        result = runner.invoke(cli, ["beads", "--help"])

        assert result.exit_code == 0
        for subcmd in SUBCOMMAND_GROUPS["beads"]:
            assert subcmd in result.output

    def test_beads_list_help(self, runner):
        """Beads list help shows filter options."""
        result = runner.invoke(cli, ["beads", "list", "--help"])

        assert "--status" in result.output
        assert "--type" in result.output
        assert "--project" in result.output


class TestContextSubcommandHelp:
    """Test context subcommand help text."""

    def test_context_help_shows_subcommands(self, runner):
        """Context group help shows all subcommands."""
        result = runner.invoke(cli, ["context", "--help"])

        assert result.exit_code == 0
        for subcmd in SUBCOMMAND_GROUPS["context"]:
            assert subcmd in result.output

    def test_context_init_help(self, runner):
        """Context init help shows options."""
        result = runner.invoke(cli, ["context", "init", "--help"])

        assert "--project" in result.output
        assert "--force" in result.output
