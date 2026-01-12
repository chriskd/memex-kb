"""Tests for 'See also' cross-references in mx command help output."""

from click.testing import CliRunner

from memex.cli import cli


class TestSeeAlsoInHelp:
    """Test that commands include 'See also' sections in --help output."""

    def test_patch_help_includes_see_also(self):
        """mx patch --help should show related commands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["patch", "--help"])

        assert result.exit_code == 0
        assert "See also:" in result.output
        assert "mx append" in result.output
        assert "mx update" in result.output

    def test_append_help_includes_see_also(self):
        """mx append --help should show related commands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["append", "--help"])

        assert result.exit_code == 0
        assert "See also:" in result.output
        assert "mx patch" in result.output
        assert "mx update" in result.output
        assert "mx add" in result.output

    def test_update_help_includes_see_also(self):
        """mx update --help should show related commands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["update", "--help"])

        assert result.exit_code == 0
        assert "See also:" in result.output
        assert "mx patch" in result.output
        assert "mx append" in result.output

    def test_add_help_includes_see_also(self):
        """mx add --help should show related commands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["add", "--help"])

        assert result.exit_code == 0
        assert "See also:" in result.output
        assert "mx append" in result.output

    def test_search_help_includes_see_also(self):
        """mx search --help should show related commands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "--help"])

        assert result.exit_code == 0
        assert "See also:" in result.output
        assert "mx get" in result.output
        assert "mx list" in result.output

    def test_get_help_includes_see_also(self):
        """mx get --help should show related commands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["get", "--help"])

        assert result.exit_code == 0
        assert "See also:" in result.output
        assert "mx search" in result.output
        assert "mx list" in result.output
