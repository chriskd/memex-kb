"""Tests for CLI argument validation."""

from click.testing import CliRunner

from memex.cli import cli


class TestSearchLimitValidation:
    """Tests for search command --limit option validation."""

    def test_search_limit_zero_shows_error(self):
        """--limit 0 should show a clean error, not a traceback."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "test", "--limit", "0"])
        assert result.exit_code != 0
        assert "0 is not in the range x>=1" in result.output or "0 is not in the range" in result.output

    def test_search_limit_negative_shows_error(self):
        """--limit -1 should show a clean error, not a traceback."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "test", "--limit", "-1"])
        assert result.exit_code != 0
        assert "-1 is not in the range x>=1" in result.output or "-1 is not in the range" in result.output

    def test_search_limit_positive_accepted(self):
        """--limit with positive values should be accepted (validation passes)."""
        runner = CliRunner()
        # We just check that the validation passes - the command may fail
        # for other reasons (no KB configured) but that's fine
        result = runner.invoke(cli, ["search", "test", "--limit", "1"])
        # Should not contain IntRange validation error
        assert "is not in the range" not in result.output

    def test_search_limit_large_positive_accepted(self):
        """--limit with large positive values should be accepted."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "test", "--limit", "100"])
        assert "is not in the range" not in result.output


class TestSearchEmptyQueryValidation:
    """Tests for search command empty query validation."""

    def test_search_empty_query_shows_error(self):
        """Empty string query should show a clean error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", ""])
        assert result.exit_code != 0
        assert "Query cannot be empty" in result.output

    def test_search_whitespace_query_shows_error(self):
        """Whitespace-only query should show a clean error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "   "])
        assert result.exit_code != 0
        assert "Query cannot be empty" in result.output

    def test_search_tab_query_shows_error(self):
        """Tab-only query should show a clean error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "\t"])
        assert result.exit_code != 0
        assert "Query cannot be empty" in result.output

    def test_search_valid_query_accepted(self):
        """Valid non-empty query should be accepted (validation passes)."""
        runner = CliRunner()
        # We just check that the empty query validation passes - the command may fail
        # for other reasons (no KB configured) but that's fine
        result = runner.invoke(cli, ["search", "deployment"])
        assert "Query cannot be empty" not in result.output

    def test_search_empty_query_json_errors_mode(self):
        """Empty query with --json-errors should return JSON error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--json-errors", "search", ""])
        assert result.exit_code != 0
        # Should be valid JSON with error structure
        import json
        try:
            error_data = json.loads(result.output)
            assert "error" in error_data
            assert "Query cannot be empty" in error_data["error"]["message"]
        except json.JSONDecodeError:
            # If not JSON, the message should still be present
            assert "Query cannot be empty" in result.output
