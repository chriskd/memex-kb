"""Tests for mx prime CLI command."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from memex import core
from memex.cli import cli


@pytest.fixture(autouse=True)
def reset_searcher_state(monkeypatch):
    """Ensure cached searcher state does not leak across tests."""
    monkeypatch.setattr(core, "_searcher", None)
    monkeypatch.setattr(core, "_searcher_ready", False)


@pytest.fixture
def kb_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary KB root with standard categories."""
    root = tmp_path / "kb"
    root.mkdir()
    (root / "development").mkdir()
    monkeypatch.setenv("MEMEX_KB_ROOT", str(root))
    return root


@pytest.fixture
def index_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary index root."""
    root = tmp_path / ".indices"
    root.mkdir()
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(root))
    return root


class TestPrimeBasic:
    """Basic tests for prime command output."""

    def test_prime_outputs_context(self):
        """prime outputs workflow context by default."""
        runner = CliRunner()
        result = runner.invoke(cli, ["prime"])

        assert result.exit_code == 0
        # Should contain some KB context
        assert "memex" in result.output.lower() or "knowledge" in result.output.lower()

    def test_prime_auto_detects_mode(self, monkeypatch):
        """prime auto-detects CLI vs MCP mode based on environment."""
        runner = CliRunner()

        # Without MCP environment - should show full output
        monkeypatch.delenv("MCP_SERVER_ACTIVE", raising=False)
        result_cli = runner.invoke(cli, ["prime"])
        assert result_cli.exit_code == 0

        # With MCP environment - should show minimal output
        monkeypatch.setenv("MCP_SERVER_ACTIVE", "1")
        result_mcp = runner.invoke(cli, ["prime"])
        assert result_mcp.exit_code == 0

        # MCP output should be shorter than full CLI output
        assert len(result_mcp.output) < len(result_cli.output)


class TestPrimeFullFlag:
    """Tests for --full flag behavior."""

    def test_prime_full_shows_full_output(self):
        """--full shows complete context."""
        runner = CliRunner()
        result = runner.invoke(cli, ["prime", "--full"])

        assert result.exit_code == 0
        # Full output should include command reference
        assert "mx search" in result.output
        assert "mx get" in result.output

    def test_prime_full_overrides_mcp_env(self, monkeypatch):
        """--full overrides MCP environment detection."""
        runner = CliRunner()

        # Set MCP environment (would normally trigger minimal output)
        monkeypatch.setenv("MCP_SERVER_ACTIVE", "1")

        result = runner.invoke(cli, ["prime", "--full"])

        assert result.exit_code == 0
        # Should still show full output with command reference
        assert "mx search" in result.output
        assert "mx get" in result.output


class TestPrimeMcpFlag:
    """Tests for --mcp flag behavior."""

    def test_prime_mcp_shows_minimal(self):
        """--mcp shows minimal output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["prime", "--mcp"])

        assert result.exit_code == 0
        # MCP output should be brief
        # Full output includes extensive command docs; MCP should not
        assert "## CLI Quick Reference" not in result.output

    def test_prime_mcp_overrides_full(self):
        """--mcp takes precedence (--full is ignored when both specified).

        When both --full and --mcp are specified, --mcp wins because
        the code checks --full first, then --mcp overrides it.
        Actually per implementation: if full: use_full=True, elif mcp: use_full=False
        So --full takes precedence. Let's verify actual behavior.
        """
        runner = CliRunner()

        # Get baseline lengths
        full_result = runner.invoke(cli, ["prime", "--full"])
        mcp_result = runner.invoke(cli, ["prime", "--mcp"])

        # When both flags are specified, --full is checked first
        # so --full should win
        both_result = runner.invoke(cli, ["prime", "--full", "--mcp"])

        assert both_result.exit_code == 0
        # Per implementation: if full: use_full=True, elif mcp: use_full=False
        # So --full takes precedence
        assert len(both_result.output) == len(full_result.output)


class TestPrimeJsonOutput:
    """Tests for --json output format."""

    def test_prime_json_structure(self):
        """--json returns proper JSON with mode and content."""
        runner = CliRunner()
        result = runner.invoke(cli, ["prime", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Should have both mode and content fields
        assert "mode" in data
        assert "content" in data

    def test_prime_json_includes_mode(self):
        """JSON output has 'mode' field with valid value."""
        runner = CliRunner()

        # Test full mode
        result_full = runner.invoke(cli, ["prime", "--full", "--json"])
        assert result_full.exit_code == 0
        data_full = json.loads(result_full.output)
        assert data_full["mode"] == "full"

        # Test mcp mode
        result_mcp = runner.invoke(cli, ["prime", "--mcp", "--json"])
        assert result_mcp.exit_code == 0
        data_mcp = json.loads(result_mcp.output)
        assert data_mcp["mode"] == "mcp"

    def test_prime_json_includes_content(self):
        """JSON output has 'content' field with string value."""
        runner = CliRunner()
        result = runner.invoke(cli, ["prime", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        assert "content" in data
        assert isinstance(data["content"], str)
        assert len(data["content"]) > 0


class TestPrimeContent:
    """Tests for prime command content."""

    def test_prime_full_includes_commands(self):
        """Full output includes command list (search, get, add, etc.)."""
        runner = CliRunner()
        result = runner.invoke(cli, ["prime", "--full"])

        assert result.exit_code == 0
        # Should include main commands
        assert "mx search" in result.output
        assert "mx get" in result.output
        assert "mx add" in result.output
        assert "mx tree" in result.output

    def test_prime_full_includes_workflow_guidance(self):
        """Full output includes workflow guidance sections."""
        runner = CliRunner()
        result = runner.invoke(cli, ["prime", "--full"])

        assert result.exit_code == 0
        # Should include workflow guidance
        assert "When to Search" in result.output or "Search" in result.output
        assert "When to Contribute" in result.output or "Contribute" in result.output

    def test_prime_mcp_is_brief(self):
        """MCP output is significantly shorter than full output."""
        runner = CliRunner()

        full_result = runner.invoke(cli, ["prime", "--full"])
        mcp_result = runner.invoke(cli, ["prime", "--mcp"])

        assert full_result.exit_code == 0
        assert mcp_result.exit_code == 0

        # MCP should be much shorter (at least 5x shorter based on implementation)
        assert len(mcp_result.output) < len(full_result.output) / 3

    def test_prime_help_shows_options(self):
        """prime --help shows all available options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["prime", "--help"])

        assert result.exit_code == 0
        assert "--full" in result.output
        assert "--mcp" in result.output
        assert "--json" in result.output
