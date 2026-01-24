"""Tests for mx info and mx config CLI commands."""

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
    for category in ("development", "architecture", "devops"):
        (root / category).mkdir()
    monkeypatch.setenv("MEMEX_USER_KB_ROOT", str(root))
    return root


@pytest.fixture
def index_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary index root."""
    root = tmp_path / ".indices"
    root.mkdir()
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(root))
    return root


class TestInfoBasic:
    """Test basic mx info functionality."""

    def test_info_shows_kb_root_path(self, kb_root, index_root):
        """Output contains KB Root path."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        assert "Primary KB:" in result.output
        assert str(kb_root) in result.output

    def test_info_shows_index_root_path(self, kb_root, index_root):
        """Output contains Index Root path."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        assert "Index Root:" in result.output
        assert str(index_root) in result.output

    def test_info_shows_entry_count(self, kb_root, index_root):
        """Output contains entry count."""
        # Create some entries
        (kb_root / "development" / "entry1.md").write_text("# Entry 1\n\nContent")
        (kb_root / "development" / "entry2.md").write_text("# Entry 2\n\nContent")
        (kb_root / "architecture" / "entry3.md").write_text("# Entry 3\n\nContent")

        runner = CliRunner()
        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        assert "Total:" in result.output
        assert "3" in result.output

    def test_info_shows_categories_list(self, kb_root, index_root):
        """Output contains categories list."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        assert "Categories:" in result.output
        assert "development" in result.output
        assert "architecture" in result.output
        assert "devops" in result.output

    def test_config_is_alias_for_info(self, kb_root, index_root):
        """mx config produces same output as mx info."""
        runner = CliRunner()

        info_result = runner.invoke(cli, ["info"])
        config_result = runner.invoke(cli, ["config"])

        assert info_result.exit_code == 0
        assert config_result.exit_code == 0
        assert info_result.output == config_result.output


class TestInfoOutputFormat:
    """Test mx info output format."""

    def test_output_contains_expected_sections(self, kb_root, index_root):
        """Output has header and all expected sections."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        assert "Memex Info" in result.output
        assert "=" * 40 in result.output
        assert "Primary KB:" in result.output
        assert "Index Root:" in result.output
        assert "Total:" in result.output
        assert "Categories:" in result.output

    def test_json_flag_returns_proper_json_structure(self, kb_root, index_root):
        """--json flag produces valid JSON output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--json"])

        assert result.exit_code == 0

        # Should be valid JSON
        data = json.loads(result.output)
        assert isinstance(data, dict)

    def test_json_contains_all_expected_fields(self, kb_root, index_root):
        """JSON output has primary_kb, index_root, categories, total_entries."""
        # Add some entries for non-zero count
        (kb_root / "development" / "test.md").write_text("# Test\n\nContent")

        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--json"])

        assert result.exit_code == 0

        data = json.loads(result.output)
        assert "primary_kb" in data
        assert "index_root" in data
        assert "categories" in data
        assert "total_entries" in data

        # Verify types
        assert isinstance(data["primary_kb"], str)
        assert isinstance(data["index_root"], str)
        assert isinstance(data["categories"], list)
        assert isinstance(data["total_entries"], int)

        # Verify values
        assert data["primary_kb"] == str(kb_root)
        assert data["index_root"] == str(index_root)
        assert data["total_entries"] == 1
        assert "development" in data["categories"]


class TestInfoEdgeCases:
    """Test mx info edge cases."""

    def test_with_empty_kb(self, kb_root, index_root):
        """Entry count is 0 when KB has no entries."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        assert "Total:      0 entries" in result.output

    def test_with_empty_kb_json(self, kb_root, index_root):
        """JSON entry_count is 0 when KB has no entries."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--json"])

        assert result.exit_code == 0

        data = json.loads(result.output)
        assert data["total_entries"] == 0

    def test_with_no_categories(self, tmp_path, monkeypatch):
        """Output shows (none) when KB has no category directories."""
        # Create empty KB root with no subdirectories
        kb_root = tmp_path / "empty_kb"
        kb_root.mkdir()
        monkeypatch.setenv("MEMEX_USER_KB_ROOT", str(kb_root))

        index_root = tmp_path / ".indices"
        index_root.mkdir()
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        # Reset searcher state
        monkeypatch.setattr(core, "_searcher", None)
        monkeypatch.setattr(core, "_searcher_ready", False)

        runner = CliRunner()
        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        # No categories means the line is omitted
        assert "Categories:" not in result.output

    def test_with_no_categories_json(self, tmp_path, monkeypatch):
        """JSON categories is empty list when KB has no directories."""
        kb_root = tmp_path / "empty_kb"
        kb_root.mkdir()
        monkeypatch.setenv("MEMEX_USER_KB_ROOT", str(kb_root))

        index_root = tmp_path / ".indices"
        index_root.mkdir()
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        monkeypatch.setattr(core, "_searcher", None)
        monkeypatch.setattr(core, "_searcher_ready", False)

        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--json"])

        assert result.exit_code == 0

        data = json.loads(result.output)
        assert data["categories"] == []

    def test_index_root_different_from_kb_root(self, tmp_path, monkeypatch):
        """Output shows correct paths when index and KB roots differ."""
        kb_root = tmp_path / "my_kb"
        kb_root.mkdir()
        (kb_root / "notes").mkdir()

        index_root = tmp_path / "separate_indices"
        index_root.mkdir()

        monkeypatch.setenv("MEMEX_USER_KB_ROOT", str(kb_root))
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))
        monkeypatch.setattr(core, "_searcher", None)
        monkeypatch.setattr(core, "_searcher_ready", False)

        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--json"])

        assert result.exit_code == 0

        data = json.loads(result.output)
        assert data["primary_kb"] == str(kb_root)
        assert data["index_root"] == str(index_root)
        assert data["primary_kb"] != data["index_root"]

    def test_excludes_hidden_and_special_files(self, kb_root, index_root):
        """Entry count excludes hidden (.) and special (_) files."""
        # Create regular entries
        (kb_root / "development" / "regular.md").write_text("# Regular\n\nContent")

        # Create hidden file (should be excluded)
        (kb_root / "development" / ".hidden.md").write_text("# Hidden\n\nContent")

        # Create special file (should be excluded)
        (kb_root / "development" / "_template.md").write_text("# Template\n\nContent")

        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--json"])

        assert result.exit_code == 0

        data = json.loads(result.output)
        # Only the regular entry should be counted
        assert data["total_entries"] == 1


class TestInfoErrors:
    """Test mx info error cases."""

    def test_error_when_no_kb_root_configured(self, tmp_path, monkeypatch):
        """Exit code 1 with error when MEMEX_USER_KB_ROOT not set."""
        # Unset MEMEX_USER_KB_ROOT
        monkeypatch.delenv("MEMEX_USER_KB_ROOT", raising=False)

        # Set index root to avoid that error
        index_root = tmp_path / ".indices"
        index_root.mkdir()
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        monkeypatch.setattr(core, "_searcher", None)
        monkeypatch.setattr(core, "_searcher_ready", False)

        runner = CliRunner()
        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 1
        assert "MEMEX_USER_KB_ROOT" in result.output or "Error" in result.output

    def test_success_when_no_index_root_configured(self, tmp_path, monkeypatch):
        """Uses {kb_root}/.indices as fallback when MEMEX_INDEX_ROOT not set."""
        # Set KB root
        kb_root = tmp_path / "kb"
        kb_root.mkdir()
        monkeypatch.setenv("MEMEX_USER_KB_ROOT", str(kb_root))

        # Unset MEMEX_INDEX_ROOT - should fallback to {kb_root}/.indices
        monkeypatch.delenv("MEMEX_INDEX_ROOT", raising=False)

        monkeypatch.setattr(core, "_searcher", None)
        monkeypatch.setattr(core, "_searcher_ready", False)

        runner = CliRunner()
        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        assert str(kb_root / ".indices") in result.output


class TestConfigAlias:
    """Test mx config alias behavior."""

    def test_config_json_flag_works(self, kb_root, index_root):
        """mx config --json produces same JSON as mx info --json."""
        runner = CliRunner()

        info_result = runner.invoke(cli, ["info", "--json"])
        config_result = runner.invoke(cli, ["config", "--json"])

        assert info_result.exit_code == 0
        assert config_result.exit_code == 0
        assert info_result.output == config_result.output

    def test_config_error_handling_matches_info(self, tmp_path, monkeypatch):
        """mx config handles errors same way as mx info."""
        monkeypatch.delenv("MEMEX_USER_KB_ROOT", raising=False)
        monkeypatch.delenv("MEMEX_INDEX_ROOT", raising=False)

        monkeypatch.setattr(core, "_searcher", None)
        monkeypatch.setattr(core, "_searcher_ready", False)

        runner = CliRunner()

        info_result = runner.invoke(cli, ["info"])
        config_result = runner.invoke(cli, ["config"])

        assert info_result.exit_code == config_result.exit_code
        # Both should error with same type of message
        assert info_result.exit_code == 1
