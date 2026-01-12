"""Tests for --json output flag on CLI commands."""

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
    """Create a temporary KB root."""
    root = tmp_path / "kb"
    root.mkdir()
    monkeypatch.setenv("MEMEX_KB_ROOT", str(root))
    return root


@pytest.fixture
def index_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary index root."""
    root = tmp_path / ".indices"
    root.mkdir()
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(root))
    return root


class TestReindexJsonOutput:
    """Tests for mx reindex --json output."""

    def test_reindex_json_output_structure(self, kb_root, index_root):
        """reindex --json returns proper JSON structure."""
        runner = CliRunner()
        result = runner.invoke(cli, ["reindex", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Verify structure
        assert "kb_files" in data
        assert "whoosh_docs" in data
        assert "chroma_docs" in data

        # All should be integers
        assert isinstance(data["kb_files"], int)
        assert isinstance(data["whoosh_docs"], int)
        assert isinstance(data["chroma_docs"], int)

    def test_reindex_json_no_progress_message(self, kb_root, index_root):
        """reindex --json should not include progress message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["reindex", "--json"])

        assert result.exit_code == 0
        # Should not have the progress message
        assert "Reindexing" not in result.output
        # Should be valid JSON
        json.loads(result.output)

    def test_reindex_without_json_shows_progress(self, kb_root, index_root):
        """reindex without --json shows progress message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["reindex"])

        assert result.exit_code == 0
        assert "Reindexing knowledge base..." in result.output
        assert "Indexed" in result.output


class TestContextInitJsonOutput:
    """Tests for mx context init --json output."""

    def test_context_init_json_output_structure(self, tmp_path):
        """context init --json returns proper JSON structure."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["context", "init", "--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)

            # Verify structure
            assert "created" in data
            assert "primary" in data
            assert "default_tags" in data

            # created should be a path string
            assert isinstance(data["created"], str)
            assert ".kbcontext" in data["created"]

            # primary should be a directory path
            assert isinstance(data["primary"], str)

            # default_tags should be a list
            assert isinstance(data["default_tags"], list)

    def test_context_init_json_with_project(self, tmp_path):
        """context init --json with --project includes project in output."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["context", "init", "--project=myapp", "--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)

            assert data["primary"] == "projects/myapp"
            assert "myapp" in data["default_tags"]

    def test_context_init_json_with_directory(self, tmp_path):
        """context init --json with --directory includes custom directory."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["context", "init", "--directory=custom/path", "--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)

            assert data["primary"] == "custom/path"

    def test_context_init_json_error_existing_file(self, tmp_path):
        """context init --json outputs JSON error when file exists."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create the file first
            Path(".kbcontext").write_text("# existing")

            result = runner.invoke(cli, ["context", "init", "--json"])

            assert result.exit_code == 1
            data = json.loads(result.output)

            assert "error" in data
            assert "already exists" in data["error"]

    def test_context_init_json_force_overwrites(self, tmp_path):
        """context init --json --force overwrites existing file."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create the file first
            Path(".kbcontext").write_text("# existing")

            result = runner.invoke(cli, ["context", "init", "--force", "--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)

            assert "created" in data

    def test_context_init_without_json_shows_text(self, tmp_path):
        """context init without --json shows text output."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["context", "init"])

            assert result.exit_code == 0
            assert "Created .kbcontext" in result.output
            assert "Primary directory:" in result.output
            assert "Default tags:" in result.output
