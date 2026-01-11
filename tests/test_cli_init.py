"""Tests for mx init CLI command."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from memex.cli import cli
from memex.cli import LOCAL_KB_DIR
from memex.context import LOCAL_KB_CONFIG_FILENAME


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


class TestInitBasic:
    """Test basic mx init functionality."""

    def test_init_creates_default_kb_directory(self, runner, tmp_path):
        """Creates .kb/ directory in current directory."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init"])

            assert result.exit_code == 0
            assert "Initialized local KB" in result.output

            kb_path = Path.cwd() / LOCAL_KB_DIR
            assert kb_path.exists()
            assert kb_path.is_dir()

    def test_init_creates_readme(self, runner, tmp_path):
        """Creates README.md in the local KB directory."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init"])

            readme_path = Path.cwd() / LOCAL_KB_DIR / "README.md"
            assert readme_path.exists()

            content = readme_path.read_text()
            assert "Local Knowledge Base" in content
            assert "mx add" in content

    def test_init_creates_config(self, runner, tmp_path):
        """Creates config file in the local KB directory."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init"])

            config_path = Path.cwd() / LOCAL_KB_DIR / LOCAL_KB_CONFIG_FILENAME
            assert config_path.exists()

            content = config_path.read_text()
            assert "Local KB Configuration" in content
            assert "default_tags" in content


class TestInitCustomPath:
    """Test mx init with custom path."""

    def test_init_custom_path(self, runner, tmp_path):
        """Creates KB at custom path when --path is provided."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init", "--path", "docs/kb"])

            assert result.exit_code == 0
            assert "docs/kb" in result.output

            kb_path = Path.cwd() / "docs" / "kb"
            assert kb_path.exists()
            assert (kb_path / "README.md").exists()
            assert (kb_path / LOCAL_KB_CONFIG_FILENAME).exists()

    def test_init_nested_custom_path(self, runner, tmp_path):
        """Creates nested directories when needed."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init", "--path", "deeply/nested/kb/dir"])

            assert result.exit_code == 0

            kb_path = Path.cwd() / "deeply" / "nested" / "kb" / "dir"
            assert kb_path.exists()


class TestInitExistingKB:
    """Test mx init when KB already exists."""

    def test_init_fails_if_kb_exists(self, runner, tmp_path):
        """Fails if .kb/ already exists without --force."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create existing KB
            kb_path = Path.cwd() / LOCAL_KB_DIR
            kb_path.mkdir()

            result = runner.invoke(cli, ["init"])

            assert result.exit_code == 1
            assert "already exists" in result.output

    def test_init_force_reinitializes(self, runner, tmp_path):
        """Reinitializes with --force."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create existing KB with old content
            kb_path = Path.cwd() / LOCAL_KB_DIR
            kb_path.mkdir()
            old_file = kb_path / "old.md"
            old_file.write_text("old content")

            result = runner.invoke(cli, ["init", "--force"])

            assert result.exit_code == 0
            assert "Initialized" in result.output

            # New files should exist
            assert (kb_path / "README.md").exists()
            assert (kb_path / LOCAL_KB_CONFIG_FILENAME).exists()

            # Old file should still be there (we don't delete)
            assert old_file.exists()


class TestInitJsonOutput:
    """Test mx init JSON output."""

    def test_init_json_output(self, runner, tmp_path):
        """Returns JSON when --json flag is provided."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init", "--json"])

            assert result.exit_code == 0

            data = json.loads(result.output)
            assert "created" in data
            assert "files" in data
            assert "README.md" in data["files"]
            assert LOCAL_KB_CONFIG_FILENAME in data["files"]
            assert "hint" in data

    def test_init_json_error_on_existing(self, runner, tmp_path):
        """Returns JSON error when KB exists."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create existing KB
            kb_path = Path.cwd() / LOCAL_KB_DIR
            kb_path.mkdir()

            result = runner.invoke(cli, ["init", "--json"])

            assert result.exit_code == 1

            data = json.loads(result.output)
            assert "error" in data
            assert "hint" in data
