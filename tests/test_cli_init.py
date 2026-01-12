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
        """Creates kb/ directory in current directory."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init"])

            assert result.exit_code == 0
            assert "Initialized project KB" in result.output

            kb_path = Path.cwd() / LOCAL_KB_DIR
            assert kb_path.exists()
            assert kb_path.is_dir()

    def test_init_creates_readme(self, runner, tmp_path):
        """Creates README.md in the project KB directory."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init"])

            readme_path = Path.cwd() / LOCAL_KB_DIR / "README.md"
            assert readme_path.exists()

            content = readme_path.read_text()
            assert "Project Knowledge Base" in content
            assert "mx add" in content

    def test_init_creates_config(self, runner, tmp_path):
        """Creates config file in the project KB directory."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init"])

            config_path = Path.cwd() / LOCAL_KB_DIR / LOCAL_KB_CONFIG_FILENAME
            assert config_path.exists()

            content = config_path.read_text()
            assert "Project KB Configuration" in content
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


class TestInitContentQuality:
    """Test the quality of generated files."""

    def test_readme_contains_usage_examples(self, runner, tmp_path):
        """README includes practical usage examples."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init"])

            readme = (Path.cwd() / LOCAL_KB_DIR / "README.md").read_text()

            # Should have code examples
            assert "```bash" in readme
            assert "mx add" in readme
            assert "mx search" in readme

    def test_config_is_valid_yaml_comment(self, runner, tmp_path):
        """Config file has valid YAML structure (all commented)."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init"])

            config = (Path.cwd() / LOCAL_KB_DIR / LOCAL_KB_CONFIG_FILENAME).read_text()

            # Should be commented YAML
            assert "# " in config
            assert "default_tags" in config
            assert "exclude" in config

    def test_config_includes_project_name(self, runner, tmp_path):
        """Config suggests project name as default tag."""
        import os

        # Create a named directory and run init from there
        project_dir = tmp_path / "my-cool-project"
        project_dir.mkdir()

        old_cwd = os.getcwd()
        try:
            os.chdir(project_dir)
            runner.invoke(cli, ["init"])

            config = (project_dir / LOCAL_KB_DIR / LOCAL_KB_CONFIG_FILENAME).read_text()
            assert "my-cool-project" in config
        finally:
            os.chdir(old_cwd)


class TestInitEdgeCases:
    """Test edge cases and error handling."""

    def test_init_with_absolute_path(self, runner, tmp_path):
        """Works with absolute path."""
        target = tmp_path / "absolute" / "path" / "kb"

        result = runner.invoke(cli, ["init", "--path", str(target)])

        assert result.exit_code == 0
        assert target.exists()
        assert (target / "README.md").exists()

    def test_init_preserves_existing_files_on_force(self, runner, tmp_path):
        """Force reinit doesn't delete existing entries."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            kb_path = Path.cwd() / LOCAL_KB_DIR
            kb_path.mkdir()

            # Create an existing entry
            entry = kb_path / "my-entry.md"
            entry.write_text("# My Entry\n\nImportant content")

            runner.invoke(cli, ["init", "--force"])

            # Entry should still exist
            assert entry.exists()
            assert "Important content" in entry.read_text()

    def test_init_help_shows_options(self, runner):
        """Help text documents all options."""
        result = runner.invoke(cli, ["init", "--help"])

        assert result.exit_code == 0
        assert "--path" in result.output
        assert "--force" in result.output
        assert "--json" in result.output
        assert "--user" in result.output
        assert "kb/" in result.output  # Default mentioned


class TestInitUserScope:
    """Test mx init --user for user-scope KB."""

    def test_init_user_creates_user_kb_directory(self, runner, tmp_path, monkeypatch):
        """Creates ~/.memex/kb/ directory with --user flag."""
        # Use tmp_path as fake home directory
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        # Need to reimport to pick up new HOME
        import importlib
        from memex import context
        importlib.reload(context)

        result = runner.invoke(cli, ["init", "--user"])

        assert result.exit_code == 0
        assert "Initialized user KB" in result.output

        user_kb_path = fake_home / ".memex" / "kb"
        assert user_kb_path.exists()
        assert user_kb_path.is_dir()

    def test_init_user_creates_user_readme(self, runner, tmp_path, monkeypatch):
        """User KB README has user-specific content."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        import importlib
        from memex import context
        importlib.reload(context)

        runner.invoke(cli, ["init", "--user"])

        readme = (fake_home / ".memex" / "kb" / "README.md").read_text()
        assert "User Knowledge Base" in readme
        assert "personal" in readme
        assert "~/.memex/kb/" in readme

    def test_init_user_creates_user_config(self, runner, tmp_path, monkeypatch):
        """User KB config has user-specific defaults."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        import importlib
        from memex import context
        importlib.reload(context)

        runner.invoke(cli, ["init", "--user"])

        config = (fake_home / ".memex" / "kb" / LOCAL_KB_CONFIG_FILENAME).read_text()
        assert "User KB Configuration" in config
        assert "personal" in config  # Suggested default tag

    def test_init_user_json_output(self, runner, tmp_path, monkeypatch):
        """User scope is included in JSON output."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        import importlib
        from memex import context
        importlib.reload(context)

        result = runner.invoke(cli, ["init", "--user", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["scope"] == "user"
        assert ".memex/kb" in data["created"]

    def test_init_user_and_path_mutually_exclusive(self, runner):
        """Cannot use --user and --path together."""
        result = runner.invoke(cli, ["init", "--user", "--path", "custom/kb"])

        assert result.exit_code == 1
        assert "mutually exclusive" in result.output

    def test_init_user_and_path_mutually_exclusive_json(self, runner):
        """Mutual exclusion error in JSON format."""
        result = runner.invoke(cli, ["init", "--user", "--path", "custom/kb", "--json"])

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert "mutually exclusive" in data["error"]

    def test_init_user_force_reinitializes(self, runner, tmp_path, monkeypatch):
        """--force works with user scope."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        import importlib
        from memex import context
        importlib.reload(context)

        # Create existing user KB
        user_kb = fake_home / ".memex" / "kb"
        user_kb.mkdir(parents=True)

        result = runner.invoke(cli, ["init", "--user", "--force"])

        assert result.exit_code == 0
        assert "Initialized user KB" in result.output
