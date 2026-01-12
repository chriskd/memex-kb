"""Tests for mx context subcommands (show, init, validate)."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from memex import core
from memex.cli import cli
from memex.context import CONTEXT_FILENAME, clear_context_cache


@pytest.fixture(autouse=True)
def reset_searcher_state(monkeypatch):
    """Ensure cached searcher state does not leak across tests."""
    monkeypatch.setattr(core, "_searcher", None)
    monkeypatch.setattr(core, "_searcher_ready", False)


@pytest.fixture(autouse=True)
def reset_context_cache():
    """Clear context cache before each test."""
    clear_context_cache()
    yield
    clear_context_cache()


@pytest.fixture
def kb_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary KB root with standard directories."""
    root = tmp_path / "kb"
    root.mkdir()
    (root / "development").mkdir()
    (root / "projects" / "myproject").mkdir(parents=True)
    monkeypatch.setenv("MEMEX_KB_ROOT", str(root))
    return root


@pytest.fixture
def index_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary index root."""
    root = tmp_path / ".indices"
    root.mkdir()
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(root))
    return root


@pytest.fixture
def project_dir(tmp_path) -> Path:
    """Create a project directory to test context init."""
    proj = tmp_path / "my-awesome-project"
    proj.mkdir()
    return proj


class TestContextShow:
    """Tests for mx context show subcommand."""

    def test_context_show_no_file(self, tmp_path, monkeypatch):
        """Handles missing .kbcontext gracefully."""
        # Change to a directory without .kbcontext
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["context", "show"])

        assert result.exit_code == 0
        assert "No .kbcontext file found" in result.output
        assert "mx init" in result.output  # Suggests using mx init

    def test_context_show_displays_primary(self, tmp_path, monkeypatch):
        """Shows primary directory from .kbcontext."""
        # Create .kbcontext with primary
        context_file = tmp_path / CONTEXT_FILENAME
        context_file.write_text("primary: projects/myproject\n")
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["context", "show"])

        assert result.exit_code == 0
        assert "Primary:" in result.output
        assert "projects/myproject" in result.output

    def test_context_show_displays_paths(self, tmp_path, monkeypatch):
        """Shows all paths from .kbcontext."""
        # Create .kbcontext with multiple paths
        context_content = """primary: projects/myproject
paths:
  - development
  - architecture
  - tooling/*
"""
        context_file = tmp_path / CONTEXT_FILENAME
        context_file.write_text(context_content)
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["context", "show"])

        assert result.exit_code == 0
        assert "Paths:" in result.output
        assert "development" in result.output
        assert "architecture" in result.output
        assert "tooling/*" in result.output

    def test_context_show_displays_project(self, tmp_path, monkeypatch):
        """Shows project name from .kbcontext."""
        # Create .kbcontext with project name
        context_content = """primary: projects/myproject
project: myproject
default_tags:
  - myproject
  - python
"""
        context_file = tmp_path / CONTEXT_FILENAME
        context_file.write_text(context_content)
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["context", "show"])

        assert result.exit_code == 0
        assert "Project:" in result.output
        assert "myproject" in result.output
        assert "Default tags:" in result.output


class TestContextValidate:
    """Tests for mx context validate subcommand."""

    def test_context_validate_valid_paths(self, kb_root, tmp_path, monkeypatch):
        """Valid config passes validation without warnings."""
        # Create .kbcontext with paths that exist
        context_content = """primary: projects/myproject
paths:
  - development
  - projects/myproject
"""
        context_file = tmp_path / CONTEXT_FILENAME
        context_file.write_text(context_content)
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["context", "validate"])

        assert result.exit_code == 0
        assert "All paths are valid" in result.output

    def test_context_validate_missing_primary_warning(self, kb_root, tmp_path, monkeypatch):
        """Warns if primary directory does not exist."""
        # Create .kbcontext with non-existent primary
        context_content = """primary: projects/nonexistent
paths:
  - development
"""
        context_file = tmp_path / CONTEXT_FILENAME
        context_file.write_text(context_content)
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["context", "validate"])

        assert result.exit_code == 0
        assert "Warnings:" in result.output
        assert "projects/nonexistent" in result.output
        assert "does not exist" in result.output

    def test_context_validate_missing_path_warning(self, kb_root, tmp_path, monkeypatch):
        """Warns if a non-glob path does not exist."""
        # Create .kbcontext with non-existent path
        context_content = """primary: development
paths:
  - development
  - missing/path
"""
        context_file = tmp_path / CONTEXT_FILENAME
        context_file.write_text(context_content)
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["context", "validate"])

        assert result.exit_code == 0
        assert "Warnings:" in result.output
        assert "missing/path" in result.output
        assert "does not exist" in result.output

    def test_context_validate_json_structure(self, kb_root, tmp_path, monkeypatch):
        """JSON output has proper structure."""
        # Create .kbcontext with valid and invalid paths
        context_content = """primary: projects/myproject
paths:
  - development
  - missing/path
"""
        context_file = tmp_path / CONTEXT_FILENAME
        context_file.write_text(context_content)
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["context", "validate", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Verify structure
        assert "valid" in data
        assert data["valid"] is True
        assert "source_file" in data
        assert "warnings" in data
        assert isinstance(data["warnings"], list)


class TestContextInit:
    """Tests for mx context init subcommand (DEPRECATED)."""

    def test_context_init_shows_deprecation_warning(self, project_dir, monkeypatch):
        """Shows deprecation warning when using mx context init."""
        monkeypatch.chdir(project_dir)

        runner = CliRunner()
        result = runner.invoke(cli, ["context", "init"])

        assert result.exit_code == 0
        # Check deprecation warning is shown
        assert "deprecated" in result.output.lower()
        assert "mx init" in result.output

    def test_context_init_hidden_from_help(self):
        """'context init' is hidden from Commands section in help output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["context", "--help"])

        assert result.exit_code == 0
        # Split output into sections and check Commands section
        lines = result.output.split("\n")
        in_commands = False
        for line in lines:
            if line.strip() == "Commands:":
                in_commands = True
            elif in_commands:
                # 'init' should NOT be listed as a command
                if line.strip().startswith("init"):
                    assert False, "init command should be hidden from help"
        # show and validate should be in Commands
        assert "show" in result.output
        assert "validate" in result.output

    def test_context_init_auto_detects_project(self, project_dir, monkeypatch):
        """Auto-detects project name from directory name (still works but deprecated)."""
        monkeypatch.chdir(project_dir)

        runner = CliRunner()
        result = runner.invoke(cli, ["context", "init"])

        assert result.exit_code == 0
        assert "Created .kbcontext" in result.output

        # Verify file contents
        context_path = project_dir / CONTEXT_FILENAME
        assert context_path.exists()
        content = context_path.read_text()
        assert "my-awesome-project" in content
        assert "primary:" in content

    def test_context_init_custom_project(self, project_dir, monkeypatch):
        """--project sets project name instead of auto-detection."""
        monkeypatch.chdir(project_dir)

        runner = CliRunner()
        result = runner.invoke(cli, ["context", "init", "--project=customname"])

        assert result.exit_code == 0
        assert "Created .kbcontext" in result.output
        assert "customname" in result.output

        # Verify file contents
        context_path = project_dir / CONTEXT_FILENAME
        content = context_path.read_text()
        assert "customname" in content
        assert "projects/customname" in content

    def test_context_init_custom_directory(self, project_dir, monkeypatch):
        """--directory sets KB directory path."""
        monkeypatch.chdir(project_dir)

        runner = CliRunner()
        result = runner.invoke(
            cli, ["context", "init", "--directory=custom/kb/path"]
        )

        assert result.exit_code == 0
        assert "custom/kb/path" in result.output

        # Verify file contents
        context_path = project_dir / CONTEXT_FILENAME
        content = context_path.read_text()
        assert "primary: custom/kb/path" in content
