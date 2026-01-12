"""Tests for mx list CLI command."""

from datetime import datetime, timezone
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
    monkeypatch.setenv("MEMEX_KB_ROOT", str(root))
    return root


@pytest.fixture
def index_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary index root."""
    root = tmp_path / ".indices"
    root.mkdir()
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(root))
    return root


def _create_entry(
    path: Path,
    title: str,
    tags: list[str],
    created: datetime,
):
    """Helper to create a KB entry with frontmatter."""
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    content = f"""---
title: {title}
tags:
{tags_yaml}
created: {created.isoformat()}
---

## Content

Some content here.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class TestListCommandCategoryValidation:
    """Test mx list --category validation."""

    def test_list_invalid_category_shows_helpful_error(self, kb_root, index_root):
        """Invalid category shows error with valid categories listed."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--category=nonexistent"])

        assert result.exit_code == 1
        assert "Error: Invalid category 'nonexistent'" in result.output
        assert "Valid categories:" in result.output
        # Should list actual categories
        assert "architecture" in result.output
        assert "development" in result.output
        assert "devops" in result.output

    def test_list_invalid_category_no_traceback(self, kb_root, index_root):
        """Invalid category does not produce a Python traceback."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--category=nonexistent"])

        assert result.exit_code == 1
        # Should not contain traceback indicators
        assert "Traceback" not in result.output
        assert "ValueError" not in result.output
        assert "File \"" not in result.output

    def test_list_valid_category_works(self, kb_root, index_root):
        """Valid category returns entries from that category."""
        today = datetime.now(timezone.utc)
        _create_entry(
            kb_root / "development" / "entry.md",
            "Dev Entry",
            ["python"],
            created=today,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--category=development"])

        assert result.exit_code == 0
        assert "Dev Entry" in result.output

    def test_list_valid_category_no_entries(self, kb_root, index_root):
        """Valid but empty category shows 'No entries found'."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--category=architecture"])

        assert result.exit_code == 0
        assert "No entries found" in result.output

    def test_list_no_category_lists_all(self, kb_root, index_root):
        """Without category filter, lists all entries."""
        today = datetime.now(timezone.utc)
        _create_entry(
            kb_root / "development" / "dev.md",
            "Dev Entry",
            ["python"],
            created=today,
        )
        _create_entry(
            kb_root / "architecture" / "arch.md",
            "Arch Entry",
            ["design"],
            created=today,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "Dev Entry" in result.output
        assert "Arch Entry" in result.output

    def test_list_empty_kb_shows_no_categories(self, tmp_path, monkeypatch):
        """When KB has no categories, error message says so."""
        # Create an empty KB root
        empty_root = tmp_path / "empty_kb"
        empty_root.mkdir()
        monkeypatch.setenv("MEMEX_KB_ROOT", str(empty_root))

        # Reset searcher to pick up new root
        monkeypatch.setattr(core, "_searcher", None)
        monkeypatch.setattr(core, "_searcher_ready", False)

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--category=nonexistent"])

        assert result.exit_code == 1
        assert "Error: Invalid category 'nonexistent'" in result.output
        assert "No categories exist yet" in result.output


class TestListTagFilter:
    """Test mx list --tag filtering."""

    def test_list_filters_by_tag(self, kb_root, index_root):
        """--tag=python shows only python-tagged entries."""
        today = datetime.now(timezone.utc)
        _create_entry(
            kb_root / "development" / "python_entry.md",
            "Python Entry",
            ["python", "programming"],
            created=today,
        )
        _create_entry(
            kb_root / "development" / "rust_entry.md",
            "Rust Entry",
            ["rust", "programming"],
            created=today,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--tag=python"])

        assert result.exit_code == 0
        assert "Python Entry" in result.output
        assert "Rust Entry" not in result.output

    def test_list_tag_no_matches(self, kb_root, index_root):
        """--tag=nonexistent shows 'No entries found'."""
        today = datetime.now(timezone.utc)
        _create_entry(
            kb_root / "development" / "entry.md",
            "Some Entry",
            ["python"],
            created=today,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--tag=nonexistent"])

        assert result.exit_code == 0
        assert "No entries found" in result.output

    def test_list_tag_case_sensitive(self, kb_root, index_root):
        """Tag matching is case-sensitive."""
        today = datetime.now(timezone.utc)
        _create_entry(
            kb_root / "development" / "entry.md",
            "Python Entry",
            ["Python"],
            created=today,
        )

        runner = CliRunner()
        # Lowercase should not match uppercase tag
        result = runner.invoke(cli, ["list", "--tag=python"])

        assert result.exit_code == 0
        assert "No entries found" in result.output

        # Exact case should match
        result_match = runner.invoke(cli, ["list", "--tag=Python"])
        assert result_match.exit_code == 0
        assert "Python Entry" in result_match.output


class TestListLimit:
    """Test mx list --limit functionality."""

    def test_list_default_limit_20(self, kb_root, index_root):
        """Default shows up to 20 entries."""
        today = datetime.now(timezone.utc)
        # Create 25 entries
        for i in range(25):
            _create_entry(
                kb_root / "development" / f"entry_{i:02d}.md",
                f"Entry {i:02d}",
                ["test"],
                created=today,
            )

        runner = CliRunner()
        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        # Count how many entries appear (each entry has its title in output)
        matches = [line for line in result.output.splitlines() if "Entry" in line]
        assert len(matches) == 20

    def test_list_custom_limit(self, kb_root, index_root):
        """--limit=5 shows only 5 entries."""
        today = datetime.now(timezone.utc)
        for i in range(10):
            _create_entry(
                kb_root / "development" / f"entry_{i:02d}.md",
                f"Entry {i:02d}",
                ["test"],
                created=today,
            )

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--limit=5"])

        assert result.exit_code == 0
        matches = [line for line in result.output.splitlines() if "Entry" in line]
        assert len(matches) == 5

    def test_list_limit_exceeds_entries(self, kb_root, index_root):
        """--limit=100 shows all entries when fewer exist."""
        today = datetime.now(timezone.utc)
        for i in range(3):
            _create_entry(
                kb_root / "development" / f"entry_{i:02d}.md",
                f"Entry {i:02d}",
                ["test"],
                created=today,
            )

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--limit=100"])

        assert result.exit_code == 0
        matches = [line for line in result.output.splitlines() if "Entry" in line]
        assert len(matches) == 3


class TestListJsonOutput:
    """Test mx list --json output format."""

    def test_list_json_structure(self, kb_root, index_root):
        """--json outputs proper JSON array."""
        import json

        today = datetime.now(timezone.utc)
        _create_entry(
            kb_root / "development" / "entry.md",
            "Test Entry",
            ["python"],
            created=today,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_list_json_includes_path(self, kb_root, index_root):
        """Each JSON entry has 'path' field."""
        import json

        today = datetime.now(timezone.utc)
        _create_entry(
            kb_root / "development" / "my_entry.md",
            "Test Entry",
            ["python"],
            created=today,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert "path" in data[0]
        assert "development/my_entry.md" in data[0]["path"]

    def test_list_json_includes_tags(self, kb_root, index_root):
        """Each JSON entry has 'tags' field."""
        import json

        today = datetime.now(timezone.utc)
        _create_entry(
            kb_root / "development" / "entry.md",
            "Test Entry",
            ["python", "testing"],
            created=today,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert "tags" in data[0]
        assert "python" in data[0]["tags"]
        assert "testing" in data[0]["tags"]


class TestListCombined:
    """Test mx list with multiple filters combined."""

    def test_list_tag_and_category(self, kb_root, index_root):
        """--tag combined with --category filters by both."""
        today = datetime.now(timezone.utc)
        _create_entry(
            kb_root / "development" / "python_dev.md",
            "Python Dev",
            ["python"],
            created=today,
        )
        _create_entry(
            kb_root / "development" / "rust_dev.md",
            "Rust Dev",
            ["rust"],
            created=today,
        )
        _create_entry(
            kb_root / "architecture" / "python_arch.md",
            "Python Arch",
            ["python"],
            created=today,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--category=development", "--tag=python"])

        assert result.exit_code == 0
        assert "Python Dev" in result.output
        assert "Rust Dev" not in result.output
        assert "Python Arch" not in result.output

    def test_list_tag_and_limit(self, kb_root, index_root):
        """--tag combined with --limit respects both filters."""
        today = datetime.now(timezone.utc)
        for i in range(10):
            _create_entry(
                kb_root / "development" / f"python_entry_{i:02d}.md",
                f"Python Entry {i:02d}",
                ["python"],
                created=today,
            )
        for i in range(5):
            _create_entry(
                kb_root / "development" / f"rust_entry_{i:02d}.md",
                f"Rust Entry {i:02d}",
                ["rust"],
                created=today,
            )

        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--tag=python", "--limit=3"])

        assert result.exit_code == 0
        matches = [line for line in result.output.splitlines() if "Python Entry" in line]
        assert len(matches) == 3
        # No rust entries should appear
        assert "Rust Entry" not in result.output
