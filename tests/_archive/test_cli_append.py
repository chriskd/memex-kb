"""Tests for mx append CLI command."""

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
    for category in ("development", "architecture"):
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
    content: str = "Original content.",
):
    """Helper to create a KB entry with frontmatter."""
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    text = f"""---
title: {title}
tags:
{tags_yaml}
created: {datetime.now(timezone.utc).isoformat()}
---

{content}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class TestAppendExisting:
    """Tests for appending to existing entries."""

    def test_append_to_existing_entry_by_title(self, kb_root, index_root):
        """Finds and appends to an existing entry by title."""
        entry_path = kb_root / "development" / "my-entry.md"
        _create_entry(entry_path, "My Entry", ["python"], "Original content.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["append", "My Entry", "--content", "Appended content."],
        )

        assert result.exit_code == 0
        assert "Appended to:" in result.output

        # Verify content was appended
        updated = entry_path.read_text()
        assert "Original content." in updated
        assert "Appended content." in updated

    def test_append_case_insensitive_title_match(self, kb_root, index_root):
        """Matches titles case-insensitively."""
        entry_path = kb_root / "development" / "my-entry.md"
        _create_entry(entry_path, "My Entry", ["python"], "Original content.")

        runner = CliRunner()

        # Try lowercase
        result = runner.invoke(
            cli,
            ["append", "my entry", "--content", "Appended via lowercase."],
        )
        assert result.exit_code == 0
        assert "Appended to:" in result.output

        # Try uppercase
        result = runner.invoke(
            cli,
            ["append", "MY ENTRY", "--content", "Appended via uppercase."],
        )
        assert result.exit_code == 0
        assert "Appended to:" in result.output

        updated = entry_path.read_text()
        assert "Appended via lowercase." in updated
        assert "Appended via uppercase." in updated

    def test_append_preserves_existing_content(self, kb_root, index_root):
        """Appending does not overwrite existing content."""
        entry_path = kb_root / "development" / "my-entry.md"
        original = "This is the original content that should be preserved."
        _create_entry(entry_path, "My Entry", ["python"], original)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["append", "My Entry", "--content", "New appended section."],
        )

        assert result.exit_code == 0

        updated = entry_path.read_text()
        assert original in updated
        assert "New appended section." in updated

    def test_append_adds_separator(self, kb_root, index_root):
        """Appending adds proper separator between old and new content."""
        entry_path = kb_root / "development" / "my-entry.md"
        _create_entry(entry_path, "My Entry", ["python"], "Original content.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["append", "My Entry", "--content", "Appended content."],
        )

        assert result.exit_code == 0

        updated = entry_path.read_text()
        # Should have double newline separator as per core.py implementation
        assert "Original content.\n\nAppended content." in updated

    def test_append_from_file(self, kb_root, index_root, tmp_path):
        """Reads append content from a file."""
        entry_path = kb_root / "development" / "my-entry.md"
        _create_entry(entry_path, "My Entry", ["python"], "Original content.")

        # Create file with content to append
        content_file = tmp_path / "append_content.txt"
        content_file.write_text("Content from file.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["append", "My Entry", "--file", str(content_file)],
        )

        assert result.exit_code == 0

        updated = entry_path.read_text()
        assert "Original content." in updated
        assert "Content from file." in updated


class TestAppendCreateNew:
    """Tests for creating new entries when not found."""

    def test_append_creates_new_when_not_found(self, kb_root, index_root):
        """Creates a new entry if no matching title exists."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "append",
                "Brand New Entry",
                "--content",
                "Initial content.",
                "--tags",
                "python,testing",
                "--category",
                "development",
            ],
        )

        assert result.exit_code == 0
        assert "Created:" in result.output

        # Find the created file
        created_files = list(kb_root.rglob("*.md"))
        assert len(created_files) == 1
        content = created_files[0].read_text()
        assert "Brand New Entry" in content
        assert "Initial content." in content

    def test_append_requires_tags_for_new(self, kb_root, index_root):
        """Errors if creating new entry without tags."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "append",
                "New Entry Without Tags",
                "--content",
                "Some content.",
            ],
        )

        assert result.exit_code == 1
        assert "Tags are required" in result.output or "tags" in result.output.lower()

    def test_append_uses_category_for_new(self, kb_root, index_root):
        """Puts new entry in specified category."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "append",
                "New Arch Entry",
                "--content",
                "Architecture content.",
                "--tags",
                "design",
                "--category",
                "architecture",
            ],
        )

        assert result.exit_code == 0
        assert "Created:" in result.output

        # Verify it was created in architecture category
        arch_files = list((kb_root / "architecture").rglob("*.md"))
        assert len(arch_files) == 1
        assert "New Arch Entry" in arch_files[0].read_text()

    def test_append_uses_directory_for_new(self, kb_root, index_root):
        """Puts new entry in specified directory."""
        # Create a custom directory
        custom_dir = kb_root / "custom-dir"
        custom_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "append",
                "Custom Dir Entry",
                "--content",
                "Custom directory content.",
                "--tags",
                "custom",
                "--directory",
                "custom-dir",
            ],
        )

        assert result.exit_code == 0
        assert "Created:" in result.output

        # Verify it was created in the custom directory
        custom_files = list(custom_dir.rglob("*.md"))
        assert len(custom_files) == 1
        assert "Custom Dir Entry" in custom_files[0].read_text()


class TestAppendNoCreate:
    """Tests for --no-create flag behavior."""

    def test_append_no_create_fails_when_not_found(self, kb_root, index_root):
        """--no-create errors when entry not found."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "append",
                "Nonexistent Entry",
                "--content",
                "Some content.",
                "--no-create",
            ],
        )

        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_append_no_create_succeeds_when_found(self, kb_root, index_root):
        """--no-create succeeds when entry exists."""
        entry_path = kb_root / "development" / "existing.md"
        _create_entry(entry_path, "Existing Entry", ["python"], "Original content.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "append",
                "Existing Entry",
                "--content",
                "Appended with no-create.",
                "--no-create",
            ],
        )

        assert result.exit_code == 0
        assert "Appended to:" in result.output

        updated = entry_path.read_text()
        assert "Appended with no-create." in updated


class TestAppendJsonOutput:
    """Tests for JSON output format."""

    def test_append_json_shows_action_appended(self, kb_root, index_root):
        """JSON output shows action: appended for existing entries."""
        entry_path = kb_root / "development" / "my-entry.md"
        _create_entry(entry_path, "My Entry", ["python"], "Original content.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["append", "My Entry", "--content", "Appended content.", "--json"],
        )

        assert result.exit_code == 0
        import json

        data = json.loads(result.output)
        assert data["action"] == "appended"

    def test_append_json_shows_action_created(self, kb_root, index_root):
        """JSON output shows action: created for new entries."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "append",
                "New Entry",
                "--content",
                "Initial content.",
                "--tags",
                "python",
                "--category",
                "development",
                "--json",
            ],
        )

        assert result.exit_code == 0
        import json

        data = json.loads(result.output)
        assert data["action"] == "created"

    def test_append_json_returns_path(self, kb_root, index_root):
        """JSON output includes path key."""
        entry_path = kb_root / "development" / "my-entry.md"
        _create_entry(entry_path, "My Entry", ["python"], "Original content.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["append", "My Entry", "--content", "Appended content.", "--json"],
        )

        assert result.exit_code == 0
        import json

        data = json.loads(result.output)
        assert "path" in data
        assert data["path"].endswith(".md")


class TestAppendStdin:
    """Tests for stdin input."""

    def test_append_reads_from_stdin(self, kb_root, index_root):
        """Stdin content works for appending."""
        entry_path = kb_root / "development" / "my-entry.md"
        _create_entry(entry_path, "My Entry", ["python"], "Original content.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["append", "My Entry", "--stdin"],
            input="Content from stdin.",
        )

        assert result.exit_code == 0

        updated = entry_path.read_text()
        assert "Original content." in updated
        assert "Content from stdin." in updated

    def test_append_stdin_multiline(self, kb_root, index_root):
        """Multi-line stdin content works."""
        entry_path = kb_root / "development" / "my-entry.md"
        _create_entry(entry_path, "My Entry", ["python"], "Original content.")

        multiline_content = """Line 1
Line 2
Line 3

## Section Header

More content here."""

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["append", "My Entry", "--stdin"],
            input=multiline_content,
        )

        assert result.exit_code == 0

        updated = entry_path.read_text()
        assert "Line 1" in updated
        assert "Line 2" in updated
        assert "Line 3" in updated
        assert "## Section Header" in updated
        assert "More content here." in updated


class TestAppendMutualExclusivity:
    """Tests for content source mutual exclusivity."""

    def test_append_content_and_file_mutual_exclusivity(self, kb_root, index_root, tmp_path):
        """--content and --file are mutually exclusive."""
        entry_path = kb_root / "development" / "my-entry.md"
        _create_entry(entry_path, "My Entry", ["python"], "Original.")

        content_file = tmp_path / "content.md"
        content_file.write_text("File content")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "append",
                "My Entry",
                "--content=Inline content",
                f"--file={content_file}",
            ],
        )

        assert result.exit_code == 1
        assert "only one of" in result.output.lower()

    def test_append_content_and_stdin_mutual_exclusivity(self, kb_root, index_root):
        """--content and --stdin are mutually exclusive."""
        entry_path = kb_root / "development" / "my-entry.md"
        _create_entry(entry_path, "My Entry", ["python"], "Original.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "append",
                "My Entry",
                "--content=Inline content",
                "--stdin",
            ],
            input="Stdin content",
        )

        assert result.exit_code == 1
        assert "only one of" in result.output.lower()

    def test_append_file_and_stdin_mutual_exclusivity(self, kb_root, index_root, tmp_path):
        """--file and --stdin are mutually exclusive."""
        entry_path = kb_root / "development" / "my-entry.md"
        _create_entry(entry_path, "My Entry", ["python"], "Original.")

        content_file = tmp_path / "content.md"
        content_file.write_text("File content")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "append",
                "My Entry",
                f"--file={content_file}",
                "--stdin",
            ],
            input="Stdin content",
        )

        assert result.exit_code == 1
        assert "only one of" in result.output.lower()

    def test_append_all_three_content_sources_fails(self, kb_root, index_root, tmp_path):
        """Providing --content, --file, and --stdin all together fails."""
        entry_path = kb_root / "development" / "my-entry.md"
        _create_entry(entry_path, "My Entry", ["python"], "Original.")

        content_file = tmp_path / "content.md"
        content_file.write_text("File content")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "append",
                "My Entry",
                "--content=Inline content",
                f"--file={content_file}",
                "--stdin",
            ],
            input="Stdin content",
        )

        assert result.exit_code == 1
        assert "only one of" in result.output.lower()


class TestAppendEdgeCases:
    """Tests for edge cases and special content."""

    def test_append_unicode_content(self, kb_root, index_root):
        """Handles unicode content correctly."""
        entry_path = kb_root / "development" / "unicode-entry.md"
        _create_entry(entry_path, "Unicode Entry", ["i18n"], "Original: Hello World")

        unicode_content = "Appended: 你好世界 Привет мир مرحبا بالعالم 🌍"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["append", "Unicode Entry", "--content", unicode_content],
        )

        assert result.exit_code == 0

        updated = entry_path.read_text()
        assert "Original: Hello World" in updated
        assert "你好世界" in updated
        assert "Привет мир" in updated
        assert "مرحبا بالعالم" in updated
        assert "🌍" in updated

    def test_append_preserves_frontmatter(self, kb_root, index_root):
        """Appending does not corrupt YAML frontmatter."""
        entry_path = kb_root / "development" / "my-entry.md"
        _create_entry(entry_path, "My Entry", ["python", "testing"], "Original content.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["append", "My Entry", "--content", "Appended content."],
        )

        assert result.exit_code == 0

        updated = entry_path.read_text()

        # Verify frontmatter structure is intact
        assert updated.startswith("---\n")
        assert "\n---\n" in updated

        # Extract and verify frontmatter is valid YAML
        parts = updated.split("---\n", 2)
        assert len(parts) >= 3  # Empty before first ---, frontmatter, content

        import yaml

        frontmatter = yaml.safe_load(parts[1])
        assert frontmatter["title"] == "My Entry"
        assert "python" in frontmatter["tags"]
        assert "testing" in frontmatter["tags"]
