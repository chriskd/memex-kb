"""Tests for mx quick-add CLI command."""

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


class TestQuickAddBasic:
    """Test basic mx quick-add functionality."""

    def test_quick_add_with_content_creates_entry(self, kb_root, index_root):
        """Creates entry with --content flag and auto-generated metadata."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=# My Test Entry\n\nThis is test content about development.",
                "--confirm",
            ],
        )

        assert result.exit_code == 0
        assert "Created:" in result.output

        # Verify file exists
        entry_path = kb_root / "development" / "my-test-entry.md"
        assert entry_path.exists()

        content = entry_path.read_text()
        assert "title: My Test Entry" in content
        assert "This is test content" in content

    def test_quick_add_with_stdin_reads_content(self, kb_root, index_root):
        """Reads content from stdin when using --stdin."""
        runner = CliRunner()
        stdin_content = "# Stdin Entry\n\nLoaded from stdin about architecture."

        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--stdin",
                "--confirm",
            ],
            input=stdin_content,
        )

        assert result.exit_code == 0
        assert "Created:" in result.output

        # Verify content was read from stdin
        entry_path = kb_root / "architecture" / "stdin-entry.md"
        assert entry_path.exists()

        content = entry_path.read_text()
        assert "# Stdin Entry" in content
        assert "Loaded from stdin" in content

    def test_quick_add_with_file_reads_content(self, kb_root, index_root, tmp_path):
        """Reads content from file path when using --file."""
        # Create a content file
        content_file = tmp_path / "content.md"
        content_file.write_text("# From File\n\nContent loaded from file about devops.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                f"--file={content_file}",
                "--confirm",
            ],
        )

        assert result.exit_code == 0
        assert "Created:" in result.output

        # Verify content was read from file
        entry_path = kb_root / "devops" / "from-file.md"
        assert entry_path.exists()

        content = entry_path.read_text()
        assert "# From File" in content
        assert "Content loaded from file" in content


class TestTitleExtraction:
    """Test title extraction from content."""

    def test_title_from_h1_heading(self, kb_root, index_root):
        """Extracts title from first H1 heading."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=# Primary Heading\n\nSome content.\n\n## Secondary",
                "--category=development",
                "--confirm",
            ],
        )

        assert result.exit_code == 0

        entry_path = kb_root / "development" / "primary-heading.md"
        assert entry_path.exists()

        content = entry_path.read_text()
        assert "title: Primary Heading" in content

    def test_title_from_h2_when_no_h1(self, kb_root, index_root):
        """Falls back to H2 heading when no H1 present."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=Some intro text.\n\n## Secondary Heading\n\nMore content.",
                "--category=development",
                "--confirm",
            ],
        )

        assert result.exit_code == 0

        entry_path = kb_root / "development" / "secondary-heading.md"
        assert entry_path.exists()

        content = entry_path.read_text()
        assert "title: Secondary Heading" in content

    def test_title_from_first_line_when_no_headings(self, kb_root, index_root):
        """Uses first non-empty line when no headings present."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=This is the first meaningful line\n\nMore content below.",
                "--category=development",
                "--confirm",
            ],
        )

        assert result.exit_code == 0

        # Should use slugified first line as filename
        files = list((kb_root / "development").glob("*.md"))
        assert len(files) == 1

        content = files[0].read_text()
        assert "This is the first meaningful line" in content


class TestTagSuggestion:
    """Test tag suggestion from content."""

    def test_tags_suggested_from_existing_kb_tags(self, kb_root, index_root):
        """Suggests tags based on matches with existing KB tags."""
        # Create an existing entry with specific tags
        existing_entry = kb_root / "development" / "existing.md"
        existing_entry.write_text(
            "---\ntitle: Existing\ntags:\n  - python\n  - testing\n---\nContent."
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=# Python Testing Guide\n\nA guide about python testing frameworks.",
                "--category=development",
                "--confirm",
            ],
        )

        assert result.exit_code == 0

        # Find the created entry (not the existing one)
        entry_path = kb_root / "development" / "python-testing-guide.md"
        assert entry_path.exists()

        content = entry_path.read_text()
        # Should include matched existing tags
        assert "python" in content or "testing" in content

    def test_uncategorized_tag_when_no_matches(self, kb_root, index_root):
        """Uses 'uncategorized' tag when no existing tags match."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=# Unique Content\n\nCompletely unique topic with no matching tags.",
                "--category=development",
                "--confirm",
            ],
        )

        assert result.exit_code == 0

        entry_path = kb_root / "development" / "unique-content.md"
        assert entry_path.exists()

        content = entry_path.read_text()
        assert "uncategorized" in content


class TestCategoryHandling:
    """Test category handling in quick-add."""

    def test_explicit_category_used(self, kb_root, index_root):
        """Uses --category when provided explicitly."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=# Architecture Doc\n\nContent about systems.",
                "--category=architecture",
                "--confirm",
            ],
        )

        assert result.exit_code == 0
        assert "architecture/" in result.output

        entry_path = kb_root / "architecture" / "architecture-doc.md"
        assert entry_path.exists()

    def test_category_suggested_from_content(self, kb_root, index_root):
        """Suggests category based on content keywords."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=# DevOps Guide\n\nThis is about devops automation and deployment.",
                "--confirm",
            ],
        )

        assert result.exit_code == 0
        assert "devops/" in result.output

        entry_path = kb_root / "devops" / "devops-guide.md"
        assert entry_path.exists()

    def test_first_category_used_when_no_match(self, kb_root, index_root):
        """Falls back to first available category when no content match."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=# Random Topic\n\nGeneric content with no category keywords.",
                "--confirm",
            ],
        )

        assert result.exit_code == 0
        # Should use one of the available categories
        created_files = list(kb_root.rglob("random-topic.md"))
        assert len(created_files) == 1


class TestEdgeCases:
    """Test edge cases in quick-add."""

    def test_empty_content_fails(self, kb_root, index_root):
        """Empty content produces error."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=",
                "--category=development",
                "--confirm",
            ],
        )

        assert result.exit_code != 0
        assert "empty" in result.output.lower() or "content" in result.output.lower()

    def test_whitespace_only_content_fails(self, kb_root, index_root):
        """Whitespace-only content produces error."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=   \n\n   \t   ",
                "--category=development",
                "--confirm",
            ],
        )

        assert result.exit_code != 0
        assert "empty" in result.output.lower()

    def test_unicode_content_handling(self, kb_root, index_root):
        """Handles unicode content correctly."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=# Unicode Test\n\nContent with unicode: \u00e9\u00e0\u00fc\u4e2d\u6587\u65e5\u672c\u8a9e",
                "--category=development",
                "--confirm",
            ],
        )

        assert result.exit_code == 0

        entry_path = kb_root / "development" / "unicode-test.md"
        assert entry_path.exists()

        content = entry_path.read_text(encoding="utf-8")
        assert "\u00e9\u00e0\u00fc\u4e2d\u6587\u65e5\u672c\u8a9e" in content

    def test_title_override(self, kb_root, index_root):
        """--title overrides auto-detected title."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=# Original Title\n\nContent.",
                "--title=Custom Override Title",
                "--category=development",
                "--confirm",
            ],
        )

        assert result.exit_code == 0

        entry_path = kb_root / "development" / "custom-override-title.md"
        assert entry_path.exists()

        content = entry_path.read_text()
        assert "title: Custom Override Title" in content

    def test_tags_override(self, kb_root, index_root):
        """--tags overrides auto-suggested tags."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=# Test Entry\n\nContent.",
                "--tags=custom,manual,tags",
                "--category=development",
                "--confirm",
            ],
        )

        assert result.exit_code == 0

        entry_path = kb_root / "development" / "test-entry.md"
        assert entry_path.exists()

        content = entry_path.read_text()
        assert "custom" in content
        assert "manual" in content
        assert "tags" in content

    def test_multiline_stdin_content(self, kb_root, index_root):
        """Handles multi-line stdin content correctly."""
        multiline_content = """# Multi-line Stdin

This is line 1.
This is line 2.
This is line 3.

## Section

More content here about development.
"""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--stdin",
                "--confirm",
            ],
            input=multiline_content,
        )

        assert result.exit_code == 0

        entry_path = kb_root / "development" / "multi-line-stdin.md"
        assert entry_path.exists()

        content = entry_path.read_text()
        assert "This is line 1." in content
        assert "This is line 2." in content
        assert "## Section" in content


class TestJsonOutput:
    """Test mx quick-add --json output format."""

    def test_json_output_has_required_fields(self, kb_root, index_root):
        """JSON output contains title, tags, category, and content_preview."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=# JSON Test\n\nContent for JSON output testing.",
                "--json",
            ],
        )

        assert result.exit_code == 0

        data = json.loads(result.output)
        assert "title" in data
        assert "tags" in data
        assert "category" in data
        assert "content_preview" in data
        assert "categories_available" in data

    def test_json_output_title_extraction(self, kb_root, index_root):
        """JSON output correctly extracts title."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=# Extracted Title\n\nSome content.",
                "--json",
            ],
        )

        assert result.exit_code == 0

        data = json.loads(result.output)
        assert data["title"] == "Extracted Title"

    def test_json_output_category_suggestion(self, kb_root, index_root):
        """JSON output includes suggested category."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=# Architecture Decision\n\nThis is about system architecture.",
                "--json",
            ],
        )

        assert result.exit_code == 0

        data = json.loads(result.output)
        assert data["category"] == "architecture"

    def test_json_output_content_preview_truncated(self, kb_root, index_root):
        """JSON output truncates long content in preview."""
        long_content = "# Long Content\n\n" + "x" * 500
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                f"--content={long_content}",
                "--json",
            ],
        )

        assert result.exit_code == 0

        data = json.loads(result.output)
        assert len(data["content_preview"]) <= 203  # 200 + "..."
        assert data["content_preview"].endswith("...")

    def test_json_output_lists_available_categories(self, kb_root, index_root):
        """JSON output includes available categories."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=# Test\n\nContent.",
                "--json",
            ],
        )

        assert result.exit_code == 0

        data = json.loads(result.output)
        assert isinstance(data["categories_available"], list)
        assert "development" in data["categories_available"]
        assert "architecture" in data["categories_available"]
        assert "devops" in data["categories_available"]


class TestErrorCases:
    """Test error handling in quick-add."""

    def test_missing_kb_root_error(self, tmp_path, monkeypatch, index_root):
        """Reports error when MEMEX_USER_KB_ROOT not set."""
        monkeypatch.delenv("MEMEX_USER_KB_ROOT", raising=False)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=# Test\n\nContent.",
                "--confirm",
            ],
        )

        assert result.exit_code != 0
        # Error may be in output or in the exception
        error_text = result.output + str(result.exception or "")
        assert "MEMEX_USER_KB_ROOT" in error_text or "not set" in error_text.lower()

    def test_missing_content_source_error(self, kb_root, index_root):
        """Reports error when no content source provided."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--confirm",
            ],
        )

        assert result.exit_code != 0
        assert "content" in result.output.lower() or "file" in result.output.lower() or "stdin" in result.output.lower()

    def test_nonexistent_file_error(self, kb_root, index_root):
        """Reports error when --file path doesn't exist."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--file=/nonexistent/path/file.md",
                "--confirm",
            ],
        )

        assert result.exit_code != 0
        # Click reports file doesn't exist

    def test_invalid_category_creates_directory(self, kb_root, index_root):
        """Invalid category creates new directory."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=# New Category Entry\n\nContent.",
                "--category=newcategory",
                "--confirm",
            ],
        )

        # Should succeed and create new category directory
        assert result.exit_code == 0
        assert (kb_root / "newcategory").is_dir()
        assert (kb_root / "newcategory" / "new-category-entry.md").exists()


class TestInteractiveMode:
    """Test interactive prompting behavior."""

    def test_without_confirm_prompts_user(self, kb_root, index_root):
        """Without --confirm, prompts for confirmation."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=# Interactive Test\n\nContent about development.",
            ],
            input="y\n",  # Confirm the prompt
        )

        assert result.exit_code == 0
        assert "Quick Add Analysis" in result.output
        assert "Create entry with these settings?" in result.output
        assert "Created:" in result.output

    def test_declined_prompt_aborts(self, kb_root, index_root):
        """Declining prompt aborts without creating entry."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=# Aborted Entry\n\nContent.",
            ],
            input="n\n",  # Decline the prompt
        )

        assert "Aborted" in result.output

        # Verify no file was created
        files = list(kb_root.rglob("aborted-entry.md"))
        assert len(files) == 0

    def test_missing_category_prompts_in_interactive(self, kb_root, index_root):
        """Prompts for category when none can be suggested."""
        # Create KB with no matching categories
        for d in kb_root.iterdir():
            if d.is_dir():
                import shutil
                shutil.rmtree(d)

        # Create a single category
        (kb_root / "notes").mkdir()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "quick-add",
                "--content=# Random Entry\n\nNo category keywords here.",
            ],
            input="notes\ny\n",  # Select category then confirm
        )

        assert result.exit_code == 0
        assert "Created:" in result.output
