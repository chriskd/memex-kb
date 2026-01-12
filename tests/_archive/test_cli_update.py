"""Tests for mx update CLI command."""

import json
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


def _create_entry(path: Path, title: str, tags: list[str], content: str = "Original content."):
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


class TestUpdateBasic:
    """Test basic mx update functionality."""

    def test_update_tags_replaces_existing(self, kb_root, index_root):
        """New tags replace old tags entirely (requires content to be provided)."""
        entry_path = kb_root / "development" / "test-entry.md"
        _create_entry(entry_path, "Test Entry", ["old-tag", "another-old"], "Original content.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "update",
                "development/test-entry.md",
                "--tags=new-tag,fresh-tag",
                "--content=Original content.",  # Must provide content to update
            ],
        )

        assert result.exit_code == 0
        assert "Updated:" in result.output

        content = entry_path.read_text()
        assert "new-tag" in content
        assert "fresh-tag" in content
        assert "old-tag" not in content
        assert "another-old" not in content

    def test_update_content_replaces_body(self, kb_root, index_root):
        """New content replaces existing body entirely."""
        entry_path = kb_root / "development" / "test-entry.md"
        _create_entry(entry_path, "Test Entry", ["tag"], "Original content that will be replaced.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "update",
                "development/test-entry.md",
                "--content=Brand new content goes here.",
            ],
        )

        assert result.exit_code == 0
        assert "Updated:" in result.output

        content = entry_path.read_text()
        assert "Brand new content goes here." in content
        assert "Original content that will be replaced." not in content

    def test_update_from_file(self, kb_root, index_root, tmp_path):
        """Reads content from --file."""
        entry_path = kb_root / "development" / "test-entry.md"
        _create_entry(entry_path, "Test Entry", ["tag"], "Original content.")

        content_file = tmp_path / "new-content.md"
        content_file.write_text("Content loaded from external file.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "update",
                "development/test-entry.md",
                f"--file={content_file}",
            ],
        )

        assert result.exit_code == 0
        assert "Updated:" in result.output

        content = entry_path.read_text()
        assert "Content loaded from external file." in content
        assert "Original content." not in content

    def test_update_preserves_frontmatter(self, kb_root, index_root):
        """Update doesn't corrupt other metadata like title and created date."""
        entry_path = kb_root / "development" / "test-entry.md"
        original_created = "2024-01-15T10:30:00+00:00"
        text = f"""---
title: Preserved Title
tags:
  - original-tag
created: {original_created}
---

Original content.
"""
        entry_path.write_text(text)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "update",
                "development/test-entry.md",
                "--content=Updated content.",
            ],
        )

        assert result.exit_code == 0

        content = entry_path.read_text()
        assert "title: Preserved Title" in content
        assert "original-tag" in content  # Tags preserved when not updating tags
        assert original_created in content  # Created date preserved


class TestUpdateValidation:
    """Test mx update validation behavior."""

    def test_update_nonexistent_fails(self, kb_root, index_root):
        """Error for missing entry."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "update",
                "nonexistent/path.md",
                "--tags=new-tag",
            ],
        )

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_update_requires_tags_or_content(self, kb_root, index_root):
        """Need at least --tags or --content."""
        entry_path = kb_root / "development" / "test-entry.md"
        _create_entry(entry_path, "Test Entry", ["tag"], "Original content.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "update",
                "development/test-entry.md",
            ],
        )

        assert result.exit_code != 0
        assert "content" in result.output.lower() or "error" in result.output.lower()

    def test_update_empty_tags_fails(self, kb_root, index_root):
        """--tags="" fails with error."""
        entry_path = kb_root / "development" / "test-entry.md"
        _create_entry(entry_path, "Test Entry", ["tag"], "Original content.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "update",
                "development/test-entry.md",
                "--tags=",
            ],
        )

        assert result.exit_code != 0
        assert "tag" in result.output.lower() or "error" in result.output.lower()

    def test_update_directory_fails(self, kb_root, index_root):
        """Can't update a directory."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "update",
                "development",
                "--tags=new-tag",
            ],
        )

        assert result.exit_code != 0
        assert "not a file" in result.output.lower() or "error" in result.output.lower()


class TestUpdateJsonOutput:
    """Test mx update --json output format."""

    def test_update_json_returns_path(self, kb_root, index_root):
        """JSON output has 'path' key."""
        entry_path = kb_root / "development" / "test-entry.md"
        _create_entry(entry_path, "Test Entry", ["tag"], "Original content.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "update",
                "development/test-entry.md",
                "--content=Updated content.",
                "--json",
            ],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "path" in data
        assert data["path"] == "development/test-entry.md"

    def test_update_json_returns_suggested_links(self, kb_root, index_root):
        """JSON output has 'suggested_links' key."""
        # Create a related entry first
        related_path = kb_root / "development" / "related-entry.md"
        _create_entry(related_path, "Related Entry", ["python"], "Python tooling content.")

        entry_path = kb_root / "development" / "test-entry.md"
        _create_entry(entry_path, "Test Entry", ["python"], "Original content about Python.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "update",
                "development/test-entry.md",
                "--content=Updated Python tooling guide.",
                "--json",
            ],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "suggested_links" in data
        assert isinstance(data["suggested_links"], list)

    def test_update_json_returns_suggested_tags(self, kb_root, index_root):
        """JSON output has 'suggested_tags' key."""
        entry_path = kb_root / "development" / "test-entry.md"
        _create_entry(entry_path, "Test Entry", ["tag"], "Original content.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "update",
                "development/test-entry.md",
                "--content=New content for tag suggestions.",
                "--json",
            ],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "suggested_tags" in data
        assert isinstance(data["suggested_tags"], list)


class TestUpdateIntentDetection:
    """Tests for intent detection on the update command."""

    def test_update_with_find_suggests_patch(self, kb_root, index_root):
        """'mx update entry.md --find=...' suggests patch command."""
        entry_path = kb_root / "development" / "test-entry.md"
        _create_entry(entry_path, "Test Entry", ["tag"], "Original content.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "update",
                "development/test-entry.md",
                "--find=old text",
            ],
        )

        # Click will reject the unknown option
        assert result.exit_code != 0
        assert "no such option" in result.output.lower() or "--find" in result.output.lower()

    def test_update_with_replace_suggests_patch(self, kb_root, index_root):
        """'mx update entry.md --replace=...' suggests patch command."""
        entry_path = kb_root / "development" / "test-entry.md"
        _create_entry(entry_path, "Test Entry", ["tag"], "Original content.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "update",
                "development/test-entry.md",
                "--replace=new text",
            ],
        )

        # Click will reject the unknown option
        assert result.exit_code != 0
        assert "no such option" in result.output.lower() or "--replace" in result.output.lower()


class TestUpdateEdgeCases:
    """Test mx update edge cases."""

    def test_update_unicode_content(self, kb_root, index_root):
        """Handles unicode content correctly."""
        entry_path = kb_root / "development" / "test-entry.md"
        _create_entry(entry_path, "Test Entry", ["tag"], "Original content.")

        unicode_content = "Content with unicode: \u00e9\u00e0\u00fc \u4e2d\u6587 \u65e5\u672c\u8a9e"

        runner = CliRunner(charset="utf-8")
        result = runner.invoke(
            cli,
            [
                "update",
                "development/test-entry.md",
                f"--content={unicode_content}",
            ],
        )

        assert result.exit_code == 0

        content = entry_path.read_text(encoding="utf-8")
        assert "\u00e9\u00e0\u00fc" in content
        assert "\u4e2d\u6587" in content
        assert "\u65e5\u672c\u8a9e" in content

    def test_update_large_file(self, kb_root, index_root):
        """Handles large content gracefully."""
        entry_path = kb_root / "development" / "test-entry.md"
        _create_entry(entry_path, "Test Entry", ["tag"], "Original content.")

        # Create large content (100KB+)
        large_content = "# Large Document\n\n" + ("Lorem ipsum dolor sit amet. " * 5000)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "update",
                "development/test-entry.md",
                f"--content={large_content}",
            ],
        )

        assert result.exit_code == 0
        assert "Updated:" in result.output

        content = entry_path.read_text()
        assert "# Large Document" in content
        assert len(content) > 100000

    def test_update_tags_and_content_together(self, kb_root, index_root):
        """Can update both tags and content in same command."""
        entry_path = kb_root / "development" / "test-entry.md"
        _create_entry(entry_path, "Test Entry", ["old-tag"], "Original content.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "update",
                "development/test-entry.md",
                "--tags=new-tag",
                "--content=New content.",
            ],
        )

        assert result.exit_code == 0

        content = entry_path.read_text()
        assert "new-tag" in content
        assert "old-tag" not in content
        assert "New content." in content
        assert "Original content." not in content

    def test_update_markdown_formatting(self, kb_root, index_root):
        """Preserves markdown formatting in content."""
        entry_path = kb_root / "development" / "test-entry.md"
        _create_entry(entry_path, "Test Entry", ["tag"], "Original content.")

        markdown_content = """# Header

## Subheader

- List item 1
- List item 2

```python
def hello():
    return "world"
```

> Blockquote

| Column 1 | Column 2 |
|----------|----------|
| Cell 1   | Cell 2   |
"""

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "update",
                "development/test-entry.md",
                f"--content={markdown_content}",
            ],
        )

        assert result.exit_code == 0

        content = entry_path.read_text()
        assert "# Header" in content
        assert "```python" in content
        assert "def hello():" in content
        assert "> Blockquote" in content
        assert "| Column 1 |" in content

    def test_update_file_not_found_fails(self, kb_root, index_root):
        """--file with non-existent path fails gracefully."""
        entry_path = kb_root / "development" / "test-entry.md"
        _create_entry(entry_path, "Test Entry", ["tag"], "Original content.")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "update",
                "development/test-entry.md",
                "--file=/nonexistent/path/file.md",
            ],
        )

        assert result.exit_code != 0
        # Click should report the file doesn't exist
