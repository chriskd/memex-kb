"""Tests for mx summarize CLI command."""

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
    content: str = "",
    description: str | None = None,
    created: datetime | None = None,
):
    """Helper to create a KB entry with frontmatter."""
    created = created or datetime.now(timezone.utc)
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    description_line = f"\ndescription: {description}" if description else ""
    text = f"""---
title: {title}
tags:
{tags_yaml}
created: {created.isoformat()}{description_line}
---

{content if content else f"Content for {title}."}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class TestSummarizeBasic:
    """Test basic mx summarize functionality."""

    def test_summarize_finds_entries_without_descriptions(self, kb_root, index_root):
        """Summarize identifies entries missing descriptions."""
        # Create entry without description
        _create_entry(
            kb_root / "development" / "no-desc.md",
            "Entry Without Description",
            ["test"],
            "This is the first sentence. More content follows here.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["summarize", "--dry-run"])

        assert result.exit_code == 0
        assert "no-desc.md" in result.output or "Entry Without Description" in result.output
        assert "would be updated" in result.output

    def test_summarize_generates_descriptions_from_content(self, kb_root, index_root):
        """Summarize creates descriptions from entry content."""
        _create_entry(
            kb_root / "development" / "needs-desc.md",
            "Needs Description",
            ["test"],
            "This is a clear first sentence. It explains the topic well.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["summarize"])

        assert result.exit_code == 0
        assert "Generated descriptions for" in result.output

        # Verify the file was updated with a description
        entry_path = kb_root / "development" / "needs-desc.md"
        content = entry_path.read_text()
        assert "description:" in content

    def test_summarize_dry_run_does_not_modify_files(self, kb_root, index_root):
        """--dry-run shows changes without modifying files."""
        _create_entry(
            kb_root / "development" / "dry-run-test.md",
            "Dry Run Test",
            ["test"],
            "This content should not be modified. The dry run option prevents writes.",
        )

        original_content = (kb_root / "development" / "dry-run-test.md").read_text()

        runner = CliRunner()
        result = runner.invoke(cli, ["summarize", "--dry-run"])

        assert result.exit_code == 0
        assert "Preview" in result.output
        assert "would be updated" in result.output

        # Verify file was not modified
        current_content = (kb_root / "development" / "dry-run-test.md").read_text()
        assert current_content == original_content
        assert "description:" not in current_content

    def test_summarize_limit_restricts_processed_entries(self, kb_root, index_root):
        """--limit restricts number of entries processed."""
        # Create multiple entries without descriptions
        for i in range(5):
            _create_entry(
                kb_root / "development" / f"entry-{i}.md",
                f"Entry {i}",
                ["test"],
                f"Content for entry number {i}. This is meaningful text.",
            )

        runner = CliRunner()
        result = runner.invoke(cli, ["summarize", "--limit", "2"])

        assert result.exit_code == 0
        assert "Generated descriptions for 2 entries" in result.output


class TestDescriptionExtraction:
    """Test description extraction logic."""

    def test_extracts_first_meaningful_sentence(self, kb_root, index_root):
        """Extracts first complete sentence as description."""
        _create_entry(
            kb_root / "development" / "sentence-test.md",
            "Sentence Test",
            ["test"],
            "This is the first sentence. This is the second sentence. Third sentence here.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["summarize", "--dry-run", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Find the entry in results
        entry = next((r for r in data if "sentence-test.md" in r["path"]), None)
        assert entry is not None
        assert entry["description"] == "This is the first sentence."

    def test_skips_entries_with_existing_descriptions(self, kb_root, index_root):
        """Entries with descriptions are skipped."""
        _create_entry(
            kb_root / "development" / "has-desc.md",
            "Has Description",
            ["test"],
            "Some content here.",
            description="Existing description",
        )
        _create_entry(
            kb_root / "development" / "no-desc.md",
            "No Description",
            ["test"],
            "Content without description. More text follows.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["summarize", "--dry-run", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Only the entry without description should be in results
        paths = [r["path"] for r in data]
        assert "development/no-desc.md" in paths
        assert "development/has-desc.md" not in paths

    def test_handles_entries_with_only_headings(self, kb_root, index_root):
        """Handles entries that have headings but no paragraph content."""
        _create_entry(
            kb_root / "development" / "headings-only.md",
            "Headings Only",
            ["test"],
            "# Main Heading\n\n## Sub Heading\n\n### Another Heading",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["summarize", "--dry-run", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Entry should either be skipped or have a generated description
        entry = next((r for r in data if "headings-only.md" in r["path"]), None)
        if entry:
            # If processed, should have some description or be marked as skipped
            assert entry["status"] in ("preview", "skipped")


class TestOutputFormat:
    """Test output formatting."""

    def test_shows_count_of_updated_entries(self, kb_root, index_root):
        """Output shows count of entries that were updated."""
        for i in range(3):
            _create_entry(
                kb_root / "development" / f"count-test-{i}.md",
                f"Count Test {i}",
                ["test"],
                f"Content for count test entry {i}. This is a full sentence.",
            )

        runner = CliRunner()
        result = runner.invoke(cli, ["summarize"])

        assert result.exit_code == 0
        assert "Generated descriptions for 3 entries" in result.output

    def test_json_returns_proper_structure(self, kb_root, index_root):
        """--json returns properly structured JSON output."""
        _create_entry(
            kb_root / "development" / "json-test.md",
            "JSON Test Entry",
            ["test"],
            "Content for JSON output test. This validates the structure.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["summarize", "--dry-run", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        assert isinstance(data, list)
        assert len(data) > 0

        # Check structure of each result
        for item in data:
            assert "path" in item
            assert "title" in item
            assert "description" in item
            assert "status" in item
            assert item["status"] in ("preview", "updated", "skipped", "error")

    def test_json_output_includes_path_and_title(self, kb_root, index_root):
        """JSON output includes path and title for each entry."""
        _create_entry(
            kb_root / "development" / "path-title-test.md",
            "Path Title Test",
            ["test"],
            "Content for path and title test entry.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["summarize", "--dry-run", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        entry = data[0]
        assert entry["path"] == "development/path-title-test.md"
        assert entry["title"] == "Path Title Test"


class TestEdgeCases:
    """Test edge cases."""

    def test_all_entries_have_descriptions(self, kb_root, index_root):
        """Reports nothing to do when all entries have descriptions."""
        _create_entry(
            kb_root / "development" / "complete-a.md",
            "Complete A",
            ["test"],
            "Content A.",
            description="Description for A",
        )
        _create_entry(
            kb_root / "development" / "complete-b.md",
            "Complete B",
            ["test"],
            "Content B.",
            description="Description for B",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["summarize"])

        assert result.exit_code == 0
        assert "All entries already have descriptions" in result.output

    def test_empty_kb(self, kb_root, index_root):
        """Handles empty KB gracefully."""
        # KB exists but has no entries (only empty directories)
        runner = CliRunner()
        result = runner.invoke(cli, ["summarize"])

        assert result.exit_code == 0
        assert "All entries already have descriptions" in result.output

    def test_entries_with_very_short_content(self, kb_root, index_root):
        """Handles entries with very short content."""
        _create_entry(
            kb_root / "development" / "short-content.md",
            "Short Content",
            ["test"],
            "Hi.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["summarize", "--dry-run", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Entry should either be processed or skipped
        if data:
            entry = data[0]
            assert entry["status"] in ("preview", "skipped")

    def test_entry_content_starts_with_title(self, kb_root, index_root):
        """Handles entries where content repeats the title."""
        _create_entry(
            kb_root / "development" / "title-repeat.md",
            "Important Topic",
            ["test"],
            "# Important Topic\n\nThis is the actual content. It explains the topic.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["summarize", "--dry-run", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        if data:
            entry = next((r for r in data if "title-repeat.md" in r["path"]), None)
            if entry and entry["description"]:
                # Description should not start with the title
                assert not entry["description"].startswith("Important Topic")


class TestErrorCases:
    """Test error handling."""

    def test_error_when_no_kb_root_configured(self, tmp_path, monkeypatch):
        """Fails gracefully when KB root is not configured."""
        # Clear the MEMEX_KB_ROOT environment variable
        monkeypatch.delenv("MEMEX_KB_ROOT", raising=False)

        # Reset searcher state
        monkeypatch.setattr(core, "_searcher", None)
        monkeypatch.setattr(core, "_searcher_ready", False)

        runner = CliRunner()
        result = runner.invoke(cli, ["summarize"])

        # Either fails with non-zero exit code or raises ConfigurationError
        assert result.exit_code != 0 or result.exception is not None
        # The error message is in the exception, not output
        if result.exception:
            assert "MEMEX_KB_ROOT" in str(result.exception)

    def test_handles_malformed_frontmatter(self, kb_root, index_root):
        """Handles entries with malformed frontmatter gracefully."""
        # Create an entry with malformed YAML
        malformed_path = kb_root / "development" / "malformed.md"
        malformed_path.write_text("""---
title: Malformed Entry
tags: [not, proper, yaml
---

Content here.
""")

        runner = CliRunner()
        result = runner.invoke(cli, ["summarize", "--json"])

        # Should not crash, may report errors
        assert result.exit_code == 0


class TestSummarizeWithLimit:
    """Test --limit behavior in detail."""

    def test_limit_greater_than_available(self, kb_root, index_root):
        """--limit larger than entry count processes all entries."""
        _create_entry(
            kb_root / "development" / "limit-test.md",
            "Limit Test",
            ["test"],
            "Content for limit test. This is a sentence.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["summarize", "--limit", "100"])

        assert result.exit_code == 0
        # Should process the one available entry
        assert "Generated descriptions for 1 entries" in result.output

    def test_limit_one_processes_single_entry(self, kb_root, index_root):
        """--limit 1 processes exactly one entry."""
        for i in range(3):
            _create_entry(
                kb_root / "development" / f"limit-one-{i}.md",
                f"Limit One Test {i}",
                ["test"],
                f"Content for limit one test {i}. Full sentence here.",
            )

        runner = CliRunner()
        result = runner.invoke(cli, ["summarize", "--limit", "1"])

        assert result.exit_code == 0
        assert "Generated descriptions for 1 entries" in result.output


class TestDryRunOutput:
    """Test --dry-run output formatting."""

    def test_dry_run_shows_preview_header(self, kb_root, index_root):
        """--dry-run shows preview header."""
        _create_entry(
            kb_root / "development" / "preview-header.md",
            "Preview Header Test",
            ["test"],
            "Content for preview header test. This is a sentence.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["summarize", "--dry-run"])

        assert result.exit_code == 0
        assert "Preview" in result.output

    def test_dry_run_shows_entry_details(self, kb_root, index_root):
        """--dry-run shows path, title, and description for each entry."""
        _create_entry(
            kb_root / "development" / "details-test.md",
            "Details Test Entry",
            ["test"],
            "This is the description content. More text follows after.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["summarize", "--dry-run"])

        assert result.exit_code == 0
        assert "details-test.md" in result.output
        assert "Details Test Entry" in result.output
        assert "Description:" in result.output
