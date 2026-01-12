"""Tests for restored core functions.

Tests for:
- detect_potential_duplicates() - Find similar existing entries before add
- preview_add_entry() - Preview entry before committing
- generate_descriptions() - Bulk AI description generation
- resolve_entry_by_title() - Resolve entry by title with fuzzy matching
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from memex import core
from memex.core import (
    AmbiguousMatchError,
    detect_potential_duplicates,
    generate_descriptions,
    preview_add_entry,
    resolve_entry_by_title,
)
from memex.models import PotentialDuplicate, SearchResult, UpsertMatch


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


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
    content: str = "Original content.",
    description: str | None = None,
    aliases: list[str] | None = None,
):
    """Helper to create a KB entry with frontmatter."""
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    frontmatter_lines = [
        "---",
        f"title: {title}",
        "tags:",
        tags_yaml,
        f"created: {datetime.now(timezone.utc).isoformat()}",
    ]
    if description:
        frontmatter_lines.insert(2, f"description: {description}")
    if aliases:
        aliases_yaml = "\n".join(f"  - {a}" for a in aliases)
        frontmatter_lines.insert(2, f"aliases:\n{aliases_yaml}")
    frontmatter_lines.append("---")
    frontmatter_lines.append("")
    frontmatter_lines.append(content)

    text = "\n".join(frontmatter_lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class DummySearcher:
    """Mock HybridSearcher for controlled testing."""

    def __init__(self, results: list[SearchResult] | None = None):
        self.results = results or []

    def search(self, query: str, limit: int = 10, mode: str = "hybrid", **kwargs):
        return self.results[:limit]

    def index_document(self, *args, **kwargs):
        pass

    def remove_document(self, *args, **kwargs):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# detect_potential_duplicates Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectPotentialDuplicates:
    """Tests for detect_potential_duplicates function."""

    def test_returns_empty_when_no_similar_entries(self):
        """Returns empty list when no entries exceed threshold."""
        searcher = DummySearcher(results=[])
        result = detect_potential_duplicates(
            title="New Unique Entry",
            content="This is completely unique content.",
            searcher=searcher,
        )
        assert result == []

    def test_finds_exact_title_match(self):
        """Returns duplicate when entry with same title exists."""
        searcher = DummySearcher(
            results=[
                SearchResult(
                    path="development/python-basics.md",
                    title="Python Basics",
                    snippet="Introduction to Python programming.",
                    score=0.95,
                    tags=["python", "tutorial"],
                )
            ]
        )
        result = detect_potential_duplicates(
            title="Python Basics",
            content="Introduction to Python programming language.",
            searcher=searcher,
        )
        assert len(result) == 1
        assert result[0].title == "Python Basics"
        assert result[0].score >= 0.75

    def test_finds_semantically_similar_entries(self):
        """Returns duplicates based on semantic similarity, not just title."""
        searcher = DummySearcher(
            results=[
                SearchResult(
                    path="development/intro-to-python.md",
                    title="Introduction to Python",
                    snippet="Getting started with Python programming.",
                    score=0.85,
                    tags=["python", "beginner"],
                )
            ]
        )
        result = detect_potential_duplicates(
            title="Python Getting Started Guide",
            content="Getting started with Python programming. Learn the basics.",
            searcher=searcher,
        )
        assert len(result) == 1
        assert result[0].path == "development/intro-to-python.md"

    def test_respects_score_threshold(self):
        """Only returns duplicates above the specified min_score."""
        searcher = DummySearcher(
            results=[
                SearchResult(
                    path="development/high-score.md",
                    title="High Score Match",
                    snippet="Very similar content.",
                    score=0.90,
                    tags=["test"],
                ),
                SearchResult(
                    path="development/low-score.md",
                    title="Low Score Match",
                    snippet="Somewhat related content.",
                    score=0.50,
                    tags=["test"],
                ),
            ]
        )
        # Default threshold is 0.75
        result = detect_potential_duplicates(
            title="Similar Entry",
            content="Very similar content.",
            searcher=searcher,
        )
        assert len(result) == 1
        assert result[0].path == "development/high-score.md"

        # Higher threshold filters out more
        result = detect_potential_duplicates(
            title="Similar Entry",
            content="Very similar content.",
            searcher=searcher,
            min_score=0.95,
        )
        assert len(result) == 0

    def test_respects_limit_parameter(self):
        """Only returns up to the specified limit."""
        searcher = DummySearcher(
            results=[
                SearchResult(
                    path=f"development/entry-{i}.md",
                    title=f"Entry {i}",
                    snippet="Content.",
                    score=0.90 - (i * 0.01),
                    tags=["test"],
                )
                for i in range(10)
            ]
        )
        result = detect_potential_duplicates(
            title="Similar Entry",
            content="Content.",
            searcher=searcher,
            limit=3,
        )
        assert len(result) == 3
        assert result[0].path == "development/entry-0.md"

    def test_handles_empty_kb_gracefully(self):
        """Returns empty list when KB has no entries."""
        searcher = DummySearcher(results=[])
        result = detect_potential_duplicates(
            title="New Entry",
            content="New content for empty KB.",
            searcher=searcher,
        )
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# preview_add_entry Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPreviewAddEntry:
    """Tests for preview_add_entry function."""

    @pytest.mark.asyncio
    async def test_returns_correct_path_without_creating_file(
        self, kb_root, index_root, monkeypatch
    ):
        """Returns expected path but doesn't actually create the file."""
        # Mock get_searcher to avoid index initialization
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        preview = await preview_add_entry(
            title="Test Entry",
            content="This is test content.",
            tags=["test", "preview"],
            category="development",
            check_duplicates=False,
        )

        assert preview.path == "development/test-entry.md"
        assert preview.absolute_path.endswith("development/test-entry.md")

        # File should NOT be created
        expected_file = kb_root / "development" / "test-entry.md"
        assert not expected_file.exists()

    @pytest.mark.asyncio
    async def test_returns_correct_frontmatter(self, kb_root, index_root, monkeypatch):
        """Returns properly formatted YAML frontmatter."""
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        preview = await preview_add_entry(
            title="Frontmatter Test",
            content="Content here.",
            tags=["yaml", "frontmatter"],
            category="development",
            check_duplicates=False,
        )

        assert "---" in preview.frontmatter
        assert "title: Frontmatter Test" in preview.frontmatter
        assert "yaml" in preview.frontmatter
        assert "frontmatter" in preview.frontmatter
        assert "created:" in preview.frontmatter

    @pytest.mark.asyncio
    async def test_returns_content_preview(self, kb_root, index_root, monkeypatch):
        """Returns the content that would be written."""
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        preview = await preview_add_entry(
            title="Content Test",
            content="# Main Header\n\nParagraph content.",
            tags=["content"],
            category="development",
            check_duplicates=False,
        )

        assert "# Main Header" in preview.content
        assert "Paragraph content." in preview.content

    @pytest.mark.asyncio
    async def test_includes_duplicate_warnings_when_similar_entries_exist(
        self, kb_root, index_root, monkeypatch
    ):
        """Warns when potential duplicates are detected."""
        # Create a searcher that returns similar entries
        similar_searcher = DummySearcher(
            results=[
                SearchResult(
                    path="development/existing-entry.md",
                    title="Existing Entry",
                    snippet="Similar content.",
                    score=0.85,
                    tags=["test"],
                )
            ]
        )
        monkeypatch.setattr(core, "get_searcher", lambda: similar_searcher)

        preview = await preview_add_entry(
            title="Similar Entry",
            content="Similar content.",
            tags=["test"],
            category="development",
            check_duplicates=True,
        )

        assert len(preview.potential_duplicates) > 0
        assert preview.warning is not None
        assert "duplicate" in preview.warning.lower()
        assert "Existing Entry" in preview.warning

    @pytest.mark.asyncio
    async def test_handles_various_categories(self, kb_root, index_root, monkeypatch):
        """Works with different category directories."""
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        for category in ["development", "architecture", "devops"]:
            preview = await preview_add_entry(
                title=f"{category.title()} Entry",
                content="Content.",
                tags=["test"],
                category=category,
                check_duplicates=False,
            )
            assert preview.path.startswith(f"{category}/")

    @pytest.mark.asyncio
    async def test_slug_generation_is_correct(self, kb_root, index_root, monkeypatch):
        """Title is correctly slugified for filename."""
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        test_cases = [
            ("My Test Title", "my-test-title"),
            ("Hello World 123", "hello-world-123"),
            ("Special_Chars-Test", "special-chars-test"),
            ("  Trimmed  Spaces  ", "trimmed-spaces"),
        ]

        for title, expected_slug in test_cases:
            preview = await preview_add_entry(
                title=title,
                content="Content.",
                tags=["test"],
                category="development",
                check_duplicates=False,
            )
            assert f"{expected_slug}.md" in preview.path, f"Failed for title: {title}"

    @pytest.mark.asyncio
    async def test_raises_error_when_entry_exists(
        self, kb_root, index_root, monkeypatch
    ):
        """Raises ValueError when entry already exists at path."""
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        # Create existing entry
        existing_path = kb_root / "development" / "existing.md"
        _create_entry(existing_path, "Existing", ["test"])

        with pytest.raises(ValueError, match="already exists"):
            await preview_add_entry(
                title="Existing",
                content="New content.",
                tags=["test"],
                category="development",
                check_duplicates=False,
            )


# ─────────────────────────────────────────────────────────────────────────────
# generate_descriptions Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerateDescriptions:
    """Tests for generate_descriptions function."""

    @pytest.mark.asyncio
    async def test_generates_description_from_first_paragraph(
        self, kb_root, index_root
    ):
        """Extracts first sentence as description."""
        entry_path = kb_root / "development" / "test-entry.md"
        _create_entry(
            entry_path,
            "Test Entry",
            ["python"],
            "This is the first sentence of the entry. Second sentence here.",
        )

        results = await generate_descriptions(dry_run=False, limit=1)

        assert len(results) == 1
        assert results[0]["status"] == "updated"
        assert results[0]["description"] is not None
        # Description should be based on content
        assert "first sentence" in results[0]["description"].lower()

    @pytest.mark.asyncio
    async def test_skips_entries_with_existing_descriptions(
        self, kb_root, index_root
    ):
        """Does not overwrite existing descriptions."""
        entry_path = kb_root / "development" / "has-desc.md"
        _create_entry(
            entry_path,
            "Has Description",
            ["python"],
            "Content here.",
            description="Existing description that should be kept.",
        )

        results = await generate_descriptions(dry_run=False)

        # Entry with existing description should not appear in results
        # (it's skipped entirely in the iteration)
        paths = [r["path"] for r in results]
        assert "development/has-desc.md" not in paths

    @pytest.mark.asyncio
    async def test_handles_entries_with_only_headings(self, kb_root, index_root):
        """Handles entries that only have headings, no paragraph text."""
        entry_path = kb_root / "development" / "headings-only.md"
        _create_entry(
            entry_path,
            "Headings Only",
            ["test"],
            "# Section One\n\n## Section Two\n\n### Section Three",
        )

        results = await generate_descriptions(dry_run=False, limit=1)

        # Should either generate something or skip gracefully
        assert len(results) == 1
        # Could be "skipped" if no meaningful text found
        assert results[0]["status"] in ("updated", "skipped")

    @pytest.mark.asyncio
    async def test_respects_limit_parameter(self, kb_root, index_root):
        """Only processes up to limit entries."""
        # Create multiple entries
        for i in range(5):
            entry_path = kb_root / "development" / f"entry-{i}.md"
            _create_entry(
                entry_path,
                f"Entry {i}",
                ["test"],
                f"This is content for entry number {i}.",
            )

        results = await generate_descriptions(dry_run=False, limit=2)

        # Should only process 2 entries
        updated_count = sum(1 for r in results if r["status"] in ("updated", "preview"))
        assert updated_count <= 2

    @pytest.mark.asyncio
    async def test_dry_run_mode_does_not_modify_files(self, kb_root, index_root):
        """dry_run=True previews changes without writing."""
        entry_path = kb_root / "development" / "dry-run-test.md"
        _create_entry(
            entry_path,
            "Dry Run Test",
            ["test"],
            "Content that should generate a description.",
        )

        original_content = entry_path.read_text()

        results = await generate_descriptions(dry_run=True, limit=1)

        # Should have preview status
        assert len(results) == 1
        assert results[0]["status"] == "preview"
        assert results[0]["description"] is not None

        # File should NOT be modified
        assert entry_path.read_text() == original_content


# ─────────────────────────────────────────────────────────────────────────────
# resolve_entry_by_title Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveEntryByTitle:
    """Tests for resolve_entry_by_title function."""

    def test_exact_title_match(self, kb_root, index_root, monkeypatch):
        """Finds entry by exact title match."""
        entry_path = kb_root / "development" / "python-guide.md"
        _create_entry(entry_path, "Python Guide", ["python"])

        # Mock the searcher to return empty (exact match should work without it)
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        result = resolve_entry_by_title("Python Guide", kb_root)

        assert result is not None
        assert result.match_type == "exact_title"
        assert result.score == 1.0
        assert "python-guide" in result.path

    def test_alias_match(self, kb_root, index_root, monkeypatch):
        """Finds entry by alias."""
        entry_path = kb_root / "development" / "python-guide.md"
        _create_entry(
            entry_path,
            "Python Guide",
            ["python"],
            "Content.",
            aliases=["py-guide", "python-tutorial"],
        )

        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        result = resolve_entry_by_title("py-guide", kb_root)

        assert result is not None
        assert "python-guide" in result.path
        # Match type could be exact_title or alias depending on implementation
        assert result.score == 1.0

    def test_fuzzy_semantic_match(self, kb_root, index_root, monkeypatch):
        """Finds entry via semantic search when no exact match."""
        entry_path = kb_root / "development" / "python-tutorial.md"
        _create_entry(entry_path, "Python Tutorial", ["python"])

        # Mock searcher to return fuzzy match
        fuzzy_searcher = DummySearcher(
            results=[
                SearchResult(
                    path="development/python-tutorial.md",
                    title="Python Tutorial",
                    snippet="Tutorial content.",
                    score=0.85,
                    tags=["python"],
                )
            ]
        )
        monkeypatch.setattr(core, "get_searcher", lambda: fuzzy_searcher)

        result = resolve_entry_by_title("Python Learning Guide", kb_root)

        assert result is not None
        assert result.match_type == "fuzzy"
        assert result.score >= 0.6

    def test_raises_ambiguous_match_error_when_multiple_matches(
        self, kb_root, index_root, monkeypatch
    ):
        """Raises AmbiguousMatchError when multiple entries match similarly."""
        # Create multiple similar entries
        _create_entry(
            kb_root / "development" / "python-basics.md", "Python Basics", ["python"]
        )
        _create_entry(
            kb_root / "development" / "python-intro.md",
            "Python Introduction",
            ["python"],
        )

        # Mock searcher to return multiple similar-scoring results
        ambiguous_searcher = DummySearcher(
            results=[
                SearchResult(
                    path="development/python-basics.md",
                    title="Python Basics",
                    snippet="Basic Python.",
                    score=0.82,
                    tags=["python"],
                ),
                SearchResult(
                    path="development/python-intro.md",
                    title="Python Introduction",
                    snippet="Intro to Python.",
                    score=0.80,
                    tags=["python"],
                ),
            ]
        )
        monkeypatch.setattr(core, "get_searcher", lambda: ambiguous_searcher)

        with pytest.raises(AmbiguousMatchError) as exc_info:
            resolve_entry_by_title("Python Getting Started", kb_root)

        assert len(exc_info.value.matches) >= 2

    def test_returns_none_when_no_match_found(self, kb_root, index_root, monkeypatch):
        """Returns None when no match above threshold."""
        # Empty KB - no entries
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        result = resolve_entry_by_title("Nonexistent Entry", kb_root)

        assert result is None

    def test_handles_case_insensitive_matching(self, kb_root, index_root, monkeypatch):
        """Matches titles regardless of case."""
        entry_path = kb_root / "development" / "python-guide.md"
        _create_entry(entry_path, "Python Guide", ["python"])

        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        # Try various case combinations
        for title_variant in ["python guide", "PYTHON GUIDE", "PyThOn GuIdE"]:
            result = resolve_entry_by_title(title_variant, kb_root)
            assert result is not None, f"Failed for: {title_variant}"
            assert result.score == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCoreIntegration:
    """Integration tests combining multiple core functions."""

    @pytest.mark.asyncio
    async def test_preview_then_resolve_workflow(
        self, kb_root, index_root, monkeypatch
    ):
        """Preview an entry, then verify it can be resolved after creation."""
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        # Preview the entry
        preview = await preview_add_entry(
            title="Integration Test Entry",
            content="This is integration test content.",
            tags=["integration", "test"],
            category="development",
            check_duplicates=False,
        )

        # Actually create the entry (simulating what add_entry would do)
        file_path = Path(preview.absolute_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(f"{preview.frontmatter}{preview.content}")

        # Now resolve should find it
        result = resolve_entry_by_title("Integration Test Entry", kb_root)
        assert result is not None
        assert "integration-test-entry" in result.path

    @pytest.mark.asyncio
    async def test_duplicate_detection_prevents_duplicate_entries(
        self, kb_root, index_root, monkeypatch
    ):
        """Duplicate detection warns before creating similar entries."""
        # Create an existing entry
        existing_path = kb_root / "development" / "python-basics.md"
        _create_entry(existing_path, "Python Basics", ["python"], "Python fundamentals.")

        # Mock searcher to detect the duplicate
        duplicate_searcher = DummySearcher(
            results=[
                SearchResult(
                    path="development/python-basics.md",
                    title="Python Basics",
                    snippet="Python fundamentals.",
                    score=0.90,
                    tags=["python"],
                )
            ]
        )
        monkeypatch.setattr(core, "get_searcher", lambda: duplicate_searcher)

        # Preview a similar entry - should get duplicate warning
        preview = await preview_add_entry(
            title="Python Fundamentals",
            content="Python fundamentals and basics.",
            tags=["python"],
            category="development",
            check_duplicates=True,
        )

        assert len(preview.potential_duplicates) > 0
        assert preview.warning is not None
