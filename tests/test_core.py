"""Tests for core business logic in memex.core.

Test organization:
- CRUD operations (add, get, update, delete)
- Search functionality (keyword, semantic, hybrid)
- Validation and error handling
- Resolution and suggestion helpers

Design:
- Test behaviors, not implementations
- Use parametrize for variants
- Integration tests over unit tests where possible
- Every test should catch a real bug class
"""

from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from memex import core
from memex.models import SearchResult
from memex.parser import ParseError


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_searcher_state(monkeypatch):
    """Ensure cached searcher state does not leak across tests."""
    monkeypatch.setattr(core, "_searcher", None)
    monkeypatch.setattr(core, "_searcher_ready", False)


def _create_entry(
    path: Path,
    title: str,
    content_body: str,
    tags: list[str] | None = None,
    description: str | None = None,
):
    """Helper to create a KB entry with frontmatter."""
    tags = tags or ["test"]
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    desc_line = f"description: {description}\n" if description else ""
    content = f"""---
title: {title}
{desc_line}tags:
{tags_yaml}
created: {date.today().isoformat()}
---

{content_body}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class DummySearcher:
    """Mock HybridSearcher for controlled testing without semantic deps."""

    def __init__(self, results: list[SearchResult] | None = None):
        self.results = results or []
        self._indexed_chunks = []
        self._deleted_docs = []

    def search(self, query: str, limit: int = 10, mode: str = "hybrid", **kwargs):
        return self.results[:limit]

    def index_chunks(self, chunks):
        self._indexed_chunks.extend(chunks)

    def delete_document(self, path):
        self._deleted_docs.append(path)

    def reindex(self, kb_root):
        pass

    def status(self):
        from memex.models import IndexStatus
        return IndexStatus(whoosh_docs=0, chroma_docs=0, kb_files=0)


# ─────────────────────────────────────────────────────────────────────────────
# Add Entry Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAddEntry:
    """Tests for add_entry core function."""

    @pytest.mark.asyncio
    async def test_add_entry_creates_file_with_frontmatter(self, tmp_kb):
        """Basic add creates file with proper frontmatter."""
        result = await core.add_entry(
            title="Test Entry",
            content="# Test\n\nContent here",
            tags=["test"],
            category="general",
        )

        assert "path" in result
        assert result["path"] == "general/test-entry.md"

        file_path = tmp_kb / result["path"]
        assert file_path.exists()

        content = file_path.read_text()
        assert "title: Test Entry" in content
        assert "tags:" in content
        assert "created:" in content
        assert "# Test" in content

    @pytest.mark.asyncio
    async def test_add_entry_returns_suggestions(self, tmp_kb, monkeypatch):
        """add_entry returns link and tag suggestions."""
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        result = await core.add_entry(
            title="Python Guide",
            content="A guide to Python programming",
            tags=["python"],
            category="general",
        )

        assert "suggested_links" in result
        assert "suggested_tags" in result
        assert isinstance(result["suggested_links"], list)
        assert isinstance(result["suggested_tags"], list)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("invalid_title", [
        "",
        "   ",
        "\t\n",
        "!@#$%^&*()",
    ])
    async def test_add_entry_rejects_invalid_titles(self, tmp_kb, invalid_title):
        """Titles without alphanumeric chars are rejected."""
        with pytest.raises(ValueError, match="alphanumeric"):
            await core.add_entry(
                title=invalid_title,
                content="Content",
                tags=["test"],
                category="general",
            )

    @pytest.mark.asyncio
    async def test_add_entry_requires_tags(self, tmp_kb):
        """Empty tags list raises ValueError."""
        with pytest.raises(ValueError, match="At least one tag"):
            await core.add_entry(
                title="Test Entry",
                content="Content",
                tags=[],
                category="general",
            )

    @pytest.mark.asyncio
    async def test_add_entry_requires_category_or_directory(self, tmp_kb):
        """Missing both category and directory raises ValueError."""
        with pytest.raises(ValueError, match="category.*directory"):
            await core.add_entry(
                title="Test Entry",
                content="Content",
                tags=["test"],
            )

    @pytest.mark.asyncio
    async def test_add_entry_auto_creates_directory(self, tmp_kb):
        """Directory is auto-created if it doesn't exist."""
        result = await core.add_entry(
            title="Nested Entry",
            content="Content",
            tags=["test"],
            directory="development/python/frameworks",
        )

        assert "path" in result
        assert "development/python/frameworks" in result["path"]
        assert (tmp_kb / "development/python/frameworks").is_dir()

    @pytest.mark.asyncio
    async def test_add_entry_duplicate_path_raises_error(self, tmp_kb, monkeypatch):
        """Creating entry at existing path raises ValueError."""
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        await core.add_entry(
            title="Existing Entry",
            content="Content",
            tags=["test"],
            category="general",
        )

        with pytest.raises(ValueError, match="already exists"):
            await core.add_entry(
                title="Existing Entry",
                content="Different content",
                tags=["test"],
                category="general",
            )

    @pytest.mark.asyncio
    async def test_add_entry_with_links(self, tmp_kb, monkeypatch):
        """Links are added as a Related section."""
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        result = await core.add_entry(
            title="Entry With Links",
            content="Main content",
            tags=["test"],
            category="general",
            links=["general/other-entry", "development/guide"],
        )

        file_path = tmp_kb / result["path"]
        content = file_path.read_text()
        assert "## Related" in content
        assert "[[general/other-entry]]" in content
        assert "[[development/guide]]" in content

    @pytest.mark.asyncio
    async def test_add_entry_unicode_content_preserved(self, tmp_kb, monkeypatch):
        """Unicode in content is preserved correctly."""
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        unicode_content = "Content with Chinese: 你好世界\nEmoji: 🚀"
        result = await core.add_entry(
            title="Unicode Test",
            content=unicode_content,
            tags=["test"],
            category="general",
        )

        entry = await core.get_entry(result["path"])
        assert "你好世界" in entry.content
        assert "🚀" in entry.content


# ─────────────────────────────────────────────────────────────────────────────
# Get Entry Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGetEntry:
    """Tests for get_entry core function."""

    @pytest.mark.asyncio
    async def test_get_entry_returns_content_and_metadata(self, tmp_kb):
        """get_entry returns parsed entry with metadata."""
        _create_entry(
            tmp_kb / "general" / "test.md",
            "Test Entry",
            "# Heading\n\nSome content here.",
            tags=["python", "guide"],
        )

        entry = await core.get_entry("general/test.md")

        assert entry.path == "general/test.md"
        assert entry.metadata.title == "Test Entry"
        assert "python" in entry.metadata.tags
        assert "guide" in entry.metadata.tags
        assert "# Heading" in entry.content

    @pytest.mark.asyncio
    async def test_get_entry_not_found_raises_error(self, tmp_kb):
        """get_entry raises ValueError for non-existent entry."""
        with pytest.raises(ValueError, match="Entry not found"):
            await core.get_entry("general/nonexistent.md")

    @pytest.mark.asyncio
    async def test_get_entry_extracts_links(self, tmp_kb):
        """get_entry extracts wiki-style links from content."""
        _create_entry(
            tmp_kb / "general" / "with-links.md",
            "Entry With Links",
            "See [[general/other]] and [[development/guide]].",
        )

        entry = await core.get_entry("general/with-links.md")

        assert "general/other" in entry.links
        assert "development/guide" in entry.links

    @pytest.mark.asyncio
    async def test_get_entry_directory_raises_error(self, tmp_kb):
        """get_entry raises ValueError for directory path."""
        (tmp_kb / "general" / "subdir").mkdir(parents=True)

        with pytest.raises(ValueError, match="not a file"):
            await core.get_entry("general/subdir")


# ─────────────────────────────────────────────────────────────────────────────
# Update Entry Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestUpdateEntry:
    """Tests for update_entry core function."""

    @pytest.mark.asyncio
    async def test_update_entry_replaces_content(self, tmp_kb, monkeypatch):
        """update_entry replaces content while preserving title."""
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        _create_entry(
            tmp_kb / "general" / "update-me.md",
            "Update Me",
            "Original content",
        )

        result = await core.update_entry(
            path="general/update-me.md",
            content="New content here",
        )

        assert result["path"] == "general/update-me.md"

        entry = await core.get_entry("general/update-me.md")
        assert "New content" in entry.content
        assert entry.metadata.title == "Update Me"

    @pytest.mark.asyncio
    async def test_update_entry_updates_tags(self, tmp_kb, monkeypatch):
        """update_entry can update tags."""
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        _create_entry(
            tmp_kb / "general" / "tags-test.md",
            "Tags Test",
            "Content",
            tags=["old-tag"],
        )

        await core.update_entry(
            path="general/tags-test.md",
            content="Content",
            tags=["new-tag", "another-tag"],
        )

        entry = await core.get_entry("general/tags-test.md")
        assert "new-tag" in entry.metadata.tags
        assert "another-tag" in entry.metadata.tags
        assert "old-tag" not in entry.metadata.tags

    @pytest.mark.asyncio
    async def test_update_entry_not_found_raises_error(self, tmp_kb):
        """update_entry raises ValueError for non-existent entry."""
        with pytest.raises(ValueError, match="Entry not found"):
            await core.update_entry(
                path="general/nonexistent.md",
                content="New content",
            )

    @pytest.mark.asyncio
    async def test_update_entry_requires_content_or_sections(self, tmp_kb):
        """update_entry requires either content or section_updates."""
        _create_entry(
            tmp_kb / "general" / "test.md",
            "Test",
            "Content",
        )

        with pytest.raises(ValueError, match="content or section_updates"):
            await core.update_entry(path="general/test.md")

    @pytest.mark.asyncio
    async def test_update_entry_section_updates_existing(self, tmp_kb, monkeypatch):
        """section_updates modifies existing section content."""
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        _create_entry(
            tmp_kb / "general" / "sections.md",
            "Sections Test",
            "## Overview\n\nOld overview content\n\n## Details\n\nDetails here",
        )

        await core.update_entry(
            path="general/sections.md",
            section_updates={"Overview": "New overview content"},
        )

        entry = await core.get_entry("general/sections.md")
        assert "New overview content" in entry.content
        assert "Details here" in entry.content

    @pytest.mark.asyncio
    async def test_update_entry_section_updates_adds_new(self, tmp_kb, monkeypatch):
        """section_updates adds new section if not found."""
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        _create_entry(
            tmp_kb / "general" / "add-section.md",
            "Add Section",
            "## Existing\n\nExisting content",
        )

        await core.update_entry(
            path="general/add-section.md",
            section_updates={"New Section": "New section content"},
        )

        entry = await core.get_entry("general/add-section.md")
        assert "## Existing" in entry.content
        assert "## New Section" in entry.content
        assert "New section content" in entry.content

    @pytest.mark.asyncio
    async def test_update_entry_rejects_empty_tags(self, tmp_kb):
        """update_entry rejects empty tags list."""
        _create_entry(
            tmp_kb / "general" / "test.md",
            "Test",
            "Content",
        )

        with pytest.raises(ValueError, match="At least one tag"):
            await core.update_entry(
                path="general/test.md",
                content="New content",
                tags=[],
            )


# ─────────────────────────────────────────────────────────────────────────────
# Delete Entry Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDeleteEntry:
    """Tests for delete_entry core function."""

    @pytest.mark.asyncio
    async def test_delete_entry_removes_file(self, tmp_kb, monkeypatch):
        """delete_entry removes the file from disk."""
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        _create_entry(
            tmp_kb / "general" / "delete-me.md",
            "Delete Me",
            "Content to delete",
        )

        result = await core.delete_entry("general/delete-me.md")

        assert result["deleted"] == "general/delete-me.md"
        assert not (tmp_kb / "general" / "delete-me.md").exists()

    @pytest.mark.asyncio
    async def test_delete_entry_not_found_raises_error(self, tmp_kb):
        """delete_entry raises ValueError for non-existent entry."""
        with pytest.raises(ValueError, match="Entry not found"):
            await core.delete_entry("general/nonexistent.md")

    @pytest.mark.asyncio
    @pytest.mark.semantic
    async def test_delete_entry_with_backlinks_requires_force(self, tmp_kb):
        """delete_entry fails if entry has backlinks without force flag."""
        _create_entry(
            tmp_kb / "general" / "target.md",
            "Target Entry",
            "This is the target",
        )
        _create_entry(
            tmp_kb / "general" / "source.md",
            "Source Entry",
            "Links to [[general/target]]",
        )

        await core.reindex()

        with pytest.raises(ValueError, match="backlink"):
            await core.delete_entry("general/target.md", force=False)

    @pytest.mark.asyncio
    @pytest.mark.semantic
    async def test_delete_entry_with_force_succeeds(self, tmp_kb):
        """delete_entry with force=True deletes even with backlinks."""
        _create_entry(
            tmp_kb / "general" / "target.md",
            "Target Entry",
            "This is the target",
        )
        _create_entry(
            tmp_kb / "general" / "source.md",
            "Source Entry",
            "Links to [[general/target]]",
        )

        await core.reindex()

        result = await core.delete_entry("general/target.md", force=True)

        assert result["deleted"] == "general/target.md"
        assert len(result["had_backlinks"]) > 0
        assert not (tmp_kb / "general" / "target.md").exists()

    @pytest.mark.asyncio
    async def test_delete_directory_raises_error(self, tmp_kb):
        """delete_entry raises ValueError for directory path."""
        (tmp_kb / "general" / "subdir").mkdir(parents=True)

        with pytest.raises(ValueError, match="not a file"):
            await core.delete_entry("general/subdir")


# ─────────────────────────────────────────────────────────────────────────────
# Search Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSearch:
    """Tests for search core function."""

    @pytest.mark.asyncio
    @pytest.mark.semantic
    async def test_search_returns_matching_entries(self, tmp_kb):
        """Basic search returns entries matching query."""
        _create_entry(
            tmp_kb / "general" / "python-guide.md",
            "Python Guide",
            "A guide to Python programming language.",
            tags=["python", "guide"],
        )
        _create_entry(
            tmp_kb / "general" / "rust-guide.md",
            "Rust Guide",
            "A guide to Rust programming language.",
            tags=["rust", "guide"],
        )

        await core.reindex()

        result = await core.search("python programming")

        assert len(result.results) >= 1
        paths = [r.path for r in result.results]
        assert "general/python-guide.md" in paths

    @pytest.mark.asyncio
    @pytest.mark.semantic
    @pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
    async def test_search_modes(self, tmp_kb, mode):
        """Search works in all three modes."""
        _create_entry(
            tmp_kb / "general" / "test.md",
            "Test Entry",
            "Content for testing search modes.",
        )

        await core.reindex()

        result = await core.search("test", mode=mode)

        assert isinstance(result.results, list)

    @pytest.mark.asyncio
    @pytest.mark.semantic
    async def test_search_filters_by_tags(self, tmp_kb):
        """Search filters results by tag."""
        _create_entry(
            tmp_kb / "general" / "python.md",
            "Python Entry",
            "Python content",
            tags=["python"],
        )
        _create_entry(
            tmp_kb / "general" / "rust.md",
            "Rust Entry",
            "Rust content",
            tags=["rust"],
        )

        await core.reindex()

        result = await core.search("content", tags=["python"])

        paths = [r.path for r in result.results]
        assert "general/python.md" in paths
        # rust entry should be filtered out
        for r in result.results:
            assert "python" in r.tags

    @pytest.mark.asyncio
    @pytest.mark.semantic
    async def test_search_empty_query_returns_results(self, tmp_kb):
        """Empty query returns results (search everything)."""
        _create_entry(
            tmp_kb / "general" / "test.md",
            "Test Entry",
            "Content",
        )

        await core.reindex()

        result = await core.search("")

        assert isinstance(result.results, list)

    @pytest.mark.asyncio
    @pytest.mark.semantic
    async def test_search_empty_kb_returns_empty(self, tmp_kb):
        """Search on empty KB returns empty results."""
        await core.reindex()

        result = await core.search("anything")

        assert result.results == []

    @pytest.mark.asyncio
    @pytest.mark.semantic
    async def test_search_respects_limit(self, tmp_kb):
        """Search respects limit parameter."""
        for i in range(10):
            _create_entry(
                tmp_kb / "general" / f"entry-{i}.md",
                f"Entry {i}",
                f"Content for entry {i}",
            )

        await core.reindex()

        result = await core.search("entry", limit=3)

        assert len(result.results) <= 3

    @pytest.mark.asyncio
    @pytest.mark.semantic
    async def test_search_includes_content_when_requested(self, tmp_kb):
        """Search hydrates content when include_content=True."""
        _create_entry(
            tmp_kb / "general" / "test.md",
            "Test Entry",
            "Full content body here.",
        )

        await core.reindex()

        result = await core.search("test", include_content=True)

        assert len(result.results) >= 1
        assert result.results[0].content is not None
        assert "Full content body" in result.results[0].content

    @pytest.mark.asyncio
    @pytest.mark.semantic
    @pytest.mark.parametrize("limit", [0, -1])
    async def test_search_invalid_limit_raises_error(self, tmp_kb, limit):
        """Search with limit < 1 raises ValueError."""
        await core.reindex()

        with pytest.raises(ValueError, match="limit must be >= 1"):
            await core.search("test", limit=limit)


# ─────────────────────────────────────────────────────────────────────────────
# Append Entry Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAppendEntry:
    """Tests for append_entry core function."""

    @pytest.mark.asyncio
    async def test_append_to_existing_entry(self, tmp_kb, monkeypatch):
        """append_entry adds content to existing entry."""
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        _create_entry(
            tmp_kb / "general" / "existing.md",
            "Existing Entry",
            "Original content",
        )

        result = await core.append_entry(
            title="Existing Entry",
            content="Appended content",
        )

        assert result["action"] == "appended"

        entry = await core.get_entry("general/existing.md")
        assert "Original content" in entry.content
        assert "Appended content" in entry.content

    @pytest.mark.asyncio
    async def test_append_creates_new_when_not_found(self, tmp_kb, monkeypatch):
        """append_entry creates new entry when title not found."""
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        result = await core.append_entry(
            title="New Entry",
            content="New content",
            tags=["test"],
            directory="general",
        )

        assert result["action"] == "created"
        assert "path" in result

    @pytest.mark.asyncio
    async def test_append_no_create_raises_when_not_found(self, tmp_kb):
        """append_entry with no_create=True raises when entry not found."""
        with pytest.raises(ValueError, match="not found"):
            await core.append_entry(
                title="Nonexistent",
                content="Content",
                no_create=True,
            )

    @pytest.mark.asyncio
    async def test_append_requires_tags_for_new_entry(self, tmp_kb):
        """append_entry requires tags when creating new entry."""
        with pytest.raises(ValueError, match="Tags are required"):
            await core.append_entry(
                title="New Entry",
                content="Content",
                directory="general",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Preview Add Entry Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPreviewAddEntry:
    """Tests for preview_add_entry core function."""

    @pytest.mark.asyncio
    async def test_preview_returns_path_without_creating(self, tmp_kb, monkeypatch):
        """preview_add_entry returns path without creating file."""
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        preview = await core.preview_add_entry(
            title="Preview Entry",
            content="Content",
            tags=["test"],
            category="general",
            check_duplicates=False,
        )

        assert preview.path == "general/preview-entry.md"
        assert not (tmp_kb / "general" / "preview-entry.md").exists()

    @pytest.mark.asyncio
    async def test_preview_returns_frontmatter(self, tmp_kb, monkeypatch):
        """preview_add_entry returns generated frontmatter."""
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        preview = await core.preview_add_entry(
            title="Preview Entry",
            content="Content",
            tags=["test", "preview"],
            category="general",
            check_duplicates=False,
        )

        assert "---" in preview.frontmatter
        assert "title: Preview Entry" in preview.frontmatter
        assert "test" in preview.frontmatter
        assert "preview" in preview.frontmatter

    @pytest.mark.asyncio
    async def test_preview_detects_duplicates(self, tmp_kb, monkeypatch):
        """preview_add_entry warns about potential duplicates."""
        similar_searcher = DummySearcher(
            results=[
                SearchResult(
                    path="general/existing.md",
                    title="Existing Entry",
                    snippet="Similar content.",
                    score=0.85,
                    tags=["test"],
                )
            ]
        )
        monkeypatch.setattr(core, "get_searcher", lambda: similar_searcher)

        preview = await core.preview_add_entry(
            title="Similar Entry",
            content="Similar content.",
            tags=["test"],
            category="general",
            check_duplicates=True,
        )

        assert len(preview.potential_duplicates) > 0
        assert preview.warning is not None
        assert "duplicate" in preview.warning.lower()

    @pytest.mark.asyncio
    async def test_preview_raises_if_exists(self, tmp_kb, monkeypatch):
        """preview_add_entry raises if file already exists."""
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        _create_entry(
            tmp_kb / "general" / "exists.md",
            "Exists",
            "Content",
        )

        with pytest.raises(ValueError, match="already exists"):
            await core.preview_add_entry(
                title="Exists",
                content="New content",
                tags=["test"],
                category="general",
                check_duplicates=False,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Path Validation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPathValidation:
    """Tests for path validation and security."""

    @pytest.mark.parametrize("invalid_path", [
        "../etc/passwd",
        "/absolute/path",
        "development/.hidden/file.md",
        "development/_private/file.md",
    ])
    def test_validate_nested_path_rejects_invalid(self, tmp_kb, invalid_path):
        """Invalid paths are rejected with ValueError."""
        with pytest.raises(ValueError):
            core.validate_nested_path(invalid_path)

    def test_validate_nested_path_accepts_valid(self, tmp_kb):
        """Valid nested paths are accepted."""
        abs_path, normalized = core.validate_nested_path("development/python/test.md")

        assert abs_path.is_absolute()
        assert normalized == "development/python/test.md"


# ─────────────────────────────────────────────────────────────────────────────
# Slugify Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSlugify:
    """Tests for title slugification."""

    @pytest.mark.parametrize("title,expected", [
        ("Test Entry", "test-entry"),
        ("Hello World 123", "hello-world-123"),
        ("Special_Chars-Test", "special-chars-test"),
        ("  Trimmed  Spaces  ", "trimmed-spaces"),
        ("test---entry", "test-entry"),
        ("-leading-trailing-", "leading-trailing"),
        ("Hello 你好 World", "hello-world"),
        ("", ""),
        ("   ", ""),
        ("!@#$%", ""),
    ])
    def test_slugify(self, title, expected):
        """Slugify produces expected output for various inputs."""
        assert core.slugify(title) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate Detection Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDuplicateDetection:
    """Tests for detect_potential_duplicates function."""

    def test_returns_empty_when_no_similar(self):
        """Returns empty list when no entries exceed threshold."""
        searcher = DummySearcher(results=[])

        result = core.detect_potential_duplicates(
            title="Unique Entry",
            content="Unique content.",
            searcher=searcher,
        )

        assert result == []

    def test_finds_high_scoring_matches(self):
        """Returns duplicates above score threshold."""
        searcher = DummySearcher(
            results=[
                SearchResult(
                    path="general/similar.md",
                    title="Similar Entry",
                    snippet="Similar content.",
                    score=0.90,
                    tags=["test"],
                ),
                SearchResult(
                    path="general/unrelated.md",
                    title="Unrelated",
                    snippet="Different.",
                    score=0.50,
                    tags=["test"],
                ),
            ]
        )

        result = core.detect_potential_duplicates(
            title="Similar Entry",
            content="Similar content.",
            searcher=searcher,
        )

        assert len(result) == 1
        assert result[0].path == "general/similar.md"

    def test_respects_limit(self):
        """Returns at most limit duplicates."""
        searcher = DummySearcher(
            results=[
                SearchResult(
                    path=f"general/entry-{i}.md",
                    title=f"Entry {i}",
                    snippet="Content.",
                    score=0.90 - (i * 0.01),
                    tags=["test"],
                )
                for i in range(10)
            ]
        )

        result = core.detect_potential_duplicates(
            title="Entry",
            content="Content.",
            searcher=searcher,
            limit=3,
        )

        assert len(result) <= 3


# ─────────────────────────────────────────────────────────────────────────────
# Generate Descriptions Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerateDescriptions:
    """Tests for generate_descriptions function."""

    @pytest.mark.asyncio
    @pytest.mark.semantic
    async def test_dry_run_previews_without_modifying(self, tmp_kb):
        """dry_run=True previews descriptions without writing."""
        _create_entry(
            tmp_kb / "general" / "no-desc.md",
            "No Description",
            "This is the first sentence. More content follows.",
        )

        original_content = (tmp_kb / "general" / "no-desc.md").read_text()

        results = await core.generate_descriptions(dry_run=True)

        assert len(results) == 1
        assert results[0]["status"] == "preview"
        assert results[0]["description"] is not None

        # File should not be modified
        assert (tmp_kb / "general" / "no-desc.md").read_text() == original_content

    @pytest.mark.asyncio
    @pytest.mark.semantic
    async def test_updates_files_when_not_dry_run(self, tmp_kb):
        """Non-dry-run mode updates files with descriptions."""
        _create_entry(
            tmp_kb / "general" / "update-me.md",
            "Update Me",
            "This entry will get a description. It has content.",
        )

        results = await core.generate_descriptions(dry_run=False)

        assert len(results) == 1
        assert results[0]["status"] == "updated"

        content = (tmp_kb / "general" / "update-me.md").read_text()
        assert "description:" in content

    @pytest.mark.asyncio
    @pytest.mark.semantic
    async def test_skips_entries_with_descriptions(self, tmp_kb):
        """Entries with existing descriptions are skipped."""
        _create_entry(
            tmp_kb / "general" / "has-desc.md",
            "Has Description",
            "Content here.",
            description="Already has one",
        )

        results = await core.generate_descriptions(dry_run=True)

        assert len(results) == 0

    @pytest.mark.asyncio
    @pytest.mark.semantic
    async def test_respects_limit(self, tmp_kb):
        """Limit parameter restricts number of entries processed."""
        for i in range(5):
            _create_entry(
                tmp_kb / "general" / f"entry-{i}.md",
                f"Entry {i}",
                f"Content for entry {i}. This is text.",
            )

        results = await core.generate_descriptions(dry_run=True, limit=2)

        assert len(results) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Resolve Entry By Title Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveEntryByTitle:
    """Tests for resolve_entry_by_title function."""

    def test_exact_title_match(self, tmp_kb, monkeypatch):
        """Finds entry by exact title match."""
        _create_entry(
            tmp_kb / "general" / "python-guide.md",
            "Python Guide",
            "Content",
        )
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        result = core.resolve_entry_by_title("Python Guide", tmp_kb)

        assert result is not None
        assert result.match_type == "exact_title"
        assert result.score == 1.0
        assert "python-guide" in result.path

    def test_case_insensitive_matching(self, tmp_kb, monkeypatch):
        """Matches titles regardless of case."""
        _create_entry(
            tmp_kb / "general" / "python-guide.md",
            "Python Guide",
            "Content",
        )
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        for variant in ["python guide", "PYTHON GUIDE", "PyThOn GuIdE"]:
            result = core.resolve_entry_by_title(variant, tmp_kb)
            assert result is not None, f"Failed for: {variant}"
            assert result.score == 1.0

    def test_returns_none_when_no_match(self, tmp_kb, monkeypatch):
        """Returns None when no match found."""
        monkeypatch.setattr(core, "get_searcher", lambda: DummySearcher())

        result = core.resolve_entry_by_title("Nonexistent Entry", tmp_kb)

        assert result is None

    def test_fuzzy_match_via_search(self, tmp_kb, monkeypatch):
        """Falls back to semantic search for fuzzy matching."""
        _create_entry(
            tmp_kb / "general" / "python-tutorial.md",
            "Python Tutorial",
            "Content",
        )

        fuzzy_searcher = DummySearcher(
            results=[
                SearchResult(
                    path="general/python-tutorial.md",
                    title="Python Tutorial",
                    snippet="Content.",
                    score=0.85,
                    tags=["python"],
                )
            ]
        )
        monkeypatch.setattr(core, "get_searcher", lambda: fuzzy_searcher)

        result = core.resolve_entry_by_title("Python Learning Guide", tmp_kb)

        assert result is not None
        assert result.match_type == "fuzzy"
        assert result.score >= 0.6


# ─────────────────────────────────────────────────────────────────────────────
# List and Find Entry Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestListEntries:
    """Tests for list_entries and find_entries_by_title."""

    @pytest.mark.asyncio
    async def test_list_entries_returns_all(self, tmp_kb):
        """list_entries returns entries from KB."""
        _create_entry(tmp_kb / "general" / "entry1.md", "Entry 1", "Content", tags=["test"])
        _create_entry(tmp_kb / "general" / "entry2.md", "Entry 2", "Content", tags=["test"])

        results = await core.list_entries()

        assert len(results) == 2
        titles = [r["title"] for r in results]
        assert "Entry 1" in titles
        assert "Entry 2" in titles

    @pytest.mark.asyncio
    async def test_list_entries_filters_by_tag(self, tmp_kb):
        """list_entries filters by tag."""
        _create_entry(tmp_kb / "general" / "python.md", "Python", "Content", tags=["python"])
        _create_entry(tmp_kb / "general" / "rust.md", "Rust", "Content", tags=["rust"])

        results = await core.list_entries(tag="python")

        assert len(results) == 1
        assert results[0]["title"] == "Python"

    @pytest.mark.asyncio
    async def test_list_entries_filters_by_directory(self, tmp_kb):
        """list_entries filters by directory."""
        _create_entry(tmp_kb / "development" / "dev.md", "Dev", "Content")
        _create_entry(tmp_kb / "general" / "gen.md", "Gen", "Content")

        results = await core.list_entries(directory="development")

        assert len(results) == 1
        assert results[0]["title"] == "Dev"

    @pytest.mark.asyncio
    async def test_find_entries_by_title_exact(self, tmp_kb):
        """find_entries_by_title finds exact matches."""
        _create_entry(tmp_kb / "general" / "python.md", "Python Guide", "Content")
        _create_entry(tmp_kb / "general" / "python2.md", "Python Tutorial", "Content")

        results = await core.find_entries_by_title("Python Guide", exact=True)

        assert len(results) == 1
        assert results[0]["title"] == "Python Guide"

    @pytest.mark.asyncio
    async def test_find_entries_by_title_partial(self, tmp_kb):
        """find_entries_by_title with exact=False finds partial matches."""
        _create_entry(tmp_kb / "general" / "python.md", "Python Guide", "Content")
        _create_entry(tmp_kb / "general" / "python2.md", "Python Tutorial", "Content")

        results = await core.find_entries_by_title("Python", exact=False)

        assert len(results) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Reindex Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestReindex:
    """Tests for reindex function."""

    @pytest.mark.asyncio
    @pytest.mark.semantic
    async def test_reindex_returns_status(self, tmp_kb):
        """reindex returns IndexStatus."""
        _create_entry(
            tmp_kb / "general" / "test.md",
            "Test",
            "Content",
        )

        status = await core.reindex()

        assert hasattr(status, "kb_files")
        assert hasattr(status, "whoosh_docs")
        assert hasattr(status, "chroma_docs")


# ─────────────────────────────────────────────────────────────────────────────
# Git Helper Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGitHelpers:
    """Tests for git helper functions."""

    def test_get_current_project_returns_string_or_none(self):
        """get_current_project returns project name or None."""
        result = core.get_current_project()
        assert result is None or isinstance(result, str)

    def test_get_current_contributor_returns_string_or_none(self):
        """get_current_contributor returns contributor or None."""
        result = core.get_current_contributor()
        assert result is None or isinstance(result, str)

    def test_get_git_branch_returns_string_or_none(self):
        """get_git_branch returns branch name or None."""
        result = core.get_git_branch()
        assert result is None or isinstance(result, str)

    def test_get_llm_model_reads_env_vars(self, monkeypatch):
        """get_llm_model reads from environment variables."""
        monkeypatch.setenv("LLM_MODEL", "claude-3-opus")

        result = core.get_llm_model()

        assert result == "claude-3-opus"

    def test_get_actor_identity_reads_env_vars(self, monkeypatch):
        """get_actor_identity reads BD_ACTOR or USER."""
        monkeypatch.setenv("BD_ACTOR", "claude-opus")

        result = core.get_actor_identity()

        assert result == "claude-opus"
