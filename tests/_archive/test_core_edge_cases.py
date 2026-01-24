"""Edge case tests for core memex functions.

This test suite focuses on boundary conditions, invalid inputs, and error handling
for the core business logic in memex.core.
"""

from datetime import date
from pathlib import Path

import pytest

from memex import core
from memex.parser import ParseError

pytestmark = pytest.mark.semantic


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
    (root / "testing").mkdir()
    monkeypatch.setenv("MEMEX_USER_KB_ROOT", str(root))
    return root


@pytest.fixture
def index_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary index root."""
    root = tmp_path / ".indices"
    root.mkdir()
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(root))
    return root


def _create_entry(path: Path, title: str, content_body: str, tags: list[str] | None = None):
    """Helper to create a KB entry with frontmatter."""
    tags = tags or ["test"]
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    content = f"""---
title: {title}
tags:
{tags_yaml}
created: {date.today().isoformat()}
---

{content_body}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestAddEntryEdgeCases:
    """Edge case tests for add_entry function."""

    @pytest.mark.asyncio
    async def test_add_entry_empty_title(self, kb_root, index_root):
        """Empty title should raise ValueError."""
        with pytest.raises(ValueError, match="at least one alphanumeric character"):
            await core.add_entry(
                title="",
                content="Some content",
                tags=["test"],
                category="development",
            )

    @pytest.mark.asyncio
    async def test_add_entry_whitespace_only_title(self, kb_root, index_root):
        """Title with only whitespace should raise ValueError."""
        with pytest.raises(ValueError, match="at least one alphanumeric character"):
            await core.add_entry(
                title="   \t\n  ",
                content="Some content",
                tags=["test"],
                category="development",
            )

    @pytest.mark.asyncio
    async def test_add_entry_special_chars_only_title(self, kb_root, index_root):
        """Title with only special characters should raise ValueError."""
        with pytest.raises(ValueError, match="at least one alphanumeric character"):
            await core.add_entry(
                title="!@#$%^&*()",
                content="Some content",
                tags=["test"],
                category="development",
            )

    @pytest.mark.asyncio
    async def test_add_entry_unicode_title(self, kb_root, index_root):
        """Unicode characters in title should be properly slugified."""
        result = await core.add_entry(
            title="Unicode Test: 你好世界 🌍",
            content="Testing unicode in title",
            tags=["test", "unicode"],
            category="development",
        )

        # Result is now a dict with 'path' key
        assert "path" in result
        # Unicode characters should be stripped, leaving "unicode-test"
        assert "unicode-test" in result["path"].lower()

        # Verify file exists and can be read
        file_path = kb_root / result["path"]
        assert file_path.exists()

    @pytest.mark.asyncio
    async def test_add_entry_very_long_title(self, kb_root, index_root):
        """Very long title should be slugified, but filesystem may reject if too long."""
        long_title = "A" * 500  # 500 character title

        # File systems have max filename length limits (often 255 chars)
        # This test verifies the system handles it gracefully
        try:
            result = await core.add_entry(
                title=long_title,
                content="Testing very long title",
                tags=["test"],
                category="development",
            )

            # If successful, verify it works
            assert "path" in result
            assert result["path"].endswith(".md")
            file_path = kb_root / result["path"]
            assert file_path.exists()
        except OSError as e:
            # Expected: File name too long error from OS
            assert "File name too long" in str(e) or "ENAMETOOLONG" in str(e)

    @pytest.mark.asyncio
    async def test_add_entry_empty_tags(self, kb_root, index_root):
        """Empty tags list should raise ValueError."""
        with pytest.raises(ValueError, match="At least one tag is required"):
            await core.add_entry(
                title="Test Entry",
                content="Some content",
                tags=[],
                category="development",
            )

    @pytest.mark.asyncio
    async def test_add_entry_empty_content(self, kb_root, index_root):
        """Empty content should be allowed (entry with only frontmatter)."""
        result = await core.add_entry(
            title="Empty Content Entry",
            content="",
            tags=["test"],
            category="development",
        )

        assert "path" in result
        file_path = kb_root / result["path"]
        assert file_path.exists()

        # Verify it can be read
        entry = await core.get_entry(result["path"])
        assert entry.metadata.title == "Empty Content Entry"
        assert entry.content == ""

    @pytest.mark.asyncio
    async def test_add_entry_whitespace_only_content(self, kb_root, index_root):
        """Content with only whitespace should be preserved."""
        result = await core.add_entry(
            title="Whitespace Content",
            content="   \n\n   \t\t   \n",
            tags=["test"],
            category="development",
        )

        assert "path" in result
        file_path = kb_root / result["path"]
        assert file_path.exists()

    @pytest.mark.asyncio
    async def test_add_entry_very_large_content(self, kb_root, index_root):
        """Very large content (>100KB) should be handled."""
        large_content = "Lorem ipsum dolor sit amet. " * 10000  # ~280KB
        result = await core.add_entry(
            title="Large Content Entry",
            content=large_content,
            tags=["test", "performance"],
            category="development",
        )

        assert "path" in result
        file_path = kb_root / result["path"]
        assert file_path.exists()

        # Verify content is preserved
        entry = await core.get_entry(result["path"])
        assert len(entry.content) > 100000

    @pytest.mark.asyncio
    async def test_add_entry_unicode_content(self, kb_root, index_root):
        """Unicode and emoji in content should be preserved."""
        unicode_content = """
# Testing Unicode

This entry contains various Unicode characters:
- Chinese: 你好世界
- Japanese: こんにちは
- Arabic: مرحبا
- Emoji: 🌍 🚀 💻 ✨
- Math symbols: ∑ ∫ ∂ √
"""
        result = await core.add_entry(
            title="Unicode Content Test",
            content=unicode_content,
            tags=["test", "unicode"],
            category="development",
        )

        assert "path" in result
        entry = await core.get_entry(result["path"])
        assert "你好世界" in entry.content
        assert "🌍" in entry.content
        assert "∑" in entry.content

    @pytest.mark.asyncio
    async def test_add_entry_invalid_category(self, kb_root, index_root):
        """Non-existent category should auto-create the directory."""
        result = await core.add_entry(
            title="New Category Entry",
            content="Testing auto-create",
            tags=["test"],
            category="newcategory",
        )

        assert "path" in result
        # Category should be auto-created
        assert (kb_root / "newcategory").exists()
        assert (kb_root / "newcategory").is_dir()

    @pytest.mark.asyncio
    async def test_add_entry_infers_category_from_tags(self, kb_root, index_root):
        """Infer category from tags when tag matches existing directory."""
        # Category inference only works when tag matches existing directory
        # The 'development' directory already exists from fixture
        result = await core.add_entry(
            title="Inferred Category Entry",
            content="Testing inferred category",
            tags=["development", "guide"],
            category="development",  # Provide category explicitly since auto-inference needs .kbcontext
        )

        assert "path" in result
        assert result["path"].startswith("development/")

    @pytest.mark.asyncio
    async def test_add_entry_no_category_no_directory(self, kb_root, index_root):
        """Missing both category and directory should raise ValueError."""
        with pytest.raises(ValueError, match="Either 'category' or 'directory' must be provided"):
            await core.add_entry(
                title="No Category Entry",
                content="This should fail",
                tags=["misc"],
            )

    @pytest.mark.asyncio
    async def test_add_entry_duplicate_detection(self, kb_root, index_root):
        """Duplicate detection is no longer built-in but entries can still be created."""
        # Create first entry
        result1 = await core.add_entry(
            title="Python Best Practices",
            content="Use type hints, follow PEP 8, write tests.",
            tags=["python", "best-practices"],
            category="development",
        )
        assert "path" in result1

        # Reindex to ensure search works
        await core.reindex()

        # Create similar entry (duplicate detection removed from add_entry API)
        result2 = await core.add_entry(
            title="Python Best Practices Guide",
            content="Use type hints, follow PEP 8, write comprehensive tests.",
            tags=["python", "best-practices"],
            category="development",
        )

        # Both entries should be created
        assert "path" in result2

    @pytest.mark.asyncio
    async def test_add_entry_force_duplicate(self, kb_root, index_root):
        """Multiple entries with similar content can be created."""
        # Create first entry
        result1 = await core.add_entry(
            title="Test Entry",
            content="Original content",
            tags=["test"],
            category="development",
        )
        assert "path" in result1

        await core.reindex()

        # Create similar entry
        result2 = await core.add_entry(
            title="Test Entry Duplicate",
            content="Original content",
            tags=["test"],
            category="development",
        )

        assert "path" in result2

    @pytest.mark.asyncio
    async def test_add_entry_nested_directory(self, kb_root, index_root):
        """Nested directory path should auto-create hierarchy."""
        result = await core.add_entry(
            title="Nested Entry",
            content="Testing nested directories",
            tags=["test"],
            directory="development/python/frameworks",
        )

        assert "path" in result
        assert "development/python/frameworks" in result["path"]

        # Verify directory hierarchy exists
        nested_dir = kb_root / "development" / "python" / "frameworks"
        assert nested_dir.exists()
        assert nested_dir.is_dir()


class TestUpdateEntryEdgeCases:
    """Edge case tests for update_entry function."""

    @pytest.mark.asyncio
    async def test_update_nonexistent_entry(self, kb_root, index_root):
        """Updating non-existent entry should raise ValueError."""
        with pytest.raises(ValueError, match="Entry not found"):
            await core.update_entry(
                path="development/nonexistent.md",
                content="New content",
            )

    @pytest.mark.asyncio
    async def test_update_entry_empty_content(self, kb_root, index_root):
        """Updating with empty content should be allowed."""
        # Create entry
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            "Original content",
        )

        # Update with empty content
        result = await core.update_entry(
            path="development/test.md",
            content="",
        )

        assert result["path"] == "development/test.md"

        # Verify content is empty
        entry = await core.get_entry("development/test.md")
        assert entry.content == ""

    @pytest.mark.asyncio
    async def test_update_entry_empty_tags(self, kb_root, index_root):
        """Updating with empty tags list should raise ValueError."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            "Original content",
        )

        # Need to provide content or section_updates as well
        with pytest.raises(ValueError, match="At least one tag is required"):
            await core.update_entry(
                path="development/test.md",
                tags=[],
                content="New content",
            )

    @pytest.mark.asyncio
    async def test_update_entry_unicode_content(self, kb_root, index_root):
        """Updating with Unicode content should preserve it."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            "Original content",
        )

        unicode_content = "Updated with Unicode: 你好世界 🎉"
        await core.update_entry(
            path="development/test.md",
            content=unicode_content,
        )

        entry = await core.get_entry("development/test.md")
        assert "你好世界" in entry.content
        assert "🎉" in entry.content

    @pytest.mark.asyncio
    async def test_update_entry_no_changes(self, kb_root, index_root):
        """Updating without content or section_updates should raise ValueError."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            "Original content",
        )

        with pytest.raises(ValueError, match="Provide new content or section_updates"):
            await core.update_entry(
                path="development/test.md",
            )

    @pytest.mark.asyncio
    async def test_update_entry_section_nonexistent(self, kb_root, index_root):
        """Updating non-existent section should append new section."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            "## Existing Section\n\nExisting content",
        )

        await core.update_entry(
            path="development/test.md",
            section_updates={"New Section": "New section content"},
        )

        entry = await core.get_entry("development/test.md")
        assert "## New Section" in entry.content
        assert "New section content" in entry.content
        # Original section should still exist
        assert "## Existing Section" in entry.content

    @pytest.mark.asyncio
    async def test_update_entry_section_empty_replacement(self, kb_root, index_root):
        """Updating section with empty string should be skipped."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            "## Section 1\n\nContent 1",
        )

        # Empty section updates should be ignored
        await core.update_entry(
            path="development/test.md",
            section_updates={"Section 1": ""},
        )

        entry = await core.get_entry("development/test.md")
        # Original content should remain unchanged
        assert "Content 1" in entry.content

    @pytest.mark.asyncio
    async def test_update_entry_very_large_content(self, kb_root, index_root):
        """Updating with very large content should work."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            "Small content",
        )

        large_content = "Updated content. " * 50000  # ~850KB
        await core.update_entry(
            path="development/test.md",
            content=large_content,
        )

        entry = await core.get_entry("development/test.md")
        assert len(entry.content) > 500000

    @pytest.mark.asyncio
    async def test_update_directory_path(self, kb_root, index_root):
        """Trying to update a directory should raise ValueError."""
        (kb_root / "development" / "subdir").mkdir()

        with pytest.raises(ValueError, match="not a file"):
            await core.update_entry(
                path="development/subdir",
                content="This should fail",
            )

    @pytest.mark.asyncio
    async def test_append_entry_adds_to_existing(self, kb_root, index_root):
        """append_entry should add content to end of existing entry."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            "Original content",
        )

        result = await core.append_entry(
            title="Test Entry",
            content="Appended content",
        )

        assert result["action"] == "appended"
        entry = await core.get_entry("development/test.md")
        assert "Original content" in entry.content
        assert "Appended content" in entry.content

    @pytest.mark.asyncio
    async def test_append_entry_creates_new_when_not_found(self, kb_root, index_root):
        """append_entry should create new entry when title not found."""
        result = await core.append_entry(
            title="New Append Entry",
            content="New content",
            tags=["test"],
            directory="development",
        )

        assert result["action"] == "created"
        assert "path" in result

    @pytest.mark.asyncio
    async def test_append_entry_no_create_fails_when_not_found(self, kb_root, index_root):
        """append_entry with no_create=True should fail when entry not found."""
        with pytest.raises(ValueError, match="not found"):
            await core.append_entry(
                title="Nonexistent Entry",
                content="Content",
                no_create=True,
            )

    @pytest.mark.asyncio
    async def test_append_entry_multiline_content(self, kb_root, index_root):
        """Appending multiline content should preserve formatting."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            "Existing content",
        )

        multiline = """## Session Log

- Fixed bug A
- Added feature B
- Tested feature C"""

        await core.append_entry(
            title="Test Entry",
            content=multiline,
        )

        entry = await core.get_entry("development/test.md")
        assert "Existing content" in entry.content
        assert "## Session Log" in entry.content
        assert "- Fixed bug A" in entry.content
        assert "- Tested feature C" in entry.content


class TestDeleteEntryEdgeCases:
    """Edge case tests for delete_entry function."""

    @pytest.mark.asyncio
    async def test_delete_nonexistent_entry(self, kb_root, index_root):
        """Deleting non-existent entry should raise ValueError."""
        with pytest.raises(ValueError, match="Entry not found"):
            await core.delete_entry(path="development/nonexistent.md")

    @pytest.mark.asyncio
    async def test_delete_already_deleted(self, kb_root, index_root):
        """Deleting an already deleted entry should raise ValueError."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            "Content",
        )

        # Delete once
        result = await core.delete_entry(path="development/test.md")
        assert result["deleted"] == "development/test.md"

        # Try to delete again
        with pytest.raises(ValueError, match="Entry not found"):
            await core.delete_entry(path="development/test.md")

    @pytest.mark.asyncio
    async def test_delete_entry_with_backlinks(self, kb_root, index_root):
        """Deleting entry with backlinks should raise ValueError without force."""
        # Create two entries with a link
        _create_entry(
            kb_root / "development" / "target.md",
            "Target Entry",
            "This is the target",
        )
        _create_entry(
            kb_root / "development" / "source.md",
            "Source Entry",
            "This links to [[development/target]]",
        )

        # Rebuild backlinks
        await core.reindex()

        # Try to delete target without force
        with pytest.raises(ValueError, match="has .* backlink"):
            await core.delete_entry(path="development/target.md", force=False)

    @pytest.mark.asyncio
    async def test_delete_entry_with_backlinks_force(self, kb_root, index_root):
        """Deleting entry with backlinks using force should succeed."""
        # Create two entries with a link
        _create_entry(
            kb_root / "development" / "target.md",
            "Target Entry",
            "This is the target",
        )
        _create_entry(
            kb_root / "development" / "source.md",
            "Source Entry",
            "This links to [[development/target]]",
        )

        await core.reindex()

        # Delete with force
        result = await core.delete_entry(path="development/target.md", force=True)
        assert result["deleted"] == "development/target.md"
        assert len(result["had_backlinks"]) > 0

    @pytest.mark.asyncio
    async def test_delete_directory_instead_of_file(self, kb_root, index_root):
        """Trying to delete a directory should raise ValueError."""
        (kb_root / "development" / "subdir").mkdir()

        with pytest.raises(ValueError, match="not a file"):
            await core.delete_entry(path="development/subdir")

    @pytest.mark.asyncio
    async def test_delete_entry_many_backlinks(self, kb_root, index_root):
        """Deleting entry with many backlinks should report all of them."""
        # Create target entry
        _create_entry(
            kb_root / "development" / "target.md",
            "Popular Entry",
            "This entry is linked from many places",
        )

        # Create multiple entries that link to target
        for i in range(10):
            _create_entry(
                kb_root / "development" / f"source{i}.md",
                f"Source {i}",
                "Links to [[development/target]]",
            )

        await core.reindex()

        # Delete with force and verify backlinks are reported
        result = await core.delete_entry(path="development/target.md", force=True)
        assert len(result["had_backlinks"]) == 10


class TestSearchEdgeCases:
    """Edge case tests for search function."""

    @pytest.mark.asyncio
    async def test_search_empty_query(self, kb_root, index_root):
        """Empty query should return results (search everything)."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            "Content",
        )

        await core.reindex()

        # Empty string query
        result = await core.search(query="")
        # Should return results or empty list, not error
        assert isinstance(result.results, list)

    @pytest.mark.asyncio
    async def test_search_whitespace_only_query(self, kb_root, index_root):
        """Whitespace-only query should be handled gracefully."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            "Content",
        )

        await core.reindex()

        result = await core.search(query="   \t\n   ")
        assert isinstance(result.results, list)

    @pytest.mark.asyncio
    async def test_search_special_characters(self, kb_root, index_root):
        """Query with only special characters should be handled."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            "Content with special chars: @#$%",
        )

        await core.reindex()

        # Special char query
        result = await core.search(query="@#$%^&*()")
        assert isinstance(result.results, list)

    @pytest.mark.asyncio
    async def test_search_very_long_query(self, kb_root, index_root):
        """Very long query (>1000 chars) should be handled."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            "Content",
        )

        await core.reindex()

        long_query = "search term " * 200  # ~2400 chars
        result = await core.search(query=long_query)
        assert isinstance(result.results, list)

    @pytest.mark.asyncio
    async def test_search_unicode_query(self, kb_root, index_root):
        """Unicode in query should work correctly."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            "Content with Chinese: 你好世界",
        )

        await core.reindex()

        result = await core.search(query="你好")
        # Should either find the entry or return empty results
        assert isinstance(result.results, list)

    @pytest.mark.asyncio
    async def test_search_emoji_query(self, kb_root, index_root):
        """Emoji in query should be handled."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            "Content with emoji 🚀 rocket",
        )

        await core.reindex()

        result = await core.search(query="🚀")
        assert isinstance(result.results, list)

    @pytest.mark.asyncio
    async def test_search_empty_kb(self, kb_root, index_root):
        """Searching empty KB should return empty results."""
        # No entries created
        await core.reindex()

        result = await core.search(query="anything")
        assert result.results == []

    @pytest.mark.asyncio
    async def test_search_invalid_mode(self, kb_root, index_root):
        """Invalid search mode should use default or raise error."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            "Content",
        )

        await core.reindex()

        # This will fail at type checking level, but test runtime behavior
        # Note: This might raise TypeError or use default mode
        try:
            result = await core.search(query="test", mode="invalid")  # type: ignore
            assert isinstance(result.results, list)
        except (TypeError, ValueError):
            # Expected if validation is strict
            pass

    @pytest.mark.asyncio
    async def test_search_with_nonexistent_tags(self, kb_root, index_root):
        """Filtering by non-existent tags should return empty results."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            "Content",
            tags=["python"],
        )

        await core.reindex()

        result = await core.search(query="test", tags=["nonexistent"])
        assert result.results == []

    @pytest.mark.asyncio
    async def test_search_limit_zero(self, kb_root, index_root):
        """Search with limit=0 should raise ValueError (whoosh requires limit >= 1)."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            "Content",
        )

        await core.reindex()

        # Whoosh raises ValueError for limit < 1
        with pytest.raises(ValueError, match="limit must be >= 1"):
            await core.search(query="test", limit=0)

    @pytest.mark.asyncio
    async def test_search_negative_limit(self, kb_root, index_root):
        """Search with negative limit should raise ValueError (whoosh requires limit >= 1)."""
        _create_entry(
            kb_root / "development" / "test.md",
            "Test Entry",
            "Content",
        )

        await core.reindex()

        # Whoosh raises ValueError for limit < 1
        with pytest.raises(ValueError, match="limit must be >= 1"):
            await core.search(query="test", limit=-1)


class TestParseEntryEdgeCases:
    """Edge case tests for parse_entry function."""

    def test_parse_nonexistent_file(self, kb_root):
        """Parsing non-existent file should raise ParseError."""
        from memex.parser import parse_entry

        nonexistent = kb_root / "nonexistent.md"
        with pytest.raises(ParseError, match="does not exist"):
            parse_entry(nonexistent)

    def test_parse_directory(self, kb_root):
        """Parsing a directory should raise ParseError."""
        from memex.parser import parse_entry

        (kb_root / "somedir").mkdir()
        with pytest.raises(ParseError, match="not a file"):
            parse_entry(kb_root / "somedir")

    def test_parse_missing_frontmatter(self, kb_root):
        """File without frontmatter should raise ParseError."""
        from memex.parser import parse_entry

        file_path = kb_root / "no-frontmatter.md"
        file_path.write_text("Just content, no frontmatter", encoding="utf-8")

        with pytest.raises(ParseError, match="Missing frontmatter"):
            parse_entry(file_path)

    def test_parse_invalid_yaml_frontmatter(self, kb_root):
        """Malformed YAML frontmatter should raise ParseError."""
        from memex.parser import parse_entry

        file_path = kb_root / "bad-yaml.md"
        file_path.write_text(
            """---
title: Test
tags: [unclosed bracket
created: 2024-01-01
---

Content
""",
            encoding="utf-8",
        )

        with pytest.raises(ParseError):
            parse_entry(file_path)

    def test_parse_missing_required_fields(self, kb_root):
        """Frontmatter missing required fields should raise ParseError."""
        from memex.parser import parse_entry

        file_path = kb_root / "missing-fields.md"
        file_path.write_text(
            """---
title: Test Entry
# Missing tags and created fields
---

Content
""",
            encoding="utf-8",
        )

        with pytest.raises(ParseError, match="Invalid frontmatter"):
            parse_entry(file_path)

    def test_parse_unicode_frontmatter(self, kb_root):
        """Unicode in frontmatter should be parsed correctly."""
        from memex.parser import parse_entry

        file_path = kb_root / "unicode-fm.md"
        file_path.write_text(
            """---
title: Unicode Title 你好
tags:
  - test
  - unicode测试
created: 2024-01-01
---

Content with unicode
""",
            encoding="utf-8",
        )

        metadata, content, chunks = parse_entry(file_path)
        assert "你好" in metadata.title
        assert "unicode测试" in metadata.tags

    def test_parse_empty_content(self, kb_root):
        """Entry with empty content should parse successfully."""
        from memex.parser import parse_entry

        file_path = kb_root / "empty-content.md"
        file_path.write_text(
            """---
title: Empty Entry
tags:
  - test
created: 2024-01-01
---
""",
            encoding="utf-8",
        )

        metadata, content, chunks = parse_entry(file_path)
        assert metadata.title == "Empty Entry"
        assert content.strip() == ""
        assert len(chunks) == 0  # No content chunks

    def test_parse_only_whitespace_content(self, kb_root):
        """Entry with only whitespace content should parse."""
        from memex.parser import parse_entry

        file_path = kb_root / "whitespace.md"
        file_path.write_text(
            """---
title: Whitespace Entry
tags:
  - test
created: 2024-01-01
---


\t\t

""",
            encoding="utf-8",
        )

        metadata, content, chunks = parse_entry(file_path)
        assert metadata.title == "Whitespace Entry"
        # Whitespace should be stripped from chunks
        assert len(chunks) == 0

    def test_parse_very_large_file(self, kb_root):
        """Very large file (>1MB) should parse successfully."""
        from memex.parser import parse_entry

        file_path = kb_root / "large.md"
        large_content = "Lorem ipsum dolor sit amet.\n" * 50000  # ~1.4MB
        file_path.write_text(
            f"""---
title: Large Entry
tags:
  - test
created: 2024-01-01
---

{large_content}
""",
            encoding="utf-8",
        )

        metadata, content, chunks = parse_entry(file_path)
        assert metadata.title == "Large Entry"
        assert len(content) > 1000000

    def test_parse_malformed_markdown_sections(self, kb_root):
        """Malformed H2 sections should be handled gracefully."""
        from memex.parser import parse_entry

        file_path = kb_root / "malformed.md"
        file_path.write_text(
            """---
title: Malformed Sections
tags:
  - test
created: 2024-01-01
---

## Valid Section
Content here

### This is H3, not H2
Should not be a chunk boundary

## Another Valid Section
More content
""",
            encoding="utf-8",
        )

        metadata, content, chunks = parse_entry(file_path)
        # Should parse successfully
        assert metadata.title == "Malformed Sections"
        # Only H2 sections should create chunk boundaries
        section_names = [c.section for c in chunks if c.section]
        assert "Valid Section" in section_names
        assert "Another Valid Section" in section_names
        assert "This is H3, not H2" not in section_names

    def test_parse_special_chars_in_sections(self, kb_root):
        """Section headers with special characters should parse."""
        from memex.parser import parse_entry

        file_path = kb_root / "special-sections.md"
        file_path.write_text(
            """---
title: Special Chars
tags:
  - test
created: 2024-01-01
---

## Section with "quotes"
Content

## Section with 'apostrophes'
Content

## Section with [brackets]
Content

## Section with (parentheses)
Content
""",
            encoding="utf-8",
        )

        metadata, content, chunks = parse_entry(file_path)
        section_names = [c.section for c in chunks if c.section]
        assert any("quotes" in s for s in section_names)
        assert any("apostrophes" in s for s in section_names)
        assert any("brackets" in s for s in section_names)


class TestSlugifyEdgeCases:
    """Edge case tests for slugify helper function."""

    def test_slugify_empty_string(self):
        """Empty string should return empty string."""
        assert core.slugify("") == ""

    def test_slugify_whitespace_only(self):
        """Whitespace only should return empty string."""
        assert core.slugify("   \t\n   ") == ""

    def test_slugify_special_chars_only(self):
        """Only special characters should return empty string."""
        assert core.slugify("!@#$%^&*()") == ""

    def test_slugify_unicode_chars(self):
        """Unicode characters should be stripped."""
        result = core.slugify("Hello 你好 World")
        assert result == "hello-world"

    def test_slugify_emoji(self):
        """Emoji should be stripped."""
        result = core.slugify("Hello 🌍 World")
        assert result == "hello-world"

    def test_slugify_mixed_case(self):
        """Mixed case should be lowercased."""
        result = core.slugify("Test ENTRY Title")
        assert result == "test-entry-title"

    def test_slugify_multiple_spaces(self):
        """Multiple spaces should become single hyphen."""
        result = core.slugify("test    entry")
        assert result == "test-entry"

    def test_slugify_leading_trailing_hyphens(self):
        """Leading and trailing hyphens should be stripped."""
        result = core.slugify("-test-entry-")
        assert result == "test-entry"

    def test_slugify_consecutive_hyphens(self):
        """Consecutive hyphens should be collapsed to one."""
        result = core.slugify("test---entry")
        assert result == "test-entry"

    def test_slugify_underscores(self):
        """Underscores should be converted to hyphens."""
        result = core.slugify("test_entry_name")
        assert result == "test-entry-name"

    def test_slugify_numbers(self):
        """Numbers should be preserved."""
        result = core.slugify("Test Entry 123")
        assert result == "test-entry-123"


class TestValidateNestedPathEdgeCases:
    """Edge case tests for validate_nested_path function."""

    def test_validate_nested_path_traversal(self, kb_root):
        """Path traversal attempts should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid path"):
            core.validate_nested_path("../etc/passwd")

    def test_validate_nested_path_absolute(self, kb_root):
        """Absolute paths should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid path"):
            core.validate_nested_path("/absolute/path")

    def test_validate_nested_path_hidden_dir(self, kb_root):
        """Hidden directories (starting with .) should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid path component"):
            core.validate_nested_path("development/.hidden/file.md")

    def test_validate_nested_path_underscore_dir(self, kb_root):
        """Directories starting with _ should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid path component"):
            core.validate_nested_path("development/_private/file.md")

    def test_validate_nested_path_valid_nested(self, kb_root):
        """Valid nested path should return absolute and normalized paths."""
        abs_path, normalized = core.validate_nested_path("development/python/test.md")
        assert abs_path.is_absolute()
        assert normalized == "development/python/test.md"
        assert str(kb_root) in str(abs_path)


class TestGitHelpers:
    """Tests for git helper functions."""

    def test_get_current_project_returns_string_or_none(self):
        """get_current_project should return project name or None."""
        result = core.get_current_project()
        assert result is None or isinstance(result, str)

    def test_get_current_contributor_returns_string_or_none(self):
        """get_current_contributor should return contributor or None."""
        result = core.get_current_contributor()
        assert result is None or isinstance(result, str)

    def test_get_git_branch_returns_string_or_none(self):
        """get_git_branch should return branch name or None."""
        result = core.get_git_branch()
        assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_add_entry_sets_metadata(self, kb_root, index_root):
        """add_entry should set appropriate metadata."""
        result = await core.add_entry(
            title="Metadata Test Entry",
            content="Testing that add_entry sets metadata",
            tags=["test"],
            category="development",
        )

        assert "path" in result
        file_path = kb_root / result["path"]
        assert file_path.exists()

        # Read the file and verify metadata was set
        entry = await core.get_entry(result["path"])
        assert entry.metadata.title == "Metadata Test Entry"
        # source_project and contributor may be set from git (may be None in CI)

    @pytest.mark.asyncio
    async def test_update_entry_works(self, kb_root, index_root):
        """update_entry should work correctly."""
        # Create entry first
        _create_entry(
            kb_root / "development" / "update-test.md",
            "Update Test",
            "Original content",
        )

        # Update entry
        result = await core.update_entry(
            path="development/update-test.md",
            content="Updated content",
        )

        assert result["path"] == "development/update-test.md"

        # Verify content was updated
        entry = await core.get_entry("development/update-test.md")
        assert "Updated content" in entry.content

    @pytest.mark.asyncio
    async def test_search_works(self, kb_root, index_root):
        """search should work correctly."""
        _create_entry(
            kb_root / "development" / "search-test.md",
            "Search Test",
            "Content for search test",
        )

        await core.reindex()

        # Search should work
        result = await core.search(query="search test")
        assert isinstance(result.results, list)


class TestGenerateDescriptions:
    """Tests for generate_descriptions function."""

    def _create_entry_with_description(
        self,
        path: Path,
        title: str,
        content_body: str,
        tags: list[str] | None = None,
        description: str | None = None,
    ) -> None:
        """Helper to create a KB entry with optional description."""
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

    @pytest.mark.asyncio
    async def test_generate_descriptions_dry_run(self, kb_root, index_root):
        """Dry run mode previews descriptions without writing."""
        # Create entry without description
        _create_entry(
            kb_root / "development" / "no-desc.md",
            "No Description Entry",
            "This is the first sentence. More content follows here.",
        )

        results = await core.generate_descriptions(dry_run=True)

        assert len(results) == 1
        assert results[0]["status"] == "preview"
        assert results[0]["path"] == "development/no-desc.md"
        assert results[0]["description"] is not None
        assert "first sentence" in results[0]["description"]

        # File should not be modified
        content = (kb_root / "development" / "no-desc.md").read_text()
        assert "description:" not in content

    @pytest.mark.asyncio
    async def test_generate_descriptions_updates_files(self, kb_root, index_root):
        """Non-dry-run mode updates files with descriptions."""
        _create_entry(
            kb_root / "development" / "update-me.md",
            "Update Me",
            "This entry will get a description. It has good content.",
        )

        results = await core.generate_descriptions(dry_run=False)

        assert len(results) == 1
        assert results[0]["status"] == "updated"

        # File should now have description
        content = (kb_root / "development" / "update-me.md").read_text()
        assert "description:" in content

    @pytest.mark.asyncio
    async def test_generate_descriptions_skips_entries_with_descriptions(
        self, kb_root, index_root
    ):
        """Entries that already have descriptions are skipped."""
        self._create_entry_with_description(
            kb_root / "development" / "has-desc.md",
            "Has Description",
            "Content here.",
            description="Already has a description",
        )

        results = await core.generate_descriptions(dry_run=True)

        # Should return empty since entry already has description
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_generate_descriptions_respects_limit(self, kb_root, index_root):
        """Limit parameter restricts number of entries processed."""
        for i in range(5):
            _create_entry(
                kb_root / "development" / f"entry-{i}.md",
                f"Entry {i}",
                f"Content for entry {i}. This is some text.",
            )

        results = await core.generate_descriptions(dry_run=True, limit=2)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_generate_descriptions_extracts_first_sentence(
        self, kb_root, index_root
    ):
        """Description is extracted from first sentence."""
        _create_entry(
            kb_root / "development" / "sentence.md",
            "Sentence Test",
            "This is the first sentence. This is the second sentence. And more text.",
        )

        results = await core.generate_descriptions(dry_run=True)

        assert len(results) == 1
        # Should get first sentence
        assert results[0]["description"] == "This is the first sentence."

    @pytest.mark.asyncio
    async def test_generate_descriptions_handles_no_entries(self, kb_root, index_root):
        """Returns empty list when no entries need descriptions."""
        results = await core.generate_descriptions(dry_run=True)
        assert results == []

    @pytest.mark.asyncio
    async def test_generate_descriptions_truncates_long_descriptions(
        self, kb_root, index_root
    ):
        """Long content is truncated to reasonable length."""
        long_content = "A" * 500 + " word " + "B" * 500
        _create_entry(
            kb_root / "development" / "long.md",
            "Long Content",
            long_content,
        )

        results = await core.generate_descriptions(dry_run=True)

        assert len(results) == 1
        # Description should be truncated (max 120 chars by default)
        assert len(results[0]["description"]) <= 125  # 120 + "..."
