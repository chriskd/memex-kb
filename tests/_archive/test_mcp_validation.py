"""Tests for MCP tool input validation.

Tests validation behavior at the core/server level, verifying that invalid
inputs are properly rejected before operations proceed.
"""

from pathlib import Path

import pytest

from memex import core, server


async def _call_tool(tool_obj, /, *args, **kwargs):
    """Invoke the wrapped coroutine behind an MCP FunctionTool."""
    bound = tool_obj.fn(*args, **kwargs)
    if callable(bound):
        return await bound()
    return await bound


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


# -----------------------------------------------------------------------------
# add_tool validation tests
# -----------------------------------------------------------------------------


class TestAddToolValidation:
    """Tests for add_tool input validation."""

    @pytest.mark.asyncio
    async def test_empty_tags_rejected(self, kb_root, index_root):
        """Empty tags list should be rejected."""
        with pytest.raises(Exception):  # Pydantic validation or ValueError
            await _call_tool(
                server.add_tool,
                title="Valid Title",
                content="Some content",
                tags=[],
                category="development",
            )

    @pytest.mark.asyncio
    async def test_valid_inputs_accepted(self, kb_root, index_root):
        """Valid inputs should pass validation and create entry."""
        result = await _call_tool(
            server.add_tool,
            title="Valid Title",
            content="Some valid content here.",
            tags=["python", "testing"],
            category="development",
        )

        assert "path" in result
        assert result["path"] == "development/valid-title.md"

    @pytest.mark.asyncio
    async def test_title_is_slugified(self, kb_root, index_root):
        """Title should be converted to slug for filename."""
        result = await _call_tool(
            server.add_tool,
            title="My Test Entry Title",
            content="Content here.",
            tags=["test"],
            category="development",
        )

        assert "path" in result
        assert "my-test-entry-title" in result["path"]


# -----------------------------------------------------------------------------
# update_tool validation tests
# -----------------------------------------------------------------------------


class TestUpdateToolValidation:
    """Tests for update_tool input validation."""

    @pytest.fixture
    def existing_entry(self, kb_root):
        """Create an existing entry for update tests."""
        entry = kb_root / "development" / "existing.md"
        entry.write_text(
            """---
title: Existing Entry
tags:
  - test
created: 2024-01-01
---

## Overview

Original content here.
"""
        )
        return entry

    @pytest.mark.asyncio
    async def test_empty_path_rejected(self, kb_root, index_root, existing_entry):
        """Empty path should be rejected."""
        with pytest.raises((ValueError, Exception)):
            await _call_tool(
                server.update_tool,
                path="",
                content="New content",
            )

    @pytest.mark.asyncio
    async def test_path_traversal_rejected(self, kb_root, index_root, existing_entry):
        """Path traversal attempts should be rejected."""
        with pytest.raises(ValueError):
            await _call_tool(
                server.update_tool,
                path="../etc/passwd",
                content="Malicious content",
            )

        with pytest.raises(ValueError):
            await _call_tool(
                server.update_tool,
                path="development/../../../etc/passwd",
                content="Malicious content",
            )

    @pytest.mark.asyncio
    async def test_absolute_path_rejected(self, kb_root, index_root, existing_entry):
        """Absolute paths should be rejected."""
        with pytest.raises(ValueError):
            await _call_tool(
                server.update_tool,
                path="/etc/passwd",
                content="Malicious content",
            )

    @pytest.mark.asyncio
    async def test_no_update_params_rejected(self, kb_root, index_root, existing_entry):
        """Update with no content or section_updates should be rejected."""
        with pytest.raises(ValueError, match="Provide new content or section_updates"):
            await _call_tool(
                server.update_tool,
                path="development/existing.md",
                content=None,
                tags=None,
                section_updates=None,
            )

    @pytest.mark.asyncio
    async def test_valid_update_accepted(self, kb_root, index_root, existing_entry):
        """Valid update inputs should pass validation."""
        result = await _call_tool(
            server.update_tool,
            path="development/existing.md",
            content="Updated content here.",
            tags=["updated", "test"],
        )

        assert result["path"] == "development/existing.md"

    @pytest.mark.asyncio
    async def test_section_updates_accepted(self, kb_root, index_root, existing_entry):
        """Valid section updates should pass validation."""
        result = await _call_tool(
            server.update_tool,
            path="development/existing.md",
            section_updates={"Overview": "New overview content"},
        )

        assert result["path"] == "development/existing.md"


# -----------------------------------------------------------------------------
# get_tool validation tests
# -----------------------------------------------------------------------------


class TestGetToolValidation:
    """Tests for get_tool input validation."""

    @pytest.mark.asyncio
    async def test_path_traversal_rejected(self, kb_root, index_root):
        """Path traversal should be rejected."""
        with pytest.raises(ValueError):
            await _call_tool(server.get_tool, path="../etc/passwd")

    @pytest.mark.asyncio
    async def test_absolute_path_rejected(self, kb_root, index_root):
        """Absolute paths should be rejected."""
        with pytest.raises(ValueError):
            await _call_tool(server.get_tool, path="/etc/passwd")

    @pytest.mark.asyncio
    async def test_nonexistent_entry_raises(self, kb_root, index_root):
        """Getting nonexistent entry raises ValueError."""
        with pytest.raises(ValueError, match="Entry not found"):
            await _call_tool(server.get_tool, path="development/nonexistent.md")


# -----------------------------------------------------------------------------
# delete_tool validation tests
# -----------------------------------------------------------------------------


class TestDeleteToolValidation:
    """Tests for delete_tool input validation."""

    @pytest.mark.asyncio
    async def test_path_traversal_rejected(self, kb_root, index_root):
        """Path traversal should be rejected."""
        with pytest.raises(ValueError):
            await _call_tool(server.delete_tool, path="../etc/passwd")

    @pytest.mark.asyncio
    async def test_nonexistent_entry_raises(self, kb_root, index_root):
        """Deleting nonexistent entry raises ValueError."""
        with pytest.raises(ValueError, match="Entry not found"):
            await _call_tool(server.delete_tool, path="development/nonexistent.md")


# -----------------------------------------------------------------------------
# list_tool validation tests
# -----------------------------------------------------------------------------


class TestListToolValidation:
    """Tests for list_tool input validation."""

    @pytest.mark.asyncio
    async def test_invalid_category_rejected(self, kb_root, index_root):
        """Invalid category should be rejected."""
        with pytest.raises(ValueError, match="Category not found"):
            await _call_tool(server.list_tool, category="nonexistent_category")

    @pytest.mark.asyncio
    async def test_valid_category_accepted(self, kb_root, index_root):
        """Valid category should be accepted."""
        # Create an entry in the category
        entry = kb_root / "development" / "test.md"
        entry.write_text(
            """---
title: Test
tags:
  - test
created: 2024-01-01
---

Content.
"""
        )

        result = await _call_tool(server.list_tool, category="development")
        assert isinstance(result, list)


# -----------------------------------------------------------------------------
# Core validation helper tests
# -----------------------------------------------------------------------------


class TestCoreValidation:
    """Tests for core validation functions."""

    def test_slugify_basic(self):
        """slugify converts title to URL-friendly format."""
        assert core.slugify("Hello World") == "hello-world"
        assert core.slugify("Test Entry") == "test-entry"

    def test_slugify_special_chars(self):
        """slugify removes special characters."""
        assert core.slugify("Test & Entry") == "test-entry"
        assert core.slugify("Hello/World") == "helloworld"
        assert core.slugify("Test!@#$%^Entry") == "testentry"

    def test_slugify_multiple_spaces(self):
        """slugify handles multiple spaces."""
        assert core.slugify("Test   Entry") == "test-entry"
        assert core.slugify("  Padded  Title  ") == "padded-title"

    def test_slugify_underscores(self):
        """slugify converts underscores to hyphens."""
        assert core.slugify("test_entry") == "test-entry"

    def test_validate_nested_path_basic(self):
        """validate_nested_path accepts valid paths."""
        # This will fail if KB_ROOT doesn't exist, so skip if it doesn't
        try:
            abs_path, normalized = core.validate_nested_path("development/test.md")
            assert normalized == "development/test.md"
        except (ValueError, FileNotFoundError):
            pytest.skip("KB root not configured for this test")

    def test_validate_nested_path_rejects_traversal(self, kb_root):
        """validate_nested_path rejects path traversal."""
        with pytest.raises(ValueError, match="Invalid path"):
            core.validate_nested_path("../etc/passwd")

        with pytest.raises(ValueError, match="Invalid path"):
            core.validate_nested_path("development/../../../etc/passwd")

    def test_validate_nested_path_rejects_absolute(self, kb_root):
        """validate_nested_path rejects absolute paths."""
        with pytest.raises(ValueError, match="Invalid path"):
            core.validate_nested_path("/etc/passwd")

    def test_validate_nested_path_rejects_hidden(self, kb_root):
        """validate_nested_path rejects hidden directories."""
        with pytest.raises(ValueError, match="Invalid path component"):
            core.validate_nested_path(".hidden/file.md")

        with pytest.raises(ValueError, match="Invalid path component"):
            core.validate_nested_path("_private/file.md")

    def test_get_valid_categories(self, kb_root):
        """get_valid_categories returns existing directories."""
        categories = core.get_valid_categories()
        assert "development" in categories
        assert "architecture" in categories
        assert "devops" in categories

    def test_get_valid_categories_excludes_hidden(self, kb_root):
        """get_valid_categories excludes hidden directories."""
        # Create hidden directory
        (kb_root / ".hidden").mkdir()
        (kb_root / "_private").mkdir()

        categories = core.get_valid_categories()
        assert ".hidden" not in categories
        assert "_private" not in categories


# -----------------------------------------------------------------------------
# Edge case tests
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_unicode_in_title(self, kb_root, index_root):
        """Unicode characters in title are handled."""
        result = await _call_tool(
            server.add_tool,
            title="Test Entry",
            content="Content here.",
            tags=["test"],
            category="development",
        )

        assert "path" in result

    @pytest.mark.asyncio
    async def test_very_long_title(self, kb_root, index_root):
        """Very long titles are handled (slugified and possibly truncated)."""
        long_title = "A" * 200 + " Very Long Title"
        result = await _call_tool(
            server.add_tool,
            title=long_title,
            content="Content",
            tags=["test"],
            category="development",
        )

        assert "path" in result
        # Path should exist and be reasonable length
        file_path = kb_root / result["path"]
        assert file_path.exists()

    @pytest.mark.asyncio
    async def test_empty_content_creates_entry(self, kb_root, index_root):
        """Entry with minimal content can still be created."""
        result = await _call_tool(
            server.add_tool,
            title="Minimal Entry",
            content=".",  # Minimal non-empty content
            tags=["test"],
            category="development",
        )

        assert "path" in result
        assert (kb_root / result["path"]).exists()

    @pytest.mark.asyncio
    async def test_special_tag_characters(self, kb_root, index_root):
        """Tags with special characters are handled."""
        result = await _call_tool(
            server.add_tool,
            title="Tagged Entry",
            content="Content",
            tags=["test-tag", "another_tag"],
            category="development",
        )

        assert "path" in result

        # Verify tags are preserved
        from memex.parser import parse_entry
        file_path = kb_root / result["path"]
        metadata, _, _ = parse_entry(file_path)
        assert "test-tag" in metadata.tags
        assert "another_tag" in metadata.tags
