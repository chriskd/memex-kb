"""Tests for MCP tool input validation."""

from pathlib import Path

import pytest

from memex import core, server
from memex.server import ValidationError


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
def dummy_searcher(monkeypatch):
    """Mock searcher to avoid actual indexing during tests."""

    class DummySearcher:
        def __init__(self):
            self.deleted = []
            self.indexed = []

        def delete_document(self, path: str) -> None:
            self.deleted.append(path)

        def index_chunks(self, chunks):
            self.indexed.append(chunks)

        def status(self):
            class Status:
                kb_files = 1
                whoosh_docs = 1
                chroma_docs = 1

            return Status()

        def preload(self):
            return None

        def search(self, query, limit=10, mode="hybrid", project_context=None, kb_context=None):
            return []

    dummy = DummySearcher()
    monkeypatch.setattr(core, "get_searcher", lambda: dummy)
    monkeypatch.setattr(core, "rebuild_backlink_cache", lambda *_args, **_kwargs: None)
    return dummy


# ─────────────────────────────────────────────────────────────────────────────
# add_tool validation tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAddToolValidation:
    """Tests for add_tool input validation."""

    @pytest.mark.asyncio
    async def test_empty_title_rejected(self, kb_root, dummy_searcher):
        """Empty title should be rejected."""
        with pytest.raises(ValidationError, match="title cannot be empty"):
            await _call_tool(
                server.add_tool,
                title="",
                content="Some content",
                tags=["test"],
                category="development",
            )

    @pytest.mark.asyncio
    async def test_whitespace_title_rejected(self, kb_root, dummy_searcher):
        """Whitespace-only title should be rejected."""
        with pytest.raises(ValidationError, match="title cannot be empty"):
            await _call_tool(
                server.add_tool,
                title="   \t\n  ",
                content="Some content",
                tags=["test"],
                category="development",
            )

    @pytest.mark.asyncio
    async def test_empty_content_rejected(self, kb_root, dummy_searcher):
        """Empty content should be rejected."""
        with pytest.raises(ValidationError, match="content cannot be empty"):
            await _call_tool(
                server.add_tool,
                title="Valid Title",
                content="",
                tags=["test"],
                category="development",
            )

    @pytest.mark.asyncio
    async def test_whitespace_content_rejected(self, kb_root, dummy_searcher):
        """Whitespace-only content should be rejected."""
        with pytest.raises(ValidationError, match="content cannot be empty"):
            await _call_tool(
                server.add_tool,
                title="Valid Title",
                content="   \n\t   ",
                tags=["test"],
                category="development",
            )

    @pytest.mark.asyncio
    async def test_empty_tags_rejected(self, kb_root, dummy_searcher):
        """Empty tags list should be rejected."""
        with pytest.raises(ValidationError, match="tags cannot be empty"):
            await _call_tool(
                server.add_tool,
                title="Valid Title",
                content="Some content",
                tags=[],
                category="development",
            )

    @pytest.mark.asyncio
    async def test_whitespace_tag_rejected(self, kb_root, dummy_searcher):
        """Whitespace-only tag in list should be rejected."""
        with pytest.raises(ValidationError, match=r"tags\[1\] cannot be empty"):
            await _call_tool(
                server.add_tool,
                title="Valid Title",
                content="Some content",
                tags=["valid", "   ", "also-valid"],
                category="development",
            )

    @pytest.mark.asyncio
    async def test_valid_inputs_accepted(self, kb_root, dummy_searcher):
        """Valid inputs should pass validation and create entry."""
        result = await _call_tool(
            server.add_tool,
            title="Valid Title",
            content="Some valid content here.",
            tags=["python", "testing"],
            category="development",
        )

        assert result.created is True
        assert result.path == "development/valid-title.md"

    @pytest.mark.asyncio
    async def test_inputs_are_stripped(self, kb_root, dummy_searcher):
        """Leading/trailing whitespace should be stripped from inputs."""
        result = await _call_tool(
            server.add_tool,
            title="  Padded Title  ",
            content="  Content with padding  ",
            tags=["  padded-tag  "],
            category="development",
        )

        assert result.created is True
        # Title should be stripped for slug generation
        assert result.path == "development/padded-title.md"


# ─────────────────────────────────────────────────────────────────────────────
# update_tool validation tests
# ─────────────────────────────────────────────────────────────────────────────


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
    async def test_empty_path_rejected(self, kb_root, dummy_searcher, existing_entry):
        """Empty path should be rejected."""
        with pytest.raises(ValidationError, match="path cannot be empty"):
            await _call_tool(
                server.update_tool,
                path="",
                content="New content",
            )

    @pytest.mark.asyncio
    async def test_path_traversal_rejected(self, kb_root, dummy_searcher, existing_entry):
        """Path traversal attempts should be rejected."""
        with pytest.raises(ValidationError, match="path traversal not allowed"):
            await _call_tool(
                server.update_tool,
                path="../etc/passwd",
                content="Malicious content",
            )

        with pytest.raises(ValidationError, match="path traversal not allowed"):
            await _call_tool(
                server.update_tool,
                path="development/../../../etc/passwd",
                content="Malicious content",
            )

    @pytest.mark.asyncio
    async def test_absolute_path_rejected(self, kb_root, dummy_searcher, existing_entry):
        """Absolute paths should be rejected."""
        with pytest.raises(ValidationError, match="path must be relative"):
            await _call_tool(
                server.update_tool,
                path="/etc/passwd",
                content="Malicious content",
            )

    @pytest.mark.asyncio
    async def test_whitespace_content_rejected(self, kb_root, dummy_searcher, existing_entry):
        """Whitespace-only content should be rejected (but empty string allowed)."""
        with pytest.raises(ValidationError, match="content cannot be whitespace-only"):
            await _call_tool(
                server.update_tool,
                path="development/existing.md",
                content="   \n\t   ",
            )

    @pytest.mark.asyncio
    async def test_empty_tags_rejected_when_provided(self, kb_root, dummy_searcher, existing_entry):
        """Empty tags list should be rejected when tags parameter is provided."""
        with pytest.raises(ValidationError, match="tags cannot be empty"):
            await _call_tool(
                server.update_tool,
                path="development/existing.md",
                tags=[],
                content="New content",
            )

    @pytest.mark.asyncio
    async def test_section_updates_with_empty_key_rejected(
        self, kb_root, dummy_searcher, existing_entry
    ):
        """Section updates with empty key should be rejected."""
        with pytest.raises(ValidationError, match="section_updates key cannot be empty"):
            await _call_tool(
                server.update_tool,
                path="development/existing.md",
                section_updates={"": "content", "Valid": "content"},
            )

    @pytest.mark.asyncio
    async def test_section_updates_with_whitespace_key_rejected(
        self, kb_root, dummy_searcher, existing_entry
    ):
        """Section updates with whitespace-only key should be rejected."""
        with pytest.raises(ValidationError, match="section_updates key cannot be empty"):
            await _call_tool(
                server.update_tool,
                path="development/existing.md",
                section_updates={"   ": "content"},
            )

    @pytest.mark.asyncio
    async def test_valid_update_accepted(self, kb_root, dummy_searcher, existing_entry):
        """Valid update inputs should pass validation."""
        result = await _call_tool(
            server.update_tool,
            path="development/existing.md",
            content="Updated content here.",
            tags=["updated", "test"],
        )

        assert result["path"] == "development/existing.md"

    @pytest.mark.asyncio
    async def test_section_updates_accepted(self, kb_root, dummy_searcher, existing_entry):
        """Valid section updates should pass validation."""
        result = await _call_tool(
            server.update_tool,
            path="development/existing.md",
            section_updates={"Overview": "New overview content"},
        )

        assert result["path"] == "development/existing.md"


# ─────────────────────────────────────────────────────────────────────────────
# Validation helper function tests
# ─────────────────────────────────────────────────────────────────────────────


class TestValidationHelpers:
    """Tests for validation helper functions."""

    def test_validate_non_empty_string_returns_stripped(self):
        """_validate_non_empty_string should return stripped value."""
        assert server._validate_non_empty_string("  hello  ", "field") == "hello"

    def test_validate_non_empty_string_raises_on_empty(self):
        """_validate_non_empty_string should raise on empty."""
        with pytest.raises(ValidationError):
            server._validate_non_empty_string("", "field")

    def test_validate_tags_returns_stripped(self):
        """_validate_tags should return list of stripped tags."""
        result = server._validate_tags(["  tag1  ", "tag2  "])
        assert result == ["tag1", "tag2"]

    def test_validate_tags_raises_on_empty_list(self):
        """_validate_tags should raise on empty list."""
        with pytest.raises(ValidationError):
            server._validate_tags([])

    def test_validate_path_returns_stripped(self):
        """_validate_path should return stripped path."""
        assert server._validate_path("  path/to/file.md  ") == "path/to/file.md"

    def test_validate_path_raises_on_traversal(self):
        """_validate_path should raise on path traversal."""
        with pytest.raises(ValidationError):
            server._validate_path("../dangerous")

    def test_validate_section_updates_returns_none_for_none(self):
        """_validate_section_updates should return None for None input."""
        assert server._validate_section_updates(None) is None

    def test_validate_section_updates_returns_stripped_keys(self):
        """_validate_section_updates should strip keys."""
        result = server._validate_section_updates({"  Section  ": "content"})
        assert result == {"Section": "content"}


# ─────────────────────────────────────────────────────────────────────────────
# Warning propagation tests
# ─────────────────────────────────────────────────────────────────────────────


class TestWarningPropagation:
    """Tests that non-critical failures surface as warnings in responses."""

    @pytest.fixture
    def failing_searcher(self, monkeypatch):
        """Mock searcher that fails during indexing."""
        from memex.parser import ParseError

        class FailingSearcher:
            def __init__(self):
                self.deleted = []
                self.indexed = []
                self.should_fail = True

            def delete_document(self, path: str) -> None:
                self.deleted.append(path)

            def index_chunks(self, chunks):
                if self.should_fail:
                    raise ParseError(Path("/fake/path.md"), "Simulated indexing failure")
                self.indexed.append(chunks)

            def status(self):
                class Status:
                    kb_files = 1
                    whoosh_docs = 1
                    chroma_docs = 1

                return Status()

            def preload(self):
                return None

            def search(self, query, limit=10, mode="hybrid", project_context=None, kb_context=None):
                return []

        dummy = FailingSearcher()
        monkeypatch.setattr(core, "get_searcher", lambda: dummy)
        monkeypatch.setattr(core, "rebuild_backlink_cache", lambda *_args, **_kwargs: None)
        return dummy

    @pytest.mark.asyncio
    async def test_add_returns_warnings_on_index_failure(self, kb_root, failing_searcher):
        """add_tool should return warnings when indexing fails."""
        result = await _call_tool(
            server.add_tool,
            title="Test Entry",
            content="Some content here.",
            tags=["test"],
            category="development",
        )

        # Entry should still be created
        assert result.created is True
        assert result.path == "development/test-entry.md"
        # But warnings should be populated
        assert len(result.warnings) > 0
        assert "indexing failed" in result.warnings[0].lower()

    @pytest.mark.asyncio
    async def test_add_no_warnings_on_success(self, kb_root, dummy_searcher):
        """add_tool should return empty warnings list when indexing succeeds."""
        result = await _call_tool(
            server.add_tool,
            title="Test Entry",
            content="Some content here.",
            tags=["test"],
            category="development",
        )

        assert result.created is True
        assert result.warnings == []

    @pytest.fixture
    def existing_entry_for_update(self, kb_root):
        """Create an existing entry for update warning tests."""
        entry = kb_root / "development" / "update-test.md"
        entry.write_text(
            """---
title: Update Test Entry
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
    async def test_update_returns_warnings_on_index_failure(
        self, kb_root, failing_searcher, existing_entry_for_update
    ):
        """update_tool should return warnings when re-indexing fails."""
        result = await _call_tool(
            server.update_tool,
            path="development/update-test.md",
            content="Updated content here.",
        )

        # Entry should still be updated
        assert result["path"] == "development/update-test.md"
        # But warnings should be populated
        assert len(result["warnings"]) > 0
        assert "re-indexing failed" in result["warnings"][0].lower()

    @pytest.mark.asyncio
    async def test_update_no_warnings_on_success(
        self, kb_root, dummy_searcher, existing_entry_for_update
    ):
        """update_tool should return empty warnings list when re-indexing succeeds."""
        result = await _call_tool(
            server.update_tool,
            path="development/update-test.md",
            content="Updated content here.",
        )

        assert result["path"] == "development/update-test.md"
        assert result["warnings"] == []
