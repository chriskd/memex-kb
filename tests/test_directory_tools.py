"""Tests for KB directory management MCP tools."""

import os
from pathlib import Path
from datetime import date

import pytest

from voidlabs_kb import core, server


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
    monkeypatch.setenv("KB_ROOT", str(root))
    return root


@pytest.mark.asyncio
async def test_tree_tool_reports_titles_and_counts(kb_root):
    entry = kb_root / "development" / "guide.md"
    entry.write_text(
        """---
title: Developer Guide
tags:
  - python
created: 2024-01-01
---

## Intro

Content
"""
    )

    result = await _call_tool(server.tree_tool, path="development", depth=2)

    assert result["directories"] == 0
    assert result["files"] == 1
    assert result["tree"]["guide.md"]["title"] == "Developer Guide"


@pytest.mark.asyncio
async def test_mkdir_tool_creates_nested_directories(kb_root):
    new_path = "development/python/frameworks"
    created = await _call_tool(server.mkdir_tool, new_path)

    assert created == new_path
    assert (kb_root / new_path).is_dir()

    # mkdir can create new top-level categories
    new_category_path = "newcategory/subdir"
    created2 = await _call_tool(server.mkdir_tool, new_category_path)
    assert created2 == new_category_path
    assert (kb_root / new_category_path).is_dir()

    # mkdir raises ValueError if directory already exists
    with pytest.raises(ValueError):
        await _call_tool(server.mkdir_tool, new_path)


@pytest.fixture
def dummy_searcher(monkeypatch):
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

    dummy = DummySearcher()
    monkeypatch.setattr(core, "get_searcher", lambda: dummy)
    monkeypatch.setattr(core, "rebuild_backlink_cache", lambda *_args, **_kwargs: None)
    return dummy


@pytest.mark.asyncio
async def test_move_tool_updates_links_and_reindexes(kb_root, dummy_searcher):
    source_dir = kb_root / "development" / "python"
    dest_dir = kb_root / "architecture" / "patterns"
    source_dir.mkdir(parents=True)
    dest_dir.mkdir(parents=True)

    source_file = source_dir / "foo.md"
    source_file.write_text(
        """---
title: Foo Doc
tags:
  - sample
created: 2024-01-01
---

## Overview

Foo body
"""
    )

    referencing = kb_root / "devops" / "ref.md"
    referencing.write_text(
        """---
title: Ref Doc
tags:
  - ops
created: 2024-01-02
---

Mentions [[development/python/foo]].
"""
    )

    params = {
        "source": "development/python/foo.md",
        "destination": "architecture/patterns/foo.md",
        "update_links": True,
    }

    result = await _call_tool(server.move_tool, **params)

    assert any("development/python/foo.md -> architecture/patterns/foo.md" in entry for entry in result["moved"])
    assert result["links_updated"] == 1
    assert (kb_root / "architecture" / "patterns" / "foo.md").exists()
    assert not source_file.exists()

    updated_ref = referencing.read_text()
    assert "[[architecture/patterns/foo]]" in updated_ref

    assert dummy_searcher.deleted == [params["source"]]
    assert dummy_searcher.indexed
    assert all(chunk.path == params["destination"] for chunk in dummy_searcher.indexed[0])

    from voidlabs_kb.parser import parse_entry

    metadata, _, chunks = parse_entry(kb_root / params["destination"])
    assert metadata.title == "Foo Doc"
    assert any(chunk.content.strip() == "Foo body" for chunk in chunks)


@pytest.mark.asyncio
async def test_rmdir_tool_enforces_empty_directory(kb_root):
    """Test that rmdir enforces empty directory requirement."""
    temp_dir = kb_root / "development" / "temp"
    sub_dir = temp_dir / "child"
    sub_dir.mkdir(parents=True)

    # Can't remove non-empty directory without force
    with pytest.raises(ValueError):
        await _call_tool(server.rmdir_tool, "development/temp")

    # With force=True, removes directory and empty subdirs
    await _call_tool(server.rmdir_tool, "development/temp", force=True)
    assert not temp_dir.exists()

    # Empty directories (including category roots) can be removed
    # Note: "development" is now empty after removing temp_dir
    await _call_tool(server.rmdir_tool, "development")
    assert not (kb_root / "development").exists()
