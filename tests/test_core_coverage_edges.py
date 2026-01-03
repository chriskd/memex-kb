"""Edge-case coverage tests for core helpers and analytics."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from memex import core
from memex.context import KBContext
from memex.models import SearchResult, ViewStats


@pytest.fixture
def kb_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "kb"
    root.mkdir()
    monkeypatch.setenv("MEMEX_KB_ROOT", str(root))
    return root


@pytest.fixture
def index_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / ".indices"
    root.mkdir()
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(root))
    return root


def _write_entry(path: Path, title: str, tags: list[str], body: str = "Body") -> None:
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    content = f"""---
title: {title}
tags:
{tags_yaml}
created: 2024-01-01
---

{body}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Git/env helpers
# ─────────────────────────────────────────────────────────────────────────────


def test_get_current_project_parses_https_remote(monkeypatch):
    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="https://github.com/voidlabs/memex.git\n")

    monkeypatch.setattr(core.subprocess, "run", fake_run)

    assert core.get_current_project() == "memex"


def test_get_current_project_timeout_falls_back_to_cwd(monkeypatch):
    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=2)

    monkeypatch.setattr(core.subprocess, "run", fake_run)

    assert core.get_current_project() == Path.cwd().name


def test_get_current_contributor_env_fallback(monkeypatch):
    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(core.subprocess, "run", fake_run)
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test User")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.com")

    assert core.get_current_contributor() == "Test User <test@example.com>"


def test_get_current_contributor_env_name_only(monkeypatch):
    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(core.subprocess, "run", fake_run)
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Solo User")
    monkeypatch.delenv("GIT_AUTHOR_EMAIL", raising=False)

    assert core.get_current_contributor() == "Solo User"


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_nested_path_symlink_escape(kb_root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink = kb_root / "escape"
    os.symlink(outside, symlink)

    with pytest.raises(ValueError, match="Path escapes knowledge base"):
        core.validate_nested_path("escape/file.md", kb_root)


def test_directory_exists_and_parent_category(kb_root):
    (kb_root / "development").mkdir()
    (kb_root / "notes.txt").write_text("hi", encoding="utf-8")

    assert core.directory_exists("development", kb_root)
    assert not core.directory_exists("notes.txt", kb_root)

    with pytest.raises(ValueError, match="Empty path"):
        core.get_parent_category("")


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate detection
# ─────────────────────────────────────────────────────────────────────────────


def test_detect_potential_duplicates_handles_search_errors():
    class FailingSearcher:
        def search(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    results = core.detect_potential_duplicates(
        title="Title",
        content="Content",
        searcher=FailingSearcher(),
    )

    assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# add_entry edge paths
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_entry_directory_path_is_file(kb_root, index_root):
    file_path = kb_root / "development"
    file_path.write_text("not a dir", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory"):
        await core.add_entry(
            title="Test",
            content="Content",
            tags=["tag"],
            directory="development",
            kb_context=KBContext(),
            check_duplicates=False,
        )


@pytest.mark.asyncio
async def test_add_entry_context_primary_is_file(kb_root, index_root):
    file_path = kb_root / "project"
    file_path.write_text("not a dir", encoding="utf-8")

    with pytest.raises(ValueError, match="Context primary path exists but is not a directory"):
        await core.add_entry(
            title="Test",
            content="Content",
            tags=["tag"],
            kb_context=KBContext(primary="project"),
            check_duplicates=False,
        )


@pytest.mark.asyncio
async def test_add_entry_infers_category_from_tags(kb_root, index_root, monkeypatch):
    (kb_root / "development").mkdir()

    class StubSearcher:
        def index_chunks(self, _chunks):
            return None

    monkeypatch.setattr(core, "get_searcher", lambda: StubSearcher())
    monkeypatch.setattr(core, "compute_link_suggestions", lambda **_kwargs: [])
    monkeypatch.setattr(core, "compute_tag_suggestions", lambda **_kwargs: [])

    result = await core.add_entry(
        title="Inferred Category",
        content="Content",
        tags=["development", "guide"],
        check_duplicates=False,
    )

    assert result.created is True
    assert result.path.startswith("development/")


@pytest.mark.asyncio
async def test_add_entry_missing_category_no_tag_match(kb_root, index_root, monkeypatch):
    (kb_root / "development").mkdir()

    class StubSearcher:
        def index_chunks(self, _chunks):
            return None

    monkeypatch.setattr(core, "get_searcher", lambda: StubSearcher())
    monkeypatch.setattr(core, "compute_link_suggestions", lambda **_kwargs: [])
    monkeypatch.setattr(core, "compute_tag_suggestions", lambda **_kwargs: [])

    with pytest.raises(ValueError, match="Either 'category' or 'directory' must be provided"):
        await core.add_entry(
            title="Missing Category",
            content="Content",
            tags=["misc"],
            check_duplicates=False,
        )


@pytest.mark.asyncio
async def test_add_entry_prepends_context_default_tags(kb_root, index_root, monkeypatch):
    (kb_root / "development").mkdir()

    async def fake_project():
        return None

    async def fake_contributor():
        return None

    async def fake_branch():
        return None

    class StubSearcher:
        def index_chunks(self, _chunks):
            return None

    monkeypatch.setattr(core, "get_current_project_async", fake_project)
    monkeypatch.setattr(core, "get_current_contributor_async", fake_contributor)
    monkeypatch.setattr(core, "get_git_branch_async", fake_branch)
    monkeypatch.setattr(core, "get_searcher", lambda: StubSearcher())
    monkeypatch.setattr(core, "compute_link_suggestions", lambda **_kwargs: [])
    monkeypatch.setattr(
        core,
        "compute_tag_suggestions",
        lambda **_kwargs: [{"tag": "suggested", "score": 0.4, "reason": "semantic"}],
    )

    response = await core.add_entry(
        title="Context Test",
        content="Content",
        tags=["base", "existing"],
        directory="development",
        kb_context=KBContext(default_tags=["context", "existing"]),
        check_duplicates=False,
    )

    assert response.suggested_tags[0]["tag"] == "context"
    assert response.suggested_tags[0]["reason"] == "From project .kbcontext"
    assert response.suggested_tags[1]["tag"] == "suggested"


# ─────────────────────────────────────────────────────────────────────────────
# popular
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_popular_filters_and_skips(kb_root, index_root, monkeypatch):
    _write_entry(kb_root / "development" / "good.md", "Good", ["alpha", "beta"])
    _write_entry(kb_root / "devops" / "wrong-tag.md", "Wrong", ["other"])

    bad_path = kb_root / "development" / "bad.md"
    bad_path.write_text("---\ntitle: Bad: [\n---\n", encoding="utf-8")

    stats = ViewStats(total_views=5, last_viewed=datetime(2024, 1, 2))

    def fake_popular(**_kwargs):
        return [
            ("development/good.md", stats),
            ("development/bad.md", stats),
            ("devops/wrong-tag.md", stats),
            ("missing.md", stats),
        ]

    monkeypatch.setattr("memex.views_tracker.get_popular", fake_popular)

    results = await core.popular(limit=5, category="development", tag="alpha")

    assert results == [
        {
            "path": "development/good.md",
            "title": "Good",
            "tags": ["alpha", "beta"],
            "view_count": 5,
            "last_viewed": "2024-01-02T00:00:00",
        }
    ]


@pytest.mark.asyncio
async def test_popular_missing_root_returns_empty(monkeypatch, tmp_path):
    missing = tmp_path / "missing"
    monkeypatch.setenv("MEMEX_KB_ROOT", str(missing))

    results = await core.popular()

    assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# hubs and dead_ends
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hubs_and_dead_ends(kb_root, index_root, monkeypatch):
    _write_entry(kb_root / "development" / "a.md", "A", ["tag"], "Link to [[development/b]]")
    _write_entry(kb_root / "development" / "b.md", "B", ["tag"], "No links")

    bad_path = kb_root / "development" / "bad.md"
    bad_path.write_text("---\ninvalid: [\n---\n", encoding="utf-8")

    backlinks = {
        "development/a": ["development/b", "development/missing"],
        "development/b": ["development/a"],
        "development/missing": ["development/a"],
    }

    monkeypatch.setattr(core, "get_backlink_index", lambda _root: backlinks)

    hubs = await core.hubs(limit=10)
    dead_ends = await core.dead_ends(limit=10)

    assert any(item["path"] == "development/a.md" for item in hubs)
    assert any(item["path"] == "development/b.md" for item in hubs)
    assert any(item["title"] == "development/missing" for item in hubs)

    assert dead_ends == [
        {
            "path": "development/b.md",
            "title": "B",
            "incoming_count": 1,
        }
    ]


@pytest.mark.asyncio
async def test_hubs_missing_root_returns_empty(monkeypatch, tmp_path):
    missing = tmp_path / "missing"
    monkeypatch.setenv("MEMEX_KB_ROOT", str(missing))

    assert await core.hubs() == []
    assert await core.dead_ends() == []
