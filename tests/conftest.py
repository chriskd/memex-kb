"""Shared test fixtures for memex test suite.

Design:
- tmp_kb: Creates isolated KB in temp directory with optional seed data
- runner: CliRunner with proper isolation
- Async helpers: pytest-asyncio configured with function scope
"""

import importlib.util
import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from click.testing import CliRunner

from memex.cli import cli
from memex.config import get_kb_root


# ─────────────────────────────────────────────────────────────────────────────
# Markers
# ─────────────────────────────────────────────────────────────────────────────


def _semantic_deps_available() -> bool:
    return (
        importlib.util.find_spec("chromadb") is not None
        and importlib.util.find_spec("sentence_transformers") is not None
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "semantic: requires chromadb and sentence-transformers extras",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _semantic_deps_available():
        return

    skip_semantic = pytest.mark.skip(
        reason=(
            "semantic extras not installed; install with "
            "`uv pip install -e '.[semantic]'` to run these tests"
        )
    )

    for item in items:
        if "semantic" in item.keywords:
            item.add_marker(skip_semantic)


# ─────────────────────────────────────────────────────────────────────────────
# Core Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def runner() -> CliRunner:
    """CLI runner with isolated environment."""
    return CliRunner()


@pytest.fixture
def tmp_kb(tmp_path: Path) -> Generator[Path, None, None]:
    """Create isolated KB directory with basic structure.

    Sets MEMEX_KB_ROOT to temp directory, creates required structure,
    yields the path, then cleans up.

    Usage:
        def test_something(tmp_kb):
            entry = tmp_kb / "test.md"
            entry.write_text("# Test")
    """
    kb_root = tmp_path / "kb"
    kb_root.mkdir()

    # Create kbconfig.yaml
    kbconfig = kb_root / "kbconfig.yaml"
    kbconfig.write_text("""categories:
  - name: general
    path: .
    description: General entries
tags:
  - name: test
    description: Test tag
""")

    # Create index directories
    (kb_root / ".kb-indices").mkdir()

    # Store original and set test env
    original_kb_root = os.environ.get("MEMEX_KB_ROOT")
    os.environ["MEMEX_KB_ROOT"] = str(kb_root)

    # Clear any cached KB context
    try:
        from memex.context import _context_cache
        _context_cache.clear()
    except (ImportError, AttributeError):
        pass

    # Reset the core module's searcher singleton
    try:
        from memex import core
        core._searcher = None
        core._searcher_ready = False
    except (ImportError, AttributeError):
        pass

    yield kb_root

    # Restore
    if original_kb_root is not None:
        os.environ["MEMEX_KB_ROOT"] = original_kb_root
    else:
        os.environ.pop("MEMEX_KB_ROOT", None)

    # Clear cache again
    try:
        from memex.context import _context_cache
        _context_cache.clear()
    except (ImportError, AttributeError):
        pass

    # Reset the core module's searcher singleton again
    try:
        from memex import core
        core._searcher = None
        core._searcher_ready = False
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def tmp_kb_with_entries(tmp_kb: Path) -> Path:
    """KB with sample entries for search/integration tests.

    Creates:
    - general/python-tips.md (tags: python, tips)
    - general/rust-guide.md (tags: rust, guide)
    - general/testing-patterns.md (tags: testing, python)
    """
    entries = [
        ("general/python-tips.md", {
            "title": "Python Tips",
            "tags": ["python", "tips"],
            "content": "# Python Tips\n\nUseful Python programming tips and tricks."
        }),
        ("general/rust-guide.md", {
            "title": "Rust Guide",
            "tags": ["rust", "guide"],
            "content": "# Rust Guide\n\nGetting started with Rust programming."
        }),
        ("general/testing-patterns.md", {
            "title": "Testing Patterns",
            "tags": ["testing", "python"],
            "content": "# Testing Patterns\n\nCommon patterns for writing effective tests."
        }),
    ]

    for path, data in entries:
        entry_path = tmp_kb / path
        entry_path.parent.mkdir(parents=True, exist_ok=True)

        frontmatter = f"""---
title: {data['title']}
tags: [{', '.join(data['tags'])}]
created: 2024-01-15
---

{data['content']}
"""
        entry_path.write_text(frontmatter)

    return tmp_kb


@pytest.fixture
def cli_invoke(runner: CliRunner, tmp_kb: Path):
    """Helper for invoking CLI with proper isolation.

    Usage:
        def test_search(cli_invoke):
            result = cli_invoke(["search", "python"])
            assert result.exit_code == 0
    """
    def _invoke(args: list[str], input: str | None = None, catch_exceptions: bool = False):
        return runner.invoke(
            cli,
            args,
            input=input,
            catch_exceptions=catch_exceptions,
            env={"MEMEX_KB_ROOT": str(tmp_kb)}
        )
    return _invoke


# ─────────────────────────────────────────────────────────────────────────────
# Async Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def anyio_backend():
    """Use asyncio for async tests."""
    return "asyncio"


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions (for test code, not fixtures)
# ─────────────────────────────────────────────────────────────────────────────


def create_entry(kb_root: Path, path: str, title: str, content: str, tags: list[str] | None = None) -> Path:
    """Helper to create a test entry with frontmatter.

    Usage in tests:
        from conftest import create_entry
        entry = create_entry(tmp_kb, "test.md", "Test", "Content", ["tag1"])
    """
    entry_path = kb_root / path
    entry_path.parent.mkdir(parents=True, exist_ok=True)

    tags_str = f"[{', '.join(tags)}]" if tags else "[]"
    frontmatter = f"""---
title: {title}
tags: {tags_str}
created: 2024-01-15
---

{content}
"""
    entry_path.write_text(frontmatter)
    return entry_path
