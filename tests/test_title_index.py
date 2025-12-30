"""Tests for title_index.py - TitleIndex and resolve_link_target.

Comprehensive test coverage for:
- TitleIndex construction and population via build_title_index()
- resolve_link_target() - all edge cases:
  - Exact path matches (e.g., "foo/bar")
  - Title-based matches (e.g., "My Document Title")
  - Case-insensitive matching
  - Link to section (e.g., "foo#section")
  - Non-existent links
  - Empty index
- Thread safety
- Performance characteristics with large indices
"""

from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path

import pytest

from memex.parser.title_index import (
    TitleEntry,
    TitleIndex,
    build_title_index,
    resolve_link_target,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def kb_root(tmp_path) -> Path:
    """Create a temporary KB root directory."""
    root = tmp_path / "kb"
    root.mkdir()
    return root


def _create_entry(
    kb_root: Path,
    rel_path: str,
    content: str,
    title: str | None = None,
    aliases: list[str] | None = None,
) -> Path:
    """Helper to create a KB entry with optional frontmatter."""
    path = kb_root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)

    frontmatter_parts = []
    if title:
        frontmatter_parts.append(f"title: {title}")
    if aliases:
        frontmatter_parts.append("aliases:")
        for alias in aliases:
            frontmatter_parts.append(f"  - {alias}")

    if frontmatter_parts:
        frontmatter_str = "\n".join(frontmatter_parts)
        full_content = f"""---
{frontmatter_str}
---

{content}
"""
    else:
        full_content = content

    path.write_text(full_content, encoding="utf-8")
    return path


# -----------------------------------------------------------------------------
# TitleEntry tests
# -----------------------------------------------------------------------------


class TestTitleEntry:
    """Tests for TitleEntry NamedTuple."""

    def test_basic_creation(self):
        """TitleEntry can be created with title and path."""
        entry = TitleEntry(title="My Title", path="path/to/entry")
        assert entry.title == "My Title"
        assert entry.path == "path/to/entry"
        assert entry.is_alias is False

    def test_alias_entry(self):
        """TitleEntry can mark aliases."""
        entry = TitleEntry(title="Alias Name", path="path/to/entry", is_alias=True)
        assert entry.is_alias is True

    def test_named_tuple_unpacking(self):
        """TitleEntry supports unpacking."""
        entry = TitleEntry("Title", "path", True)
        title, path, is_alias = entry
        assert title == "Title"
        assert path == "path"
        assert is_alias is True


# -----------------------------------------------------------------------------
# TitleIndex tests
# -----------------------------------------------------------------------------


class TestTitleIndex:
    """Tests for TitleIndex NamedTuple."""

    def test_basic_creation(self):
        """TitleIndex can be created with both dicts."""
        title_to_path = {"my title": "path/to/entry"}
        filename_to_paths = {"entry": ["path/to/entry"]}
        index = TitleIndex(
            title_to_path=title_to_path, filename_to_paths=filename_to_paths
        )
        assert index.title_to_path == title_to_path
        assert index.filename_to_paths == filename_to_paths

    def test_empty_index(self):
        """TitleIndex can be empty."""
        index = TitleIndex(title_to_path={}, filename_to_paths={})
        assert len(index.title_to_path) == 0
        assert len(index.filename_to_paths) == 0

    def test_immutable(self):
        """TitleIndex fields cannot be reassigned (NamedTuple)."""
        index = TitleIndex(title_to_path={}, filename_to_paths={})
        with pytest.raises(AttributeError):
            index.title_to_path = {"new": "value"}


# -----------------------------------------------------------------------------
# build_title_index tests
# -----------------------------------------------------------------------------


class TestBuildTitleIndex:
    """Tests for build_title_index function."""

    def test_empty_kb_returns_empty_index(self, kb_root):
        """Empty KB returns TitleIndex with empty dicts."""
        index = build_title_index(kb_root)
        assert isinstance(index, TitleIndex)
        assert index.title_to_path == {}
        assert index.filename_to_paths == {}

    def test_nonexistent_kb_returns_empty_index(self, tmp_path):
        """Non-existent KB root returns empty TitleIndex."""
        nonexistent = tmp_path / "does-not-exist"
        index = build_title_index(nonexistent)
        assert isinstance(index, TitleIndex)
        assert index.title_to_path == {}

    def test_file_instead_of_directory(self, tmp_path):
        """When kb_root is a file, returns empty TitleIndex."""
        file = tmp_path / "file.txt"
        file.write_text("not a directory")
        index = build_title_index(file)
        assert isinstance(index, TitleIndex)
        assert index.title_to_path == {}

    def test_single_entry_with_title(self, kb_root):
        """Single entry with title is indexed."""
        _create_entry(kb_root, "my-entry.md", "Content", title="My Entry Title")

        index = build_title_index(kb_root)

        assert "my entry title" in index.title_to_path
        assert index.title_to_path["my entry title"] == "my-entry"

    def test_title_case_insensitive(self, kb_root):
        """Title lookup key is lowercase."""
        _create_entry(kb_root, "entry.md", "Content", title="UPPERCASE TITLE")

        index = build_title_index(kb_root)

        assert "uppercase title" in index.title_to_path
        assert "UPPERCASE TITLE" not in index.title_to_path

    def test_entry_with_aliases(self, kb_root):
        """Entry aliases are indexed."""
        _create_entry(
            kb_root,
            "entry.md",
            "Content",
            title="Main Title",
            aliases=["Alias One", "Alias Two"],
        )

        index = build_title_index(kb_root)

        assert "main title" in index.title_to_path
        assert "alias one" in index.title_to_path
        assert "alias two" in index.title_to_path
        # All point to the same path
        assert index.title_to_path["alias one"] == "entry"
        assert index.title_to_path["alias two"] == "entry"

    def test_nested_directory_paths(self, kb_root):
        """Entries in nested directories have correct paths."""
        _create_entry(
            kb_root, "projects/alpha/readme.md", "Content", title="Alpha Readme"
        )

        index = build_title_index(kb_root)

        assert "alpha readme" in index.title_to_path
        assert index.title_to_path["alpha readme"] == "projects/alpha/readme"

    def test_filename_index_populated(self, kb_root):
        """Filename index is populated for O(1) lookups."""
        _create_entry(kb_root, "docs/guide.md", "Content", title="Guide")
        _create_entry(kb_root, "projects/guide.md", "Content", title="Project Guide")

        index = build_title_index(kb_root)

        # "guide" (lowercase) should map to both paths
        assert "guide" in index.filename_to_paths
        paths = index.filename_to_paths["guide"]
        assert "docs/guide" in paths
        assert "projects/guide" in paths

    def test_underscore_files_skipped(self, kb_root):
        """Files starting with underscore are skipped."""
        _create_entry(kb_root, "_template.md", "Template content", title="Template")
        _create_entry(kb_root, "normal.md", "Normal content", title="Normal")

        index = build_title_index(kb_root)

        assert "template" not in index.title_to_path
        assert "normal" in index.title_to_path

    def test_files_without_frontmatter_skipped(self, kb_root):
        """Files without frontmatter metadata are skipped entirely."""
        (kb_root / "no-frontmatter.md").write_text("# Just markdown\n\nNo frontmatter.")

        index = build_title_index(kb_root)

        # Files without metadata are completely skipped (no title, no filename index)
        assert "no-frontmatter" not in index.filename_to_paths
        assert len(index.title_to_path) == 0

    def test_files_with_empty_frontmatter_skipped(self, kb_root):
        """Files with empty frontmatter metadata are skipped."""
        (kb_root / "empty-frontmatter.md").write_text("---\n---\n\nContent")

        index = build_title_index(kb_root)

        # Empty metadata dict, so no title indexed
        assert len(index.title_to_path) == 0

    def test_first_title_wins(self, kb_root):
        """When multiple entries have the same title, first one wins."""
        _create_entry(kb_root, "first.md", "First content", title="Duplicate Title")
        _create_entry(kb_root, "second.md", "Second content", title="Duplicate Title")

        index = build_title_index(kb_root)

        # Due to rglob ordering, we just verify one is indexed
        assert "duplicate title" in index.title_to_path
        # The path should be one of the two
        path = index.title_to_path["duplicate title"]
        assert path in ("first", "second")

    def test_backward_compatible_dict_mode(self, kb_root):
        """When include_filename_index=False, returns dict."""
        _create_entry(kb_root, "entry.md", "Content", title="Entry Title")

        result = build_title_index(kb_root, include_filename_index=False)

        assert isinstance(result, dict)
        assert "entry title" in result
        assert result["entry title"] == "entry"

    def test_whitespace_in_title_stripped(self, kb_root):
        """Whitespace in titles is stripped."""
        _create_entry(kb_root, "entry.md", "Content", title="  Padded Title  ")

        index = build_title_index(kb_root)

        assert "padded title" in index.title_to_path

    def test_malformed_file_skipped(self, kb_root):
        """Malformed files are skipped gracefully."""
        # Create a file with invalid YAML frontmatter
        bad_file = kb_root / "bad.md"
        bad_file.write_text("---\ntitle: [invalid yaml\n---\n\nContent")

        _create_entry(kb_root, "good.md", "Content", title="Good Entry")

        index = build_title_index(kb_root)

        # Good entry should still be indexed
        assert "good entry" in index.title_to_path

    def test_non_list_aliases_handled(self, kb_root):
        """Non-list aliases field is handled gracefully."""
        content = """---
title: Entry
aliases: not-a-list
---

Content
"""
        (kb_root / "entry.md").write_text(content)

        index = build_title_index(kb_root)

        # Should not crash, title should still be indexed
        assert "entry" in index.title_to_path

    def test_empty_alias_skipped(self, kb_root):
        """Empty alias strings are skipped."""
        _create_entry(
            kb_root, "entry.md", "Content", title="Entry", aliases=["", "Valid Alias"]
        )

        index = build_title_index(kb_root)

        assert "" not in index.title_to_path
        assert "valid alias" in index.title_to_path


# -----------------------------------------------------------------------------
# resolve_link_target tests - Exact Path Matches
# -----------------------------------------------------------------------------


class TestResolveLinkTargetPathMatches:
    """Tests for resolve_link_target with path-style links."""

    def test_path_with_separator_returned_as_is(self, kb_root):
        """Links containing / are returned as-is (path references)."""
        index = TitleIndex(title_to_path={}, filename_to_paths={})

        result = resolve_link_target("foo/bar", index)

        assert result == "foo/bar"

    def test_path_with_md_extension_stripped(self, kb_root):
        """Links with .md extension have it stripped."""
        index = TitleIndex(title_to_path={}, filename_to_paths={})

        result = resolve_link_target("foo/bar.md", index)

        assert result == "foo/bar"

    def test_backslash_normalized_to_forward_slash(self, kb_root):
        """Backslash path separators are normalized."""
        index = TitleIndex(title_to_path={}, filename_to_paths={})

        result = resolve_link_target("foo\\bar\\baz", index)

        assert result == "foo/bar/baz"

    def test_leading_trailing_slashes_stripped(self, kb_root):
        """Leading and trailing slashes are stripped."""
        index = TitleIndex(title_to_path={}, filename_to_paths={})

        result = resolve_link_target("/foo/bar/", index)

        assert result == "foo/bar"

    def test_whitespace_stripped(self, kb_root):
        """Whitespace around target is stripped."""
        index = TitleIndex(title_to_path={}, filename_to_paths={})

        result = resolve_link_target("  foo/bar  ", index)

        assert result == "foo/bar"


# -----------------------------------------------------------------------------
# resolve_link_target tests - Title-Based Matches
# -----------------------------------------------------------------------------


class TestResolveLinkTargetTitleMatches:
    """Tests for resolve_link_target with title-based links."""

    def test_exact_title_match(self, kb_root):
        """Title exactly matching an indexed title resolves."""
        index = TitleIndex(
            title_to_path={"my document": "path/to/doc"},
            filename_to_paths={},
        )

        result = resolve_link_target("My Document", index)

        assert result == "path/to/doc"

    def test_title_case_insensitive(self, kb_root):
        """Title matching is case-insensitive."""
        index = TitleIndex(
            title_to_path={"uppercase title": "entry"},
            filename_to_paths={},
        )

        result = resolve_link_target("UPPERCASE TITLE", index)

        assert result == "entry"

    def test_alias_match(self, kb_root):
        """Alias resolves to the correct path."""
        index = TitleIndex(
            title_to_path={
                "main title": "entry",
                "my alias": "entry",
            },
            filename_to_paths={},
        )

        result = resolve_link_target("My Alias", index)

        assert result == "entry"


# -----------------------------------------------------------------------------
# resolve_link_target tests - Filename Matches
# -----------------------------------------------------------------------------


class TestResolveLinkTargetFilenameMatches:
    """Tests for resolve_link_target with filename-based links."""

    def test_filename_match_via_index(self, kb_root):
        """Filename matches via O(1) filename index."""
        index = TitleIndex(
            title_to_path={},
            filename_to_paths={"myfile": ["docs/myfile"]},
        )

        result = resolve_link_target("myfile", index)

        assert result == "docs/myfile"

    def test_filename_match_case_insensitive(self, kb_root):
        """Filename matching is case-insensitive."""
        index = TitleIndex(
            title_to_path={},
            filename_to_paths={"myfile": ["docs/myfile"]},
        )

        result = resolve_link_target("MYFILE", index)

        assert result == "docs/myfile"

    def test_filename_match_returns_first_path(self, kb_root):
        """When multiple paths match a filename, first is returned."""
        index = TitleIndex(
            title_to_path={},
            filename_to_paths={"guide": ["docs/guide", "projects/guide"]},
        )

        result = resolve_link_target("guide", index)

        assert result == "docs/guide"


# -----------------------------------------------------------------------------
# resolve_link_target tests - Fallback O(n) Search
# -----------------------------------------------------------------------------


class TestResolveLinkTargetFallback:
    """Tests for resolve_link_target fallback O(n) search (legacy compatibility)."""

    def test_fallback_path_suffix_match(self):
        """Fallback search matches path suffixes."""
        # Using plain dict (legacy mode) without filename_index
        title_index = {"some title": "category/entry-name"}

        result = resolve_link_target("entry-name", title_index)

        assert result == "category/entry-name"

    def test_fallback_exact_path_match(self):
        """Fallback search matches exact paths."""
        title_index = {"some title": "entry-name"}

        result = resolve_link_target("entry-name", title_index)

        assert result == "entry-name"

    def test_fallback_case_insensitive(self):
        """Fallback search is case-insensitive."""
        title_index = {"some title": "Category/Entry-Name"}

        result = resolve_link_target("ENTRY-NAME", title_index)

        assert result == "Category/Entry-Name"

    def test_fallback_with_explicit_filename_index(self):
        """Explicit filename_index parameter is used."""
        title_index = {}
        filename_index = {"myentry": ["folder/myentry"]}

        result = resolve_link_target(
            "myentry", title_index, filename_index=filename_index
        )

        assert result == "folder/myentry"


# -----------------------------------------------------------------------------
# resolve_link_target tests - Section Links
# -----------------------------------------------------------------------------


class TestResolveLinkTargetSectionLinks:
    """Tests for resolve_link_target with section anchors (e.g., foo#section)."""

    def test_section_link_path_style(self):
        """Path-style link with section anchor."""
        # Note: The current implementation doesn't special-case # in paths
        # Links with # are treated as paths if they contain /
        index = TitleIndex(title_to_path={}, filename_to_paths={})

        result = resolve_link_target("foo/bar#section", index)

        # Returns as-is since it contains /
        assert result == "foo/bar#section"

    def test_section_in_simple_link(self):
        """Simple link with section anchor may need title lookup."""
        # The # is part of the lookup key, which won't match
        # This is current behavior - # is not stripped before lookup
        index = TitleIndex(
            title_to_path={"mypage": "docs/mypage"},
            filename_to_paths={"mypage": ["docs/mypage"]},
        )

        # "mypage#section" won't match "mypage" in title_to_path
        result = resolve_link_target("mypage#section", index)

        # Current behavior: returns None because "mypage#section" != "mypage"
        # This documents current behavior - section handling may be improved later
        assert result is None


# -----------------------------------------------------------------------------
# resolve_link_target tests - Non-Existent Links
# -----------------------------------------------------------------------------


class TestResolveLinkTargetNonExistent:
    """Tests for resolve_link_target with non-existent targets."""

    def test_nonexistent_returns_none(self):
        """Non-existent target returns None."""
        index = TitleIndex(
            title_to_path={"existing": "path"},
            filename_to_paths={"existing": ["path"]},
        )

        result = resolve_link_target("nonexistent", index)

        assert result is None

    def test_empty_target(self):
        """Empty target returns None (after normalization)."""
        index = TitleIndex(title_to_path={}, filename_to_paths={})

        result = resolve_link_target("", index)

        assert result is None

    def test_whitespace_only_target(self):
        """Whitespace-only target returns None."""
        index = TitleIndex(title_to_path={}, filename_to_paths={})

        result = resolve_link_target("   ", index)

        assert result is None


# -----------------------------------------------------------------------------
# resolve_link_target tests - Empty Index
# -----------------------------------------------------------------------------


class TestResolveLinkTargetEmptyIndex:
    """Tests for resolve_link_target with empty index."""

    def test_path_link_still_works(self):
        """Path-style links work even with empty index."""
        index = TitleIndex(title_to_path={}, filename_to_paths={})

        result = resolve_link_target("foo/bar", index)

        assert result == "foo/bar"

    def test_title_link_returns_none(self):
        """Title-style links return None with empty index."""
        index = TitleIndex(title_to_path={}, filename_to_paths={})

        result = resolve_link_target("Some Title", index)

        assert result is None


# -----------------------------------------------------------------------------
# Thread Safety Tests
# -----------------------------------------------------------------------------


class TestThreadSafety:
    """Tests for thread safety of TitleIndex operations."""

    def test_concurrent_reads(self, kb_root):
        """Multiple threads can read from TitleIndex simultaneously."""
        # Build a moderate-size index
        for i in range(100):
            _create_entry(kb_root, f"entry{i}.md", f"Content {i}", title=f"Entry {i}")

        index = build_title_index(kb_root)

        results = []
        errors = []

        def reader(thread_id: int):
            try:
                for i in range(50):
                    target = f"Entry {(thread_id + i) % 100}"
                    result = resolve_link_target(target, index)
                    results.append(result is not None)
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(reader, i) for i in range(10)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0
        assert all(results)

    def test_index_immutability(self, kb_root):
        """TitleIndex is a NamedTuple and thus effectively immutable."""
        _create_entry(kb_root, "entry.md", "Content", title="Entry")
        index = build_title_index(kb_root)

        # Attempting to modify the underlying dicts is possible but
        # doesn't affect the NamedTuple's immutability guarantee
        original_title_to_path = index.title_to_path.copy()

        # Direct field reassignment should fail
        with pytest.raises(AttributeError):
            index.title_to_path = {}

        # Original data unchanged
        assert index.title_to_path == original_title_to_path


# -----------------------------------------------------------------------------
# Performance Tests
# -----------------------------------------------------------------------------


class TestPerformance:
    """Tests for performance characteristics with large indices."""

    def test_large_index_build_time(self, kb_root):
        """Building index for 1000 entries completes in reasonable time."""
        # Create 1000 entries
        for i in range(1000):
            category = i % 10
            _create_entry(
                kb_root,
                f"cat{category}/entry{i}.md",
                f"Content for entry {i}",
                title=f"Entry Number {i}",
                aliases=[f"Alias {i}"],
            )

        start = time.perf_counter()
        index = build_title_index(kb_root)
        elapsed = time.perf_counter() - start

        # Should complete in under 30 seconds (generous for CI)
        assert elapsed < 30.0
        # Verify all entries indexed
        assert len(index.title_to_path) == 2000  # titles + aliases

    def test_o1_lookup_performance(self, kb_root):
        """O(1) lookups are fast even with large index."""
        # Create 1000 entries
        for i in range(1000):
            _create_entry(
                kb_root,
                f"entry{i}.md",
                f"Content {i}",
                title=f"Entry Number {i}",
            )

        index = build_title_index(kb_root)

        # Time 10000 lookups
        start = time.perf_counter()
        for _ in range(10000):
            resolve_link_target("Entry Number 500", index)
        elapsed = time.perf_counter() - start

        # 10000 lookups should complete in under 1 second
        assert elapsed < 1.0

    def test_filename_index_enables_o1_filename_lookup(self, kb_root):
        """Filename index enables O(1) lookups by filename."""
        # Create entries with same filename in different directories
        for i in range(100):
            _create_entry(
                kb_root,
                f"cat{i}/guide.md",
                f"Guide for category {i}",
                title=f"Guide Cat{i}",
            )

        index = build_title_index(kb_root)

        # All 100 paths should be in the filename index under "guide"
        assert "guide" in index.filename_to_paths
        assert len(index.filename_to_paths["guide"]) == 100

        # Lookup should be fast (O(1))
        start = time.perf_counter()
        for _ in range(10000):
            resolve_link_target("guide", index)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestIntegration:
    """Integration tests combining build_title_index and resolve_link_target."""

    def test_full_workflow(self, kb_root):
        """Full workflow: build index, resolve various link types."""
        # Create diverse KB structure
        _create_entry(kb_root, "index.md", "Main index", title="Home")
        _create_entry(
            kb_root,
            "projects/alpha.md",
            "Alpha project",
            title="Project Alpha",
            aliases=["Alpha"],
        )
        _create_entry(kb_root, "docs/guide.md", "User guide", title="User Guide")
        _create_entry(kb_root, "docs/api.md", "API docs", title="API Reference")

        index = build_title_index(kb_root)

        # Path-style link
        assert resolve_link_target("projects/alpha", index) == "projects/alpha"

        # Title-based link
        assert resolve_link_target("Project Alpha", index) == "projects/alpha"

        # Alias-based link
        assert resolve_link_target("Alpha", index) == "projects/alpha"

        # Case-insensitive
        assert resolve_link_target("USER GUIDE", index) == "docs/guide"

        # Filename-based link
        assert resolve_link_target("api", index) == "docs/api"

        # Non-existent
        assert resolve_link_target("Nonexistent", index) is None

    def test_resolve_with_legacy_dict(self, kb_root):
        """resolve_link_target works with legacy dict format."""
        _create_entry(kb_root, "entry.md", "Content", title="My Entry")

        # Get legacy dict format
        title_dict = build_title_index(kb_root, include_filename_index=False)

        # Should still resolve titles
        result = resolve_link_target("My Entry", title_dict)
        assert result == "entry"

    def test_title_index_as_titleindex_type(self, kb_root):
        """Type check: build_title_index returns TitleIndex by default."""
        _create_entry(kb_root, "entry.md", "Content", title="Entry")

        index = build_title_index(kb_root)

        assert isinstance(index, TitleIndex)
        assert hasattr(index, "title_to_path")
        assert hasattr(index, "filename_to_paths")
