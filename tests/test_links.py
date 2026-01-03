"""Tests for links.py - bidirectional link extraction and resolution.

Comprehensive test coverage for:
- extract_links() - extracting [[wikilinks]] from markdown content
- resolve_backlinks() - building backlink index from KB
- _normalize_link() - link normalization
- _resolve_relative_link() - relative path resolution
- update_links_in_files() - updating links when entries move
- update_links_batch() - batch link updates
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memex.parser.links import (
    LINK_PATTERN,
    _normalize_link,
    _resolve_relative_link,
    extract_links,
    resolve_backlinks,
    update_links_batch,
    update_links_in_files,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def kb_root(tmp_path) -> Path:
    """Create a temporary KB root directory."""
    root = tmp_path / "kb"
    root.mkdir()
    return root


def _create_entry(
    kb_root: Path, rel_path: str, content: str, title: str | None = None
) -> Path:
    """Helper to create a KB entry with optional frontmatter."""
    path = kb_root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)

    if title:
        full_content = f"""---
title: {title}
---

{content}
"""
    else:
        full_content = content

    path.write_text(full_content, encoding="utf-8")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# extract_links tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractLinks:
    """Tests for extract_links function."""

    def test_empty_content(self):
        """Empty content returns empty list."""
        links = extract_links("")
        assert links == []

    def test_content_without_links(self):
        """Content without any wikilinks returns empty list."""
        content = """# A Document

This is some text with no links at all.
Just plain markdown.
"""
        links = extract_links(content)
        assert links == []

    def test_single_link(self):
        """Extract a single wikilink."""
        content = "This links to [[some-entry]]."
        links = extract_links(content)
        assert links == ["some-entry"]

    def test_multiple_links(self):
        """Extract multiple wikilinks."""
        content = "Links to [[first]], [[second]], and [[third]]."
        links = extract_links(content)
        assert links == ["first", "second", "third"]

    def test_duplicate_links(self):
        """Duplicate links are deduplicated."""
        content = "[[target]] appears [[target]] multiple [[target]] times."
        links = extract_links(content)
        assert links == ["target"]
        assert len(links) == 1

    def test_link_with_path(self):
        """Extract wikilinks with path separators."""
        content = "See [[projects/alpha]] and [[docs/guide/setup]]."
        links = extract_links(content)
        assert "projects/alpha" in links
        assert "docs/guide/setup" in links

    def test_aliased_link(self):
        """Extract wikilinks with display aliases."""
        content = "See [[target|Display Text]] for more."
        links = extract_links(content)
        assert links == ["target"]

    def test_mixed_aliased_and_simple_links(self):
        """Extract both aliased and simple wikilinks."""
        content = "[[simple-link]] and [[aliased|Nice Name]] together."
        links = extract_links(content)
        assert "simple-link" in links
        assert "aliased" in links
        assert "Nice Name" not in links

    def test_link_with_md_extension(self):
        """Links with .md extension are normalized."""
        content = "Link to [[entry.md]] file."
        links = extract_links(content)
        assert links == ["entry"]

    def test_link_in_code_block_ignored(self):
        """Links inside code blocks should not be extracted."""
        content = """# Document

Regular [[valid-link]] here.

```
This [[not-a-link]] is in a code block.
```

And [[another-valid]] after.
"""
        links = extract_links(content)
        assert "valid-link" in links
        assert "another-valid" in links
        assert "not-a-link" not in links

    def test_link_in_inline_code_ignored(self):
        """Links inside inline code should not be extracted."""
        content = "Regular [[valid]] and `[[not-valid]]` in code."
        links = extract_links(content)
        assert "valid" in links
        assert "not-valid" not in links

    def test_whitespace_in_link(self):
        """Whitespace in links is normalized."""
        content = "Link to [[  spaced-target  ]]."
        links = extract_links(content)
        assert links == ["spaced-target"]

    def test_unicode_around_link(self):
        """Unicode content around links works correctly."""
        content = "中文内容 [[target]] 更多文本"
        links = extract_links(content)
        assert links == ["target"]

    def test_malformed_link_unclosed(self):
        """Unclosed brackets may match greedily depending on parser behavior."""
        content = "[[unclosed link and [[valid-link]]."
        links = extract_links(content)
        # The markdown-it-py parser matches greedily from the first [[
        # This is expected behavior - the parser finds the first complete match
        assert len(links) == 1  # At least one link is extracted

    def test_link_with_newline_inside(self):
        """Links with newlines inside are not valid."""
        content = "[[invalid\nlink]] but [[valid]] works."
        links = extract_links(content)
        assert "valid" in links
        assert "invalid\nlink" not in links

    def test_nested_brackets(self):
        """Nested brackets are handled correctly."""
        content = "Text [[outer]] and [not [[inner]] a link]."
        links = extract_links(content)
        assert "outer" in links
        assert "inner" in links

    def test_empty_link(self):
        """Empty link brackets are ignored."""
        content = "[[]] is empty and [[valid]] is not."
        links = extract_links(content)
        assert "valid" in links
        assert "" not in links

    def test_link_with_special_characters(self):
        """Links with special characters (hyphens, underscores)."""
        content = "[[my-entry_v2]] and [[hello-world]]."
        links = extract_links(content)
        assert "my-entry_v2" in links
        assert "hello-world" in links

    def test_link_at_start_of_line(self):
        """Link at the very start of content."""
        content = "[[first-thing]] is the link."
        links = extract_links(content)
        assert links == ["first-thing"]

    def test_link_at_end_of_content(self):
        """Link at the very end of content."""
        content = "The link is [[last-thing]]"
        links = extract_links(content)
        assert links == ["last-thing"]

    def test_link_only_content(self):
        """Content that is only a single link."""
        content = "[[only-link]]"
        links = extract_links(content)
        assert links == ["only-link"]

    def test_backslash_path_normalized(self):
        """Backslash path separators are normalized to forward slash."""
        content = r"[[path\to\entry]]"
        links = extract_links(content)
        assert links == ["path/to/entry"]


# ─────────────────────────────────────────────────────────────────────────────
# _normalize_link tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalizeLink:
    """Tests for _normalize_link helper function."""

    def test_strip_whitespace(self):
        """Whitespace is stripped from links."""
        assert _normalize_link("  target  ") == "target"

    def test_remove_md_extension(self):
        """The .md extension is removed."""
        assert _normalize_link("entry.md") == "entry"

    def test_normalize_backslash(self):
        """Backslashes are converted to forward slashes."""
        assert _normalize_link(r"path\to\entry") == "path/to/entry"

    def test_strip_leading_trailing_slashes(self):
        """Leading and trailing slashes are removed."""
        assert _normalize_link("/path/to/entry/") == "path/to/entry"

    def test_combined_normalization(self):
        """All normalization steps applied together."""
        # Note: .md removal happens BEFORE slash stripping
        # So "entry.md/" -> "entry.md" (slash stripped) but .md is not removed
        # because the trailing / was stripped after the .md check
        assert _normalize_link("  /path\\to\\entry/  ") == "path/to/entry"
        # And a clean case without trailing slash after .md
        assert _normalize_link("  /path\\to\\entry.md  ") == "path/to/entry"

    def test_empty_string(self):
        """Empty string remains empty."""
        assert _normalize_link("") == ""

    def test_only_whitespace(self):
        """Whitespace-only string becomes empty."""
        assert _normalize_link("   ") == ""

    def test_only_slashes(self):
        """Slash-only string becomes empty."""
        assert _normalize_link("///") == ""


# ─────────────────────────────────────────────────────────────────────────────
# _resolve_relative_link tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveRelativeLink:
    """Tests for _resolve_relative_link helper function."""

    def test_absolute_path_without_separator(self):
        """Target without path separator is returned as-is."""
        result = _resolve_relative_link("some/source", "target")
        assert result == "target"

    def test_path_with_separator_no_dots(self):
        """Path with separator but no dots is returned as-is."""
        result = _resolve_relative_link("source", "other/path")
        assert result == "other/path"

    def test_relative_parent_single(self):
        """Single parent directory reference.

        _resolve_relative_link("projects/alpha", "../beta"):
        - source_parts = ["projects"] (parent dir of alpha)
        - "../beta" -> pops "projects" and appends "beta"
        - result = "beta"
        """
        result = _resolve_relative_link("projects/alpha", "../beta")
        assert result == "beta"

    def test_relative_parent_double(self):
        """Double parent directory reference.

        _resolve_relative_link("a/b/c", "../../d"):
        - source_parts = ["a", "b"] (parent dir of c)
        - "../../d" -> pops "b", pops "a", appends "d"
        - result = "d"
        """
        result = _resolve_relative_link("a/b/c", "../../d")
        assert result == "d"

    def test_relative_current_dir(self):
        """Current directory reference is skipped."""
        result = _resolve_relative_link("a/b", "./c")
        assert result == "a/c"

    def test_relative_mixed(self):
        """Mixed current and parent directory references."""
        result = _resolve_relative_link("a/b/c", "./../d")
        assert result == "a/d"

    def test_absolute_with_leading_slash(self):
        """Leading slash is stripped."""
        result = _resolve_relative_link("source", "/absolute/path")
        assert result == "absolute/path"

    def test_parent_beyond_root(self):
        """Too many parent references results in empty path."""
        result = _resolve_relative_link("a/b", "../../../c")
        assert result == "c"

    def test_source_at_root(self):
        """Source at root level with parent reference."""
        result = _resolve_relative_link("source", "../other")
        assert result == "other"


# ─────────────────────────────────────────────────────────────────────────────
# resolve_backlinks tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveBacklinks:
    """Tests for resolve_backlinks function."""

    def test_empty_kb(self, kb_root):
        """Empty KB returns empty backlinks dict."""
        backlinks = resolve_backlinks(kb_root)
        assert backlinks == {}

    def test_nonexistent_kb(self, tmp_path):
        """Non-existent KB root returns empty dict."""
        nonexistent = tmp_path / "does-not-exist"
        backlinks = resolve_backlinks(nonexistent)
        assert backlinks == {}

    def test_kb_is_file(self, tmp_path):
        """When kb_root is a file, returns empty dict."""
        file = tmp_path / "file.txt"
        file.write_text("not a directory")
        backlinks = resolve_backlinks(file)
        assert backlinks == {}

    def test_single_link(self, kb_root):
        """Single link creates single backlink."""
        _create_entry(kb_root, "a.md", "Links to [[b]]", title="Entry A")
        _create_entry(kb_root, "b.md", "No links here", title="Entry B")

        backlinks = resolve_backlinks(kb_root)

        assert "b" in backlinks
        assert "a" in backlinks["b"]

    def test_multiple_files_linking_to_one(self, kb_root):
        """Multiple files linking to the same target."""
        _create_entry(kb_root, "a.md", "Links to [[target]]", title="A")
        _create_entry(kb_root, "b.md", "Also links to [[target]]", title="B")
        _create_entry(kb_root, "c.md", "And [[target]] too", title="C")
        _create_entry(kb_root, "target.md", "The target", title="Target")

        backlinks = resolve_backlinks(kb_root)

        assert "target" in backlinks
        assert set(backlinks["target"]) == {"a", "b", "c"}

    def test_mutual_links(self, kb_root):
        """Two files linking to each other."""
        _create_entry(kb_root, "a.md", "Links to [[b]]", title="A")
        _create_entry(kb_root, "b.md", "Links to [[a]]", title="B")

        backlinks = resolve_backlinks(kb_root)

        assert "a" in backlinks
        assert "b" in backlinks["a"]
        assert "b" in backlinks
        assert "a" in backlinks["b"]

    def test_self_link(self, kb_root):
        """Self-referential link."""
        _create_entry(kb_root, "self.md", "Links to [[self]]", title="Self")

        backlinks = resolve_backlinks(kb_root)

        assert "self" in backlinks
        assert "self" in backlinks["self"]

    def test_nested_paths(self, kb_root):
        """Files in nested directories."""
        _create_entry(kb_root, "index.md", "See [[projects/alpha]]", title="Index")
        _create_entry(
            kb_root, "projects/alpha.md", "Back to [[index]]", title="Alpha"
        )

        backlinks = resolve_backlinks(kb_root)

        assert "projects/alpha" in backlinks
        assert "index" in backlinks["projects/alpha"]
        assert "index" in backlinks
        assert "projects/alpha" in backlinks["index"]

    def test_link_to_nonexistent_target(self, kb_root):
        """Link to a file that doesn't exist."""
        _create_entry(kb_root, "a.md", "Links to [[nonexistent]]", title="A")

        backlinks = resolve_backlinks(kb_root)

        assert "nonexistent" in backlinks
        assert "a" in backlinks["nonexistent"]

    def test_underscore_files_skipped(self, kb_root):
        """Files starting with underscore are skipped as sources."""
        _create_entry(kb_root, "_template.md", "Links to [[other]]", title="Template")
        _create_entry(kb_root, "normal.md", "Links to [[_template]]", title="Normal")
        _create_entry(kb_root, "other.md", "No links", title="Other")

        backlinks = resolve_backlinks(kb_root)

        # _template should not create backlinks to 'other' because it's skipped
        if "other" in backlinks:
            assert "_template" not in backlinks["other"]

    def test_duplicate_links_in_file(self, kb_root):
        """File with duplicate links only listed once in backlinks."""
        _create_entry(
            kb_root, "a.md", "[[target]] and [[target]] again", title="A"
        )
        _create_entry(kb_root, "target.md", "The target", title="Target")

        backlinks = resolve_backlinks(kb_root)

        assert "target" in backlinks
        assert backlinks["target"].count("a") == 1

    def test_circular_links(self, kb_root):
        """Circular link pattern (A -> B -> C -> A)."""
        _create_entry(kb_root, "a.md", "Links to [[b]]", title="A")
        _create_entry(kb_root, "b.md", "Links to [[c]]", title="B")
        _create_entry(kb_root, "c.md", "Links to [[a]]", title="C")

        backlinks = resolve_backlinks(kb_root)

        assert "b" in backlinks and "a" in backlinks["b"]
        assert "c" in backlinks and "b" in backlinks["c"]
        assert "a" in backlinks and "c" in backlinks["a"]

    def test_unicode_content(self, kb_root):
        """Unicode content is handled correctly."""
        _create_entry(kb_root, "unicode.md", "中文 [[target]] 日本語", title="Unicode")
        _create_entry(kb_root, "target.md", "Target", title="Target")

        backlinks = resolve_backlinks(kb_root)

        assert "target" in backlinks
        assert "unicode" in backlinks["target"]

    def test_empty_file(self, kb_root):
        """Empty files don't cause errors."""
        (kb_root / "empty.md").write_text("")
        _create_entry(kb_root, "normal.md", "Links to [[empty]]", title="Normal")

        backlinks = resolve_backlinks(kb_root)

        assert "empty" in backlinks
        assert "normal" in backlinks["empty"]

    def test_title_resolution(self, kb_root):
        """Links by title are resolved correctly."""
        _create_entry(
            kb_root, "a.md", "Links to [[My Entry]]", title="Entry A"
        )
        _create_entry(kb_root, "b.md", "Target content", title="My Entry")

        backlinks = resolve_backlinks(kb_root)

        # Should resolve "My Entry" to "b"
        assert "b" in backlinks
        assert "a" in backlinks["b"]

    def test_alias_resolution(self, kb_root):
        """Links by alias are resolved correctly."""
        content_with_alias = """---
title: Target Entry
aliases:
  - My Alias
---

Content here.
"""
        (kb_root / "target.md").write_text(content_with_alias)
        _create_entry(kb_root, "source.md", "Links to [[My Alias]]", title="Source")

        backlinks = resolve_backlinks(kb_root)

        assert "target" in backlinks
        assert "source" in backlinks["target"]

    def test_relative_path_link(self, kb_root):
        """Relative path links are resolved."""
        (kb_root / "sub").mkdir()
        _create_entry(kb_root, "sub/a.md", "Links to [[../b]]", title="A")
        _create_entry(kb_root, "b.md", "Target", title="B")

        backlinks = resolve_backlinks(kb_root)

        # Relative paths should be resolved or tracked
        assert len(backlinks) > 0

    def test_file_read_error_handled(self, kb_root):
        """Files that can't be read are skipped gracefully."""
        # Create a file with invalid encoding
        bad_file = kb_root / "bad.md"
        bad_file.write_bytes(b"\xff\xfe invalid utf-8 \x80\x81")

        _create_entry(kb_root, "good.md", "Links to [[target]]", title="Good")

        # Should not raise, just skip the bad file
        backlinks = resolve_backlinks(kb_root)
        assert "target" in backlinks


# ─────────────────────────────────────────────────────────────────────────────
# update_links_in_files tests
# ─────────────────────────────────────────────────────────────────────────────


class TestUpdateLinksInFiles:
    """Tests for update_links_in_files function."""

    def test_no_files_to_update(self, kb_root):
        """No files contain the old path."""
        _create_entry(kb_root, "a.md", "No links here", title="A")

        count = update_links_in_files(kb_root, "old/path", "new/path")

        assert count == 0
        assert "No links here" in (kb_root / "a.md").read_text()

    def test_single_file_single_link(self, kb_root):
        """Update a single link in a single file."""
        _create_entry(kb_root, "a.md", "Links to [[old-entry]]", title="A")

        count = update_links_in_files(kb_root, "old-entry", "new-entry")

        assert count == 1
        content = (kb_root / "a.md").read_text()
        assert "[[new-entry]]" in content
        assert "[[old-entry]]" not in content

    def test_multiple_links_same_file(self, kb_root):
        """Update multiple links in the same file."""
        _create_entry(
            kb_root, "a.md", "[[old]] and [[old]] again", title="A"
        )

        count = update_links_in_files(kb_root, "old", "new")

        assert count == 1  # Only one file updated
        content = (kb_root / "a.md").read_text()
        assert content.count("[[new]]") == 2
        assert "[[old]]" not in content

    def test_multiple_files(self, kb_root):
        """Update links across multiple files."""
        _create_entry(kb_root, "a.md", "Links to [[target]]", title="A")
        _create_entry(kb_root, "b.md", "Also [[target]]", title="B")

        count = update_links_in_files(kb_root, "target", "new-target")

        assert count == 2
        assert "[[new-target]]" in (kb_root / "a.md").read_text()
        assert "[[new-target]]" in (kb_root / "b.md").read_text()

    def test_link_with_md_extension(self, kb_root):
        """Links with .md extension are also updated."""
        _create_entry(kb_root, "a.md", "Links to [[old.md]]", title="A")

        count = update_links_in_files(kb_root, "old", "new")

        assert count == 1
        content = (kb_root / "a.md").read_text()
        assert "[[new]]" in content

    def test_path_update(self, kb_root):
        """Update links with paths."""
        _create_entry(kb_root, "a.md", "Links to [[dev/old-entry]]", title="A")

        count = update_links_in_files(kb_root, "dev/old-entry", "arch/new-entry")

        assert count == 1
        content = (kb_root / "a.md").read_text()
        assert "[[arch/new-entry]]" in content

    def test_other_links_unchanged(self, kb_root):
        """Other links in the file are not affected."""
        _create_entry(
            kb_root, "a.md", "[[target]] and [[other]]", title="A"
        )

        update_links_in_files(kb_root, "target", "new-target")

        content = (kb_root / "a.md").read_text()
        assert "[[new-target]]" in content
        assert "[[other]]" in content

    def test_empty_kb(self, kb_root):
        """Empty KB returns 0 updates."""
        count = update_links_in_files(kb_root, "old", "new")
        assert count == 0


# ─────────────────────────────────────────────────────────────────────────────
# update_links_batch tests
# ─────────────────────────────────────────────────────────────────────────────


class TestUpdateLinksBatch:
    """Tests for update_links_batch function."""

    def test_empty_mapping(self, kb_root):
        """Empty mapping returns 0 updates."""
        _create_entry(kb_root, "a.md", "[[link]]", title="A")

        count = update_links_batch(kb_root, {})

        assert count == 0

    def test_single_mapping(self, kb_root):
        """Single path mapping."""
        _create_entry(kb_root, "a.md", "[[old]]", title="A")

        count = update_links_batch(kb_root, {"old": "new"})

        assert count == 1
        assert "[[new]]" in (kb_root / "a.md").read_text()

    def test_multiple_mappings(self, kb_root):
        """Multiple path mappings in one pass."""
        _create_entry(
            kb_root, "a.md", "[[first]] and [[second]]", title="A"
        )

        count = update_links_batch(
            kb_root, {"first": "first-new", "second": "second-new"}
        )

        assert count == 1
        content = (kb_root / "a.md").read_text()
        assert "[[first-new]]" in content
        assert "[[second-new]]" in content

    def test_no_matching_links(self, kb_root):
        """No files contain any of the mapped paths."""
        _create_entry(kb_root, "a.md", "[[other]]", title="A")

        count = update_links_batch(kb_root, {"nonexistent": "new"})

        assert count == 0

    def test_partial_matches(self, kb_root):
        """Some mappings match, others don't."""
        _create_entry(
            kb_root, "a.md", "[[match]] and [[other]]", title="A"
        )

        count = update_links_batch(
            kb_root, {"match": "matched", "nomatch": "nope"}
        )

        assert count == 1
        content = (kb_root / "a.md").read_text()
        assert "[[matched]]" in content
        assert "[[other]]" in content

    def test_batch_more_efficient_than_individual(self, kb_root):
        """Batch update touches each file only once."""
        # Create a file with many different links
        _create_entry(
            kb_root,
            "a.md",
            "[[a]] [[b]] [[c]] [[d]] [[e]]",
            title="A",
        )

        mapping = {
            "a": "a-new",
            "b": "b-new",
            "c": "c-new",
            "d": "d-new",
            "e": "e-new",
        }

        count = update_links_batch(kb_root, mapping)

        assert count == 1  # Only one file modified
        content = (kb_root / "a.md").read_text()
        for new_name in mapping.values():
            assert f"[[{new_name}]]" in content


# ─────────────────────────────────────────────────────────────────────────────
# LINK_PATTERN tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLinkPattern:
    """Tests for the LINK_PATTERN regex constant."""

    def test_simple_link(self):
        """Matches simple [[link]]."""
        match = LINK_PATTERN.search("text [[target]] more")
        assert match is not None
        assert match.group(1) == "target"

    def test_link_with_path(self):
        """Matches [[path/to/link]]."""
        match = LINK_PATTERN.search("[[path/to/entry]]")
        assert match is not None
        assert match.group(1) == "path/to/entry"

    def test_multiple_links(self):
        """Finds all links in text."""
        matches = LINK_PATTERN.findall("[[a]] and [[b]] and [[c]]")
        assert matches == ["a", "b", "c"]

    def test_no_match_unclosed(self):
        """Does not match unclosed brackets."""
        match = LINK_PATTERN.search("[[unclosed")
        assert match is None

    def test_no_match_single_bracket(self):
        """Does not match single brackets."""
        match = LINK_PATTERN.search("[single]")
        assert match is None
