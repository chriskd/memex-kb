"""Tests for memex parsing modules.

Coverage:
- src/memex/frontmatter.py - frontmatter building and metadata management
- src/memex/parser/links.py - wikilink extraction and resolution
- src/memex/parser/markdown.py - entry parsing and chunking
- src/memex/parser/md_renderer.py - markdown rendering with wikilinks
- src/memex/core.slugify - title to slug conversion

Philosophy: Test behaviors, not regex internals. Use parametrize for variations.
Target: <2 seconds execution time.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from memex.core import slugify
from memex.frontmatter import build_frontmatter, create_new_metadata, update_metadata_for_edit
from memex.models import EntryMetadata
from memex.parser.links import (
    _resolve_relative_link,
    extract_links,
    resolve_backlinks,
    update_links_batch,
    update_links_in_files,
)
from memex.parser.markdown import ParseError, _chunk_by_h2, parse_entry
from memex.parser.md_renderer import MarkdownResult, normalize_link, render_markdown


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def kb_root(tmp_path: Path) -> Path:
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
tags:
  - test
created: 2024-01-15T00:00:00
---

{content}
"""
    else:
        full_content = content

    path.write_text(full_content, encoding="utf-8")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Frontmatter Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildFrontmatter:
    """Tests for build_frontmatter function."""

    def test_minimal_metadata_produces_valid_yaml(self):
        """Minimal metadata produces parseable YAML frontmatter."""
        metadata = EntryMetadata(
            title="Test Entry",
            tags=["python"],
            created=datetime(2024, 1, 15, 10, 30, 45),
        )

        result = build_frontmatter(metadata)

        # Verify structure
        assert result.startswith("---\n")
        assert result.endswith("---\n\n")

        # Parse YAML and verify fields
        yaml_content = result.split("---")[1]
        parsed = yaml.safe_load(yaml_content)
        assert parsed["title"] == "Test Entry"
        assert parsed["tags"] == ["python"]

    def test_optional_fields_omitted_when_empty(self):
        """Optional fields are not included when they have default values."""
        metadata = EntryMetadata(
            title="Entry",
            tags=["test"],
            created=datetime(2024, 1, 1, 0, 0, 0),
        )

        result = build_frontmatter(metadata)

        assert "contributors:" not in result
        assert "aliases:" not in result
        assert "status:" not in result  # Default is 'published'
        assert "source_project:" not in result

    def test_all_optional_fields_included_when_set(self):
        """All optional fields appear when they have values."""
        metadata = EntryMetadata(
            title="Complete Entry",
            description="A one-line summary",
            tags=["python", "testing"],
            created=datetime(2024, 1, 1, 0, 0, 0),
            updated=datetime(2024, 1, 15, 14, 0, 0),
            contributors=["Alice", "Bob"],
            aliases=["complete", "full"],
            status="draft",
            source_project="main-project",
            edit_sources=["other-project"],
            model="claude-opus-4",
            git_branch="main",
            last_edited_by="ci-agent",
            beads_issues=["beads-001"],
            beads_project="tracker",
        )

        result = build_frontmatter(metadata)

        # All optional fields should be present
        assert "description: A one-line summary" in result
        assert "contributors:" in result
        assert "- Alice" in result
        assert "aliases:" in result
        assert "status: draft" in result
        assert "source_project: main-project" in result
        assert "edit_sources:" in result
        assert "model: claude-opus-4" in result
        assert "git_branch: main" in result
        assert "last_edited_by: ci-agent" in result
        assert "beads_issues:" in result
        assert "beads_project: tracker" in result

    def test_roundtrip_preserves_data(self):
        """Frontmatter can be parsed back to equivalent data."""
        original = EntryMetadata(
            title="Roundtrip Test",
            tags=["python", "testing"],
            created=datetime(2024, 1, 15, 10, 30, 0),
            contributors=["Alice"],
            aliases=["old-name"],
        )

        fm = build_frontmatter(original)
        yaml_content = fm.split("---")[1]
        parsed = yaml.safe_load(yaml_content)

        assert parsed["title"] == original.title
        assert parsed["tags"] == original.tags
        assert parsed["contributors"] == original.contributors
        assert parsed["aliases"] == original.aliases


class TestCreateNewMetadata:
    """Tests for create_new_metadata function."""

    def test_creates_with_current_timestamp(self):
        """Created timestamp is set to current UTC time."""
        before = datetime.now(timezone.utc)
        metadata = create_new_metadata(title="New Entry", tags=["test"])
        after = datetime.now(timezone.utc)

        assert metadata.created.tzinfo == timezone.utc
        assert before <= metadata.created <= after
        assert metadata.updated is None

    def test_populates_optional_breadcrumbs(self):
        """Optional breadcrumb fields are set when provided."""
        metadata = create_new_metadata(
            title="Entry",
            tags=["test"],
            source_project="my-project",
            contributor="Alice",
            model="claude-opus-4",
            git_branch="feature/new",
            actor="ci-agent",
        )

        assert metadata.source_project == "my-project"
        assert metadata.contributors == ["Alice"]
        assert metadata.model == "claude-opus-4"
        assert metadata.git_branch == "feature/new"
        assert metadata.last_edited_by == "ci-agent"


class TestUpdateMetadataForEdit:
    """Tests for update_metadata_for_edit function."""

    @pytest.fixture
    def base_metadata(self) -> EntryMetadata:
        """Base metadata for update tests."""
        return EntryMetadata(
            title="Original Title",
            tags=["original"],
            created=datetime(2024, 1, 1, 0, 0, 0),
            source_project="original-project",
        )

    def test_preserves_immutable_fields(self, base_metadata: EntryMetadata):
        """Title, created date, and source_project are preserved."""
        updated = update_metadata_for_edit(base_metadata)

        assert updated.title == "Original Title"
        assert updated.created == datetime(2024, 1, 1, 0, 0, 0)
        assert updated.source_project == "original-project"

    def test_sets_updated_timestamp(self, base_metadata: EntryMetadata):
        """Updated timestamp is set to current UTC time."""
        before = datetime.now(timezone.utc)
        updated = update_metadata_for_edit(base_metadata)
        after = datetime.now(timezone.utc)

        assert updated.updated is not None
        assert updated.updated.tzinfo == timezone.utc
        assert before <= updated.updated <= after

    def test_adds_new_contributor_without_duplicates(self, base_metadata: EntryMetadata):
        """New contributors are added; duplicates are skipped."""
        base_metadata.contributors = ["Alice"]

        updated = update_metadata_for_edit(base_metadata, new_contributor="Bob")
        assert "Bob" in updated.contributors

        updated2 = update_metadata_for_edit(updated, new_contributor="Bob")
        assert updated2.contributors.count("Bob") == 1

    def test_adds_edit_source_unless_same_as_origin(self, base_metadata: EntryMetadata):
        """Edit sources added unless same as source_project."""
        updated = update_metadata_for_edit(base_metadata, edit_source="other-project")
        assert "other-project" in updated.edit_sources

        # Same as source_project should not be added
        updated2 = update_metadata_for_edit(base_metadata, edit_source="original-project")
        assert "original-project" not in updated2.edit_sources


# ─────────────────────────────────────────────────────────────────────────────
# Link Extraction Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractLinks:
    """Tests for extract_links function."""

    @pytest.mark.parametrize(
        "content,expected",
        [
            ("", []),
            ("No links here.", []),
            ("[[target]]", ["target"]),
            ("[[path/to/entry]]", ["path/to/entry"]),
            ("[[a]] and [[b]]", ["a", "b"]),
            ("[[target|Display]]", ["target"]),
            ("[[a]] and [[b|B]] and [[a]]", ["a", "b"]),  # Deduped
            ("[[entry.md]]", ["entry"]),  # .md stripped
        ],
    )
    def test_extract_wikilinks(self, content: str, expected: list[str]):
        """Wikilinks are extracted correctly."""
        links = extract_links(content)
        assert links == expected

    def test_links_in_code_blocks_ignored(self):
        """Links inside code blocks are not extracted."""
        content = """
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

    def test_links_in_inline_code_ignored(self):
        """Links inside inline code are not extracted."""
        content = "Regular [[valid]] and `[[not-valid]]` in code."
        links = extract_links(content)
        assert "valid" in links
        assert "not-valid" not in links


class TestNormalizeLink:
    """Tests for normalize_link function."""

    @pytest.mark.parametrize(
        "link,expected",
        [
            ("target", "target"),
            ("  target  ", "target"),
            ("entry.md", "entry"),
            (r"path\to\entry", "path/to/entry"),
            ("/path/to/entry/", "path/to/entry"),
            ("  /path\\to\\entry.md  ", "path/to/entry"),
            ("", ""),
            ("   ", ""),
            ("///", ""),
        ],
    )
    def test_normalization(self, link: str, expected: str):
        """Links are normalized correctly."""
        assert normalize_link(link) == expected


class TestResolveRelativeLink:
    """Tests for _resolve_relative_link function."""

    @pytest.mark.parametrize(
        "source,target,expected",
        [
            ("source", "target", "target"),
            ("source", "other/path", "other/path"),
            ("projects/alpha", "../beta", "beta"),
            ("a/b/c", "../../d", "d"),
            ("a/b", "./c", "a/c"),
            ("source", "/absolute/path", "absolute/path"),
            ("a/b", "../../../c", "c"),  # Beyond root
        ],
    )
    def test_resolution(self, source: str, target: str, expected: str):
        """Relative paths are resolved correctly."""
        assert _resolve_relative_link(source, target) == expected


class TestResolveBacklinks:
    """Tests for resolve_backlinks function."""

    def test_empty_kb_returns_empty_dict(self, kb_root: Path):
        """Empty KB returns empty backlinks dict."""
        assert resolve_backlinks(kb_root) == {}

    def test_single_link_creates_backlink(self, kb_root: Path):
        """A link from A to B creates a backlink in B."""
        _create_entry(kb_root, "a.md", "Links to [[b]]", title="A")
        _create_entry(kb_root, "b.md", "Target", title="B")

        backlinks = resolve_backlinks(kb_root)

        assert "b" in backlinks
        assert "a" in backlinks["b"]

    def test_multiple_files_linking_to_one(self, kb_root: Path):
        """Multiple sources linking to same target."""
        _create_entry(kb_root, "a.md", "[[target]]", title="A")
        _create_entry(kb_root, "b.md", "[[target]]", title="B")
        _create_entry(kb_root, "target.md", "Target", title="Target")

        backlinks = resolve_backlinks(kb_root)

        assert set(backlinks["target"]) == {"a", "b"}

    def test_title_based_resolution(self, kb_root: Path):
        """Links by title are resolved to paths."""
        _create_entry(kb_root, "a.md", "[[My Entry]]", title="A")
        _create_entry(kb_root, "b.md", "Content", title="My Entry")

        backlinks = resolve_backlinks(kb_root)

        assert "b" in backlinks
        assert "a" in backlinks["b"]


class TestUpdateLinksInFiles:
    """Tests for update_links_in_files function."""

    def test_updates_matching_links(self, kb_root: Path):
        """Links matching old_path are updated to new_path."""
        _create_entry(kb_root, "a.md", "See [[old-entry]] for details", title="A")

        count = update_links_in_files(kb_root, "old-entry", "new-entry")

        assert count == 1
        content = (kb_root / "a.md").read_text()
        assert "[[new-entry]]" in content
        assert "[[old-entry]]" not in content

    def test_no_matches_returns_zero(self, kb_root: Path):
        """No matching links returns count of 0."""
        _create_entry(kb_root, "a.md", "No relevant links", title="A")

        count = update_links_in_files(kb_root, "nonexistent", "new")

        assert count == 0


class TestUpdateLinksBatch:
    """Tests for update_links_batch function."""

    def test_multiple_mappings_single_pass(self, kb_root: Path):
        """Multiple path mappings are applied in a single pass."""
        _create_entry(kb_root, "a.md", "[[first]] and [[second]]", title="A")

        count = update_links_batch(
            kb_root, {"first": "first-new", "second": "second-new"}
        )

        assert count == 1
        content = (kb_root / "a.md").read_text()
        assert "[[first-new]]" in content
        assert "[[second-new]]" in content

    def test_empty_mapping_returns_zero(self, kb_root: Path):
        """Empty mapping returns 0 without modifying files."""
        _create_entry(kb_root, "a.md", "[[link]]", title="A")

        count = update_links_batch(kb_root, {})

        assert count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Markdown Parsing Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestParseEntry:
    """Tests for parse_entry function."""

    def test_parses_valid_entry(self, kb_root: Path):
        """Valid entry is parsed to metadata, content, and chunks."""
        entry = _create_entry(kb_root, "test.md", "# Content\n\nBody text", title="Test")

        metadata, content, chunks = parse_entry(entry)

        assert metadata.title == "Test"
        assert "test" in metadata.tags
        assert "Body text" in content
        assert len(chunks) >= 1

    def test_raises_on_missing_file(self, kb_root: Path):
        """ParseError raised for missing file."""
        missing = kb_root / "missing.md"

        with pytest.raises(ParseError) as exc:
            parse_entry(missing)
        assert "does not exist" in str(exc.value)

    def test_raises_on_missing_frontmatter(self, kb_root: Path):
        """ParseError raised for file without frontmatter."""
        no_fm = kb_root / "no-fm.md"
        no_fm.write_text("# Just content, no frontmatter")

        with pytest.raises(ParseError) as exc:
            parse_entry(no_fm)
        assert "frontmatter" in str(exc.value).lower()


class TestChunkByH2:
    """Tests for _chunk_by_h2 function."""

    @pytest.fixture
    def metadata(self) -> EntryMetadata:
        """Sample metadata for chunking tests."""
        return EntryMetadata(
            title="Test",
            tags=["test"],
            created=datetime(2024, 1, 1, 0, 0, 0),
        )

    def test_no_h2_headers_single_chunk(self, metadata: EntryMetadata):
        """Content without H2 headers becomes a single chunk."""
        content = "Just a paragraph.\n\nAnother paragraph."

        chunks = _chunk_by_h2("test.md", content, metadata)

        assert len(chunks) == 1
        assert chunks[0].section is None
        assert "Just a paragraph" in chunks[0].content

    def test_h2_headers_create_sections(self, metadata: EntryMetadata):
        """H2 headers split content into named sections."""
        content = """Intro text.

## Section One

Content of section one.

## Section Two

Content of section two.
"""
        chunks = _chunk_by_h2("test.md", content, metadata)

        sections = [c.section for c in chunks]
        assert None in sections  # Intro chunk
        assert "Section One" in sections
        assert "Section Two" in sections


# ─────────────────────────────────────────────────────────────────────────────
# Markdown Rendering Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRenderMarkdown:
    """Tests for render_markdown function."""

    def test_returns_markdown_result(self):
        """Returns MarkdownResult with html and links."""
        result = render_markdown("See [[target]] here.")

        assert isinstance(result, MarkdownResult)
        assert isinstance(result.html, str)
        assert isinstance(result.links, list)

    def test_extracts_wikilinks(self):
        """Wikilinks are extracted from content."""
        result = render_markdown("[[foo]] and [[bar|Display]]")

        assert result.links == ["foo", "bar"]

    def test_renders_wikilinks_to_html(self):
        """Wikilinks are rendered as anchor elements."""
        result = render_markdown("See [[target]]")

        assert 'class="wikilink"' in result.html
        assert 'data-path="target"' in result.html

    def test_renders_tables(self):
        """GFM tables are rendered correctly."""
        content = """
| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
"""
        result = render_markdown(content)

        assert "<table>" in result.html
        assert "<th>Header 1</th>" in result.html
        assert "<td>Cell 1</td>" in result.html

    def test_empty_content_returns_empty(self):
        """Empty content returns empty html and links."""
        result = render_markdown("")

        assert result.html == ""
        assert result.links == []


# ─────────────────────────────────────────────────────────────────────────────
# Slugify Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSlugify:
    """Tests for slugify function."""

    @pytest.mark.parametrize(
        "title,expected",
        [
            ("Hello World", "hello-world"),
            ("Test Entry", "test-entry"),
            ("UPPERCASE", "uppercase"),
            ("with_underscores", "with-underscores"),
            ("Test's Entry", "tests-entry"),
            ("Special! @#$% Chars", "special-chars"),
            ("Multiple   Spaces", "multiple-spaces"),
            ("Leading-Trailing-", "leading-trailing"),
            ("123 Numbers", "123-numbers"),
        ],
    )
    def test_basic_slugify(self, title: str, expected: str):
        """Titles are converted to URL-friendly slugs."""
        assert slugify(title) == expected

    @pytest.mark.parametrize(
        "title,expected",
        [
            ("Cafe", "cafe"),
            ("naive", "naive"),
        ],
    )
    def test_ascii_only_output(self, title: str, expected: str):
        """Unicode characters are removed, leaving ASCII only."""
        result = slugify(title)
        assert result == expected
        assert result.isascii()

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert slugify("") == ""

    def test_only_special_chars(self):
        """String with only special characters returns empty."""
        assert slugify("!@#$%") == ""
