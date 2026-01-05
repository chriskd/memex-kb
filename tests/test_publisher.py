"""Tests for static site publisher."""

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from memex.publisher.renderer import StaticWikilinkRenderer
from memex.publisher.search_index import build_search_index, _strip_html
from memex.publisher.templates import render_entry_page, render_index_page, render_tag_page


class TestStripHtml:
    """Tests for HTML stripping utility."""

    def test_strips_basic_tags(self):
        assert _strip_html("<p>Hello</p>") == "Hello"

    def test_strips_nested_tags(self):
        assert _strip_html("<div><p>Hello <strong>world</strong></p></div>") == "Hello world"

    def test_normalizes_whitespace(self):
        assert _strip_html("<p>Hello</p>   <p>World</p>") == "Hello World"

    def test_empty_string(self):
        assert _strip_html("") == ""


class TestStaticWikilinkRenderer:
    """Tests for wikilink resolution to HTML hrefs."""

    @pytest.fixture
    def mock_title_index(self):
        """Create a mock title index for testing."""
        return MagicMock(
            title_to_path={
                "beads": "tooling/beads",
                "python guide": "development/python",
            },
            filename_to_paths={
                "beads": ["tooling/beads"],
                "python": ["development/python"],
            },
        )

    def test_build_href_same_directory(self):
        """Links in same directory use relative path."""
        renderer = StaticWikilinkRenderer(
            title_index=None,
            source_path="tooling/memex",
            base_url="",
        )
        href = renderer._build_href("tooling/beads")
        assert href == "beads.html"

    def test_build_href_parent_directory(self):
        """Links to parent directory use ../"""
        renderer = StaticWikilinkRenderer(
            title_index=None,
            source_path="projects/memex/overview",
            base_url="",
        )
        href = renderer._build_href("tooling/beads")
        assert href == "../../tooling/beads.html"

    def test_build_href_child_directory(self):
        """Links to child directory."""
        renderer = StaticWikilinkRenderer(
            title_index=None,
            source_path="index",
            base_url="",
        )
        href = renderer._build_href("tooling/beads")
        assert href == "tooling/beads.html"

    def test_broken_link_tracking(self):
        """Broken links are tracked in the set."""
        broken = set()
        renderer = StaticWikilinkRenderer(
            title_index=MagicMock(
                title_to_path={},
                filename_to_paths={},
            ),
            source_path="test",
            base_url="",
            broken_links=broken,
        )

        # Mock the _resolve_link to return None (broken)
        renderer._resolve_link = lambda x: None

        # Create mock token
        token = MagicMock()
        token.meta = {"target": "nonexistent"}
        token.content = "Nonexistent"

        result = renderer.wikilink([token], 0, {}, {})

        assert "nonexistent" in broken
        assert 'class="wikilink wikilink-broken"' in result


class TestSearchIndex:
    """Tests for Lunr.js search index generation."""

    @pytest.fixture
    def mock_entry(self):
        """Create a mock EntryData."""
        entry = MagicMock()
        entry.title = "Test Entry"
        entry.tags = ["test", "example"]
        entry.html_content = "<p>This is test content.</p>"
        return entry

    def test_generates_valid_json(self, mock_entry):
        """Output is valid JSON."""
        entries = {"test/entry": mock_entry}
        result = build_search_index(entries)

        parsed = json.loads(result)
        assert "documents" in parsed
        assert "metadata" in parsed

    def test_documents_have_required_fields(self, mock_entry):
        """Documents have id, title, tags, content."""
        entries = {"test/entry": mock_entry}
        result = build_search_index(entries)
        parsed = json.loads(result)

        doc = parsed["documents"][0]
        assert doc["id"] == "test/entry"
        assert doc["title"] == "Test Entry"
        assert "test example" in doc["tags"]
        assert "test content" in doc["content"]

    def test_metadata_has_path(self, mock_entry):
        """Metadata includes .html path for navigation."""
        entries = {"test/entry": mock_entry}
        result = build_search_index(entries)
        parsed = json.loads(result)

        meta = parsed["metadata"]["test/entry"]
        assert meta["path"] == "test/entry.html"


class TestTemplates:
    """Tests for HTML template rendering."""

    @pytest.fixture
    def mock_entry(self):
        """Create a mock EntryData for template tests."""
        entry = MagicMock()
        entry.title = "Test Entry"
        entry.path = "test/entry"
        entry.tags = ["test"]
        entry.html_content = "<p>Content here</p>"
        entry.backlinks = ["other/entry"]
        entry.metadata = MagicMock()
        entry.metadata.created = date(2024, 1, 15)
        return entry

    def test_render_entry_page_includes_title(self, mock_entry):
        """Entry page includes title in heading."""
        html = render_entry_page(mock_entry, base_url="")
        assert "<title>Test Entry - Memex</title>" in html
        assert "<h1>Test Entry</h1>" in html

    def test_render_entry_page_includes_content(self, mock_entry):
        """Entry page includes rendered HTML content."""
        html = render_entry_page(mock_entry, base_url="")
        assert "<p>Content here</p>" in html

    def test_render_entry_page_includes_backlinks(self, mock_entry):
        """Entry page includes backlinks section."""
        html = render_entry_page(mock_entry, base_url="")
        assert "Backlinks" in html
        assert "other/entry.html" in html

    def test_render_entry_page_base_url(self, mock_entry):
        """Base URL is applied to asset paths."""
        html = render_entry_page(mock_entry, base_url="/my-kb")
        assert 'href="/my-kb/assets/style.css"' in html

    def test_render_index_page_structure(self, mock_entry):
        """Index page has expected sections."""
        html = render_index_page([mock_entry], {"test": ["test/entry"]}, base_url="")
        assert "Knowledge Base" in html
        assert "Recent Entries" in html
        assert "Tags" in html

    def test_render_tag_page_lists_entries(self, mock_entry):
        """Tag page lists entries with that tag."""
        html = render_tag_page("test", [mock_entry], base_url="")
        assert "Tag: test" in html
        assert "Test Entry" in html


class TestSiteGeneratorIntegration:
    """Integration tests for full site generation."""

    @pytest.fixture
    def temp_kb(self, tmp_path):
        """Create a temporary KB with test entries."""
        kb_root = tmp_path / "kb"
        kb_root.mkdir()

        # Create a simple entry
        entry1 = kb_root / "test.md"
        entry1.write_text("""---
title: Test Entry
tags: [test]
created: 2024-01-15
---

# Test Entry

This is a test with a [[link]].
""")

        # Create linked entry
        entry2 = kb_root / "link.md"
        entry2.write_text("""---
title: Link Target
tags: [test]
created: 2024-01-14
---

# Link Target

Content here.
""")

        return kb_root

    @pytest.mark.asyncio
    async def test_generates_site_structure(self, temp_kb, tmp_path):
        """Site generation creates expected files."""
        from memex.publisher import PublishConfig, SiteGenerator

        output_dir = tmp_path / "site"
        config = PublishConfig(output_dir=output_dir, base_url="")
        generator = SiteGenerator(config, temp_kb)

        result = await generator.generate()

        assert result.entries_published == 2
        assert (output_dir / "index.html").exists()
        assert (output_dir / "test.html").exists()
        assert (output_dir / "link.html").exists()
        assert (output_dir / "search-index.json").exists()
        assert (output_dir / "assets" / "style.css").exists()

    @pytest.mark.asyncio
    async def test_excludes_drafts_by_default(self, tmp_path):
        """Draft entries are excluded unless flag is set."""
        from memex.publisher import PublishConfig, SiteGenerator

        kb_root = tmp_path / "kb"
        kb_root.mkdir()

        # Create a draft entry
        draft = kb_root / "draft.md"
        draft.write_text("""---
title: Draft Entry
tags: [test]
created: 2024-01-15
status: draft
---

# Draft

This is a draft.
""")

        output_dir = tmp_path / "site"
        config = PublishConfig(output_dir=output_dir, include_drafts=False)
        generator = SiteGenerator(config, kb_root)

        result = await generator.generate()

        assert result.entries_published == 0
        assert not (output_dir / "draft.html").exists()

    @pytest.mark.asyncio
    async def test_includes_drafts_when_flag_set(self, tmp_path):
        """Draft entries are included when flag is set."""
        from memex.publisher import PublishConfig, SiteGenerator

        kb_root = tmp_path / "kb"
        kb_root.mkdir()

        # Create a draft entry
        draft = kb_root / "draft.md"
        draft.write_text("""---
title: Draft Entry
tags: [test]
created: 2024-01-15
status: draft
---

# Draft

This is a draft.
""")

        output_dir = tmp_path / "site"
        config = PublishConfig(output_dir=output_dir, include_drafts=True)
        generator = SiteGenerator(config, kb_root)

        result = await generator.generate()

        assert result.entries_published == 1
        assert (output_dir / "draft.html").exists()
