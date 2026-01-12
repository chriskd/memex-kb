"""Tests for markdown-it-py based renderer."""

import pytest

from memex.parser.md_renderer import MarkdownResult, render_markdown


class TestWikilinks:
    """Test wikilink parsing and rendering."""

    def test_simple_wikilink(self):
        """Simple [[target]] link."""
        result = render_markdown("See [[foo/bar]] for details.")
        assert result.links == ["foo/bar"]
        assert 'data-path="foo/bar"' in result.html
        assert ">foo/bar</a>" in result.html

    def test_aliased_wikilink(self):
        """[[target|Display Text]] link."""
        result = render_markdown("Check [[path/to/doc|the documentation]].")
        assert result.links == ["path/to/doc"]
        assert 'data-path="path/to/doc"' in result.html
        assert ">the documentation</a>" in result.html

    def test_multiple_wikilinks(self):
        """Multiple links in content."""
        result = render_markdown("See [[foo]] and [[bar]] and [[foo]] again.")
        # Should deduplicate
        assert result.links == ["foo", "bar"]

    def test_wikilink_with_md_extension(self):
        """Link with .md extension should be normalized."""
        result = render_markdown("See [[doc.md]].")
        assert result.links == ["doc"]

    def test_wikilink_path_normalization(self):
        """Paths should be normalized."""
        result = render_markdown("See [[/path/to/doc/]].")
        assert result.links == ["path/to/doc"]

    def test_wikilink_html_escaping(self):
        """Special characters should be escaped."""
        result = render_markdown('See [[path/with"quote]].')
        assert 'data-path="path/with&quot;quote"' in result.html

    def test_wikilink_in_paragraph(self):
        """Wikilink in paragraph context."""
        result = render_markdown("First paragraph.\n\nSee [[link]] here.\n\nLast paragraph.")
        assert result.links == ["link"]
        assert "<p>" in result.html


class TestTables:
    """Test GFM table rendering."""

    def test_simple_table(self):
        """Basic GFM table."""
        content = """
| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
"""
        result = render_markdown(content)
        assert "<table>" in result.html
        assert "<thead>" in result.html
        assert "<th>Header 1</th>" in result.html
        assert "<td>Cell 1</td>" in result.html

    def test_table_with_wikilinks(self):
        """Table containing wikilinks."""
        content = """
| Topic | Link |
|-------|------|
| Foo   | [[foo/doc]] |
"""
        result = render_markdown(content)
        assert "<table>" in result.html
        assert result.links == ["foo/doc"]
        assert 'class="wikilink"' in result.html


class TestCodeBlocks:
    """Test code block rendering."""

    def test_fenced_code_block(self):
        """Fenced code block with language."""
        content = """
```python
def hello():
    print("world")
```
"""
        result = render_markdown(content)
        assert "<pre>" in result.html
        assert "<code" in result.html
        assert "def hello():" in result.html

    def test_wikilink_in_code_not_parsed(self):
        """Wikilinks inside code blocks should NOT be parsed."""
        content = """
```
This [[is not a link]]
```
"""
        result = render_markdown(content)
        # Should not extract as link
        assert result.links == []
        # Should appear as raw text in code
        assert "[[is not a link]]" in result.html

    def test_inline_code(self):
        """Inline code formatting."""
        result = render_markdown("Use `[[link]]` syntax.")
        assert "<code>" in result.html
        # Links in inline code should not be parsed as wikilinks
        # (they're just displayed as text)


class TestMixedContent:
    """Test complex mixed content."""

    def test_headers_and_links(self):
        """Headers with wikilinks."""
        content = """
# Main Title

See [[overview]] for introduction.

## Section

More about [[details]].
"""
        result = render_markdown(content)
        assert "<h1>Main Title</h1>" in result.html
        assert "<h2>Section</h2>" in result.html
        assert result.links == ["overview", "details"]

    def test_lists_with_links(self):
        """Lists containing wikilinks."""
        content = """
- First item with [[link1]]
- Second item with [[link2]]
"""
        result = render_markdown(content)
        assert "<ul>" in result.html
        assert "<li>" in result.html
        assert result.links == ["link1", "link2"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_content(self):
        """Empty string input."""
        result = render_markdown("")
        assert result.html == ""
        assert result.links == []

    def test_no_wikilinks(self):
        """Content with no wikilinks."""
        result = render_markdown("Just plain text.")
        assert result.links == []
        assert "<p>Just plain text.</p>" in result.html

    def test_malformed_wikilink(self):
        """Incomplete wikilink syntax."""
        result = render_markdown("See [[incomplete")
        assert result.links == []
        assert "[[incomplete" in result.html

    def test_nested_brackets(self):
        """Nested bracket patterns."""
        result = render_markdown("array[i][j] is not [[a link]]")
        assert result.links == ["a link"]


class TestReturnType:
    """Test MarkdownResult dataclass."""

    def test_result_type(self):
        """Result should be MarkdownResult."""
        result = render_markdown("test")
        assert isinstance(result, MarkdownResult)
        assert isinstance(result.html, str)
        assert isinstance(result.links, list)
