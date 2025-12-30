"""Markdown parsing with frontmatter and link extraction."""

from ..models import DocumentChunk, EntryMetadata
from .links import (
    extract_links,
    resolve_backlinks,
    update_links_batch,
    update_links_in_files,
)
from .markdown import ParseError, parse_entry
from .md_renderer import MarkdownResult, render_markdown
from .title_index import TitleIndex, build_title_index, resolve_link_target

__all__ = [
    "parse_entry",
    "ParseError",
    "EntryMetadata",
    "DocumentChunk",
    "extract_links",
    "resolve_backlinks",
    "update_links_in_files",
    "update_links_batch",
    "build_title_index",
    "resolve_link_target",
    "TitleIndex",
    "render_markdown",
    "MarkdownResult",
]
