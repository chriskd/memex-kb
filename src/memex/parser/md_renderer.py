"""Markdown rendering with markdown-it-py and wikilink support.

This module provides AST-based markdown parsing that:
1. Renders markdown to HTML with proper GFM table support
2. Extracts [[wikilinks]] in a single pass
3. Supports aliased links: [[target|Display Text]]
"""

import html
import re
from dataclasses import dataclass

from markdown_it import MarkdownIt
from markdown_it.renderer import RendererHTML
from markdown_it.rules_inline import StateInline

# Pattern for [[target]] or [[target|alias]]
# Groups: 1=target, 2=alias (optional, without the |)
WIKILINK_PATTERN = re.compile(r"\[\[([^|\]\n]+)(?:\|([^\]\n]+))?\]\]")


@dataclass
class MarkdownResult:
    """Result of markdown parsing."""

    html: str
    links: list[str]


def _wikilink_rule(state: StateInline, silent: bool) -> bool:
    """Inline rule to parse [[wikilinks]].

    Args:
        state: Parser state.
        silent: If True, don't emit tokens (just validate).

    Returns:
        True if a wikilink was matched, False otherwise.
    """
    # Quick check: must start with [[
    if state.src[state.pos : state.pos + 2] != "[[":
        return False

    # Try to match the full pattern
    match = WIKILINK_PATTERN.match(state.src, state.pos)
    if not match:
        return False

    if not silent:
        target = match.group(1).strip()
        alias = match.group(2)
        if alias:
            alias = alias.strip()

        token = state.push("wikilink", "", 0)
        token.meta = {"target": target, "alias": alias}
        token.content = alias if alias else target

    state.pos = match.end()
    return True


class WikilinkRenderer(RendererHTML):
    """Custom renderer that handles wikilink tokens."""

    def wikilink(self, tokens, idx, options, env):
        """Render a wikilink token to HTML."""
        token = tokens[idx]
        target = token.meta.get("target", "")
        display = token.content

        # Escape for HTML attributes and content
        target_escaped = html.escape(target, quote=True)
        display_escaped = html.escape(display)

        return f'<a href="#" class="wikilink" data-path="{target_escaped}">{display_escaped}</a>'


def _extract_wikilinks(tokens: list) -> list[str]:
    """Extract wikilink targets from token stream.

    Args:
        tokens: List of markdown-it tokens.

    Returns:
        List of unique wikilink targets (normalized).
    """
    seen: set[str] = set()
    links: list[str] = []

    def walk_tokens(token_list):
        for token in token_list:
            if token.type == "wikilink":
                target = token.meta.get("target", "")
                normalized = normalize_link(target)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    links.append(normalized)
            # Check inline children
            if token.children:
                walk_tokens(token.children)

    walk_tokens(tokens)
    return links


def normalize_link(link: str) -> str:
    """Normalize a link target.

    This is the canonical link normalization function used throughout the parser.

    - Strips whitespace
    - Removes .md extension
    - Normalizes path separators (backslash to forward slash)
    - Removes leading/trailing slashes

    Args:
        link: Raw link target.

    Returns:
        Normalized link target.
    """
    link = link.strip()

    # Remove .md extension if present
    if link.endswith(".md"):
        link = link[:-3]

    # Normalize path separators (use forward slashes)
    link = link.replace("\\", "/")

    # Remove leading/trailing slashes
    link = link.strip("/")

    return link


# Module-level parser instance (created lazily)
_parser: MarkdownIt | None = None


def _get_parser() -> MarkdownIt:
    """Get or create the markdown-it parser instance."""
    global _parser
    if _parser is None:
        _parser = MarkdownIt(renderer_cls=WikilinkRenderer)
        _parser.enable("table")
        _parser.inline.ruler.push("wikilink", _wikilink_rule)
    return _parser


def render_markdown(content: str) -> MarkdownResult:
    """Parse markdown content, returning HTML and extracted links.

    This performs a single parse to extract both the rendered HTML
    and any [[wikilinks]] in the content.

    Args:
        content: Markdown content to parse.

    Returns:
        MarkdownResult with html and links fields.
    """
    md = _get_parser()

    # Parse to get token stream
    tokens = md.parse(content)

    # Extract links from tokens
    links = _extract_wikilinks(tokens)

    # Render to HTML
    html_output = md.render(content)

    return MarkdownResult(html=html_output, links=links)


def extract_links_only(content: str) -> list[str]:
    """Extract wikilinks from markdown content (no HTML rendering).

    This is optimized to only parse tokens and extract links,
    avoiding the overhead of full HTML rendering.

    Args:
        content: Markdown content.

    Returns:
        List of normalized wikilink targets.
    """
    md = _get_parser()
    tokens = md.parse(content)
    return _extract_wikilinks(tokens)
