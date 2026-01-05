"""Static site wikilink renderer with href resolution.

Converts [[wikilinks]] to proper HTML anchors with resolved paths,
suitable for static site generation.
"""

from __future__ import annotations

import html
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from markdown_it.renderer import RendererHTML

if TYPE_CHECKING:
    from ..parser.title_index import TitleIndex


class StaticWikilinkRenderer(RendererHTML):
    """Renderer that resolves wikilinks to relative HTML paths.

    Unlike the standard WikilinkRenderer which outputs data-path attributes
    for client-side resolution, this renderer resolves links at build time
    to actual relative HTML file paths.
    """

    def __init__(
        self,
        parser=None,
        *,
        title_index: TitleIndex = None,
        source_path: str = "",
        base_url: str = "",
        broken_links: set[str] | None = None,
    ):
        """Initialize renderer with link resolution context.

        Args:
            parser: markdown-it parser instance (passed by MarkdownIt)
            title_index: Index for resolving [[Title]] style links
            source_path: Path of the source file (relative, no .md extension)
            base_url: Base URL prefix for generated links (e.g., "/my-kb")
            broken_links: Set to collect broken link targets (mutated in place)
        """
        super().__init__(parser)
        self.title_index = title_index
        self.source_path = source_path
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.broken_links = broken_links if broken_links is not None else set()
        self.resolved_links: set[str] = set()  # Track successfully resolved links

    def wikilink(self, tokens, idx, options, env):
        """Render wikilink to HTML anchor with resolved href."""
        token = tokens[idx]
        target = token.meta.get("target", "")
        display = token.content

        # Resolve the link target
        resolved = self._resolve_link(target)

        if resolved:
            href = self._build_href(resolved)
            css_class = "wikilink"
            self.resolved_links.add(resolved)  # Track for graph
        else:
            # Broken link - mark but don't fail
            href = "#"
            css_class = "wikilink wikilink-broken"
            self.broken_links.add(target)

        display_escaped = html.escape(display)
        return f'<a href="{href}" class="{css_class}">{display_escaped}</a>'

    def _resolve_link(self, target: str) -> str | None:
        """Resolve a wikilink target to a path.

        Args:
            target: The wikilink target (path, title, or alias)

        Returns:
            Resolved path without .md extension, or None if not found
        """
        from ..parser.title_index import resolve_link_target

        return resolve_link_target(target, self.title_index, self.source_path)

    def _build_href(self, target_path: str) -> str:
        """Build href from source to target.

        Args:
            target_path: Target path (relative, no .md extension)

        Returns:
            Relative or absolute href to the .html file
        """
        source = PurePosixPath(self.source_path)
        target = PurePosixPath(target_path)

        # Calculate relative path from source's directory to target
        source_dir = source.parent
        try:
            # Try to compute relative path
            rel_parts = []

            # Walk up from source to common ancestor
            src_parts = list(source_dir.parts)
            tgt_parts = list(target.parts)

            # Find common prefix length
            common = 0
            for a, b in zip(src_parts, tgt_parts):
                if a != b:
                    break
                common += 1

            # Add ".." for each level up from source
            up_count = len(src_parts) - common
            rel_parts.extend([".."] * up_count)

            # Add path down to target
            rel_parts.extend(tgt_parts[common:])

            if rel_parts:
                rel_path = "/".join(rel_parts)
            else:
                rel_path = target.name

        except Exception:
            # Fallback to absolute path
            rel_path = str(target)

        # Add .html extension
        href = f"{rel_path}.html"

        # Prepend base URL if set and using absolute paths
        if self.base_url and href.startswith("/"):
            href = f"{self.base_url}{href}"

        return href
