"""Static site generator for memex knowledge base.

Main orchestrator that processes KB entries and generates a complete
static HTML site with resolved wikilinks, search index, and theme assets.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from markdown_it import MarkdownIt

if TYPE_CHECKING:
    from ..models import EntryMetadata


@dataclass
class PublishConfig:
    """Configuration for site generation."""

    output_dir: Path = field(default_factory=lambda: Path("_site"))
    base_url: str = ""
    include_drafts: bool = False
    include_archived: bool = False
    clean: bool = True  # Remove output dir before build


@dataclass
class EntryData:
    """Processed entry for rendering."""

    path: str  # Relative path without .md
    title: str
    html_content: str
    metadata: EntryMetadata
    tags: list[str]
    backlinks: list[str] = field(default_factory=list)


@dataclass
class PublishResult:
    """Result of site generation."""

    entries_published: int
    broken_links: list[dict]  # [{source, target}]
    output_dir: str
    search_index_path: str


class SiteGenerator:
    """Generates static HTML site from memex KB.

    Orchestrates the full publishing pipeline:
    1. Build indices (title index, backlinks)
    2. Process markdown entries with link resolution
    3. Render HTML pages via Jinja2 templates
    4. Generate Lunr.js search index
    5. Copy theme assets
    """

    def __init__(self, config: PublishConfig, kb_root: Path):
        """Initialize generator.

        Args:
            config: Publishing configuration
            kb_root: Path to knowledge base root directory
        """
        self.config = config
        self.kb_root = kb_root
        self.title_index = None
        self.entries: dict[str, EntryData] = {}
        self.broken_links: list[dict] = []
        self.tags_index: dict[str, list[str]] = {}  # tag -> [paths]

    async def generate(self) -> PublishResult:
        """Generate the complete static site.

        Returns:
            PublishResult with statistics and output paths
        """
        from ..parser.links import resolve_backlinks
        from ..parser.title_index import build_title_index

        # Phase 1: Build indices
        self.title_index = build_title_index(self.kb_root, include_filename_index=True)
        backlinks_index = resolve_backlinks(self.kb_root, self.title_index)

        # Phase 2: Clean output directory
        if self.config.clean and self.config.output_dir.exists():
            shutil.rmtree(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 3: Process all entries
        await self._process_entries(backlinks_index)

        # Phase 4: Render all pages
        self._render_all_pages()

        # Phase 5: Generate search index
        search_index_path = self._generate_search_index()

        # Phase 6: Copy theme assets
        self._copy_theme_assets()

        return PublishResult(
            entries_published=len(self.entries),
            broken_links=self.broken_links,
            output_dir=str(self.config.output_dir),
            search_index_path=search_index_path,
        )

    async def _process_entries(self, backlinks_index: dict[str, list[str]]) -> None:
        """Process all markdown entries.

        Args:
            backlinks_index: Pre-computed backlinks index
        """
        from ..parser import ParseError, parse_entry

        for md_file in self.kb_root.rglob("*.md"):
            # Skip hidden files and directories
            if any(part.startswith("_") or part.startswith(".") for part in md_file.parts):
                continue

            rel_path = md_file.relative_to(self.kb_root)
            path_key = str(rel_path.with_suffix(""))

            try:
                metadata, content, _ = parse_entry(md_file)
            except ParseError:
                continue

            # Filter by status
            if metadata.status == "draft" and not self.config.include_drafts:
                continue
            if metadata.status == "archived" and not self.config.include_archived:
                continue

            # Render markdown with resolved wikilinks
            broken_links: set[str] = set()
            html_content = self._render_markdown(content, path_key, broken_links)

            # Track broken links
            for broken in broken_links:
                self.broken_links.append({"source": path_key, "target": broken})

            # Build entry data
            self.entries[path_key] = EntryData(
                path=path_key,
                title=metadata.title,
                html_content=html_content,
                metadata=metadata,
                tags=list(metadata.tags),
                backlinks=backlinks_index.get(path_key, []),
            )

            # Build tags index
            for tag in metadata.tags:
                if tag not in self.tags_index:
                    self.tags_index[tag] = []
                self.tags_index[tag].append(path_key)

    def _render_markdown(
        self, content: str, source_path: str, broken_links: set[str]
    ) -> str:
        """Render markdown content with resolved wikilinks.

        Args:
            content: Raw markdown content
            source_path: Path of source file (for relative link calculation)
            broken_links: Set to collect unresolved link targets

        Returns:
            Rendered HTML string
        """
        from ..parser.md_renderer import _wikilink_rule
        from .renderer import StaticWikilinkRenderer

        # Create a renderer class factory that captures our parameters
        title_index = self.title_index
        base_url = self.config.base_url

        class ConfiguredRenderer(StaticWikilinkRenderer):
            def __init__(self, parser=None):
                super().__init__(
                    parser,
                    title_index=title_index,
                    source_path=source_path,
                    base_url=base_url,
                    broken_links=broken_links,
                )

        # Create parser with our custom renderer class
        md = MarkdownIt(renderer_cls=ConfiguredRenderer)
        md.enable("table")
        md.inline.ruler.push("wikilink", _wikilink_rule)

        return md.render(content)

    def _render_all_pages(self) -> None:
        """Render all HTML pages."""
        from .templates import render_entry_page, render_index_page, render_tag_page

        # Entry pages
        for path_key, entry in self.entries.items():
            html_path = self.config.output_dir / f"{path_key}.html"
            html_path.parent.mkdir(parents=True, exist_ok=True)

            html = render_entry_page(
                entry=entry,
                base_url=self.config.base_url,
            )
            html_path.write_text(html, encoding="utf-8")

        # Index page
        index_html = render_index_page(
            entries=list(self.entries.values()),
            tags_index=self.tags_index,
            base_url=self.config.base_url,
        )
        (self.config.output_dir / "index.html").write_text(index_html, encoding="utf-8")

        # Tag pages
        tags_dir = self.config.output_dir / "tags"
        tags_dir.mkdir(exist_ok=True)

        for tag, paths in self.tags_index.items():
            tag_entries = [self.entries[p] for p in paths if p in self.entries]
            tag_html = render_tag_page(
                tag=tag,
                entries=tag_entries,
                base_url=self.config.base_url,
            )
            (tags_dir / f"{tag}.html").write_text(tag_html, encoding="utf-8")

    def _generate_search_index(self) -> str:
        """Generate Lunr.js search index.

        Returns:
            Path to the generated search index file
        """
        from .search_index import build_search_index

        index_data = build_search_index(self.entries)
        index_path = self.config.output_dir / "search-index.json"
        index_path.write_text(index_data, encoding="utf-8")

        return str(index_path)

    def _copy_theme_assets(self) -> None:
        """Copy CSS and JS theme assets to output directory."""
        assets_dir = self.config.output_dir / "assets"
        assets_dir.mkdir(exist_ok=True)

        # Theme files are bundled with the package
        theme_dir = Path(__file__).parent / "theme"

        for asset in theme_dir.glob("*"):
            if asset.is_file():
                shutil.copy(asset, assets_dir / asset.name)
