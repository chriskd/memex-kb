"""HTML templates for static site generation.

Uses Jinja2 for templating with inline template definitions.
Templates include: base layout, entry page, index page, and tag pages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jinja2 import Environment, BaseLoader, select_autoescape

if TYPE_CHECKING:
    from .generator import EntryData


def _base_wrapper(title: str, base_url: str, content: str) -> str:
    """Wrap content in the base HTML template.

    This is a simple string formatting approach that avoids Jinja
    parsing issues with user content that might contain {{ }} syntax.
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_escape_html(title)} - Memex</title>
    <link rel="stylesheet" href="{base_url}/assets/style.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/styles/github-dark.min.css">
    <script src="https://cdn.jsdelivr.net/npm/lunr@2.3.9/lunr.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/highlight.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
</head>
<body>
    <nav class="nav">
        <a href="{base_url}/" class="nav-brand">Memex</a>
        <a href="{base_url}/graph.html" class="nav-link">Graph</a>
        <div class="search-container">
            <input type="text" id="search-input" placeholder="Search..." autocomplete="off">
            <div id="search-results"></div>
        </div>
    </nav>
    <main class="main">
        {content}
    </main>
    <script>window.BASE_URL = "{base_url}";</script>
    <script src="{base_url}/assets/search.js"></script>
    <script>hljs.highlightAll(); mermaid.initialize({{startOnLoad: true, theme: 'dark'}});</script>
</body>
</html>
"""


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# Entry page template - shows a single KB entry with metadata and backlinks
ENTRY_TEMPLATE = """
<article class="entry">
    <header class="entry-header">
        <h1>{{ entry.title }}</h1>
        <div class="entry-meta">
            {% if entry.metadata.created %}
            <span class="entry-date">{{ entry.metadata.created }}</span>
            {% endif %}
            {% if entry.tags %}
            <div class="entry-tags">
                {% for tag in entry.tags %}
                <a href="{{ base_url }}/tags/{{ tag }}.html" class="tag">{{ tag }}</a>
                {% endfor %}
            </div>
            {% endif %}
        </div>
    </header>
    <div class="entry-content">
        {{ html_content }}
    </div>
    {% if entry.backlinks %}
    <footer class="entry-backlinks">
        <h2>Backlinks</h2>
        <ul>
            {% for bl in entry.backlinks %}
            <li><a href="{{ base_url }}/{{ bl }}.html">{{ bl }}</a></li>
            {% endfor %}
        </ul>
    </footer>
    {% endif %}
</article>
"""

# Index page template - homepage with recent entries and tag cloud
INDEX_TEMPLATE = """
<div class="index">
    <h1>Knowledge Base</h1>
    <section class="recent-entries">
        <h2>Recent Entries</h2>
        <ul class="entry-list">
            {% for entry in recent_entries %}
            <li>
                <a href="{{ base_url }}/{{ entry.path }}.html">{{ entry.title }}</a>
                {% if entry.metadata.created %}
                <span class="entry-date">{{ entry.metadata.created }}</span>
                {% endif %}
            </li>
            {% endfor %}
        </ul>
    </section>
    <section class="tags-cloud">
        <h2>Tags</h2>
        <div class="tags">
            {% for tag, count in tags_with_counts %}
            <a href="{{ base_url }}/tags/{{ tag }}.html" class="tag">
                {{ tag }} ({{ count }})
            </a>
            {% endfor %}
        </div>
    </section>
</div>
"""

# Tag page template - lists all entries with a specific tag
TAG_TEMPLATE = """
<div class="tag-page">
    <h1>Tag: {{ tag }}</h1>
    <p class="tag-count">{{ entries | length }} entries</p>
    <ul class="entry-list">
        {% for entry in entries %}
        <li>
            <a href="{{ base_url }}/{{ entry.path }}.html">{{ entry.title }}</a>
            {% if entry.metadata.created %}
            <span class="entry-date">{{ entry.metadata.created }}</span>
            {% endif %}
        </li>
        {% endfor %}
    </ul>
    <p><a href="{{ base_url }}/">Back to index</a></p>
</div>
"""


def _get_env() -> Environment:
    """Create Jinja2 environment with autoescape enabled."""
    return Environment(
        loader=BaseLoader(),
        autoescape=select_autoescape(default=True, default_for_string=True),
    )


def _safe(html: str) -> str:
    """Mark HTML as safe for Jinja2 (won't be escaped)."""
    from markupsafe import Markup
    return Markup(html)


def render_entry_page(entry: "EntryData", base_url: str) -> str:
    """Render a single entry page.

    Args:
        entry: Entry data including content and metadata
        base_url: Base URL for links

    Returns:
        Complete HTML page string
    """
    env = _get_env()

    # Render the entry template (but NOT the html_content - it's already HTML)
    tmpl = env.from_string(ENTRY_TEMPLATE)
    # Pass html_content separately and mark it safe to avoid double-escaping
    content = tmpl.render(
        entry=entry,
        base_url=base_url,
        # html_content is already rendered HTML, mark as safe to prevent escaping
        html_content=_safe(entry.html_content),
    )

    return _base_wrapper(entry.title, base_url, content)


def render_index_page(
    entries: list["EntryData"],
    tags_index: dict[str, list[str]],
    base_url: str,
) -> str:
    """Render the main index page.

    Args:
        entries: All entry data
        tags_index: Dict mapping tag -> list of entry paths
        base_url: Base URL for links

    Returns:
        Complete HTML page string
    """
    env = _get_env()

    # Sort entries by created date (newest first), handling None dates
    recent_entries = sorted(
        entries,
        key=lambda e: str(e.metadata.created) if e.metadata.created else "",
        reverse=True
    )[:20]

    # Build tags with counts, sorted by count then name
    tags_with_counts = sorted(
        [(tag, len(paths)) for tag, paths in tags_index.items()],
        key=lambda x: (-x[1], x[0])
    )

    tmpl = env.from_string(INDEX_TEMPLATE)
    content = tmpl.render(
        base_url=base_url,
        recent_entries=recent_entries,
        tags_with_counts=tags_with_counts,
    )

    return _base_wrapper("Home", base_url, content)


def render_tag_page(
    tag: str,
    entries: list["EntryData"],
    base_url: str,
) -> str:
    """Render a tag listing page.

    Args:
        tag: The tag name
        entries: Entries with this tag
        base_url: Base URL for links

    Returns:
        Complete HTML page string
    """
    env = _get_env()

    # Sort entries alphabetically by title
    sorted_entries = sorted(entries, key=lambda e: e.title.lower())

    tmpl = env.from_string(TAG_TEMPLATE)
    content = tmpl.render(
        tag=tag,
        base_url=base_url,
        entries=sorted_entries,
    )

    return _base_wrapper(f"Tag: {tag}", base_url, content)


def render_graph_page(base_url: str) -> str:
    """Render the graph visualization page.

    Args:
        base_url: Base URL for links

    Returns:
        Complete HTML page string with D3.js graph visualization
    """
    # Full-page graph with D3.js force simulation
    # Graph data is loaded from graph.json
    content = """
<div class="graph-container">
    <div id="graph"></div>
    <div id="graph-tooltip" class="graph-tooltip"></div>
</div>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script>
(function() {
    const baseUrl = window.BASE_URL || '';

    fetch(baseUrl + '/graph.json')
        .then(r => r.json())
        .then(data => {
            const container = document.getElementById('graph');
            const tooltip = document.getElementById('graph-tooltip');
            const width = container.clientWidth || 960;
            const height = container.clientHeight || 600;

            const svg = d3.select('#graph')
                .append('svg')
                .attr('width', '100%')
                .attr('height', '100%')
                .attr('viewBox', [0, 0, width, height]);

            // Zoom behavior
            const g = svg.append('g');
            svg.call(d3.zoom()
                .extent([[0, 0], [width, height]])
                .scaleExtent([0.1, 4])
                .on('zoom', (event) => g.attr('transform', event.transform)));

            // Force simulation
            const simulation = d3.forceSimulation(data.nodes)
                .force('link', d3.forceLink(data.edges).id(d => d.id).distance(80))
                .force('charge', d3.forceManyBody().strength(-200))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(20));

            // Links
            const link = g.append('g')
                .attr('class', 'links')
                .selectAll('line')
                .data(data.edges)
                .join('line')
                .attr('stroke', '#666')
                .attr('stroke-opacity', 0.6)
                .attr('stroke-width', 1);

            // Nodes
            const node = g.append('g')
                .attr('class', 'nodes')
                .selectAll('g')
                .data(data.nodes)
                .join('g')
                .call(d3.drag()
                    .on('start', (event, d) => {
                        if (!event.active) simulation.alphaTarget(0.3).restart();
                        d.fx = d.x;
                        d.fy = d.y;
                    })
                    .on('drag', (event, d) => {
                        d.fx = event.x;
                        d.fy = event.y;
                    })
                    .on('end', (event, d) => {
                        if (!event.active) simulation.alphaTarget(0);
                        d.fx = null;
                        d.fy = null;
                    }));

            node.append('circle')
                .attr('r', 8)
                .attr('fill', '#4dabf7')
                .attr('stroke', '#228be6')
                .attr('stroke-width', 2);

            node.append('text')
                .text(d => d.title)
                .attr('x', 12)
                .attr('y', 4)
                .attr('font-size', '12px')
                .attr('fill', '#e9ecef');

            // Hover effects
            node.on('mouseover', (event, d) => {
                    tooltip.style.display = 'block';
                    tooltip.innerHTML = '<strong>' + d.title + '</strong>' +
                        (d.tags.length ? '<br>Tags: ' + d.tags.join(', ') : '');
                    tooltip.style.left = (event.pageX + 10) + 'px';
                    tooltip.style.top = (event.pageY + 10) + 'px';
                })
                .on('mouseout', () => {
                    tooltip.style.display = 'none';
                })
                .on('click', (event, d) => {
                    window.location.href = baseUrl + '/' + d.url;
                });

            // Simulation tick
            simulation.on('tick', () => {
                link.attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
                node.attr('transform', d => `translate(${d.x},${d.y})`);
            });
        });
})();
</script>
"""
    return _base_wrapper("Graph", base_url, content)
