"""HTML templates for static site generation.

Uses Jinja2 for templating with inline template definitions.
Templates include: base layout with 3-column grid, entry page, index page, and tag pages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jinja2 import Environment, BaseLoader, select_autoescape

if TYPE_CHECKING:
    from .generator import EntryData


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _safe(html: str) -> str:
    """Mark HTML as safe for Jinja2 (won't be escaped)."""
    from markupsafe import Markup
    return Markup(html)


def _build_file_tree(entries: list["EntryData"], current_path: str = "", base_url: str = "") -> str:
    """Build HTML for the sidebar file tree navigation.

    Args:
        entries: All entry data
        current_path: Currently viewed entry path (for highlighting)
        base_url: Base URL prefix for links (e.g., "/my-kb")

    Returns:
        HTML string for the tree structure
    """
    # Group entries by their top-level folder
    folders: dict[str, list["EntryData"]] = {}
    root_entries: list["EntryData"] = []

    for entry in sorted(entries, key=lambda e: e.path.lower()):
        parts = entry.path.split("/")
        if len(parts) > 1:
            folder = parts[0]
            if folder not in folders:
                folders[folder] = []
            folders[folder].append(entry)
        else:
            root_entries.append(entry)

    html_parts = ['<div class="tree" id="tree">']

    # Render folders
    for folder in sorted(folders.keys()):
        html_parts.append(f'''
            <div class="tree-folder">
                <div class="tree-folder-header">
                    <span class="tree-icon folder">📁</span>
                    <span class="tree-label">{_escape_html(folder)}</span>
                </div>
                <div class="tree-children">''')

        for entry in folders[folder]:
            active_class = ' active' if entry.path == current_path else ''
            # Get just the filename part for display
            filename = entry.path.split("/")[-1]
            html_parts.append(f'''
                    <a href="{base_url}/{entry.path}.html" class="tree-item{active_class}">
                        <span class="tree-icon file">◇</span>
                        <span class="tree-label">{_escape_html(filename)}</span>
                    </a>''')

        html_parts.append('''
                </div>
            </div>''')

    # Render root entries
    for entry in root_entries:
        active_class = ' active' if entry.path == current_path else ''
        html_parts.append(f'''
            <a href="{base_url}/{entry.path}.html" class="tree-item{active_class}">
                <span class="tree-icon file">◇</span>
                <span class="tree-label">{_escape_html(entry.path)}</span>
            </a>''')

    html_parts.append('</div>')
    return ''.join(html_parts)


def _build_recent_list(entries: list["EntryData"], current_path: str = "", base_url: str = "", limit: int = 20) -> str:
    """Build HTML for the recent entries list in sidebar.

    Args:
        entries: All entry data
        current_path: Currently viewed entry path (for highlighting)
        base_url: Base URL prefix for links
        limit: Maximum number of recent entries to show

    Returns:
        HTML string for the recent entries list
    """
    # Sort by created date, newest first
    sorted_entries = sorted(
        entries,
        key=lambda e: str(e.metadata.created) if e.metadata.created else "",
        reverse=True
    )[:limit]

    html_parts = ['<div class="tree" id="recent-list">']

    for entry in sorted_entries:
        active_class = ' active' if entry.path == current_path else ''
        html_parts.append(f'''
            <a href="{base_url}/{entry.path}.html" class="tree-item{active_class}">
                <span class="tree-icon file">◇</span>
                <span class="tree-label">{_escape_html(entry.title)}</span>
            </a>''')

    html_parts.append('</div>')
    return ''.join(html_parts)


def _build_tabbed_sidebar(entries: list["EntryData"], current_path: str = "", base_url: str = "") -> str:
    """Build HTML for the tabbed sidebar with Browse and Recent tabs.

    Args:
        entries: All entry data
        current_path: Currently viewed entry path (for highlighting)
        base_url: Base URL prefix for links

    Returns:
        HTML string for the complete tabbed sidebar content
    """
    tree_html = _build_file_tree(entries, current_path, base_url)
    recent_html = _build_recent_list(entries, current_path, base_url)

    return f'''
            <div class="nav-tabs">
                <button class="nav-tab active" data-tab="tree">Browse</button>
                <button class="nav-tab" data-tab="recent">Recent</button>
            </div>

            <div class="sidebar-section" id="tree-section">
                <div class="sidebar-header">Categories</div>
                {tree_html}
            </div>

            <div class="sidebar-section" id="recent-section" style="display: none;">
                <div class="sidebar-header">Recent Updates</div>
                {recent_html}
            </div>'''


def _build_link_panel(
    outlinks: list[str],
    backlinks: list[str],
    entries_dict: dict[str, "EntryData"],
    base_url: str,
) -> str:
    """Build HTML for the right panel with outlinks and backlinks.

    Args:
        outlinks: List of outgoing link paths
        backlinks: List of incoming link paths
        entries_dict: Dict mapping path to EntryData for title lookup
        base_url: Base URL for links

    Returns:
        HTML string for the panel content
    """
    html_parts = []

    # Outgoing links section
    html_parts.append('''
        <div class="panel-section">
            <div class="panel-header">Outgoing Links</div>
            <ul class="link-list">''')

    if outlinks:
        for path in sorted(outlinks):
            title = entries_dict.get(path, None)
            title_text = title.title if title else path.split("/")[-1]
            html_parts.append(f'''
                <li class="link-item">
                    <a href="{base_url}/{path}.html">
                        <div class="link-title">{_escape_html(title_text)}</div>
                        <div class="link-path">{_escape_html(path)}</div>
                    </a>
                </li>''')
    else:
        html_parts.append('<li class="empty-state">No outgoing links</li>')

    html_parts.append('''
            </ul>
        </div>''')

    # Backlinks section
    html_parts.append('''
        <div class="panel-section">
            <div class="panel-header">Backlinks</div>
            <ul class="link-list">''')

    if backlinks:
        for path in sorted(backlinks):
            title = entries_dict.get(path, None)
            title_text = title.title if title else path.split("/")[-1]
            html_parts.append(f'''
                <li class="link-item">
                    <a href="{base_url}/{path}.html">
                        <div class="link-title">{_escape_html(title_text)}</div>
                        <div class="link-path">{_escape_html(path)}</div>
                    </a>
                </li>''')
    else:
        html_parts.append('<li class="empty-state">No backlinks</li>')

    html_parts.append('''
            </ul>
        </div>''')

    return ''.join(html_parts)


def _base_layout(
    title: str,
    base_url: str,
    sidebar_html: str,
    main_html: str,
    panel_html: str,
    current_view: str = "reader",
    site_title: str = "Memex",
) -> str:
    """Generate complete HTML page with 3-column layout.

    Args:
        title: Page title
        base_url: Base URL for assets
        sidebar_html: HTML for left sidebar content
        main_html: HTML for main content area
        panel_html: HTML for right panel content
        current_view: Current view for nav highlighting ("reader" or "graph")
        site_title: Site title for header and <title> tag
    """
    reader_active = ' active' if current_view == 'reader' else ''
    graph_active = ' active' if current_view == 'graph' else ''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_escape_html(title)} - {_escape_html(site_title)}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="{base_url}/assets/style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdn.jsdelivr.net/npm/lunr@2.3.9/lunr.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
</head>
<body>
    <div class="app">
        <!-- Header -->
        <header class="header">
            <a href="{base_url}/" class="logo">
                <div class="logo-mark"></div>
                <div class="logo-text">{_escape_html(site_title)}<span> / knowledge</span></div>
            </a>

            <div class="search-container">
                <svg class="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="11" cy="11" r="8"></circle>
                    <path d="m21 21-4.35-4.35"></path>
                </svg>
                <input type="text" id="search-input" placeholder="Search knowledge base..." autocomplete="off">
                <div id="search-results"></div>
            </div>

            <div class="header-nav">
                <a href="{base_url}/" class="nav-link{reader_active}">Reader</a>
                <a href="{base_url}/graph.html" class="nav-link{graph_active}">Graph</a>
            </div>
        </header>

        <!-- Sidebar -->
        <aside class="sidebar">
            {sidebar_html}
            <div class="sidebar-footer">
                <a href="https://github.com/aaronsb/memex" target="_blank" rel="noopener">Powered by memex</a>
            </div>
        </aside>

        <!-- Main content -->
        <main class="main">
            {main_html}
        </main>

        <!-- Right panel -->
        <aside class="panel">
            {panel_html}
        </aside>
    </div>

    <script>window.BASE_URL = "{base_url}";</script>
    <script src="{base_url}/assets/search.js"></script>
    <script src="{base_url}/assets/sidebar.js"></script>
    <script>hljs.highlightAll(); mermaid.initialize({{startOnLoad: true, theme: 'dark'}});</script>
</body>
</html>
'''


# Entry page template content (inside reader-container)
ENTRY_TEMPLATE = """
<div class="reader-container">
    <article class="entry">
        <header class="entry-header">
            <div class="entry-path">{{ entry.path }}</div>
            <h1 class="entry-title">{{ entry.title }}</h1>
            <div class="entry-meta">
                {% if entry.metadata.created %}
                <div class="entry-meta-item">Created: <span>{{ entry.metadata.created }}</span></div>
                {% endif %}
            </div>
            {% if entry.tags %}
            <div class="entry-tags">
                {% for tag in entry.tags %}
                <a href="{{ base_url }}/tags/{{ tag }}.html" class="tag">{{ tag }}</a>
                {% endfor %}
            </div>
            {% endif %}
        </header>
        <div class="entry-content">
            {{ html_content }}
        </div>
    </article>
</div>
"""

# Index page template content
INDEX_TEMPLATE = """
<div class="index-container">
    <h1>Knowledge Base</h1>
    <section class="index-section">
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
    <section class="index-section">
        <h2>Tags</h2>
        <div class="tags-cloud">
            {% for tag, count in tags_with_counts %}
            <a href="{{ base_url }}/tags/{{ tag }}.html" class="tag">
                {{ tag }} ({{ count }})
            </a>
            {% endfor %}
        </div>
    </section>
</div>
"""

# Tag page template content
TAG_TEMPLATE = """
<div class="tag-page-container">
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
    <p class="back-link"><a href="{{ base_url }}/">← Back to index</a></p>
</div>
"""


def _get_env() -> Environment:
    """Create Jinja2 environment with autoescape enabled."""
    return Environment(
        loader=BaseLoader(),
        autoescape=select_autoescape(default=True, default_for_string=True),
    )


def render_entry_page(
    entry: "EntryData",
    base_url: str,
    site_title: str = "Memex",
    all_entries: list["EntryData"] | None = None,
    entries_dict: dict[str, "EntryData"] | None = None,
) -> str:
    """Render a single entry page with full 3-column layout.

    Args:
        entry: Entry data including content and metadata
        base_url: Base URL for links
        site_title: Site title for header and <title> tag
        all_entries: All entries for sidebar navigation
        entries_dict: Dict mapping path to EntryData for title lookup

    Returns:
        Complete HTML page string
    """
    env = _get_env()
    all_entries = all_entries or []
    entries_dict = entries_dict or {}

    # Build tabbed sidebar
    sidebar_html = _build_tabbed_sidebar(all_entries, entry.path, base_url)

    # Build main content
    tmpl = env.from_string(ENTRY_TEMPLATE)
    main_html = tmpl.render(
        entry=entry,
        base_url=base_url,
        html_content=_safe(entry.html_content),
    )

    # Build right panel
    panel_html = _build_link_panel(
        entry.outlinks,
        entry.backlinks,
        entries_dict,
        base_url,
    )

    return _base_layout(
        title=entry.title,
        base_url=base_url,
        sidebar_html=sidebar_html,
        main_html=main_html,
        panel_html=panel_html,
        current_view="reader",
        site_title=site_title,
    )


def render_index_page(
    entries: list["EntryData"],
    tags_index: dict[str, list[str]],
    base_url: str,
    site_title: str = "Memex",
) -> str:
    """Render the main index page with full 3-column layout.

    Args:
        entries: All entry data
        tags_index: Dict mapping tag -> list of entry paths
        base_url: Base URL for links
        site_title: Site title for header and <title> tag

    Returns:
        Complete HTML page string
    """
    env = _get_env()

    # Build tabbed sidebar
    sidebar_html = _build_tabbed_sidebar(entries, base_url=base_url)

    # Sort entries by created date (newest first)
    recent_entries = sorted(
        entries,
        key=lambda e: str(e.metadata.created) if e.metadata.created else "",
        reverse=True
    )[:20]

    # Build tags with counts
    tags_with_counts = sorted(
        [(tag, len(paths)) for tag, paths in tags_index.items()],
        key=lambda x: (-x[1], x[0])
    )

    tmpl = env.from_string(INDEX_TEMPLATE)
    main_html = tmpl.render(
        base_url=base_url,
        recent_entries=recent_entries,
        tags_with_counts=tags_with_counts,
    )

    # Empty panel for index page
    panel_html = '''
        <div class="panel-section">
            <div class="panel-header">Welcome</div>
            <p class="empty-state">Select an entry to view its connections</p>
        </div>
    '''

    return _base_layout(
        title="Home",
        base_url=base_url,
        sidebar_html=sidebar_html,
        main_html=main_html,
        panel_html=panel_html,
        current_view="reader",
        site_title=site_title,
    )


def render_tag_page(
    tag: str,
    entries: list["EntryData"],
    base_url: str,
    site_title: str = "Memex",
    all_entries: list["EntryData"] | None = None,
) -> str:
    """Render a tag listing page with full 3-column layout.

    Args:
        tag: The tag name
        entries: Entries with this tag
        base_url: Base URL for links
        site_title: Site title for header and <title> tag
        all_entries: All entries for sidebar navigation

    Returns:
        Complete HTML page string
    """
    env = _get_env()
    all_entries = all_entries or entries

    # Build tabbed sidebar
    sidebar_html = _build_tabbed_sidebar(all_entries, base_url=base_url)

    # Sort entries alphabetically by title
    sorted_entries = sorted(entries, key=lambda e: e.title.lower())

    tmpl = env.from_string(TAG_TEMPLATE)
    main_html = tmpl.render(
        tag=tag,
        base_url=base_url,
        entries=sorted_entries,
    )

    # Empty panel for tag page
    panel_html = '''
        <div class="panel-section">
            <div class="panel-header">Tag Info</div>
            <p class="empty-state">Entries tagged with this topic</p>
        </div>
    '''

    return _base_layout(
        title=f"Tag: {tag}",
        base_url=base_url,
        sidebar_html=sidebar_html,
        main_html=main_html,
        panel_html=panel_html,
        current_view="reader",
        site_title=site_title,
    )


def render_graph_page(
    base_url: str,
    site_title: str = "Memex",
    all_entries: list["EntryData"] | None = None,
) -> str:
    """Render the graph visualization page with full 3-column layout.

    Args:
        base_url: Base URL for links
        site_title: Site title for header and <title> tag
        all_entries: All entries for sidebar navigation

    Returns:
        Complete HTML page string with D3.js graph visualization
    """
    all_entries = all_entries or []

    # Build tabbed sidebar
    sidebar_html = _build_tabbed_sidebar(all_entries, base_url=base_url)

    # Full-page graph with D3.js force simulation
    main_html = """
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
                .attr('stroke', '#262b3a')
                .attr('stroke-opacity', 0.4)
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

            // Category colors matching webapp
            const categoryColors = {
                'devops': '#f59e0b',
                'infrastructure': '#22d3ee',
                'projects': '#10b981',
                'tooling': '#f43f5e',
                'development': '#a78bfa',
                'best-practices': '#f472b6',
                'default': '#8b5cf6'
            };

            function getNodeColor(d) {
                const path = d.id || '';
                const category = path.split('/')[0];
                return categoryColors[category] || categoryColors['default'];
            }

            node.append('circle')
                .attr('r', 8)
                .attr('fill', d => getNodeColor(d))
                .attr('stroke', d => getNodeColor(d))
                .attr('stroke-opacity', 0.3)
                .attr('stroke-width', 8)
                .style('filter', 'drop-shadow(0 0 4px currentColor)');

            node.append('text')
                .text(d => d.title)
                .attr('x', 14)
                .attr('y', 4)
                .attr('font-family', "'JetBrains Mono', monospace")
                .attr('font-size', '10px')
                .attr('fill', '#9ca3af')
                .attr('opacity', 0.8);

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

    # Empty panel for graph page
    panel_html = '''
        <div class="panel-section">
            <div class="panel-header">Graph View</div>
            <p class="empty-state">Click a node to navigate to that entry</p>
        </div>
    '''

    return _base_layout(
        title="Graph",
        base_url=base_url,
        sidebar_html=sidebar_html,
        main_html=main_html,
        panel_html=panel_html,
        current_view="graph",
        site_title=site_title,
    )
