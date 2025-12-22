"""REST API for KB Explorer web application."""

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..backlinks_cache import ensure_backlink_cache
from ..config import get_kb_root
from ..indexer import HybridSearcher
from ..beads_client import (
    get_beads_client,
    get_client_for_issue,
    get_kanban_for_project,
    list_registered_projects,
    resolve_issue,
    resolve_issues,
)
from ..models import BeadsIssue, IndexStatus, KBEntry, SearchResult
from ..parser import ParseError, extract_links, parse_entry, render_markdown


app = FastAPI(
    title="Voidlabs KB Explorer",
    description="Knowledge base explorer with graph visualization",
    version="1.0.0",
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-initialized searcher
_searcher: HybridSearcher | None = None


def _get_searcher() -> HybridSearcher:
    """Get the HybridSearcher, initializing lazily."""
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher()
        kb_root = get_kb_root()
        status = _searcher.status()
        if status.kb_files > 0 and (status.whoosh_docs == 0 or status.chroma_docs == 0):
            if kb_root.exists():
                _searcher.reindex(kb_root)
    return _searcher


def _get_backlink_index() -> dict[str, list[str]]:
    """Return cached backlink index."""
    kb_root = get_kb_root()
    return ensure_backlink_cache(kb_root)


# Response models
class SearchResponseAPI(BaseModel):
    """Search results response."""
    results: list[SearchResult]
    total: int


class EntryResponse(BaseModel):
    """Full entry response."""
    path: str
    title: str
    content: str
    content_html: str
    tags: list[str]
    created: str | None
    updated: str | None
    links: list[str]
    backlinks: list[str]


class TreeNode(BaseModel):
    """Tree node response."""
    name: str
    type: Literal["directory", "file"]
    path: str
    title: str | None = None
    children: list["TreeNode"] = []


class GraphNode(BaseModel):
    """Node in the knowledge graph."""
    id: str
    label: str
    path: str
    tags: list[str] = []
    group: str = "default"


class GraphEdge(BaseModel):
    """Edge in the knowledge graph."""
    source: str
    target: str


class GraphData(BaseModel):
    """Full graph data."""
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class StatsResponse(BaseModel):
    """KB statistics response."""
    total_entries: int
    total_tags: int
    total_links: int
    categories: list[dict]
    recent_entries: list[dict]


# Beads response models


class RegisteredProjectResponse(BaseModel):
    """Registered beads project."""
    prefix: str
    path: str
    available: bool


class BeadsConfigResponse(BaseModel):
    """Beads configuration info."""
    available: bool
    project: str | None = None
    beads_root: str | None = None
    registered_projects: list[RegisteredProjectResponse] = []


class BeadsIssueResponse(BaseModel):
    """Issue for API response."""
    id: str
    title: str
    description: str | None
    status: str
    priority: int
    priority_label: str  # "critical", "high", "medium", "low", "backlog"
    issue_type: str
    created_at: str
    updated_at: str
    closed_at: str | None
    close_reason: str | None = None
    dependency_count: int
    dependent_count: int


class BeadsCommentResponse(BaseModel):
    """Comment for API response."""
    id: str
    content: str
    content_html: str  # Rendered markdown
    author: str
    created_at: str


class BeadsIssueDetailResponse(BaseModel):
    """Detailed issue with comments for modal view."""
    issue: BeadsIssueResponse
    comments: list[BeadsCommentResponse]


class BeadsKanbanColumnResponse(BaseModel):
    """Kanban column for API response."""
    status: str
    label: str
    issues: list[BeadsIssueResponse]


class BeadsKanbanResponse(BaseModel):
    """Kanban board data."""
    project: str
    columns: list[BeadsKanbanColumnResponse]
    total_issues: int


class EntryBeadsResponse(BaseModel):
    """Beads info for a KB entry."""
    linked_issues: list[BeadsIssueResponse]
    project_issues: list[BeadsIssueResponse] | None = None


# API Routes

@app.get("/api/search", response_model=SearchResponseAPI)
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    mode: Literal["hybrid", "keyword", "semantic"] = "hybrid",
):
    """Search the knowledge base."""
    searcher = _get_searcher()
    results = searcher.search(q, limit=limit, mode=mode)
    return SearchResponseAPI(results=results, total=len(results))


# NOTE: This route must come BEFORE the generic /api/entries/{path:path} route
# because FastAPI's {path:path} is greedy and would match "/beads" as part of the path
@app.get("/api/entries/{path:path}/beads", response_model=EntryBeadsResponse)
async def get_entry_beads(path: str):
    """Get beads issues linked from a KB entry.

    Uses the beads registry to resolve issues from any registered project.
    """
    kb_root = get_kb_root()

    # Ensure .md extension
    if not path.endswith(".md"):
        path = f"{path}.md"

    file_path = kb_root / path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Entry not found: {path}")

    try:
        metadata, content, _ = parse_entry(file_path)
    except ParseError as e:
        raise HTTPException(status_code=500, detail=f"Parse error: {e}")

    # Get specifically linked issues using registry-aware resolution
    linked_issues = []
    if metadata.beads_issues:
        linked_issues = resolve_issues(metadata.beads_issues)

    # Get project issues if linked to a project
    project_issues = None
    if metadata.beads_project:
        kanban = get_kanban_for_project(metadata.beads_project)
        if kanban:
            # Flatten all columns into a single list
            project_issues = []
            for col in kanban.columns:
                project_issues.extend(col.issues)

    return EntryBeadsResponse(
        linked_issues=[_format_beads_issue(i) for i in linked_issues],
        project_issues=[_format_beads_issue(i) for i in project_issues]
        if project_issues
        else None,
    )


@app.get("/api/entries/{path:path}", response_model=EntryResponse)
async def get_entry(path: str):
    """Get a single KB entry."""
    kb_root = get_kb_root()

    # Ensure .md extension
    if not path.endswith(".md"):
        path = f"{path}.md"

    file_path = kb_root / path

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Entry not found: {path}")

    try:
        metadata, content, _ = parse_entry(file_path)
    except ParseError as e:
        raise HTTPException(status_code=500, detail=f"Parse error: {e}")

    # Single-pass: get HTML and links from AST
    md_result = render_markdown(content)

    # Get backlinks
    all_backlinks = _get_backlink_index()
    path_key = path[:-3] if path.endswith(".md") else path
    backlinks = all_backlinks.get(path_key, [])

    return EntryResponse(
        path=path,
        title=metadata.title,
        content=content,
        content_html=md_result.html,
        tags=list(metadata.tags),
        created=metadata.created.isoformat() if metadata.created else None,
        updated=metadata.updated.isoformat() if metadata.updated else None,
        links=md_result.links,
        backlinks=backlinks,
    )


@app.get("/api/tree")
async def get_tree():
    """Get the KB directory tree."""
    kb_root = get_kb_root()

    def build_tree(path: Path, rel_path: str = "") -> list[TreeNode]:
        nodes = []
        try:
            items = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return nodes

        for item in items:
            if item.name.startswith(".") or item.name.startswith("_"):
                continue

            item_rel = f"{rel_path}/{item.name}" if rel_path else item.name

            if item.is_dir():
                children = build_tree(item, item_rel)
                nodes.append(TreeNode(
                    name=item.name,
                    type="directory",
                    path=item_rel,
                    children=children,
                ))
            elif item.suffix == ".md":
                title = None
                try:
                    metadata, _, _ = parse_entry(item)
                    title = metadata.title
                except ParseError:
                    pass
                nodes.append(TreeNode(
                    name=item.name,
                    type="file",
                    path=item_rel,
                    title=title,
                ))

        return nodes

    return build_tree(kb_root)


@app.get("/api/graph", response_model=GraphData)
async def get_graph():
    """Get the full knowledge graph."""
    kb_root = get_kb_root()

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    seen_nodes: set[str] = set()

    # First pass: build nodes and create title->path mapping
    title_to_path: dict[str, str] = {}
    entry_links: dict[str, list[str]] = {}  # path_key -> raw links

    for md_file in kb_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue

        rel_path = str(md_file.relative_to(kb_root))
        path_key = rel_path[:-3] if rel_path.endswith(".md") else rel_path

        if path_key in seen_nodes:
            continue
        seen_nodes.add(path_key)

        try:
            metadata, content, _ = parse_entry(md_file)
            links = extract_links(content)
        except ParseError:
            continue

        # Build title->path mapping
        title_to_path[metadata.title] = path_key
        entry_links[path_key] = links

        # Determine group from category
        parts = Path(rel_path).parts
        group = parts[0] if len(parts) > 1 else "root"

        nodes.append(GraphNode(
            id=path_key,
            label=metadata.title,
            path=rel_path,
            tags=list(metadata.tags),
            group=group,
        ))

    # Second pass: resolve links to actual node IDs
    for source_path, links in entry_links.items():
        for link in links:
            # Try to resolve the link target
            link_key = link[:-3] if link.endswith(".md") else link

            # Check if it's already a valid path
            if link_key in seen_nodes:
                target = link_key
            # Check if it's a title
            elif link in title_to_path:
                target = title_to_path[link]
            elif link_key in title_to_path:
                target = title_to_path[link_key]
            else:
                # Link target doesn't exist, skip it
                continue

            edges.append(GraphEdge(source=source_path, target=target))

    return GraphData(nodes=nodes, edges=edges)


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get KB statistics."""
    kb_root = get_kb_root()

    total_entries = 0
    all_tags: set[str] = set()
    total_links = 0
    categories: dict[str, int] = {}
    recent_entries: list[dict] = []

    for md_file in kb_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue

        rel_path = str(md_file.relative_to(kb_root))
        total_entries += 1

        # Count by category
        parts = Path(rel_path).parts
        cat = parts[0] if len(parts) > 1 else "root"
        categories[cat] = categories.get(cat, 0) + 1

        try:
            metadata, content, _ = parse_entry(md_file)
            links = extract_links(content)
            total_links += len(links)
            all_tags.update(metadata.tags)

            recent_entries.append({
                "path": rel_path,
                "title": metadata.title,
                "created": metadata.created.isoformat() if metadata.created else None,
                "updated": metadata.updated.isoformat() if metadata.updated else None,
            })
        except ParseError:
            continue

    # Sort recent entries
    recent_entries.sort(
        key=lambda x: x.get("updated") or x.get("created") or "",
        reverse=True
    )

    return StatsResponse(
        total_entries=total_entries,
        total_tags=len(all_tags),
        total_links=total_links,
        categories=[{"name": k, "count": v} for k, v in sorted(categories.items())],
        recent_entries=recent_entries[:10],
    )


@app.get("/api/tags")
async def get_tags():
    """Get all tags with counts."""
    kb_root = get_kb_root()
    tag_counts: dict[str, int] = {}

    for md_file in kb_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue

        try:
            metadata, _, _ = parse_entry(md_file)
            for tag in metadata.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        except ParseError:
            continue

    return [
        {"tag": tag, "count": count}
        for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])
    ]


@app.get("/api/recent")
async def get_recent(limit: int = 10):
    """Get recently updated entries."""
    kb_root = get_kb_root()
    entries: list[dict] = []

    for md_file in kb_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue

        rel_path = str(md_file.relative_to(kb_root))

        try:
            metadata, _, _ = parse_entry(md_file)
            entries.append({
                "path": rel_path,
                "title": metadata.title,
                "tags": list(metadata.tags),
                "created": metadata.created.isoformat() if metadata.created else None,
                "updated": metadata.updated.isoformat() if metadata.updated else None,
            })
        except ParseError:
            continue

    entries.sort(
        key=lambda x: x.get("updated") or x.get("created") or "",
        reverse=True
    )

    return entries[:limit]


# Beads API Routes


def _format_beads_issue(issue: BeadsIssue) -> BeadsIssueResponse:
    """Format issue for API response."""
    priority_labels = {
        0: "critical",
        1: "high",
        2: "medium",
        3: "low",
        4: "backlog",
    }
    return BeadsIssueResponse(
        id=issue.id,
        title=issue.title,
        description=issue.description,
        status=issue.status,
        priority=issue.priority,
        priority_label=priority_labels.get(issue.priority, "medium"),
        issue_type=issue.issue_type,
        created_at=issue.created_at.isoformat(),
        updated_at=issue.updated_at.isoformat(),
        closed_at=issue.closed_at.isoformat() if issue.closed_at else None,
        close_reason=issue.close_reason,
        dependency_count=issue.dependency_count,
        dependent_count=issue.dependent_count,
    )


@app.get("/api/beads/config", response_model=BeadsConfigResponse)
async def get_beads_config():
    """Get beads integration status and configuration."""
    client = get_beads_client()
    registered = list_registered_projects()
    return BeadsConfigResponse(
        available=client.is_available,
        project=client.get_project_name() if client.is_available else None,
        beads_root=str(client.beads_root) if client.beads_root else None,
        registered_projects=[
            RegisteredProjectResponse(**p) for p in registered
        ],
    )


@app.get("/api/beads/issues", response_model=list[BeadsIssueResponse])
async def list_beads_issues(
    status: Literal["open", "in_progress", "closed", "all"] = "all",
    limit: int = Query(50, ge=1, le=500),
    priority: int | None = Query(None, ge=0, le=4),
):
    """List beads issues with optional filters."""
    client = get_beads_client()
    if not client.is_available:
        raise HTTPException(status_code=404, detail="Beads not available for this KB")

    issues = client.list_issues(status=status, limit=limit, priority=priority)
    return [_format_beads_issue(i) for i in issues]


@app.get("/api/beads/kanban", response_model=BeadsKanbanResponse)
async def get_beads_kanban(project: str | None = None):
    """Get issues grouped by status for kanban display.

    Args:
        project: Project prefix to show (e.g., "dv", "kb").
                 Uses registry for cross-project lookups.
                 If None, uses the default KB beads project.
    """
    kanban_data = None

    if project:
        # Try registry-based lookup for cross-project
        kanban_data = get_kanban_for_project(project)
    else:
        # Use default KB client
        client = get_beads_client()
        if client.is_available:
            kanban_data = client.get_kanban_data()

    if not kanban_data:
        raise HTTPException(
            status_code=404,
            detail=f"Beads project not found: {project}" if project else "No beads project available"
        )

    return BeadsKanbanResponse(
        project=kanban_data.project,
        columns=[
            BeadsKanbanColumnResponse(
                status=col.status,
                label=col.label,
                issues=[_format_beads_issue(i) for i in col.issues],
            )
            for col in kanban_data.columns
        ],
        total_issues=kanban_data.total_issues,
    )


@app.get("/api/beads/issues/{issue_id}", response_model=BeadsIssueDetailResponse)
async def get_beads_issue_detail(issue_id: str):
    """Get detailed issue information including comments.

    This endpoint fetches the full issue data plus any comments,
    supporting cross-project issue resolution via the registry.
    """
    # Get the appropriate client for this issue (handles cross-project)
    client = get_client_for_issue(issue_id)
    if client is None:
        # Fall back to default client
        client = get_beads_client()

    if not client or not client.is_available:
        raise HTTPException(status_code=404, detail="Beads not available")

    # Get the issue
    issue = client.get_issue(issue_id)
    if not issue:
        raise HTTPException(status_code=404, detail=f"Issue not found: {issue_id}")

    # Get comments for the issue
    comments = client.get_comments(issue_id)

    return BeadsIssueDetailResponse(
        issue=_format_beads_issue(issue),
        comments=[
            BeadsCommentResponse(
                id=c.id,
                content=c.content,
                content_html=render_markdown(c.content).html,
                author=c.author,
                created_at=c.created_at.isoformat(),
            )
            for c in comments
        ],
    )


# Mount static files and serve index
static_dir = Path(__file__).parent / "static"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Serve the main app."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "KB Explorer API", "docs": "/docs"}


def main():
    """Run the webapp server."""
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
