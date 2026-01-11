"""REST API for KB Explorer web application."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query

log = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .events import Event, EventType, get_broadcaster

from ..backlinks_cache import ensure_backlink_cache
from ..config import get_kb_root
from ..indexer import HybridSearcher
from ..models import IndexStatus, KBEntry, SearchResult
from ..parser import ParseError, extract_links, parse_entry, render_markdown


app = FastAPI(
    title="Memex KB Explorer",
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


# API Routes


# File watcher for live reload
_file_watcher = None


@app.on_event("startup")
async def startup_event():
    """Start the file watcher on server startup."""
    global _file_watcher
    from .._logging import configure_logging
    from ..indexer.watcher import FileWatcher

    configure_logging()

    searcher = _get_searcher()
    kb_root = get_kb_root()

    _file_watcher = FileWatcher(
        searcher=searcher,
        kb_root=kb_root,
        debounce_seconds=2.0,  # Faster for live reload
    )
    _file_watcher.start()
    log.info("File watcher started for %s", kb_root)

    # Start heartbeat task
    asyncio.create_task(_heartbeat_loop())


@app.on_event("shutdown")
async def shutdown_event():
    """Stop the file watcher on server shutdown."""
    global _file_watcher
    if _file_watcher is not None:
        _file_watcher.stop()
        _file_watcher = None


async def _heartbeat_loop():
    """Send periodic heartbeat to keep SSE connections alive."""
    broadcaster = get_broadcaster()
    while True:
        await asyncio.sleep(30)
        await broadcaster.broadcast(Event(type=EventType.HEARTBEAT))


@app.get("/api/events")
async def events():
    """Server-Sent Events endpoint for live reload.

    Clients connect here to receive real-time notifications when
    KB files change. Events include:
    - file_changed: A markdown file was modified
    - file_created: A new markdown file was added
    - file_deleted: A markdown file was removed
    - reindex_complete: Re-indexing finished
    - heartbeat: Keep-alive signal (every 30s)
    """
    broadcaster = get_broadcaster()

    async def event_generator():
        # Send initial connection event
        yield Event(type=EventType.HEARTBEAT, data={"connected": True}).to_sse()

        async for event in broadcaster.subscribe():
            yield event.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


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
                except ParseError as e:
                    log.debug("Could not parse %s for tree: %s", item, e)
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
