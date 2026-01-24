"""Unified relations graph built from wikilinks and frontmatter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Literal, cast

from .config import get_index_root, get_kb_roots_for_indexing, parse_scoped_path
from .models import RelationEdge, RelationNode, RelationsGraph, RelationsQueryResult
from .parser import ParseError, extract_links, parse_entry
from .parser.links import resolve_wikilink_target
from .parser.title_index import TitleIndex, build_title_index, resolve_link_target

CACHE_BASENAME = "relations_graph"


def _cache_path(scope: str | None) -> Path:
    root = get_index_root()
    root.mkdir(parents=True, exist_ok=True)
    suffix = f"_{scope}" if scope else ""
    return root / f"{CACHE_BASENAME}{suffix}.json"


def _kb_tree_mtime(kb_root: Path) -> float:
    latest = 0.0
    if not kb_root.exists():
        return latest
    for md_file in kb_root.rglob("*.md"):
        try:
            latest = max(latest, md_file.stat().st_mtime)
        except OSError:
            continue
    return latest


def _aggregate_kb_mtime(kb_roots: Iterable[Path]) -> float:
    latest = 0.0
    for root in kb_roots:
        latest = max(latest, _kb_tree_mtime(root))
    return latest


def _ensure_md_extension(path: str) -> str:
    if path.endswith(".md"):
        return path
    return f"{path}.md"


def _scoped_path(scope_label: str | None, relative_path: str) -> str:
    if scope_label:
        return f"@{scope_label}/{relative_path}"
    return relative_path


def _normalize_link_target(
    target: str,
    *,
    default_scope: str | None,
    scope_indices: dict[str | None, TitleIndex],
    source_no_ext: str,
) -> str | None:
    scope_label, relative = parse_scoped_path(target)

    # Scoped link explicitly targets another KB
    if scope_label is not None:
        title_index = scope_indices.get(scope_label)
        if title_index is None:
            return None
        resolved = resolve_link_target(relative, title_index, None)
        resolved = resolved or relative
        return _scoped_path(scope_label, _ensure_md_extension(resolved))

    # Unscoped link resolves within the current KB
    title_index = scope_indices.get(default_scope)
    if title_index is None:
        return None

    resolved = resolve_wikilink_target(source_no_ext, target, title_index)
    if resolved is None:
        return None
    return _scoped_path(default_scope, _ensure_md_extension(resolved))


def build_relations_graph(scope: str | None = None) -> RelationsGraph:
    """Build a relations graph from wikilinks and typed relations."""
    kb_roots = get_kb_roots_for_indexing(scope=scope)
    if not kb_roots:
        return RelationsGraph()

    scope_indices: dict[str | None, TitleIndex] = {}
    scope_files: dict[str | None, list[Path]] = {}
    known_nodes: set[str] = set()

    for scope_label, kb_root in kb_roots:
        scope_indices[scope_label] = cast(
            TitleIndex,
            build_title_index(kb_root, include_filename_index=True),
        )
        files = [
            md_file
            for md_file in kb_root.rglob("*.md")
            if not md_file.name.startswith("_")
        ]
        scope_files[scope_label] = files
        for md_file in files:
            rel_path = str(md_file.relative_to(kb_root))
            known_nodes.add(_scoped_path(scope_label, rel_path))

    nodes: dict[str, RelationNode] = {}
    edges: list[RelationEdge] = []
    edge_keys: set[tuple] = set()

    for scope_label, kb_root in kb_roots:
        files = scope_files.get(scope_label, [])
        for md_file in files:
            rel_path = str(md_file.relative_to(kb_root))
            source_node = _scoped_path(scope_label, rel_path)
            source_no_ext = str(Path(rel_path).with_suffix(""))

            try:
                metadata, content, _ = parse_entry(md_file)
            except (ParseError, OSError):
                continue

            nodes[source_node] = RelationNode(
                path=source_node,
                title=metadata.title,
                tags=list(metadata.tags),
                scope=scope_label,
            )

            # Wikilink edges (untyped)
            for target in extract_links(content):
                resolved = _normalize_link_target(
                    target,
                    default_scope=scope_label,
                    scope_indices=scope_indices,
                    source_no_ext=source_no_ext,
                )
                if resolved is None or resolved not in known_nodes:
                    continue
                key = (source_node, resolved, "wikilink", None, None)
                if key in edge_keys:
                    continue
                edge_keys.add(key)
                edges.append(
                    RelationEdge(
                        source=source_node,
                        target=resolved,
                        origin="wikilink",
                    )
                )

            # Typed relations (frontmatter)
            for relation in metadata.relations:
                target_scope, target_rel = parse_scoped_path(relation.path)
                target_scope = target_scope if target_scope is not None else scope_label
                target_path = _scoped_path(target_scope, _ensure_md_extension(target_rel))
                if target_path not in known_nodes:
                    continue
                key = (source_node, target_path, "relations", relation.type, None)
                if key in edge_keys:
                    continue
                edge_keys.add(key)
                edges.append(
                    RelationEdge(
                        source=source_node,
                        target=target_path,
                        origin="relations",
                        relation_type=relation.type,
                    )
                )

    return RelationsGraph(nodes=nodes, edges=edges)


def load_relations_graph(scope: str | None = None) -> tuple[RelationsGraph, float]:
    path = _cache_path(scope)
    if not path.exists():
        return RelationsGraph(), 0.0

    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return RelationsGraph(), 0.0

    kb_mtime = float(payload.get("kb_mtime", 0.0))
    graph = RelationsGraph.model_validate(payload.get("graph", {}))
    return graph, kb_mtime


def save_relations_graph(graph: RelationsGraph, kb_mtime: float, scope: str | None = None) -> None:
    path = _cache_path(scope)
    payload = {"kb_mtime": kb_mtime, "graph": graph.model_dump()}
    path.write_text(json.dumps(payload, indent=2))


def ensure_relations_graph(scope: str | None = None) -> RelationsGraph:
    graph, cached_mtime = load_relations_graph(scope=scope)
    kb_roots = [kb_root for _, kb_root in get_kb_roots_for_indexing(scope=scope)]
    current_mtime = _aggregate_kb_mtime(kb_roots)

    if current_mtime > cached_mtime or not graph.nodes:
        graph = build_relations_graph(scope=scope)
        save_relations_graph(graph, current_mtime, scope=scope)

    return graph


def _normalize_query_path(path: str, graph: RelationsGraph) -> str:
    scope_label, relative = parse_scoped_path(path)
    relative = _ensure_md_extension(relative)

    if scope_label is not None:
        candidate = _scoped_path(scope_label, relative)
        if candidate in graph.nodes:
            return candidate
        raise ValueError(f"Entry not found in graph: {candidate}")

    if relative in graph.nodes:
        return relative

    scoped_candidates = [
        _scoped_path(scope, relative)
        for scope in ("project", "user")
        if _scoped_path(scope, relative) in graph.nodes
    ]
    if len(scoped_candidates) == 1:
        return scoped_candidates[0]
    if len(scoped_candidates) > 1:
        raise ValueError(f"Ambiguous path '{path}'. Use @project/ or @user/ prefix.")
    raise ValueError(f"Entry not found in graph: {path}")


def query_relations_graph(
    path: str,
    *,
    depth: int = 1,
    direction: Literal["outgoing", "incoming", "both"] = "both",
    origin: set[Literal["wikilink", "relations"]] | None = None,
    relation_types: set[str] | None = None,
    scope: str | None = None,
) -> RelationsQueryResult:
    """Query the relations graph for a subgraph around a path."""
    graph = ensure_relations_graph(scope=scope)
    root = _normalize_query_path(path, graph)

    outgoing: dict[str, list[RelationEdge]] = {}
    incoming: dict[str, list[RelationEdge]] = {}
    for edge in graph.edges:
        outgoing.setdefault(edge.source, []).append(edge)
        incoming.setdefault(edge.target, []).append(edge)

    visited_nodes: set[str] = {root}
    collected_edges: list[RelationEdge] = []
    collected_edge_keys: set[tuple] = set()
    queue: list[tuple[str, int]] = [(root, 0)]

    def edge_allowed(edge: RelationEdge) -> bool:
        if origin and edge.origin not in origin:
            return False
        if relation_types and edge.relation_type not in relation_types:
            return False
        return True

    while queue:
        node, current_depth = queue.pop(0)
        if current_depth >= depth:
            continue

        neighbors: list[tuple[str, RelationEdge]] = []
        if direction in ("outgoing", "both"):
            for edge in outgoing.get(node, []):
                if edge_allowed(edge):
                    neighbors.append((edge.target, edge))
        if direction in ("incoming", "both"):
            for edge in incoming.get(node, []):
                if edge_allowed(edge):
                    neighbors.append((edge.source, edge))

        for neighbor, edge in neighbors:
            edge_key = (edge.source, edge.target, edge.origin, edge.relation_type, edge.score)
            if edge_key not in collected_edge_keys:
                collected_edge_keys.add(edge_key)
                collected_edges.append(edge)
            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                queue.append((neighbor, current_depth + 1))

    nodes = [graph.nodes[node] for node in visited_nodes if node in graph.nodes]
    return RelationsQueryResult(root=root, depth=depth, nodes=nodes, edges=collected_edges)

