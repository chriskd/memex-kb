"""Hybrid search combining Whoosh BM25 and ChromaDB semantic search."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ..config import DEFAULT_SEARCH_LIMIT, RRF_K, get_kb_root
from ..models import DocumentChunk, IndexStatus, SearchResult
from .chroma_index import ChromaIndex
from .whoosh_index import WhooshIndex

if TYPE_CHECKING:
    from ..context import KBContext

SearchMode = Literal["hybrid", "keyword", "semantic"]


class HybridSearcher:
    """Hybrid search combining keyword (Whoosh) and semantic (Chroma) indices."""

    def __init__(
        self,
        whoosh_index: WhooshIndex | None = None,
        chroma_index: ChromaIndex | None = None,
    ):
        """Initialize the hybrid searcher.

        Args:
            whoosh_index: Whoosh index instance. Created if not provided.
            chroma_index: Chroma index instance. Created if not provided.
        """
        self._whoosh = whoosh_index or WhooshIndex()
        self._chroma = chroma_index or ChromaIndex()
        self._last_indexed: datetime | None = None

    def search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        mode: SearchMode = "hybrid",
        project_context: str | None = None,
        kb_context: KBContext | None = None,
    ) -> list[SearchResult]:
        """Search the knowledge base.

        Args:
            query: Search query string.
            limit: Maximum number of results.
            mode: Search mode - "hybrid", "keyword", or "semantic".
            project_context: Current project name for context boosting.
            kb_context: Project context for path-based boosting.

        Returns:
            List of search results, deduplicated by document path.
        """
        # Fetch more results to allow for deduplication
        fetch_limit = limit * 3

        if mode == "keyword":
            results = self._whoosh.search(query, limit=fetch_limit)
            results = self._apply_ranking_adjustments(query, results, project_context, kb_context)
        elif mode == "semantic":
            results = self._chroma.search(query, limit=fetch_limit)
            results = self._apply_ranking_adjustments(query, results, project_context, kb_context)
        else:
            results = self._hybrid_search(query, limit=fetch_limit, project_context=project_context, kb_context=kb_context)

        # Deduplicate by path, keeping highest-scoring chunk per document
        return self._deduplicate_by_path(results, limit)

    def _hybrid_search(
        self,
        query: str,
        limit: int,
        project_context: str | None = None,
        kb_context: KBContext | None = None,
    ) -> list[SearchResult]:
        """Perform hybrid search using Reciprocal Rank Fusion.

        Args:
            query: Search query string.
            limit: Maximum number of results.
            project_context: Current project name for context boosting.
            kb_context: Project context for path-based boosting.

        Returns:
            List of merged search results.
        """
        # Get results from both indices (fetch more to have good RRF merge)
        fetch_limit = limit * 3
        whoosh_results = self._whoosh.search(query, limit=fetch_limit)
        chroma_results = self._chroma.search(query, limit=fetch_limit)

        # If one index is empty, return the other
        if not whoosh_results and not chroma_results:
            return []
        if not whoosh_results:
            return self._apply_ranking_adjustments(query, chroma_results[:limit], project_context, kb_context)
        if not chroma_results:
            return self._apply_ranking_adjustments(query, whoosh_results[:limit], project_context, kb_context)

        # Apply RRF
        return self._rrf_merge(query, whoosh_results, chroma_results, limit, project_context, kb_context)

    def _rrf_merge(
        self,
        query: str,
        whoosh_results: list[SearchResult],
        chroma_results: list[SearchResult],
        limit: int,
        project_context: str | None = None,
        kb_context: KBContext | None = None,
    ) -> list[SearchResult]:
        """Merge results using Reciprocal Rank Fusion.

        RRF score: score(d) = sum(1 / (k + rank)) for each ranking list.

        Args:
            whoosh_results: Results from keyword search.
            chroma_results: Results from semantic search.
            limit: Maximum number of results to return.
            project_context: Current project name for context boosting.
            kb_context: Project context for path-based boosting.

        Returns:
            Merged and re-ranked results.
        """
        # Build result map for deduplication (key: path#section)
        result_map: dict[str, SearchResult] = {}
        rrf_scores: dict[str, float] = defaultdict(float)

        # Process Whoosh results
        for rank, result in enumerate(whoosh_results, start=1):
            key = f"{result.path}#{result.section or ''}"
            rrf_scores[key] += 1.0 / (RRF_K + rank)
            if key not in result_map:
                result_map[key] = result

        # Process Chroma results
        for rank, result in enumerate(chroma_results, start=1):
            key = f"{result.path}#{result.section or ''}"
            rrf_scores[key] += 1.0 / (RRF_K + rank)
            if key not in result_map:
                result_map[key] = result

        # Sort by RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        # Normalize scores to 0-1 range
        max_score = rrf_scores[sorted_keys[0]] if sorted_keys else 1.0
        max_score = max_score if max_score > 0 else 1.0

        # Build final results
        final_results = []
        for key in sorted_keys[:limit]:
            result = result_map[key]
            normalized_score = rrf_scores[key] / max_score
            final_results.append(
                SearchResult(
                    path=result.path,
                    title=result.title,
                    snippet=result.snippet,
                    score=normalized_score,
                    tags=result.tags,
                    section=result.section,
                    created=result.created,
                    updated=result.updated,
                    token_count=result.token_count,
                    source_project=result.source_project,
                )
            )

        return self._apply_ranking_adjustments(query, final_results, project_context, kb_context)

    def _apply_ranking_adjustments(
        self,
        query: str,
        results: list[SearchResult],
        project_context: str | None = None,
        kb_context: KBContext | None = None,
    ) -> list[SearchResult]:
        """Boost results with matching tags, project context, and KB context paths.

        Applies three types of boosts:
        1. Tag boost: +0.05 per matching tag in query
        2. Project context boost: +0.15 for entries from current project
        3. KB context path boost: +0.12 for entries matching .kbcontext paths
        """
        if not results:
            return results

        # Tag boost: +0.05 per matching tag
        tokens = {tok for tok in re.split(r"\W+", query.lower()) if tok}
        if tokens:
            for result in results:
                tag_tokens = {tag.lower() for tag in result.tags}
                overlap = tokens.intersection(tag_tokens)
                if overlap:
                    result.score += 0.05 * len(overlap)

        # Project context boost: +0.15 for entries from current project
        if project_context:
            for result in results:
                if result.source_project and result.source_project == project_context:
                    result.score += 0.15

        # KB context path boost: +0.12 for entries matching .kbcontext paths
        if kb_context:
            from ..context import matches_glob

            boost_paths = kb_context.get_all_boost_paths()
            if boost_paths:
                for result in results:
                    for pattern in boost_paths:
                        if matches_glob(result.path, pattern):
                            result.score += 0.12
                            break  # Only apply once per result

        # Renormalize scores to 0-1
        max_score = max((res.score for res in results), default=1.0)
        if max_score <= 0:
            return results

        for res in results:
            res.score = min(1.0, res.score / max_score)

        # Re-sort by adjusted scores
        results.sort(key=lambda r: r.score, reverse=True)

        return results

    def _deduplicate_by_path(
        self, results: list[SearchResult], limit: int
    ) -> list[SearchResult]:
        """Deduplicate results by document path, keeping highest-scoring chunk.

        Args:
            results: List of search results (may contain duplicates).
            limit: Maximum number of results to return.

        Returns:
            Deduplicated list with at most one result per document path.
        """
        if not results:
            return results

        # Keep track of best result per path
        best_by_path: dict[str, SearchResult] = {}

        for result in results:
            path = result.path
            if path not in best_by_path or result.score > best_by_path[path].score:
                best_by_path[path] = result

        # Sort by score descending and limit
        deduplicated = sorted(best_by_path.values(), key=lambda r: r.score, reverse=True)
        return deduplicated[:limit]

    def index_document(self, chunk: DocumentChunk) -> None:
        """Index a single document chunk to both indices.

        Args:
            chunk: The document chunk to index.
        """
        self._whoosh.index_document(chunk)
        self._chroma.index_document(chunk)
        self._last_indexed = datetime.now(UTC)

    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Index multiple document chunks to both indices.

        Args:
            chunks: List of document chunks to index.
        """
        if not chunks:
            return

        self._whoosh.index_documents(chunks)
        self._chroma.index_documents(chunks)
        self._last_indexed = datetime.now(UTC)

    def delete_document(self, path: str) -> None:
        """Delete a document from both indices.

        Args:
            path: The document path to delete.
        """
        self._whoosh.delete_document(path)
        self._chroma.delete_document(path)

    def reindex(self, kb_root: Path | None = None) -> int:
        """Clear and rebuild indices from all markdown files.

        Args:
            kb_root: Knowledge base root directory. Uses config default if None.

        Returns:
            Number of chunks indexed.
        """
        kb_root = kb_root or get_kb_root()

        # Clear existing indices
        self.clear()

        # Find all markdown files
        md_files = list(kb_root.rglob("*.md"))

        if not md_files:
            return 0

        # Import parser here to avoid circular imports
        from ..parser import parse_entry

        chunks: list[DocumentChunk] = []

        for md_file in md_files:
            try:
                # Parse the file - returns (metadata, content, chunks)
                _, _, file_chunks = parse_entry(md_file)
                if not file_chunks:
                    continue

                relative_path = str(md_file.relative_to(kb_root))
                normalized_chunks = [
                    chunk.model_copy(update={"path": relative_path}) for chunk in file_chunks
                ]
                chunks.extend(normalized_chunks)
            except Exception:
                # Skip files that can't be parsed
                continue

        # Index all chunks
        if chunks:
            self.index_chunks(chunks)

        return len(chunks)

    def clear(self) -> None:
        """Clear both indices."""
        self._whoosh.clear()
        self._chroma.clear()
        self._last_indexed = None

    def status(self) -> IndexStatus:
        """Get status of the search indices.

        Returns:
            IndexStatus with document counts and last indexed time.
        """
        kb_root = get_kb_root()
        kb_files = len(list(kb_root.rglob("*.md"))) if kb_root.exists() else 0

        return IndexStatus(
            whoosh_docs=self._whoosh.doc_count(),
            chroma_docs=self._chroma.doc_count(),
            last_indexed=self._last_indexed.isoformat() if self._last_indexed else None,
            kb_files=kb_files,
        )

    def preload(self) -> None:
        """Preload the embedding model to avoid first-query latency.

        Call this at startup when KB_PRELOAD=1 is set.
        """
        self._chroma.preload()
