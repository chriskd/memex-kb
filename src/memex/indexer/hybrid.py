"""Hybrid search combining Whoosh BM25 and ChromaDB semantic search."""

from __future__ import annotations

import importlib.util
import logging
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ..config import (
    DEFAULT_SEARCH_LIMIT,
    HYBRID_SEMANTIC_FASTPATH,
    HYBRID_SEMANTIC_FASTPATH_MIN_SCORE,
    KB_PATH_CONTEXT_BOOST,
    PROJECT_CONTEXT_BOOST,
    RRF_K,
    SEMANTIC_MIN_SIMILARITY,
    SEMANTIC_STRICT_SIMILARITY,
    TAG_MATCH_BOOST,
    get_kb_root,
)
from ..models import DocumentChunk, IndexStatus, SearchResult
from ..errors import MemexError
from .chroma_index import ChromaIndex
from .whoosh_index import WhooshIndex

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..context import KBContext

SearchMode = Literal["hybrid", "keyword", "semantic"]


def _semantic_deps_available() -> bool:
    # Avoid importing heavyweight deps unless needed.
    return (
        importlib.util.find_spec("chromadb") is not None
        and importlib.util.find_spec("sentence_transformers") is not None
    )


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
        # Semantic search is optional; allow keyword-only installs to work.
        self.semantic_available = chroma_index is not None or _semantic_deps_available()
        self._chroma = chroma_index or (ChromaIndex() if self.semantic_available else None)
        self._last_indexed: datetime | None = None

    def search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        mode: SearchMode = "hybrid",
        project_context: str | None = None,
        kb_context: KBContext | None = None,
        strict: bool = False,
    ) -> list[SearchResult]:
        """Search the knowledge base.

        Args:
            query: Search query string.
            limit: Maximum number of results.
            mode: Search mode - "hybrid", "keyword", or "semantic".
            project_context: Current project name for context boosting.
            kb_context: Project context for path-based boosting.
            strict: If True, use higher similarity threshold for semantic search
                to filter out low-confidence matches.

        Returns:
            List of search results, deduplicated by document path.
        """
        # Fetch more results to allow for deduplication
        fetch_limit = limit * 3

        # Determine semantic similarity threshold based on strict mode
        min_similarity = SEMANTIC_STRICT_SIMILARITY if strict else SEMANTIC_MIN_SIMILARITY

        if mode == "keyword":
            results = self._whoosh.search(query, limit=fetch_limit)
            results = self._apply_ranking_adjustments(query, results, project_context, kb_context)
        elif mode == "semantic":
            if not self._chroma:
                raise MemexError.semantic_search_unavailable()
            results = self._chroma.search(query, limit=fetch_limit, min_similarity=min_similarity)
            results = self._apply_ranking_adjustments(query, results, project_context, kb_context)
        else:
            # If semantic deps aren't available, treat hybrid as keyword-only.
            if not self._chroma:
                results = self._whoosh.search(query, limit=fetch_limit)
                results = self._apply_ranking_adjustments(query, results, project_context, kb_context)
            else:
                results = self._hybrid_search(
                    query,
                    limit=limit,
                    project_context=project_context,
                    kb_context=kb_context,
                    min_similarity=min_similarity,
                )

        # Deduplicate by path, keeping highest-scoring chunk per document
        return self._deduplicate_by_path(results, limit)

    def _hybrid_search(
        self,
        query: str,
        limit: int,
        project_context: str | None = None,
        kb_context: KBContext | None = None,
        min_similarity: float = SEMANTIC_MIN_SIMILARITY,
    ) -> list[SearchResult]:
        """Perform hybrid search using Reciprocal Rank Fusion.

        Args:
            query: Search query string.
            limit: Maximum number of results.
            project_context: Current project name for context boosting.
            kb_context: Project context for path-based boosting.
            min_similarity: Minimum similarity threshold for semantic results.

        Returns:
            List of merged search results.
        """
        # Get results from both indices (fetch more to have good RRF merge)
        fetch_limit = limit * 3
        whoosh_results = self._whoosh.search(query, limit=fetch_limit)

        # Fast-path: if keyword results are strong enough, skip semantic search
        if (
            HYBRID_SEMANTIC_FASTPATH
            and len(whoosh_results) >= limit
            and whoosh_results[0].score >= HYBRID_SEMANTIC_FASTPATH_MIN_SCORE
        ):
            return self._apply_ranking_adjustments(
                query,
                whoosh_results[:limit],
                project_context,
                kb_context,
            )

        chroma_results: list[SearchResult] = []
        try:
            chroma_results = self._chroma.search(
                query,
                limit=fetch_limit,
                min_similarity=min_similarity,
            )
        except Exception as e:
            # Semantic search is optional; if it's broken/missing, degrade gracefully.
            log.debug("Semantic search unavailable, falling back to keyword: %s", e)
            chroma_results = []

        # If one index is empty, return the other
        if not whoosh_results and not chroma_results:
            return []
        if not whoosh_results:
            return self._apply_ranking_adjustments(
                query,
                chroma_results[:limit],
                project_context,
                kb_context,
            )
        if not chroma_results:
            return self._apply_ranking_adjustments(
                query,
                whoosh_results[:limit],
                project_context,
                kb_context,
            )

        # Apply RRF
        return self._rrf_merge(
            query,
            whoosh_results,
            chroma_results,
            limit,
            project_context,
            kb_context,
        )

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
        """Boost results with matching tags and project/path context.

        Applies two types of boosts:
        1. Tag boost: TAG_MATCH_BOOST per matching tag in query (always stacks)
        2. Context boost: MAX of PROJECT_CONTEXT_BOOST or KB_PATH_CONTEXT_BOOST
           - Project boost: entry was created from current project
           - Path boost: entry matches .kbconfig paths
           These don't stack to avoid overboosting correlated signals.
        """
        if not results:
            return results

        # Tag boost: per matching tag (always applies, stacks with context)
        tokens = {tok for tok in re.split(r"\W+", query.lower()) if tok}
        if tokens:
            for result in results:
                tag_tokens = {tag.lower() for tag in result.tags}
                overlap = tokens.intersection(tag_tokens)
                if overlap:
                    result.score += TAG_MATCH_BOOST * len(overlap)

        # Context boost: apply MAX of project_context or kb_context path boost
        # These are correlated signals so we don't stack them
        for result in results:
            project_boost = 0.0
            path_boost = 0.0

            # Check project context boost
            if project_context and result.source_project == project_context:
                project_boost = PROJECT_CONTEXT_BOOST

            # Check KB context path boost
            if kb_context:
                from ..context import matches_glob

                boost_paths = kb_context.get_all_boost_paths()
                for pattern in boost_paths:
                    if matches_glob(result.path, pattern):
                        path_boost = KB_PATH_CONTEXT_BOOST
                        break

            # Apply the higher of the two (don't stack)
            result.score += max(project_boost, path_boost)

        # Renormalize scores to 0-1
        max_score = max((res.score for res in results), default=1.0)
        if max_score <= 0:
            return results

        for res in results:
            res.score = min(1.0, res.score / max_score)

        # Re-sort by adjusted scores
        results.sort(key=lambda r: r.score, reverse=True)

        return results

    def _deduplicate_by_path(self, results: list[SearchResult], limit: int) -> list[SearchResult]:
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
        if self._chroma:
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
        if self._chroma:
            self._chroma.index_documents(chunks)
        self._last_indexed = datetime.now(UTC)

    def delete_document(self, path: str) -> None:
        """Delete a document from both indices.

        Args:
            path: The document path to delete.
        """
        self._whoosh.delete_document(path)
        if self._chroma:
            self._chroma.delete_document(path)

    def reindex(
        self,
        kb_root: Path | None = None,
        kb_roots: list[tuple[str | None, Path]] | None = None,
    ) -> int:
        """Clear and rebuild indices from all markdown files.

        Args:
            kb_root: Knowledge base root directory (single KB mode). Uses config default if None.
            kb_roots: List of (scope, path) tuples for multi-KB mode. Scope is "project" or "user".
                      If provided, kb_root is ignored.

        Returns:
            Number of chunks indexed.
        """
        # Clear existing indices
        self.clear()

        # Import parser here to avoid circular imports
        from ..parser import parse_entry

        chunks: list[DocumentChunk] = []

        # Build list of (scope, kb_path) to index
        if kb_roots:
            roots_to_index = kb_roots
        else:
            single_root = kb_root or get_kb_root()
            roots_to_index = [(None, single_root)]  # No scope prefix for single-KB mode

        for scope, root in roots_to_index:
            if not root.exists():
                continue

            # Find all markdown files in this KB
            md_files = list(root.rglob("*.md"))

            for md_file in md_files:
                try:
                    # Parse the file - returns (metadata, content, chunks)
                    _, _, file_chunks = parse_entry(md_file)
                    if not file_chunks:
                        continue

                    relative_path = str(md_file.relative_to(root))
                    # Add scope prefix for multi-KB mode
                    if scope:
                        prefixed_path = f"@{scope}/{relative_path}"
                    else:
                        prefixed_path = relative_path

                    normalized_chunks = [
                        chunk.model_copy(update={"path": prefixed_path}) for chunk in file_chunks
                    ]
                    chunks.extend(normalized_chunks)
                except Exception as e:
                    log.warning("Skipping %s during reindex: %s", md_file, e)
                    continue

        # Index all chunks
        if chunks:
            self.index_chunks(chunks)

        return len(chunks)

    def clear(self) -> None:
        """Clear both indices."""
        self._whoosh.clear()
        if self._chroma:
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
            chroma_docs=self._chroma.doc_count() if self._chroma else 0,
            last_indexed=self._last_indexed.isoformat() if self._last_indexed else None,
            kb_files=kb_files,
        )

    def preload(self) -> None:
        """Preload the embedding model to avoid first-query latency.

        Call this at startup when MEMEX_PRELOAD=1 is set.
        """
        if self._chroma:
            self._chroma.preload()
