"""Hybrid search combining Whoosh BM25 and ChromaDB semantic search."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ..config import (
    DEFAULT_SEARCH_LIMIT,
    KB_PATH_CONTEXT_BOOST,
    PROJECT_CONTEXT_BOOST,
    RRF_K,
    TAG_MATCH_BOOST,
    get_kb_root,
    get_index_root,
)
from ..models import DocumentChunk, IndexStatus, SearchResult
from .chroma_index import ChromaIndex
from .manifest import IndexManifest
from .whoosh_index import WhooshIndex

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..context import KBContext

SearchMode = Literal["hybrid", "keyword", "semantic"]


@dataclass
class ReindexStats:
    """Statistics from an incremental reindex operation."""

    total_chunks: int  # Total chunks now indexed
    added: int  # Files added (new)
    updated: int  # Files updated (modified)
    deleted: int  # Files deleted
    unchanged: int  # Files unchanged (skipped)


class HybridSearcher:
    """Hybrid search combining keyword (Whoosh) and semantic (Chroma) indices."""

    def __init__(
        self,
        whoosh_index: WhooshIndex | None = None,
        chroma_index: ChromaIndex | None = None,
        index_dir: Path | None = None,
    ):
        """Initialize the hybrid searcher.

        Args:
            whoosh_index: Whoosh index instance. Created if not provided.
            chroma_index: Chroma index instance. Created if not provided.
            index_dir: Directory for index storage. Used for manifest.
        """
        self._whoosh = whoosh_index or WhooshIndex()
        self._chroma = chroma_index or ChromaIndex()
        self._last_indexed: datetime | None = None
        self._index_dir = index_dir or get_index_root()
        self._manifest = IndexManifest(self._index_dir)

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
        """Boost results with matching tags and project/path context.

        Applies two types of boosts:
        1. Tag boost: TAG_MATCH_BOOST per matching tag in query (always stacks)
        2. Context boost: MAX of PROJECT_CONTEXT_BOOST or KB_PATH_CONTEXT_BOOST
           - Project boost: entry was created from current project
           - Path boost: entry matches .kbcontext paths
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

    def reindex(
        self,
        kb_root: Path | None = None,
        *,
        force: bool = False,
    ) -> int | ReindexStats:
        """Rebuild indices from markdown files.

        By default, performs incremental indexing - only reindexing files
        that have been added, modified, or deleted since the last index.
        Use force=True for a full rebuild.

        Args:
            kb_root: Knowledge base root directory. Uses config default if None.
            force: If True, clear and rebuild everything. If False (default),
                   perform incremental update based on file mtimes.

        Returns:
            If force=True: Number of chunks indexed (int, for backward compatibility).
            If force=False: ReindexStats with detailed counts.
        """
        kb_root = kb_root or get_kb_root()

        if force:
            return self._full_reindex(kb_root)
        else:
            return self._incremental_reindex(kb_root)

    def _full_reindex(self, kb_root: Path) -> int:
        """Perform a full reindex, clearing all existing data.

        Args:
            kb_root: Knowledge base root directory.

        Returns:
            Number of chunks indexed.
        """
        # Clear existing indices and manifest
        self.clear()

        # Find all markdown files
        md_files = list(kb_root.rglob("*.md"))

        if not md_files:
            return 0

        # Import parser here to avoid circular imports
        from ..parser import parse_entry

        BATCH_SIZE = 100
        batch: list[DocumentChunk] = []
        total_indexed = 0

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
                batch.extend(normalized_chunks)

                # Update manifest with current file state
                stat = md_file.stat()
                self._manifest.update_file(relative_path, stat.st_mtime, stat.st_size)

                # Index when batch is full
                if len(batch) >= BATCH_SIZE:
                    self.index_chunks(batch)
                    total_indexed += len(batch)
                    batch = []
            except Exception as e:
                log.warning("Skipping %s during reindex: %s", md_file, e)
                continue

        # Index remaining chunks
        if batch:
            self.index_chunks(batch)
            total_indexed += len(batch)

        # Save manifest
        self._manifest.save()

        return total_indexed

    def _incremental_reindex(self, kb_root: Path) -> ReindexStats:
        """Perform incremental reindex, only updating changed files.

        Args:
            kb_root: Knowledge base root directory.

        Returns:
            ReindexStats with counts of added, updated, deleted, unchanged files.
        """
        # Import parser here to avoid circular imports
        from ..parser import parse_entry

        # Get current files on disk
        md_files = list(kb_root.rglob("*.md"))
        current_paths = {str(f.relative_to(kb_root)) for f in md_files}

        # Get previously indexed files from manifest
        indexed_paths = self._manifest.get_all_paths()

        # Find deleted files
        deleted_paths = indexed_paths - current_paths

        # Track stats
        added_count = 0
        updated_count = 0
        unchanged_count = 0
        total_chunks = 0

        BATCH_SIZE = 100
        batch: list[DocumentChunk] = []

        # Process current files
        for md_file in md_files:
            relative_path = str(md_file.relative_to(kb_root))

            try:
                stat = md_file.stat()
                current_mtime = stat.st_mtime
                current_size = stat.st_size

                # Check if file has changed
                if not self._manifest.is_file_changed(relative_path, current_mtime, current_size):
                    unchanged_count += 1
                    continue

                # File is new or modified - parse and index
                is_new = relative_path not in indexed_paths

                # Delete old chunks first (for updates)
                if not is_new:
                    self.delete_document(relative_path)

                # Parse the file
                _, _, file_chunks = parse_entry(md_file)
                if not file_chunks:
                    # File exists but has no chunks - still track it
                    self._manifest.update_file(relative_path, current_mtime, current_size)
                    if is_new:
                        added_count += 1
                    else:
                        updated_count += 1
                    continue

                # Normalize paths in chunks
                normalized_chunks = [
                    chunk.model_copy(update={"path": relative_path}) for chunk in file_chunks
                ]
                batch.extend(normalized_chunks)

                # Update manifest
                self._manifest.update_file(relative_path, current_mtime, current_size)

                if is_new:
                    added_count += 1
                else:
                    updated_count += 1

                # Index when batch is full
                if len(batch) >= BATCH_SIZE:
                    self.index_chunks(batch)
                    total_chunks += len(batch)
                    batch = []

            except Exception as e:
                log.warning("Skipping %s during incremental reindex: %s", md_file, e)
                continue

        # Index remaining batch
        if batch:
            self.index_chunks(batch)
            total_chunks += len(batch)

        # Delete removed files from index
        for deleted_path in deleted_paths:
            self.delete_document(deleted_path)
            self._manifest.remove_file(deleted_path)

        # Save manifest
        self._manifest.save()

        # Update timestamp if we did any work
        if added_count > 0 or updated_count > 0 or deleted_paths:
            self._last_indexed = datetime.now(UTC)

        return ReindexStats(
            total_chunks=total_chunks,
            added=added_count,
            updated=updated_count,
            deleted=len(deleted_paths),
            unchanged=unchanged_count,
        )

    def clear(self) -> None:
        """Clear both indices and the manifest."""
        self._whoosh.clear()
        self._chroma.clear()
        self._manifest.clear()
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

        Call this at startup when MEMEX_PRELOAD=1 is set.
        """
        self._chroma.preload()
