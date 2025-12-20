"""Hybrid search combining Whoosh BM25 and ChromaDB semantic search."""

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from ..config import DEFAULT_SEARCH_LIMIT, RRF_K, get_kb_root
from ..models import DocumentChunk, IndexStatus, SearchResult
from .chroma_index import ChromaIndex
from .whoosh_index import WhooshIndex

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
    ) -> list[SearchResult]:
        """Search the knowledge base.

        Args:
            query: Search query string.
            limit: Maximum number of results.
            mode: Search mode - "hybrid", "keyword", or "semantic".

        Returns:
            List of search results.
        """
        if mode == "keyword":
            return self._whoosh.search(query, limit=limit)
        elif mode == "semantic":
            return self._chroma.search(query, limit=limit)
        else:
            return self._hybrid_search(query, limit=limit)

    def _hybrid_search(self, query: str, limit: int) -> list[SearchResult]:
        """Perform hybrid search using Reciprocal Rank Fusion.

        Args:
            query: Search query string.
            limit: Maximum number of results.

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
            return chroma_results[:limit]
        if not chroma_results:
            return whoosh_results[:limit]

        # Apply RRF
        return self._rrf_merge(whoosh_results, chroma_results, limit)

    def _rrf_merge(
        self,
        whoosh_results: list[SearchResult],
        chroma_results: list[SearchResult],
        limit: int,
    ) -> list[SearchResult]:
        """Merge results using Reciprocal Rank Fusion.

        RRF score: score(d) = sum(1 / (k + rank)) for each ranking list.

        Args:
            whoosh_results: Results from keyword search.
            chroma_results: Results from semantic search.
            limit: Maximum number of results to return.

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
                )
            )

        return final_results

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
                chunks.extend(file_chunks)
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
