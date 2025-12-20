"""Whoosh-based BM25 keyword search index."""

from pathlib import Path

from datetime import date

from whoosh import index
from whoosh.fields import ID, KEYWORD, STORED, TEXT, Schema
from whoosh.qparser import MultifieldParser, OrGroup

from ..config import get_index_root
from ..models import DocumentChunk, SearchResult


class WhooshIndex:
    """Fast keyword/BM25 search using Whoosh."""

    def __init__(self, index_dir: Path | None = None):
        """Initialize the Whoosh index.

        Args:
            index_dir: Directory for index storage. Defaults to INDEX_ROOT/whoosh/.
        """
        self._index_dir = index_dir or get_index_root() / "whoosh"
        self._index: index.Index | None = None
        self._schema = Schema(
            path=ID(stored=True, unique=True),
            section=ID(stored=True),
            title=TEXT(stored=True),
            content=TEXT(stored=True),
            tags=KEYWORD(stored=True, commas=True),
            chunk_id=STORED,
            created=STORED,
            updated=STORED,
        )

    def _ensure_index(self) -> index.Index:
        """Ensure index exists and return it."""
        if self._index is not None:
            return self._index

        self._index_dir.mkdir(parents=True, exist_ok=True)

        if index.exists_in(str(self._index_dir)):
            self._index = index.open_dir(str(self._index_dir))
        else:
            self._index = index.create_in(str(self._index_dir), self._schema)

        return self._index

    def index_document(self, chunk: DocumentChunk) -> None:
        """Index a document chunk.

        Args:
            chunk: The document chunk to index.
        """
        ix = self._ensure_index()
        writer = ix.writer()

        # Create unique chunk ID
        chunk_id = f"{chunk.path}#{chunk.section or 'main'}"

        writer.update_document(
            path=chunk.path,
            section=chunk.section or "",
            title=chunk.metadata.title,
            content=chunk.content,
            tags=",".join(chunk.metadata.tags),
            chunk_id=chunk_id,
            created=chunk.metadata.created.isoformat() if chunk.metadata.created else None,
            updated=chunk.metadata.updated.isoformat() if chunk.metadata.updated else None,
        )
        writer.commit()

    def index_documents(self, chunks: list[DocumentChunk]) -> None:
        """Index multiple document chunks in a single transaction.

        Args:
            chunks: List of document chunks to index.
        """
        if not chunks:
            return

        ix = self._ensure_index()
        writer = ix.writer()

        for chunk in chunks:
            chunk_id = f"{chunk.path}#{chunk.section or 'main'}"
            writer.update_document(
                path=chunk.path,
                section=chunk.section or "",
                title=chunk.metadata.title,
                content=chunk.content,
                tags=",".join(chunk.metadata.tags),
                chunk_id=chunk_id,
                created=chunk.metadata.created.isoformat() if chunk.metadata.created else None,
                updated=chunk.metadata.updated.isoformat() if chunk.metadata.updated else None,
            )

        writer.commit()

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search the index.

        Args:
            query: Search query string.
            limit: Maximum number of results.

        Returns:
            List of search results with normalized scores.
        """
        ix = self._ensure_index()

        with ix.searcher() as searcher:
            # Search across title, content, and tags
            parser = MultifieldParser(
                ["title", "content", "tags"],
                schema=self._schema,
                group=OrGroup,
            )

            try:
                parsed_query = parser.parse(query)
            except Exception:
                # If parsing fails, try a simple term query
                from whoosh.query import Term

                parsed_query = Term("content", query.lower())

            results = searcher.search(parsed_query, limit=limit)

            if not results:
                return []

            # Normalize scores to 0-1 range
            max_score = max(r.score for r in results) if results else 1.0
            max_score = max_score if max_score > 0 else 1.0

            search_results = []
            for hit in results:
                # Create snippet from content
                content = hit.get("content", "")
                snippet = content[:200] + "..." if len(content) > 200 else content

                tags = hit.get("tags", "")
                tag_list = [t.strip() for t in tags.split(",") if t.strip()]

                # Parse dates from stored ISO strings
                created_str = hit.get("created")
                updated_str = hit.get("updated")
                created_date = date.fromisoformat(created_str) if created_str else None
                updated_date = date.fromisoformat(updated_str) if updated_str else None

                search_results.append(
                    SearchResult(
                        path=hit["path"],
                        title=hit.get("title", ""),
                        snippet=snippet,
                        score=hit.score / max_score,
                        tags=tag_list,
                        section=hit.get("section") or None,
                        created=created_date,
                        updated=updated_date,
                    )
                )

            return search_results

    def clear(self) -> None:
        """Clear all documents from the index."""
        if self._index is not None:
            self._index.close()
            self._index = None

        # Remove and recreate the index
        if self._index_dir.exists():
            import shutil

            shutil.rmtree(self._index_dir)

        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._index = index.create_in(str(self._index_dir), self._schema)

    def doc_count(self) -> int:
        """Return the number of documents in the index."""
        ix = self._ensure_index()
        return ix.doc_count()

    def delete_document(self, path: str) -> None:
        """Delete all chunks for a document path.

        Args:
            path: The document path to delete.
        """
        ix = self._ensure_index()
        writer = ix.writer()
        writer.delete_by_term("path", path)
        writer.commit()
