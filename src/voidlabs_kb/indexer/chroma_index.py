"""ChromaDB-based semantic search index with embeddings."""

from pathlib import Path
from typing import TYPE_CHECKING

from ..config import EMBEDDING_MODEL, get_index_root
from ..models import DocumentChunk, SearchResult

if TYPE_CHECKING:
    import chromadb
    from sentence_transformers import SentenceTransformer


class ChromaIndex:
    """Semantic search using ChromaDB with sentence-transformers embeddings."""

    COLLECTION_NAME = "kb_chunks"

    def __init__(self, index_dir: Path | None = None):
        """Initialize the Chroma index.

        Args:
            index_dir: Directory for index storage. Defaults to INDEX_ROOT/chroma/.
        """
        self._index_dir = index_dir or get_index_root() / "chroma"
        self._client: chromadb.PersistentClient | None = None
        self._collection: chromadb.Collection | None = None
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> "SentenceTransformer":
        """Lazy-load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def _get_collection(self) -> "chromadb.Collection":
        """Get or create the Chroma collection."""
        if self._collection is not None:
            return self._collection

        import chromadb

        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._index_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def index_document(self, chunk: DocumentChunk) -> None:
        """Index a document chunk.

        Args:
            chunk: The document chunk to index.
        """
        collection = self._get_collection()

        # Create unique chunk ID
        chunk_id = f"{chunk.path}#{chunk.section or 'main'}"

        # Generate embedding
        embedding = self._embed([chunk.content])[0]

        # Prepare metadata
        metadata = {
            "path": chunk.path,
            "title": chunk.metadata.title,
            "section": chunk.section or "",
            "tags": ",".join(chunk.metadata.tags),
        }

        # Upsert to handle updates
        collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk.content],
            metadatas=[metadata],
        )

    def index_documents(self, chunks: list[DocumentChunk]) -> None:
        """Index multiple document chunks.

        Args:
            chunks: List of document chunks to index.
        """
        if not chunks:
            return

        collection = self._get_collection()

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = f"{chunk.path}#{chunk.section or 'main'}"
            ids.append(chunk_id)
            documents.append(chunk.content)
            metadatas.append(
                {
                    "path": chunk.path,
                    "title": chunk.metadata.title,
                    "section": chunk.section or "",
                    "tags": ",".join(chunk.metadata.tags),
                }
            )

        # Generate embeddings in batch
        embeddings = self._embed(documents)

        # Upsert all at once
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search the index semantically.

        Args:
            query: Search query string.
            limit: Maximum number of results.

        Returns:
            List of search results with normalized scores.
        """
        collection = self._get_collection()

        # Check if collection is empty
        if collection.count() == 0:
            return []

        # Generate query embedding
        query_embedding = self._embed([query])[0]

        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        search_results = []
        ids = results["ids"][0]
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for i, chunk_id in enumerate(ids):
            doc = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 1.0

            # Convert cosine distance to similarity score (0-1)
            # Cosine distance is 1 - cosine_similarity, so similarity = 1 - distance
            score = max(0.0, min(1.0, 1.0 - distance))

            # Create snippet
            snippet = doc[:200] + "..." if len(doc) > 200 else doc

            tags = meta.get("tags", "")
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]

            search_results.append(
                SearchResult(
                    path=meta.get("path", ""),
                    title=meta.get("title", ""),
                    snippet=snippet,
                    score=score,
                    tags=tag_list,
                    section=meta.get("section") or None,
                )
            )

        return search_results

    def clear(self) -> None:
        """Clear all documents from the index."""
        if self._client is not None:
            # Delete and recreate collection
            try:
                self._client.delete_collection(self.COLLECTION_NAME)
            except Exception:
                pass
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            # Force initialize and clear
            self._get_collection()
            if self._client:
                try:
                    self._client.delete_collection(self.COLLECTION_NAME)
                except Exception:
                    pass
                self._collection = self._client.get_or_create_collection(
                    name=self.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )

    def doc_count(self) -> int:
        """Return the number of documents in the index."""
        collection = self._get_collection()
        return collection.count()

    def delete_document(self, path: str) -> None:
        """Delete all chunks for a document path.

        Args:
            path: The document path to delete.
        """
        collection = self._get_collection()

        # Query for all chunks with this path
        results = collection.get(
            where={"path": path},
            include=[],
        )

        if results["ids"]:
            collection.delete(ids=results["ids"])
