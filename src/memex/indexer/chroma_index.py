"""ChromaDB-based semantic search index with embeddings."""

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ..config import EMBEDDING_MODEL, SEMANTIC_MIN_SIMILARITY, get_index_root
from ..models import DocumentChunk, SearchResult
from .embedding_cache import EmbeddingCache, hash_embedding_text

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    import chromadb
    from chromadb.api import ClientAPI
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
        self._client: ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._model: SentenceTransformer | None = None
        self._embedding_cache: EmbeddingCache | None = None

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

        import shutil

        import chromadb

        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._index_dir))
        try:
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except KeyError:
            # Schema incompatibility from chromadb version change - reset the index
            del self._client
            shutil.rmtree(self._index_dir, ignore_errors=True)
            self._index_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._index_dir))
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _get_embedding_cache(self) -> EmbeddingCache:
        if self._embedding_cache is None:
            self._embedding_cache = EmbeddingCache(
                index_root=self._index_dir.parent,
                model_name=EMBEDDING_MODEL,
            )
        return self._embedding_cache

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

    def _load_cached_embeddings(self, hashes: list[str]) -> dict[str, list[float]]:
        try:
            return self._get_embedding_cache().get_many(hashes)
        except Exception as e:
            log.warning("Embedding cache read failed: %s", e)
            return {}

    def _store_cached_embeddings(self, embeddings: dict[str, list[float]]) -> None:
        try:
            self._get_embedding_cache().set_many(embeddings)
        except Exception as e:
            log.warning("Embedding cache write failed: %s", e)

    def _build_embedding_text(self, content: str, keywords: list[str], tags: list[str]) -> str:
        """Build text for embedding by concatenating content, keywords, and tags.

        This follows A-Mem's approach of enriching embeddings with semantic context
        to improve search quality for keyword-rich entries.

        Args:
            content: The document content.
            keywords: LLM-extracted key concepts from metadata.
            tags: Document tags from metadata.

        Returns:
            Combined text for embedding.
        """
        parts = [content]
        if keywords:
            parts.append(f"\n\nKeywords: {', '.join(keywords)}")
        if tags:
            parts.append(f"\nTags: {', '.join(tags)}")
        return "".join(parts)

    def index_document(self, chunk: DocumentChunk) -> None:
        """Index a document chunk.

        Args:
            chunk: The document chunk to index.
        """
        collection = self._get_collection()

        # Create unique chunk ID
        chunk_id = f"{chunk.path}#{chunk.section or 'main'}"

        # Build text for embedding with keywords and tags
        embedding_text = self._build_embedding_text(
            chunk.content,
            chunk.metadata.keywords,
            chunk.metadata.tags,
        )
        embedding_hash = hash_embedding_text(embedding_text)
        cached = self._load_cached_embeddings([embedding_hash])
        embedding = cached.get(embedding_hash)
        if embedding is None:
            embedding = self._embed([embedding_text])[0]
            self._store_cached_embeddings({embedding_hash: embedding})

        # Prepare metadata
        metadata = {
            "path": chunk.path,
            "title": chunk.metadata.title,
            "section": chunk.section or "",
            "tags": ",".join(chunk.metadata.tags),
            "created": chunk.metadata.created.isoformat() if chunk.metadata.created else "",
            "updated": chunk.metadata.updated.isoformat() if chunk.metadata.updated else "",
            "token_count": chunk.token_count or 0,
            "source_project": chunk.metadata.source_project or "",
            "embedding_hash": embedding_hash,
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

        # Deduplicate chunks by ID, keeping the last occurrence
        # (handles cases where documents have duplicate sections)
        seen_ids: dict[str, int] = {}
        for i, chunk in enumerate(chunks):
            chunk_id = f"{chunk.path}#{chunk.section or 'main'}"
            seen_ids[chunk_id] = i  # Later occurrences overwrite earlier

        # Build lists using deduplicated indices
        ids = []
        documents = []
        embedding_texts = []
        embedding_hashes = []
        metadatas = []

        for chunk_id, idx in seen_ids.items():
            chunk = chunks[idx]
            ids.append(chunk_id)
            documents.append(chunk.content)
            # Build enriched text for embedding (content + keywords + tags)
            embedding_text = self._build_embedding_text(
                chunk.content,
                chunk.metadata.keywords,
                chunk.metadata.tags,
            )
            embedding_texts.append(embedding_text)
            embedding_hash = hash_embedding_text(embedding_text)
            embedding_hashes.append(embedding_hash)
            metadatas.append(
                {
                    "path": chunk.path,
                    "title": chunk.metadata.title,
                    "section": chunk.section or "",
                    "tags": ",".join(chunk.metadata.tags),
                    "created": chunk.metadata.created.isoformat() if chunk.metadata.created else "",
                    "updated": chunk.metadata.updated.isoformat() if chunk.metadata.updated else "",
                    "token_count": chunk.token_count or 0,
                    "source_project": chunk.metadata.source_project or "",
                    "embedding_hash": embedding_hash,
                }
            )

        # Reuse cached embeddings when possible
        cached = self._load_cached_embeddings(embedding_hashes)
        embeddings: list[list[float] | None] = [None] * len(embedding_texts)
        missing_texts = []
        missing_indices = []
        missing_hashes = []
        for i, embedding_hash in enumerate(embedding_hashes):
            cached_embedding = cached.get(embedding_hash)
            if cached_embedding is not None:
                embeddings[i] = cached_embedding
            else:
                missing_texts.append(embedding_texts[i])
                missing_indices.append(i)
                missing_hashes.append(embedding_hash)

        if missing_texts:
            computed = self._embed(missing_texts)
            to_cache: dict[str, list[float]] = {}
            for idx, embedding in zip(missing_indices, computed, strict=False):
                embeddings[idx] = embedding
            for embedding_hash, embedding in zip(missing_hashes, computed, strict=False):
                to_cache[embedding_hash] = embedding
            self._store_cached_embeddings(to_cache)

        # All embeddings should be populated now
        final_embeddings = [cast(list[float], emb) for emb in embeddings]

        # Upsert all at once
        collection.upsert(
            ids=ids,
            embeddings=cast(Any, final_embeddings),
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float | None = None,
    ) -> list[SearchResult]:
        """Search the index semantically.

        Args:
            query: Search query string.
            limit: Maximum number of results.
            min_similarity: Minimum similarity threshold (0-1). Results below this
                are filtered out. Defaults to SEMANTIC_MIN_SIMILARITY from config.
                Pass 0.0 to disable filtering.

        Returns:
            List of search results with raw similarity scores (not normalized).
            Scores reflect actual cosine similarity, not relative ranking.
        """
        collection = self._get_collection()

        # Check if collection is empty
        if collection.count() == 0:
            return []

        # Use config default if not specified
        if min_similarity is None:
            min_similarity = SEMANTIC_MIN_SIMILARITY

        # Generate query embedding
        query_embedding = self._embed([query])[0]

        # Fetch extra results to allow for threshold filtering
        fetch_limit = min(limit * 2, collection.count())

        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_limit,
            include=cast(Any, ["documents", "metadatas", "distances"]),
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

            # Filter out results below minimum similarity threshold
            if score < min_similarity:
                continue

            # Create snippet, stripping markdown syntax
            from . import strip_markdown_for_snippet

            snippet = strip_markdown_for_snippet(doc, max_length=200)

            tags_value = meta.get("tags") or ""
            tags_str = str(tags_value)
            tag_list = [t.strip() for t in tags_str.split(",") if t.strip()]

            # Parse datetimes from stored ISO strings
            created_str = str(meta.get("created") or "")
            updated_str = str(meta.get("updated") or "")
            created_date = datetime.fromisoformat(created_str) if created_str else None
            updated_date = datetime.fromisoformat(updated_str) if updated_str else None

            path_value = meta.get("path") or ""
            title_value = meta.get("title") or ""
            section_value = meta.get("section")
            token_count_value = meta.get("token_count")
            source_project_value = meta.get("source_project")

            section = str(section_value) if section_value else None
            token_count = (
                int(token_count_value)
                if isinstance(token_count_value, (int, float))
                and not isinstance(token_count_value, bool)
                else 0
            )
            source_project = str(source_project_value) if source_project_value else None

            search_results.append(
                SearchResult(
                    path=str(path_value),
                    title=str(title_value),
                    snippet=snippet,
                    score=score,
                    tags=tag_list,
                    section=section,
                    created=created_date,
                    updated=updated_date,
                    token_count=token_count,
                    source_project=source_project,
                )
            )

            # Stop once we have enough results
            if len(search_results) >= limit:
                break

        return search_results

    def clear(self) -> None:
        """Clear all documents from the index."""
        # Force initialize client if needed
        if self._client is None:
            self._get_collection()

        if self._client is not None:
            # Delete and recreate collection
            try:
                self._client.delete_collection(self.COLLECTION_NAME)
            except Exception as e:
                log.debug("Could not delete collection during clear: %s", e)
            # Create fresh collection and update cached reference
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

    def preload(self) -> None:
        """Preload the embedding model and collection to avoid first-query latency.

        Call this at startup to warm up the model before any searches.
        """
        # Load the embedding model (this is the slow part - 2-3s)
        self._get_model()
        # Initialize the collection
        self._get_collection()
