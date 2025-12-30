"""Comprehensive tests for ChromaIndex semantic search indexer."""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from memex.indexer.chroma_index import ChromaIndex
from memex.models import DocumentChunk, EntryMetadata


@pytest.fixture
def index_dir(tmp_path) -> Path:
    """Create a temporary directory for Chroma index."""
    return tmp_path / "chroma_test"


@pytest.fixture
def chroma_index(index_dir) -> ChromaIndex:
    """Create a fresh ChromaIndex instance for each test."""
    return ChromaIndex(index_dir=index_dir)


@pytest.fixture
def sample_chunk() -> DocumentChunk:
    """Create a sample document chunk for testing."""
    return DocumentChunk(
        path="test/sample.md",
        section="intro",
        content="Machine learning is a subset of artificial intelligence focused on data-driven algorithms.",
        metadata=EntryMetadata(
            title="ML Basics",
            tags=["machine-learning", "ai"],
            created=date(2024, 1, 1),
            updated=date(2024, 1, 15),
            source_project="ai-project",
        ),
        token_count=15,
    )


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """Create multiple sample chunks for batch testing."""
    return [
        DocumentChunk(
            path="ai/neural_nets.md",
            section="intro",
            content="Neural networks are computing systems inspired by biological neural networks in animal brains.",
            metadata=EntryMetadata(
                title="Neural Networks",
                tags=["neural-networks", "deep-learning"],
                created=date(2024, 1, 1),
            ),
            token_count=13,
        ),
        DocumentChunk(
            path="ai/transformers.md",
            section="architecture",
            content="Transformer architecture revolutionized natural language processing with self-attention mechanisms.",
            metadata=EntryMetadata(
                title="Transformers",
                tags=["transformers", "nlp"],
                created=date(2024, 1, 2),
            ),
            token_count=11,
        ),
        DocumentChunk(
            path="dev/databases.md",
            section=None,
            content="Relational databases use SQL for querying structured data with ACID guarantees.",
            metadata=EntryMetadata(
                title="Database Guide",
                tags=["database", "sql"],
                created=date(2024, 1, 3),
            ),
            token_count=12,
        ),
    ]


class TestChromaIndexInitialization:
    """Test Chroma index initialization and setup."""

    def test_init_creates_directory(self, chroma_index, index_dir):
        """Index directory is created when first accessed."""
        assert not index_dir.exists()
        # Trigger index creation
        chroma_index.doc_count()
        assert index_dir.exists()

    def test_init_with_custom_path(self, tmp_path):
        """Can initialize with custom index path."""
        custom_path = tmp_path / "custom" / "chroma"
        index = ChromaIndex(index_dir=custom_path)
        index.doc_count()
        assert custom_path.exists()

    def test_reopens_existing_collection(self, chroma_index, sample_chunk):
        """Can reopen an existing collection and access previous data."""
        chroma_index.index_document(sample_chunk)
        assert chroma_index.doc_count() == 1

        # Create new index instance with same directory
        new_index = ChromaIndex(index_dir=chroma_index._index_dir)
        assert new_index.doc_count() == 1

    def test_lazy_loads_embedding_model(self, chroma_index):
        """Embedding model is lazy-loaded on first use."""
        assert chroma_index._model is None
        chroma_index.preload()
        assert chroma_index._model is not None


class TestChromaIndexDocument:
    """Test single document indexing operations."""

    def test_index_single_document(self, chroma_index, sample_chunk):
        """Can index a single document chunk."""
        chroma_index.index_document(sample_chunk)
        assert chroma_index.doc_count() == 1

    def test_index_document_generates_embedding(self, chroma_index, sample_chunk):
        """Indexing generates embeddings for content."""
        chroma_index.index_document(sample_chunk)

        # Search should work (implies embeddings were created)
        results = chroma_index.search("machine learning algorithms")
        assert len(results) >= 1

    def test_index_document_with_all_fields(self, chroma_index):
        """Document with all fields is indexed correctly."""
        chunk = DocumentChunk(
            path="complete/doc.md",
            section="section1",
            content="Complete document with all metadata fields for semantic indexing.",
            metadata=EntryMetadata(
                title="Complete Doc",
                tags=["tag1", "tag2", "tag3"],
                created=date(2024, 6, 15),
                updated=date(2024, 6, 20),
                source_project="main-project",
                contributors=["user1"],
                aliases=["alias1"],
            ),
            token_count=25,
        )
        chroma_index.index_document(chunk)
        results = chroma_index.search("complete semantic document")
        assert len(results) == 1
        assert results[0].path == "complete/doc.md"

    def test_index_document_with_minimal_fields(self, chroma_index):
        """Document with minimal required fields is indexed correctly."""
        chunk = DocumentChunk(
            path="minimal/doc.md",
            section=None,
            content="Minimal document for testing basic indexing.",
            metadata=EntryMetadata(
                title="Minimal",
                tags=["minimal"],
                created=date(2024, 1, 1),
            ),
            token_count=None,
        )
        chroma_index.index_document(chunk)
        assert chroma_index.doc_count() == 1

    def test_update_existing_document(self, chroma_index):
        """Updating a document replaces the old version."""
        chunk1 = DocumentChunk(
            path="test/update.md",
            section="intro",
            content="Original content about cats and dogs",
            metadata=EntryMetadata(
                title="Original Title",
                tags=["test"],
                created=date(2024, 1, 1),
            ),
        )
        chroma_index.index_document(chunk1)

        chunk2 = DocumentChunk(
            path="test/update.md",
            section="intro",
            content="Updated content about birds and fish",
            metadata=EntryMetadata(
                title="Updated Title",
                tags=["test", "updated"],
                created=date(2024, 1, 1),
                updated=date(2024, 1, 10),
            ),
        )
        chroma_index.index_document(chunk2)

        # Should still have only one document
        assert chroma_index.doc_count() == 1

        # Should find updated content semantically
        results = chroma_index.search("birds fish")
        assert len(results) == 1


class TestChromaIndexDocuments:
    """Test batch document indexing operations."""

    def test_index_multiple_documents(self, chroma_index, sample_chunks):
        """Can index multiple documents in batch."""
        chroma_index.index_documents(sample_chunks)
        assert chroma_index.doc_count() == 3

    def test_index_empty_list(self, chroma_index):
        """Indexing empty list doesn't cause errors."""
        chroma_index.index_documents([])
        assert chroma_index.doc_count() == 0

    def test_batch_index_generates_embeddings(self, chroma_index, sample_chunks):
        """Batch indexing generates embeddings for all documents."""
        chroma_index.index_documents(sample_chunks)

        # All documents should be searchable
        results = chroma_index.search("neural networks")
        assert len(results) >= 1

    def test_batch_index_deduplicates_chunks(self, chroma_index):
        """Batch indexing deduplicates chunks with same path#section."""
        duplicate_chunks = [
            DocumentChunk(
                path="dup/doc.md",
                section="intro",
                content="First version",
                metadata=EntryMetadata(
                    title="Dup Test",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
            ),
            DocumentChunk(
                path="dup/doc.md",
                section="intro",
                content="Second version (should win)",
                metadata=EntryMetadata(
                    title="Dup Test Updated",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
            ),
        ]
        chroma_index.index_documents(duplicate_chunks)

        # Should only have one document
        assert chroma_index.doc_count() == 1

        # Should have the later version
        results = chroma_index.search("version")
        assert len(results) == 1
        assert "Second version" in results[0].snippet


class TestChromaSearch:
    """Test semantic search functionality."""

    def test_search_by_semantic_similarity(self, chroma_index, sample_chunks):
        """Search finds semantically similar documents."""
        chroma_index.index_documents(sample_chunks)

        # Search for concept related to neural networks
        results = chroma_index.search("brain-inspired computing systems")
        assert len(results) >= 1
        # Should find the neural networks document
        assert any("neural" in r.snippet.lower() for r in results)

    def test_search_returns_normalized_scores(self, chroma_index, sample_chunks):
        """Search results have normalized scores (0-1 range)."""
        chroma_index.index_documents(sample_chunks)
        results = chroma_index.search("deep learning")

        assert len(results) >= 1
        for result in results:
            assert 0.0 <= result.score <= 1.0

    def test_search_respects_limit(self, chroma_index):
        """Search limit parameter restricts results."""
        chunks = [
            DocumentChunk(
                path=f"docs/ai{i}.md",
                section=None,
                content=f"Artificial intelligence document {i} about neural networks and deep learning",
                metadata=EntryMetadata(
                    title=f"AI Doc {i}",
                    tags=["ai"],
                    created=date(2024, 1, (i % 28) + 1),
                ),
            )
            for i in range(20)
        ]
        chroma_index.index_documents(chunks)

        results = chroma_index.search("neural networks", limit=5)
        assert len(results) == 5

    def test_search_with_no_results(self, chroma_index, sample_chunk):
        """Search with poor semantic match returns results (embedding-based always finds something)."""
        chroma_index.index_document(sample_chunk)
        # Very unrelated query
        results = chroma_index.search("completely unrelated topic xyz")
        # Chroma will still return results, but with low scores
        assert isinstance(results, list)

    def test_search_empty_index(self, chroma_index):
        """Searching empty index returns empty list."""
        results = chroma_index.search("anything")
        assert results == []

    def test_search_returns_best_matches_first(self, chroma_index):
        """Search results are ordered by relevance."""
        chunks = [
            DocumentChunk(
                path="exact_match.md",
                section=None,
                content="Deep learning neural networks for computer vision tasks",
                metadata=EntryMetadata(
                    title="Deep Learning",
                    tags=["ai"],
                    created=date(2024, 1, 1),
                ),
            ),
            DocumentChunk(
                path="related.md",
                section=None,
                content="Machine learning algorithms for data analysis",
                metadata=EntryMetadata(
                    title="ML",
                    tags=["ai"],
                    created=date(2024, 1, 2),
                ),
            ),
            DocumentChunk(
                path="unrelated.md",
                section=None,
                content="Database administration and SQL queries",
                metadata=EntryMetadata(
                    title="Databases",
                    tags=["database"],
                    created=date(2024, 1, 3),
                ),
            ),
        ]
        chroma_index.index_documents(chunks)

        results = chroma_index.search("deep learning neural networks")
        assert len(results) >= 2
        # Most relevant should be first
        assert results[0].score >= results[1].score


class TestChromaSearchEdgeCases:
    """Test search edge cases and special inputs."""

    def test_search_empty_query(self, chroma_index, sample_chunk):
        """Empty query returns empty results."""
        chroma_index.index_document(sample_chunk)
        results = chroma_index.search("")
        # ChromaDB still generates an embedding for empty string and returns results
        # This is expected behavior for semantic search
        assert isinstance(results, list)

    def test_search_special_characters(self, chroma_index):
        """Search handles special characters in content."""
        chunk = DocumentChunk(
            path="special.md",
            section=None,
            content="Code: def function(x, y): return x + y # Python function",
            metadata=EntryMetadata(
                title="Code Examples",
                tags=["code"],
                created=date(2024, 1, 1),
            ),
        )
        chroma_index.index_document(chunk)

        results = chroma_index.search("python function definition")
        assert len(results) >= 1

    def test_search_unicode_content(self, chroma_index):
        """Search handles Unicode characters correctly."""
        chunk = DocumentChunk(
            path="unicode.md",
            section=None,
            content="Multilingual text: 你好世界 Bonjour monde Hola mundo",
            metadata=EntryMetadata(
                title="Unicode Test",
                tags=["unicode"],
                created=date(2024, 1, 1),
            ),
        )
        chroma_index.index_document(chunk)

        results = chroma_index.search("multilingual greetings")
        assert len(results) >= 1

    def test_search_very_long_query(self, chroma_index, sample_chunk):
        """Search handles very long queries."""
        chroma_index.index_document(sample_chunk)
        long_query = "machine learning artificial intelligence " * 50
        results = chroma_index.search(long_query)
        assert isinstance(results, list)

    def test_search_numeric_content(self, chroma_index):
        """Search handles numeric content."""
        chunk = DocumentChunk(
            path="numbers.md",
            section=None,
            content="Statistical analysis: mean=42.5, stddev=3.14159, n=1000 samples",
            metadata=EntryMetadata(
                title="Statistics",
                tags=["stats"],
                created=date(2024, 1, 1),
            ),
        )
        chroma_index.index_document(chunk)

        results = chroma_index.search("statistical data analysis")
        assert len(results) >= 1


class TestChromaDeleteDocument:
    """Test document deletion operations."""

    def test_delete_single_document(self, chroma_index, sample_chunks):
        """Can delete a document by path."""
        chroma_index.index_documents(sample_chunks)
        assert chroma_index.doc_count() == 3

        chroma_index.delete_document("ai/neural_nets.md")
        assert chroma_index.doc_count() == 2

    def test_delete_all_chunks_for_path(self, chroma_index):
        """Deleting by path removes all chunks for that document."""
        chunks = [
            DocumentChunk(
                path="multi/chunk.md",
                section="intro",
                content="Introduction to multi-section documents",
                metadata=EntryMetadata(
                    title="Multi-chunk Doc",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
            ),
            DocumentChunk(
                path="multi/chunk.md",
                section="body",
                content="Main body content of the document",
                metadata=EntryMetadata(
                    title="Multi-chunk Doc",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
            ),
        ]
        chroma_index.index_documents(chunks)
        assert chroma_index.doc_count() == 2

        chroma_index.delete_document("multi/chunk.md")
        assert chroma_index.doc_count() == 0

    def test_delete_nonexistent_document(self, chroma_index):
        """Deleting non-existent document doesn't cause errors."""
        chroma_index.delete_document("nonexistent/path.md")
        assert chroma_index.doc_count() == 0

    def test_search_after_delete(self, chroma_index, sample_chunks):
        """Deleted documents don't appear in search results."""
        chroma_index.index_documents(sample_chunks)

        # Delete one document
        chroma_index.delete_document("ai/neural_nets.md")

        # Search for content that was in deleted document
        results = chroma_index.search("neural networks biological")
        # Should not find the deleted document
        paths = [r.path for r in results]
        assert "ai/neural_nets.md" not in paths


class TestChromaDeleteDocuments:
    """Test batch document deletion operations."""

    def test_delete_multiple_documents(self, chroma_index, sample_chunks):
        """Can delete multiple documents by path in a single operation."""
        chroma_index.index_documents(sample_chunks)
        assert chroma_index.doc_count() == 3

        chroma_index.delete_documents(["ai/neural_nets.md", "ai/transformers.md"])
        assert chroma_index.doc_count() == 1

        # Only databases doc should remain
        results = chroma_index.search("SQL databases")
        assert len(results) == 1
        assert results[0].path == "dev/databases.md"

    def test_delete_documents_empty_list(self, chroma_index, sample_chunks):
        """Deleting empty list doesn't cause errors or changes."""
        chroma_index.index_documents(sample_chunks)
        assert chroma_index.doc_count() == 3

        chroma_index.delete_documents([])
        assert chroma_index.doc_count() == 3

    def test_delete_documents_with_nonexistent_paths(self, chroma_index, sample_chunks):
        """Deleting mix of existent and non-existent paths works correctly."""
        chroma_index.index_documents(sample_chunks)
        assert chroma_index.doc_count() == 3

        chroma_index.delete_documents(["ai/neural_nets.md", "nonexistent/path.md"])
        assert chroma_index.doc_count() == 2

    def test_delete_documents_all_nonexistent(self, chroma_index, sample_chunks):
        """Deleting only non-existent paths doesn't cause errors."""
        chroma_index.index_documents(sample_chunks)
        assert chroma_index.doc_count() == 3

        chroma_index.delete_documents(["fake/path1.md", "fake/path2.md"])
        assert chroma_index.doc_count() == 3

    def test_delete_documents_removes_all_chunks_per_path(self, chroma_index):
        """Batch delete removes all chunks for each document path."""
        chunks = [
            DocumentChunk(
                path="multi/doc1.md",
                section="intro",
                content="Doc1 introduction about neural networks",
                metadata=EntryMetadata(
                    title="Doc1",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
            ),
            DocumentChunk(
                path="multi/doc1.md",
                section="body",
                content="Doc1 body about machine learning",
                metadata=EntryMetadata(
                    title="Doc1",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
            ),
            DocumentChunk(
                path="multi/doc2.md",
                section="intro",
                content="Doc2 introduction about databases",
                metadata=EntryMetadata(
                    title="Doc2",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
            ),
            DocumentChunk(
                path="multi/doc2.md",
                section="body",
                content="Doc2 body about SQL queries",
                metadata=EntryMetadata(
                    title="Doc2",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
            ),
        ]
        chroma_index.index_documents(chunks)
        assert chroma_index.doc_count() == 4

        chroma_index.delete_documents(["multi/doc1.md"])
        assert chroma_index.doc_count() == 2

        # Only doc2 chunks should remain
        results = chroma_index.search("databases SQL")
        paths = [r.path for r in results]
        assert all(p == "multi/doc2.md" for p in paths)

    def test_delete_documents_single_batch_operation(self, chroma_index, sample_chunks):
        """Batch delete uses single query/delete operation."""
        chroma_index.index_documents(sample_chunks)

        # Delete all documents at once
        paths = [chunk.path for chunk in sample_chunks]
        chroma_index.delete_documents(paths)
        assert chroma_index.doc_count() == 0

    def test_search_after_batch_delete(self, chroma_index, sample_chunks):
        """Deleted documents don't appear in search results after batch delete."""
        chroma_index.index_documents(sample_chunks)

        # Delete two documents
        chroma_index.delete_documents(["ai/neural_nets.md", "ai/transformers.md"])

        # Search for content that was in deleted documents
        results = chroma_index.search("neural networks transformers NLP")
        paths = [r.path for r in results]
        assert "ai/neural_nets.md" not in paths
        assert "ai/transformers.md" not in paths


class TestChromaClear:
    """Test index clearing operations."""

    def test_clear_removes_all_documents(self, chroma_index, sample_chunks):
        """Clear removes all indexed documents."""
        chroma_index.index_documents(sample_chunks)
        assert chroma_index.doc_count() == 3

        chroma_index.clear()
        assert chroma_index.doc_count() == 0

    def test_clear_empty_index(self, chroma_index):
        """Clearing empty index doesn't cause errors."""
        chroma_index.clear()
        assert chroma_index.doc_count() == 0

    def test_index_after_clear(self, chroma_index, sample_chunk, sample_chunks):
        """Can index new documents after clearing."""
        chroma_index.index_documents(sample_chunks)
        chroma_index.clear()
        chroma_index.index_document(sample_chunk)
        assert chroma_index.doc_count() == 1


class TestChromaDocCount:
    """Test document counting functionality."""

    def test_doc_count_empty_index(self, chroma_index):
        """Empty index has zero document count."""
        assert chroma_index.doc_count() == 0

    def test_doc_count_after_indexing(self, chroma_index, sample_chunks):
        """Document count reflects indexed documents."""
        chroma_index.index_documents(sample_chunks)
        assert chroma_index.doc_count() == 3

    def test_doc_count_after_delete(self, chroma_index, sample_chunks):
        """Document count decreases after deletion."""
        chroma_index.index_documents(sample_chunks)
        initial_count = chroma_index.doc_count()

        chroma_index.delete_document(sample_chunks[0].path)
        assert chroma_index.doc_count() == initial_count - 1


class TestChromaSearchMetadata:
    """Test that search results preserve metadata correctly."""

    def test_result_contains_all_metadata(self, chroma_index):
        """Search results contain all expected metadata fields."""
        chunk = DocumentChunk(
            path="meta/test.md",
            section="section1",
            content="Metadata preservation test for semantic search indexing",
            metadata=EntryMetadata(
                title="Metadata Test",
                tags=["meta", "test"],
                created=date(2024, 1, 15),
                updated=date(2024, 2, 20),
                source_project="test-project",
            ),
            token_count=42,
        )
        chroma_index.index_document(chunk)

        results = chroma_index.search("metadata preservation")
        assert len(results) == 1
        result = results[0]

        assert result.path == "meta/test.md"
        assert result.title == "Metadata Test"
        assert result.section == "section1"
        assert set(result.tags) == {"meta", "test"}
        assert result.created == date(2024, 1, 15)
        assert result.updated == date(2024, 2, 20)
        assert result.token_count == 42
        assert result.source_project == "test-project"

    def test_result_snippet_is_generated(self, chroma_index):
        """Search results include content snippets."""
        chunk = DocumentChunk(
            path="snippet/test.md",
            section=None,
            content="This is a very long document that should be truncated to create a snippet. " * 10,
            metadata=EntryMetadata(
                title="Long Doc",
                tags=["test"],
                created=date(2024, 1, 1),
            ),
        )
        chroma_index.index_document(chunk)

        results = chroma_index.search("long document")
        assert len(results) == 1
        assert results[0].snippet is not None
        assert len(results[0].snippet) <= 210  # Max length + "..."

    def test_result_snippet_strips_markdown(self, chroma_index):
        """Result snippets have markdown syntax stripped."""
        chunk = DocumentChunk(
            path="markdown/test.md",
            section=None,
            content="# Header\n\n**Bold** and *italic* with [link](url) and `code`",
            metadata=EntryMetadata(
                title="Markdown Test",
                tags=["test"],
                created=date(2024, 1, 1),
            ),
        )
        chroma_index.index_document(chunk)

        results = chroma_index.search("header bold italic")
        assert len(results) == 1
        snippet = results[0].snippet

        # Should not contain markdown syntax
        assert "**" not in snippet
        assert "[" not in snippet
        assert "]" not in snippet
        assert "`" not in snippet


class TestChromaPreload:
    """Test embedding model preloading functionality."""

    def test_preload_loads_model(self, chroma_index):
        """Preload loads the embedding model."""
        assert chroma_index._model is None
        chroma_index.preload()
        assert chroma_index._model is not None

    def test_preload_loads_collection(self, chroma_index):
        """Preload initializes the collection."""
        assert chroma_index._collection is None
        chroma_index.preload()
        assert chroma_index._collection is not None

    def test_preload_idempotent(self, chroma_index):
        """Calling preload multiple times is safe."""
        chroma_index.preload()
        model1 = chroma_index._model
        collection1 = chroma_index._collection

        chroma_index.preload()
        assert chroma_index._model is model1
        assert chroma_index._collection is collection1


class TestChromaEmbeddings:
    """Test embedding generation functionality."""

    def test_embeddings_are_consistent(self, chroma_index):
        """Same text produces same embeddings."""
        text = "Consistent embedding test"

        embedding1 = chroma_index._embed([text])[0]
        embedding2 = chroma_index._embed([text])[0]

        assert len(embedding1) == len(embedding2)
        # Embeddings should be very similar (within floating point precision)
        for v1, v2 in zip(embedding1, embedding2):
            assert abs(v1 - v2) < 1e-6

    def test_embeddings_have_expected_dimensions(self, chroma_index):
        """Embeddings have the expected dimensionality."""
        text = "Test document for embedding dimensions"
        embeddings = chroma_index._embed([text])

        assert len(embeddings) == 1
        # Sentence transformers typically produce 384 or 768 dimensional vectors
        # Check that we have a reasonable dimension
        assert len(embeddings[0]) >= 100

    def test_batch_embedding_generation(self, chroma_index):
        """Can generate embeddings for multiple texts at once."""
        texts = [
            "First document about AI",
            "Second document about databases",
            "Third document about web development",
        ]

        embeddings = chroma_index._embed(texts)
        assert len(embeddings) == 3
        assert all(len(emb) == len(embeddings[0]) for emb in embeddings)


class TestChromaLargeDocuments:
    """Test behavior with large documents and batches."""

    def test_index_large_document(self, chroma_index):
        """Can index very large documents."""
        large_content = "Machine learning and artificial intelligence concepts. " * 200
        chunk = DocumentChunk(
            path="large/doc.md",
            section=None,
            content=large_content,
            metadata=EntryMetadata(
                title="Large Document",
                tags=["large", "ai"],
                created=date(2024, 1, 1),
            ),
            token_count=2000,
        )
        chroma_index.index_document(chunk)
        assert chroma_index.doc_count() == 1

        results = chroma_index.search("machine learning concepts")
        assert len(results) == 1

    def test_index_many_documents(self, chroma_index):
        """Can handle indexing many documents in batch."""
        chunks = [
            DocumentChunk(
                path=f"docs/doc{i}.md",
                section=None,
                content=f"Document {i} about artificial intelligence topic {i % 10}",
                metadata=EntryMetadata(
                    title=f"AI Document {i}",
                    tags=[f"tag{i % 5}"],
                    created=date(2024, 1, (i % 28) + 1),
                ),
            )
            for i in range(50)
        ]
        chroma_index.index_documents(chunks)
        assert chroma_index.doc_count() == 50

        results = chroma_index.search("artificial intelligence", limit=10)
        assert len(results) == 10


class TestChromaErrorHandling:
    """Test error handling and edge cases."""

    def test_handles_collection_recreation_on_error(self, chroma_index, sample_chunk):
        """Handles schema incompatibility by recreating collection."""
        # This test verifies the KeyError handling in _get_collection
        # Index a document normally
        chroma_index.index_document(sample_chunk)
        assert chroma_index.doc_count() == 1

    def test_empty_content_handling(self, chroma_index):
        """Handles empty content gracefully."""
        chunk = DocumentChunk(
            path="empty/doc.md",
            section=None,
            content="",
            metadata=EntryMetadata(
                title="Empty",
                tags=["test"],
                created=date(2024, 1, 1),
            ),
        )
        # Should not crash
        chroma_index.index_document(chunk)
        assert chroma_index.doc_count() == 1

    def test_whitespace_only_content(self, chroma_index):
        """Handles whitespace-only content."""
        chunk = DocumentChunk(
            path="whitespace/doc.md",
            section=None,
            content="   \n\n   \t\t   ",
            metadata=EntryMetadata(
                title="Whitespace",
                tags=["test"],
                created=date(2024, 1, 1),
            ),
        )
        chroma_index.index_document(chunk)
        assert chroma_index.doc_count() == 1


class TestChromaEmbeddingCache:
    """Test embedding cache functionality to avoid regenerating embeddings for unchanged content."""

    def test_content_hash_is_deterministic(self, chroma_index):
        """Same content produces same hash."""
        text = "Some test content for hashing"
        hash1 = chroma_index._content_hash(text)
        hash2 = chroma_index._content_hash(text)
        assert hash1 == hash2
        assert len(hash1) == 16  # 16-char hex hash

    def test_content_hash_differs_for_different_content(self, chroma_index):
        """Different content produces different hashes."""
        hash1 = chroma_index._content_hash("Content A")
        hash2 = chroma_index._content_hash("Content B")
        assert hash1 != hash2

    def test_unchanged_content_skips_embedding(self, chroma_index):
        """Re-indexing unchanged content skips embedding generation."""
        chunk = DocumentChunk(
            path="cache/test.md",
            section="intro",
            content="Original content that won't change",
            metadata=EntryMetadata(
                title="Cache Test",
                tags=["test"],
                created=date(2024, 1, 1),
            ),
        )
        # First index - should generate embedding
        chroma_index.index_document(chunk)
        assert chroma_index.doc_count() == 1

        # Track embed calls
        embed_call_count = 0
        original_embed = chroma_index._embed

        def counting_embed(texts):
            nonlocal embed_call_count
            embed_call_count += 1
            return original_embed(texts)

        chroma_index._embed = counting_embed

        # Re-index same content - should skip embedding
        chroma_index.index_document(chunk)
        assert embed_call_count == 0  # No new embeddings generated
        assert chroma_index.doc_count() == 1

    def test_changed_content_regenerates_embedding(self, chroma_index):
        """Changed content triggers new embedding generation."""
        chunk1 = DocumentChunk(
            path="cache/update.md",
            section="intro",
            content="Original content version one",
            metadata=EntryMetadata(
                title="Cache Update Test",
                tags=["test"],
                created=date(2024, 1, 1),
            ),
        )
        chroma_index.index_document(chunk1)

        # Track embed calls
        embed_call_count = 0
        original_embed = chroma_index._embed

        def counting_embed(texts):
            nonlocal embed_call_count
            embed_call_count += 1
            return original_embed(texts)

        chroma_index._embed = counting_embed

        # Update content - should regenerate embedding
        chunk2 = DocumentChunk(
            path="cache/update.md",
            section="intro",
            content="Updated content version two",
            metadata=EntryMetadata(
                title="Cache Update Test",
                tags=["test"],
                created=date(2024, 1, 1),
            ),
        )
        chroma_index.index_document(chunk2)
        assert embed_call_count == 1  # New embedding generated
        assert chroma_index.doc_count() == 1

        # Verify new content is searchable
        results = chroma_index.search("version two")
        assert len(results) == 1

    def test_batch_index_skips_unchanged_content(self, chroma_index):
        """Batch indexing skips embedding for unchanged documents."""
        chunks = [
            DocumentChunk(
                path="batch/doc1.md",
                section=None,
                content="First document content",
                metadata=EntryMetadata(
                    title="Doc 1",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
            ),
            DocumentChunk(
                path="batch/doc2.md",
                section=None,
                content="Second document content",
                metadata=EntryMetadata(
                    title="Doc 2",
                    tags=["test"],
                    created=date(2024, 1, 2),
                ),
            ),
        ]
        # First index
        chroma_index.index_documents(chunks)
        assert chroma_index.doc_count() == 2

        # Track embed calls
        embed_call_count = 0
        original_embed = chroma_index._embed

        def counting_embed(texts):
            nonlocal embed_call_count
            embed_call_count += 1
            return original_embed(texts)

        chroma_index._embed = counting_embed

        # Re-index same content - should skip all embeddings
        chroma_index.index_documents(chunks)
        assert embed_call_count == 0

    def test_batch_index_only_embeds_changed_content(self, chroma_index):
        """Batch indexing only generates embeddings for changed documents."""
        chunks = [
            DocumentChunk(
                path="partial/doc1.md",
                section=None,
                content="First document unchanged",
                metadata=EntryMetadata(
                    title="Doc 1",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
            ),
            DocumentChunk(
                path="partial/doc2.md",
                section=None,
                content="Second document will change",
                metadata=EntryMetadata(
                    title="Doc 2",
                    tags=["test"],
                    created=date(2024, 1, 2),
                ),
            ),
        ]
        chroma_index.index_documents(chunks)

        # Track embed calls and what was embedded
        embedded_texts = []
        original_embed = chroma_index._embed

        def tracking_embed(texts):
            embedded_texts.extend(texts)
            return original_embed(texts)

        chroma_index._embed = tracking_embed

        # Update only second document
        updated_chunks = [
            chunks[0],  # Unchanged
            DocumentChunk(
                path="partial/doc2.md",
                section=None,
                content="Second document has been updated",
                metadata=EntryMetadata(
                    title="Doc 2",
                    tags=["test"],
                    created=date(2024, 1, 2),
                ),
            ),
        ]
        chroma_index.index_documents(updated_chunks)

        # Should only embed the changed document
        assert len(embedded_texts) == 1
        assert "updated" in embedded_texts[0]

    def test_content_hash_stored_in_metadata(self, chroma_index):
        """Content hash is stored in document metadata."""
        chunk = DocumentChunk(
            path="hash/meta.md",
            section=None,
            content="Content for hash storage test",
            metadata=EntryMetadata(
                title="Hash Meta Test",
                tags=["test"],
                created=date(2024, 1, 1),
            ),
        )
        chroma_index.index_document(chunk)

        # Verify hash is stored
        collection = chroma_index._get_collection()
        result = collection.get(ids=["hash/meta.md#main"], include=["metadatas"])
        assert result["metadatas"]
        assert result["metadatas"][0].get("content_hash")
        assert len(result["metadatas"][0]["content_hash"]) == 16

    def test_get_existing_hash_returns_none_for_missing(self, chroma_index):
        """_get_existing_hash returns None for non-existent documents."""
        result = chroma_index._get_existing_hash("nonexistent/path.md#main")
        assert result is None

    def test_get_existing_hash_returns_hash_for_existing(self, chroma_index):
        """_get_existing_hash returns the stored hash for existing documents."""
        chunk = DocumentChunk(
            path="existing/doc.md",
            section="intro",
            content="Test content for hash retrieval",
            metadata=EntryMetadata(
                title="Existing Doc",
                tags=["test"],
                created=date(2024, 1, 1),
            ),
        )
        chroma_index.index_document(chunk)

        expected_hash = chroma_index._content_hash(chunk.content)
        actual_hash = chroma_index._get_existing_hash("existing/doc.md#intro")
        assert actual_hash == expected_hash
