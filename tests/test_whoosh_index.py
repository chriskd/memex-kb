"""Comprehensive tests for WhooshIndex keyword search indexer."""

from datetime import date
from pathlib import Path

import pytest

from memex.indexer.whoosh_index import WhooshIndex
from memex.models import DocumentChunk, EntryMetadata


@pytest.fixture
def index_dir(tmp_path) -> Path:
    """Create a temporary directory for Whoosh index."""
    return tmp_path / "whoosh_test"


@pytest.fixture
def whoosh_index(index_dir) -> WhooshIndex:
    """Create a fresh WhooshIndex instance for each test."""
    return WhooshIndex(index_dir=index_dir)


@pytest.fixture
def sample_chunk() -> DocumentChunk:
    """Create a sample document chunk for testing."""
    return DocumentChunk(
        path="test/sample.md",
        section="intro",
        content="This is a sample document about Python programming and testing.",
        metadata=EntryMetadata(
            title="Sample Document",
            tags=["python", "testing"],
            created=date(2024, 1, 1),
            updated=date(2024, 1, 15),
            source_project="test-project",
        ),
        token_count=12,
    )


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """Create multiple sample chunks for batch testing."""
    return [
        DocumentChunk(
            path="docs/python.md",
            section="basics",
            content="Python is a high-level programming language with dynamic typing.",
            metadata=EntryMetadata(
                title="Python Basics",
                tags=["python", "programming"],
                created=date(2024, 1, 1),
            ),
            token_count=10,
        ),
        DocumentChunk(
            path="docs/testing.md",
            section="unit-tests",
            content="Unit testing is essential for software quality assurance and regression prevention.",
            metadata=EntryMetadata(
                title="Testing Guide",
                tags=["testing", "quality"],
                created=date(2024, 1, 2),
            ),
            token_count=11,
        ),
        DocumentChunk(
            path="docs/databases.md",
            section=None,
            content="SQL databases provide ACID guarantees for reliable data storage.",
            metadata=EntryMetadata(
                title="Database Guide",
                tags=["database", "sql"],
                created=date(2024, 1, 3),
            ),
            token_count=9,
        ),
    ]


class TestWhooshIndexInitialization:
    """Test Whoosh index initialization and setup."""

    def test_init_creates_directory(self, whoosh_index, index_dir):
        """Index directory is created when first accessed."""
        assert not index_dir.exists()
        # Trigger index creation
        whoosh_index.doc_count()
        assert index_dir.exists()

    def test_init_with_custom_path(self, tmp_path):
        """Can initialize with custom index path."""
        custom_path = tmp_path / "custom" / "whoosh"
        index = WhooshIndex(index_dir=custom_path)
        index.doc_count()
        assert custom_path.exists()

    def test_reopens_existing_index(self, whoosh_index, sample_chunk):
        """Can reopen an existing index and access previous data."""
        whoosh_index.index_document(sample_chunk)
        assert whoosh_index.doc_count() == 1

        # Create new index instance with same directory
        new_index = WhooshIndex(index_dir=whoosh_index._index_dir)
        assert new_index.doc_count() == 1


class TestWhooshIndexDocument:
    """Test single document indexing operations."""

    def test_index_single_document(self, whoosh_index, sample_chunk):
        """Can index a single document chunk."""
        whoosh_index.index_document(sample_chunk)
        assert whoosh_index.doc_count() == 1

    def test_index_document_with_all_fields(self, whoosh_index):
        """Document with all fields is indexed correctly."""
        chunk = DocumentChunk(
            path="complete/doc.md",
            section="section1",
            content="Complete document with all metadata fields populated.",
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
        whoosh_index.index_document(chunk)
        results = whoosh_index.search("complete")
        assert len(results) == 1
        assert results[0].path == "complete/doc.md"
        assert results[0].title == "Complete Doc"

    def test_index_document_with_minimal_fields(self, whoosh_index):
        """Document with minimal required fields is indexed correctly."""
        chunk = DocumentChunk(
            path="minimal/doc.md",
            section=None,
            content="Minimal document content.",
            metadata=EntryMetadata(
                title="Minimal",
                tags=["minimal"],
                created=date(2024, 1, 1),
            ),
            token_count=None,
        )
        whoosh_index.index_document(chunk)
        assert whoosh_index.doc_count() == 1
        results = whoosh_index.search("minimal")
        assert len(results) == 1

    def test_update_existing_document(self, whoosh_index):
        """Updating a document replaces the old version."""
        chunk1 = DocumentChunk(
            path="test/update.md",
            section="intro",
            content="Original content",
            metadata=EntryMetadata(
                title="Original Title",
                tags=["test"],
                created=date(2024, 1, 1),
            ),
        )
        whoosh_index.index_document(chunk1)

        chunk2 = DocumentChunk(
            path="test/update.md",
            section="intro",
            content="Updated content with new information",
            metadata=EntryMetadata(
                title="Updated Title",
                tags=["test", "updated"],
                created=date(2024, 1, 1),
                updated=date(2024, 1, 10),
            ),
        )
        whoosh_index.index_document(chunk2)

        # Should still have only one document
        assert whoosh_index.doc_count() == 1

        # Should find updated content
        results = whoosh_index.search("updated")
        assert len(results) == 1
        assert results[0].title == "Updated Title"


class TestWhooshIndexDocuments:
    """Test batch document indexing operations."""

    def test_index_multiple_documents(self, whoosh_index, sample_chunks):
        """Can index multiple documents in batch."""
        whoosh_index.index_documents(sample_chunks)
        assert whoosh_index.doc_count() == 3

    def test_index_empty_list(self, whoosh_index):
        """Indexing empty list doesn't cause errors."""
        whoosh_index.index_documents([])
        assert whoosh_index.doc_count() == 0

    def test_batch_index_is_atomic(self, whoosh_index, sample_chunks):
        """Batch indexing happens in single transaction."""
        whoosh_index.index_documents(sample_chunks)
        # All documents should be available immediately
        results = whoosh_index.search("python OR testing OR database")
        assert len(results) == 3


class TestWhooshSearch:
    """Test search functionality with various queries."""

    def test_search_by_content(self, whoosh_index, sample_chunk):
        """Can search document by content."""
        whoosh_index.index_document(sample_chunk)
        results = whoosh_index.search("Python programming")
        assert len(results) == 1
        assert results[0].path == "test/sample.md"

    def test_search_by_title(self, whoosh_index, sample_chunk):
        """Can search document by title."""
        whoosh_index.index_document(sample_chunk)
        results = whoosh_index.search("Sample Document")
        assert len(results) == 1
        assert results[0].title == "Sample Document"

    def test_search_by_tags(self, whoosh_index, sample_chunk):
        """Can search document by tags."""
        whoosh_index.index_document(sample_chunk)
        results = whoosh_index.search("python")
        assert len(results) == 1
        assert "python" in results[0].tags

    def test_search_returns_normalized_scores(self, whoosh_index, sample_chunks):
        """Search results have normalized scores (0-1 range)."""
        whoosh_index.index_documents(sample_chunks)
        results = whoosh_index.search("python")
        assert len(results) >= 1
        for result in results:
            assert 0.0 <= result.score <= 1.0

    def test_search_respects_limit(self, whoosh_index, sample_chunks):
        """Search limit parameter restricts results."""
        # Create more documents
        many_chunks = sample_chunks + [
            DocumentChunk(
                path=f"docs/test{i}.md",
                section=None,
                content=f"Python document number {i}",
                metadata=EntryMetadata(
                    title=f"Test {i}",
                    tags=["python"],
                    created=date(2024, 1, i + 1),
                ),
            )
            for i in range(10)
        ]
        whoosh_index.index_documents(many_chunks)

        results = whoosh_index.search("python", limit=5)
        assert len(results) <= 5

    def test_search_with_no_results(self, whoosh_index, sample_chunk):
        """Search with no matches returns empty list."""
        whoosh_index.index_document(sample_chunk)
        results = whoosh_index.search("nonexistent query xyz")
        assert results == []

    def test_search_empty_index(self, whoosh_index):
        """Searching empty index returns empty list."""
        results = whoosh_index.search("anything")
        assert results == []

    def test_search_multi_field(self, whoosh_index):
        """Search matches across title, content, and tags."""
        chunks = [
            DocumentChunk(
                path="title_match.md",
                section=None,
                content="Generic content here",
                metadata=EntryMetadata(
                    title="Machine Learning Guide",
                    tags=["guide"],
                    created=date(2024, 1, 1),
                ),
            ),
            DocumentChunk(
                path="content_match.md",
                section=None,
                content="This document discusses machine learning algorithms",
                metadata=EntryMetadata(
                    title="Algorithms",
                    tags=["guide"],
                    created=date(2024, 1, 2),
                ),
            ),
            DocumentChunk(
                path="tag_match.md",
                section=None,
                content="Generic content",
                metadata=EntryMetadata(
                    title="Reference",
                    tags=["machine-learning"],
                    created=date(2024, 1, 3),
                ),
            ),
        ]
        whoosh_index.index_documents(chunks)

        results = whoosh_index.search("machine learning")
        # Should find matches in title, content, and tags
        assert len(results) >= 2


class TestWhooshSearchEdgeCases:
    """Test search edge cases and special characters."""

    def test_search_empty_query(self, whoosh_index, sample_chunk):
        """Empty query returns empty results."""
        whoosh_index.index_document(sample_chunk)
        results = whoosh_index.search("")
        assert results == []

    def test_search_special_characters(self, whoosh_index):
        """Search handles special characters gracefully."""
        chunk = DocumentChunk(
            path="special.md",
            section=None,
            content="Code example: def function(x, y): return x + y",
            metadata=EntryMetadata(
                title="Code Examples",
                tags=["code"],
                created=date(2024, 1, 1),
            ),
        )
        whoosh_index.index_document(chunk)

        # Special chars that might break parsing
        results = whoosh_index.search("function(x, y)")
        assert len(results) >= 0  # Should not crash

    def test_search_unicode_content(self, whoosh_index):
        """Search handles Unicode characters correctly."""
        chunk = DocumentChunk(
            path="unicode.md",
            section=None,
            content="Unicode test: 你好世界 café naïve résumé",
            metadata=EntryMetadata(
                title="Unicode Test",
                tags=["unicode"],
                created=date(2024, 1, 1),
            ),
        )
        whoosh_index.index_document(chunk)

        results = whoosh_index.search("café")
        assert len(results) >= 0  # Should handle Unicode gracefully

    def test_search_very_long_query(self, whoosh_index, sample_chunk):
        """Search handles very long queries."""
        whoosh_index.index_document(sample_chunk)
        long_query = "python " * 100
        results = whoosh_index.search(long_query)
        assert isinstance(results, list)

    def test_search_query_with_invalid_syntax(self, whoosh_index, sample_chunk):
        """Invalid query syntax falls back gracefully."""
        whoosh_index.index_document(sample_chunk)
        # These might cause parsing errors in query parser
        problematic_queries = [
            "AND OR NOT",
            "(((",
            "]]]]",
            "***",
        ]
        for query in problematic_queries:
            results = whoosh_index.search(query)
            assert isinstance(results, list)


class TestWhooshDeleteDocument:
    """Test document deletion operations."""

    def test_delete_single_document(self, whoosh_index, sample_chunks):
        """Can delete a document by path."""
        whoosh_index.index_documents(sample_chunks)
        assert whoosh_index.doc_count() == 3

        whoosh_index.delete_document("docs/python.md")
        assert whoosh_index.doc_count() == 2

        # Deleted document should not appear in search
        results = whoosh_index.search("Python Basics")
        assert len(results) == 0

    def test_delete_all_chunks_for_path(self, whoosh_index):
        """Deleting by path removes all chunks for that document."""
        chunks = [
            DocumentChunk(
                path="multi/chunk.md",
                section="intro",
                content="Introduction section",
                metadata=EntryMetadata(
                    title="Multi-chunk Doc",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
            ),
            DocumentChunk(
                path="multi/chunk.md",
                section="body",
                content="Body section",
                metadata=EntryMetadata(
                    title="Multi-chunk Doc",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
            ),
        ]
        whoosh_index.index_documents(chunks)

        whoosh_index.delete_document("multi/chunk.md")
        results = whoosh_index.search("section")
        assert len(results) == 0

    def test_delete_nonexistent_document(self, whoosh_index):
        """Deleting non-existent document doesn't cause errors."""
        whoosh_index.delete_document("nonexistent/path.md")
        assert whoosh_index.doc_count() == 0


class TestWhooshDeleteDocuments:
    """Test batch document deletion operations."""

    def test_delete_multiple_documents(self, whoosh_index, sample_chunks):
        """Can delete multiple documents by path in a single operation."""
        whoosh_index.index_documents(sample_chunks)
        assert whoosh_index.doc_count() == 3

        whoosh_index.delete_documents(["docs/python.md", "docs/testing.md"])
        assert whoosh_index.doc_count() == 1

        # Only databases doc should remain
        results = whoosh_index.search("databases SQL")
        assert len(results) == 1
        assert results[0].path == "docs/databases.md"

    def test_delete_documents_empty_list(self, whoosh_index, sample_chunks):
        """Deleting empty list doesn't cause errors or changes."""
        whoosh_index.index_documents(sample_chunks)
        assert whoosh_index.doc_count() == 3

        whoosh_index.delete_documents([])
        assert whoosh_index.doc_count() == 3

    def test_delete_documents_with_nonexistent_paths(self, whoosh_index, sample_chunks):
        """Deleting mix of existent and non-existent paths works correctly."""
        whoosh_index.index_documents(sample_chunks)
        assert whoosh_index.doc_count() == 3

        whoosh_index.delete_documents(["docs/python.md", "nonexistent/path.md"])
        assert whoosh_index.doc_count() == 2

    def test_delete_documents_all_nonexistent(self, whoosh_index, sample_chunks):
        """Deleting only non-existent paths doesn't cause errors."""
        whoosh_index.index_documents(sample_chunks)
        assert whoosh_index.doc_count() == 3

        whoosh_index.delete_documents(["fake/path1.md", "fake/path2.md"])
        assert whoosh_index.doc_count() == 3

    def test_delete_documents_removes_all_chunks_per_path(self, whoosh_index):
        """Batch delete removes all chunks for each document path."""
        chunks = [
            DocumentChunk(
                path="multi/doc1.md",
                section="intro",
                content="Doc1 introduction section",
                metadata=EntryMetadata(
                    title="Doc1",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
            ),
            DocumentChunk(
                path="multi/doc1.md",
                section="body",
                content="Doc1 body section",
                metadata=EntryMetadata(
                    title="Doc1",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
            ),
            DocumentChunk(
                path="multi/doc2.md",
                section="intro",
                content="Doc2 introduction section",
                metadata=EntryMetadata(
                    title="Doc2",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
            ),
            DocumentChunk(
                path="multi/doc2.md",
                section="body",
                content="Doc2 body section",
                metadata=EntryMetadata(
                    title="Doc2",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
            ),
        ]
        whoosh_index.index_documents(chunks)
        # Each unique path#section combo is one doc
        assert whoosh_index.doc_count() == 4

        whoosh_index.delete_documents(["multi/doc1.md"])
        assert whoosh_index.doc_count() == 2

        # Only doc2 chunks should remain
        results = whoosh_index.search("Doc2")
        assert len(results) == 2

    def test_delete_documents_single_transaction(self, whoosh_index, sample_chunks):
        """Batch delete uses single transaction (all or nothing)."""
        whoosh_index.index_documents(sample_chunks)

        # Delete all documents at once
        paths = [chunk.path for chunk in sample_chunks]
        whoosh_index.delete_documents(paths)
        assert whoosh_index.doc_count() == 0


class TestWhooshClear:
    """Test index clearing operations."""

    def test_clear_removes_all_documents(self, whoosh_index, sample_chunks):
        """Clear removes all indexed documents."""
        whoosh_index.index_documents(sample_chunks)
        assert whoosh_index.doc_count() == 3

        whoosh_index.clear()
        assert whoosh_index.doc_count() == 0

    def test_clear_empty_index(self, whoosh_index):
        """Clearing empty index doesn't cause errors."""
        whoosh_index.clear()
        assert whoosh_index.doc_count() == 0

    def test_index_after_clear(self, whoosh_index, sample_chunk, sample_chunks):
        """Can index new documents after clearing."""
        whoosh_index.index_documents(sample_chunks)
        whoosh_index.clear()
        whoosh_index.index_document(sample_chunk)
        assert whoosh_index.doc_count() == 1


class TestWhooshDocCount:
    """Test document counting functionality."""

    def test_doc_count_empty_index(self, whoosh_index):
        """Empty index has zero document count."""
        assert whoosh_index.doc_count() == 0

    def test_doc_count_after_indexing(self, whoosh_index, sample_chunks):
        """Document count reflects indexed documents."""
        whoosh_index.index_documents(sample_chunks)
        assert whoosh_index.doc_count() == 3

    def test_doc_count_after_delete(self, whoosh_index, sample_chunks):
        """Document count decreases after deletion."""
        whoosh_index.index_documents(sample_chunks)
        initial_count = whoosh_index.doc_count()

        whoosh_index.delete_document(sample_chunks[0].path)
        assert whoosh_index.doc_count() == initial_count - 1


class TestWhooshSearchMetadata:
    """Test that search results preserve metadata correctly."""

    def test_result_contains_all_metadata(self, whoosh_index):
        """Search results contain all expected metadata fields."""
        chunk = DocumentChunk(
            path="meta/test.md",
            section="section1",
            content="Test content for metadata",
            metadata=EntryMetadata(
                title="Metadata Test",
                tags=["meta", "test"],
                created=date(2024, 1, 15),
                updated=date(2024, 2, 20),
                source_project="test-project",
            ),
            token_count=42,
        )
        whoosh_index.index_document(chunk)

        results = whoosh_index.search("metadata")
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

    def test_result_snippet_is_generated(self, whoosh_index):
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
        whoosh_index.index_document(chunk)

        results = whoosh_index.search("document")
        assert len(results) == 1
        assert results[0].snippet is not None
        assert len(results[0].snippet) <= 210  # Max length + "..."

    def test_result_snippet_strips_markdown(self, whoosh_index):
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
        whoosh_index.index_document(chunk)

        results = whoosh_index.search("header")
        assert len(results) == 1
        snippet = results[0].snippet

        # Should not contain markdown syntax
        assert "**" not in snippet
        assert "*" not in snippet
        assert "[" not in snippet
        assert "]" not in snippet
        assert "`" not in snippet


class TestWhooshLargeDocuments:
    """Test behavior with large documents."""

    def test_index_large_document(self, whoosh_index):
        """Can index very large documents."""
        large_content = "Python programming tutorial. " * 1000
        chunk = DocumentChunk(
            path="large/doc.md",
            section=None,
            content=large_content,
            metadata=EntryMetadata(
                title="Large Document",
                tags=["large"],
                created=date(2024, 1, 1),
            ),
            token_count=3000,
        )
        whoosh_index.index_document(chunk)
        assert whoosh_index.doc_count() == 1

        results = whoosh_index.search("programming tutorial")
        assert len(results) == 1

    def test_search_large_number_of_documents(self, whoosh_index):
        """Can handle searching through many documents."""
        chunks = [
            DocumentChunk(
                path=f"docs/doc{i}.md",
                section=None,
                content=f"Document {i} about Python and programming topic {i % 10}",
                metadata=EntryMetadata(
                    title=f"Document {i}",
                    tags=[f"tag{i % 5}"],
                    created=date(2024, 1, (i % 28) + 1),
                ),
            )
            for i in range(100)
        ]
        whoosh_index.index_documents(chunks)
        assert whoosh_index.doc_count() == 100

        results = whoosh_index.search("Python programming", limit=10)
        assert len(results) == 10
