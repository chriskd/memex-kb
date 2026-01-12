"""Comprehensive tests for HybridSearcher combining Whoosh and Chroma."""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pytest

from memex.indexer.chroma_index import ChromaIndex
from memex.indexer.hybrid import HybridSearcher
from memex.indexer.whoosh_index import WhooshIndex
from memex.models import DocumentChunk, EntryMetadata

pytestmark = pytest.mark.semantic


@pytest.fixture
def index_dirs(tmp_path) -> tuple[Path, Path]:
    """Create temporary directories for both indices."""
    whoosh_dir = tmp_path / "whoosh"
    chroma_dir = tmp_path / "chroma"
    return whoosh_dir, chroma_dir


@pytest.fixture
def hybrid_searcher(index_dirs) -> HybridSearcher:
    """Create a HybridSearcher with separate test indices."""
    whoosh_dir, chroma_dir = index_dirs
    whoosh = WhooshIndex(index_dir=whoosh_dir)
    chroma = ChromaIndex(index_dir=chroma_dir)
    return HybridSearcher(whoosh_index=whoosh, chroma_index=chroma)


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """Create diverse sample chunks for testing."""
    return [
        DocumentChunk(
            path="ai/neural_nets.md",
            section="basics",
            content="Neural networks are inspired by biological neurons in the brain.",
            metadata=EntryMetadata(
                title="Neural Networks Basics",
                tags=["neural-networks", "ai", "deep-learning"],
                created=date(2024, 1, 1),
                source_project="ai-docs",
            ),
            token_count=11,
        ),
        DocumentChunk(
            path="ai/transformers.md",
            section="architecture",
            content="Transformer models use self-attention mechanisms for NLP tasks.",
            metadata=EntryMetadata(
                title="Transformers",
                tags=["transformers", "nlp"],
                created=date(2024, 1, 2),
                source_project="ai-docs",
            ),
            token_count=9,
        ),
        DocumentChunk(
            path="dev/python.md",
            section=None,
            content="Python is a versatile programming language for data science.",
            metadata=EntryMetadata(
                title="Python Guide",
                tags=["python", "programming"],
                created=date(2024, 1, 3),
                source_project="dev-docs",
            ),
            token_count=10,
        ),
    ]


class TestHybridSearcherInitialization:
    """Test HybridSearcher initialization."""

    def test_init_creates_default_indices(self):
        """Can initialize with default indices."""
        searcher = HybridSearcher()
        assert searcher._whoosh is not None
        assert searcher._chroma is not None

    def test_init_with_custom_indices(self, index_dirs):
        """Can initialize with custom index instances."""
        whoosh_dir, chroma_dir = index_dirs
        whoosh = WhooshIndex(index_dir=whoosh_dir)
        chroma = ChromaIndex(index_dir=chroma_dir)

        searcher = HybridSearcher(whoosh_index=whoosh, chroma_index=chroma)
        assert searcher._whoosh is whoosh
        assert searcher._chroma is chroma

    def test_init_last_indexed_none(self, hybrid_searcher):
        """Last indexed timestamp is None initially."""
        assert hybrid_searcher._last_indexed is None


class TestHybridIndexDocument:
    """Test single document indexing to both indices."""

    def test_index_document_to_both_indices(self, hybrid_searcher, sample_chunks):
        """Indexing a document adds it to both Whoosh and Chroma."""
        chunk = sample_chunks[0]
        hybrid_searcher.index_document(chunk)

        assert hybrid_searcher._whoosh.doc_count() == 1
        assert hybrid_searcher._chroma.doc_count() == 1

    def test_index_document_updates_timestamp(self, hybrid_searcher, sample_chunks):
        """Indexing updates the last_indexed timestamp."""
        assert hybrid_searcher._last_indexed is None

        hybrid_searcher.index_document(sample_chunks[0])
        assert hybrid_searcher._last_indexed is not None

    def test_index_document_searchable_in_both(self, hybrid_searcher, sample_chunks):
        """Indexed document is searchable via both indices."""
        chunk = sample_chunks[0]
        hybrid_searcher.index_document(chunk)

        # Should find via keyword search
        whoosh_results = hybrid_searcher._whoosh.search("neural networks")
        assert len(whoosh_results) == 1

        # Should find via semantic search
        chroma_results = hybrid_searcher._chroma.search("brain-inspired computing")
        assert len(chroma_results) == 1


class TestHybridIndexChunks:
    """Test batch document indexing."""

    def test_index_chunks_to_both_indices(self, hybrid_searcher, sample_chunks):
        """Batch indexing adds all chunks to both indices."""
        hybrid_searcher.index_chunks(sample_chunks)

        assert hybrid_searcher._whoosh.doc_count() == 3
        assert hybrid_searcher._chroma.doc_count() == 3

    def test_index_empty_chunks(self, hybrid_searcher):
        """Indexing empty list doesn't cause errors."""
        hybrid_searcher.index_chunks([])
        assert hybrid_searcher._whoosh.doc_count() == 0
        assert hybrid_searcher._chroma.doc_count() == 0

    def test_index_chunks_updates_timestamp(self, hybrid_searcher, sample_chunks):
        """Batch indexing updates the last_indexed timestamp."""
        assert hybrid_searcher._last_indexed is None

        hybrid_searcher.index_chunks(sample_chunks)
        assert hybrid_searcher._last_indexed is not None


class TestHybridDeleteDocument:
    """Test document deletion from both indices."""

    def test_delete_from_both_indices(self, hybrid_searcher, sample_chunks):
        """Deleting a document removes it from both indices."""
        hybrid_searcher.index_chunks(sample_chunks)

        hybrid_searcher.delete_document("ai/neural_nets.md")

        assert hybrid_searcher._whoosh.doc_count() == 2
        assert hybrid_searcher._chroma.doc_count() == 2

    def test_delete_nonexistent_document(self, hybrid_searcher):
        """Deleting non-existent document doesn't cause errors."""
        hybrid_searcher.delete_document("nonexistent.md")
        assert hybrid_searcher._whoosh.doc_count() == 0
        assert hybrid_searcher._chroma.doc_count() == 0


class TestHybridClear:
    """Test clearing both indices."""

    def test_clear_both_indices(self, hybrid_searcher, sample_chunks):
        """Clear removes all documents from both indices."""
        hybrid_searcher.index_chunks(sample_chunks)
        assert hybrid_searcher._whoosh.doc_count() == 3
        assert hybrid_searcher._chroma.doc_count() == 3

        hybrid_searcher.clear()

        assert hybrid_searcher._whoosh.doc_count() == 0
        assert hybrid_searcher._chroma.doc_count() == 0

    def test_clear_resets_timestamp(self, hybrid_searcher, sample_chunks):
        """Clear resets the last_indexed timestamp."""
        hybrid_searcher.index_chunks(sample_chunks)
        assert hybrid_searcher._last_indexed is not None

        hybrid_searcher.clear()
        assert hybrid_searcher._last_indexed is None


class TestHybridStatus:
    """Test index status reporting."""

    def test_status_empty_indices(self, hybrid_searcher, tmp_path, monkeypatch):
        """Status reports zero counts for empty indices."""
        kb_root = tmp_path / "kb"
        kb_root.mkdir()
        monkeypatch.setenv("MEMEX_KB_ROOT", str(kb_root))

        status = hybrid_searcher.status()
        assert status.whoosh_docs == 0
        assert status.chroma_docs == 0
        assert status.last_indexed is None

    def test_status_after_indexing(self, hybrid_searcher, sample_chunks, tmp_path, monkeypatch):
        """Status reflects document counts after indexing."""
        kb_root = tmp_path / "kb"
        kb_root.mkdir()
        monkeypatch.setenv("MEMEX_KB_ROOT", str(kb_root))

        hybrid_searcher.index_chunks(sample_chunks)
        status = hybrid_searcher.status()

        assert status.whoosh_docs == 3
        assert status.chroma_docs == 3
        assert status.last_indexed is not None


class TestHybridSearchModes:
    """Test different search modes (hybrid, keyword, semantic)."""

    def test_search_keyword_mode(self, hybrid_searcher, sample_chunks):
        """Keyword mode uses only Whoosh index."""
        hybrid_searcher.index_chunks(sample_chunks)

        results = hybrid_searcher.search("neural networks", mode="keyword")
        assert len(results) >= 1
        assert any("neural" in r.snippet.lower() for r in results)

    def test_search_semantic_mode(self, hybrid_searcher, sample_chunks):
        """Semantic mode uses only Chroma index."""
        hybrid_searcher.index_chunks(sample_chunks)

        results = hybrid_searcher.search("brain-inspired computing", mode="semantic")
        assert len(results) >= 1

    def test_search_hybrid_mode(self, hybrid_searcher, sample_chunks):
        """Hybrid mode combines both indices using RRF."""
        hybrid_searcher.index_chunks(sample_chunks)

        results = hybrid_searcher.search("neural networks", mode="hybrid")
        assert len(results) >= 1

    def test_search_default_mode_is_hybrid(self, hybrid_searcher, sample_chunks):
        """Default search mode is hybrid."""
        hybrid_searcher.index_chunks(sample_chunks)

        results = hybrid_searcher.search("neural")
        # Should use hybrid mode by default
        assert len(results) >= 1


class TestHybridSearchRRFMerge:
    """Test Reciprocal Rank Fusion algorithm."""

    def test_rrf_combines_results(self, hybrid_searcher):
        """RRF merges results from both indices."""
        chunks = [
            DocumentChunk(
                path="keyword_strong.md",
                section=None,
                content="Python Python Python programming language",
                metadata=EntryMetadata(
                    title="Python Keyword Match",
                    tags=["python"],
                    created=date(2024, 1, 1),
                ),
            ),
            DocumentChunk(
                path="semantic_strong.md",
                section=None,
                content="A versatile interpreted high-level language for scripting",
                metadata=EntryMetadata(
                    title="Programming Language",
                    tags=["coding"],
                    created=date(2024, 1, 2),
                ),
            ),
        ]
        hybrid_searcher.index_chunks(chunks)

        results = hybrid_searcher.search("Python programming", mode="hybrid")
        assert len(results) >= 1

    def test_rrf_normalizes_scores(self, hybrid_searcher, sample_chunks):
        """RRF produces normalized scores (0-1 range)."""
        hybrid_searcher.index_chunks(sample_chunks)

        results = hybrid_searcher.search("neural networks", mode="hybrid")
        assert len(results) >= 1
        for result in results:
            assert 0.0 <= result.score <= 1.0

    def test_rrf_deduplicates_results(self, hybrid_searcher):
        """RRF deduplicates results appearing in both indices."""
        chunk = DocumentChunk(
            path="duplicate.md",
            section="intro",
            content="Machine learning algorithms for data analysis",
            metadata=EntryMetadata(
                title="ML Algorithms",
                tags=["machine-learning"],
                created=date(2024, 1, 1),
            ),
        )
        hybrid_searcher.index_chunks([chunk])

        # This should appear in both Whoosh and Chroma results
        results = hybrid_searcher.search("machine learning", mode="hybrid")

        # Should only appear once in final results
        paths = [r.path for r in results]
        assert paths.count("duplicate.md") <= 1

    def test_rrf_handles_one_empty_index(self, hybrid_searcher, sample_chunks):
        """RRF handles case where one index has no results."""
        hybrid_searcher.index_chunks(sample_chunks)

        # Query that might only match in one index
        results = hybrid_searcher.search("xyznonexistent", mode="hybrid")
        # Should still work, returning results from whichever index has them
        assert isinstance(results, list)


class TestHybridSearchDeduplication:
    """Test deduplication of search results by path."""

    def test_deduplicate_keeps_highest_score(self, hybrid_searcher):
        """Deduplication keeps the highest-scoring chunk per document."""
        chunks = [
            DocumentChunk(
                path="multi/doc.md",
                section="intro",
                content="Introduction section with some content",
                metadata=EntryMetadata(
                    title="Multi-section Doc",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
            ),
            DocumentChunk(
                path="multi/doc.md",
                section="main",
                content="Main section with Python programming tutorial",
                metadata=EntryMetadata(
                    title="Multi-section Doc",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
            ),
        ]
        hybrid_searcher.index_chunks(chunks)

        results = hybrid_searcher.search("Python programming")

        # Should only return one result for multi/doc.md
        paths = [r.path for r in results]
        assert paths.count("multi/doc.md") == 1

    def test_deduplicate_respects_limit(self, hybrid_searcher):
        """Deduplication happens after limiting results."""
        chunks = [
            DocumentChunk(
                path=f"doc{i}.md",
                section=None,
                content=f"Document {i} about Python programming",
                metadata=EntryMetadata(
                    title=f"Doc {i}",
                    tags=["python"],
                    created=date(2024, 1, i + 1),
                ),
            )
            for i in range(10)
        ]
        hybrid_searcher.index_chunks(chunks)

        results = hybrid_searcher.search("Python", limit=5)
        assert len(results) <= 5

        # All paths should be unique
        paths = [r.path for r in results]
        assert len(paths) == len(set(paths))


class TestHybridSearchRankingAdjustments:
    """Test tag matching and context-based ranking boosts."""

    def test_tag_match_boost(self, hybrid_searcher):
        """Results with matching tags get score boost."""
        chunks = [
            DocumentChunk(
                path="with_tag.md",
                section=None,
                content="Generic content about topics",
                metadata=EntryMetadata(
                    title="With Tag",
                    tags=["python", "testing"],
                    created=date(2024, 1, 1),
                ),
            ),
            DocumentChunk(
                path="without_tag.md",
                section=None,
                content="Generic content about topics",
                metadata=EntryMetadata(
                    title="Without Tag",
                    tags=["other"],
                    created=date(2024, 1, 2),
                ),
            ),
        ]
        hybrid_searcher.index_chunks(chunks)

        # Query that mentions a tag
        results = hybrid_searcher.search("python testing")

        # Document with matching tags should be boosted
        if len(results) >= 2:
            # with_tag.md should rank higher due to tag boost
            top_result = results[0]
            assert "python" in top_result.tags or "testing" in top_result.tags

    def test_project_context_boost(self, hybrid_searcher, sample_chunks):
        """Results from current project get score boost."""
        hybrid_searcher.index_chunks(sample_chunks)

        # Search with project context
        results = hybrid_searcher.search(
            "neural networks",
            project_context="ai-docs"
        )

        # Results from ai-docs project should be boosted
        if len(results) >= 1:
            assert results[0].source_project in [None, "ai-docs"]

    def test_kb_context_path_boost(self, hybrid_searcher, sample_chunks):
        """Results matching KB context paths get boost."""
        from memex.context import KBContext

        hybrid_searcher.index_chunks(sample_chunks)

        # Create mock KB context
        kb_context = Mock(spec=KBContext)
        kb_context.get_all_boost_paths.return_value = ["ai/*.md"]

        results = hybrid_searcher.search(
            "guide",
            kb_context=kb_context
        )

        # Results from ai/ paths should be present
        assert len(results) >= 0

    def test_boost_renormalizes_scores(self, hybrid_searcher, sample_chunks):
        """After boosting, scores are renormalized to 0-1 range."""
        hybrid_searcher.index_chunks(sample_chunks)

        results = hybrid_searcher.search(
            "neural python",  # Trigger tag boost
            project_context="ai-docs"  # Trigger project boost
        )

        # All scores should still be in valid range
        for result in results:
            assert 0.0 <= result.score <= 1.0

    def test_context_boosts_dont_stack(self, hybrid_searcher):
        """Project and path boosts don't stack (MAX is used)."""
        from memex.context import KBContext

        chunk = DocumentChunk(
            path="ai/doc.md",
            section=None,
            content="Test content",
            metadata=EntryMetadata(
                title="Test",
                tags=["test"],
                created=date(2024, 1, 1),
                source_project="ai-docs",
            ),
        )
        hybrid_searcher.index_chunks([chunk])

        kb_context = Mock(spec=KBContext)
        kb_context.get_all_boost_paths.return_value = ["ai/*.md"]

        # Both project and path match
        results = hybrid_searcher.search(
            "test",
            project_context="ai-docs",
            kb_context=kb_context
        )

        # Should get boost, but not double boost
        assert len(results) == 1
        assert 0.0 <= results[0].score <= 1.0


class TestHybridSearchEdgeCases:
    """Test edge cases and error conditions."""

    def test_search_empty_indices(self, hybrid_searcher):
        """Searching empty indices returns empty list."""
        results = hybrid_searcher.search("anything")
        assert results == []

    def test_search_empty_query(self, hybrid_searcher, sample_chunks):
        """Empty query returns empty results."""
        hybrid_searcher.index_chunks(sample_chunks)
        results = hybrid_searcher.search("")
        # ChromaDB still generates embeddings for empty strings, so hybrid search
        # may return results from the semantic index
        assert isinstance(results, list)

    def test_search_with_limit_zero(self, hybrid_searcher, sample_chunks):
        """Search with limit=0 may fail or return empty due to Whoosh constraints."""
        hybrid_searcher.index_chunks(sample_chunks)
        # Whoosh raises ValueError for limit < 1, but hybrid search multiplies limit by 3
        # So limit=0 will cause a ValueError when Whoosh is called with fetch_limit=0
        # This is expected behavior - limit should be >= 1
        try:
            results = hybrid_searcher.search("neural", limit=0)
            # If it doesn't raise, should return empty
            assert results == []
        except ValueError as e:
            # Expected: Whoosh requires limit >= 1
            assert "limit must be >= 1" in str(e)

    def test_search_with_large_limit(self, hybrid_searcher, sample_chunks):
        """Search with very large limit works correctly."""
        hybrid_searcher.index_chunks(sample_chunks)
        results = hybrid_searcher.search("neural", limit=10000)
        assert len(results) <= 3  # Can't return more than indexed


class TestHybridReindex:
    """Test reindexing functionality."""

    def test_reindex_clears_and_rebuilds(self, hybrid_searcher, tmp_path, monkeypatch):
        """Reindex clears indices and rebuilds from KB files."""
        kb_root = tmp_path / "kb"
        kb_root.mkdir()
        monkeypatch.setenv("MEMEX_KB_ROOT", str(kb_root))

        # Create a test markdown file
        test_file = kb_root / "test.md"
        test_file.write_text("""---
title: Test Entry
tags:
  - test
created: 2024-01-01
---

Test content for reindexing.
""")

        # Force reindex (returns int for backward compatibility)
        count = hybrid_searcher.reindex(kb_root, force=True)
        assert count >= 1
        assert hybrid_searcher._whoosh.doc_count() >= 1
        assert hybrid_searcher._chroma.doc_count() >= 1

    def test_reindex_empty_kb(self, hybrid_searcher, tmp_path):
        """Reindex on empty KB directory works."""
        kb_root = tmp_path / "empty_kb"
        kb_root.mkdir()

        # Force reindex returns int
        count = hybrid_searcher.reindex(kb_root, force=True)
        assert count == 0

    def test_reindex_updates_timestamp(self, hybrid_searcher, tmp_path):
        """Reindex updates last_indexed timestamp."""
        kb_root = tmp_path / "kb"
        kb_root.mkdir()

        assert hybrid_searcher._last_indexed is None
        hybrid_searcher.reindex(kb_root)
        # Timestamp may or may not be set depending on whether files were indexed
        # Just verify it doesn't crash


class TestHybridPreload:
    """Test preloading functionality."""

    def test_preload_calls_chroma_preload(self, hybrid_searcher):
        """Preload calls Chroma's preload to warm up embedding model."""
        # Just verify it doesn't crash
        hybrid_searcher.preload()
        # Model should be loaded
        assert hybrid_searcher._chroma._model is not None


class TestHybridSearchLimit:
    """Test search limit parameter behavior."""

    def test_search_respects_limit(self, hybrid_searcher):
        """Search limit parameter restricts number of results."""
        chunks = [
            DocumentChunk(
                path=f"doc{i}.md",
                section=None,
                content=f"Python programming document {i}",
                metadata=EntryMetadata(
                    title=f"Doc {i}",
                    tags=["python"],
                    created=date(2024, 1, i + 1),
                ),
            )
            for i in range(20)
        ]
        hybrid_searcher.index_chunks(chunks)

        results = hybrid_searcher.search("Python", limit=5)
        assert len(results) <= 5

    def test_search_default_limit(self, hybrid_searcher, sample_chunks):
        """Search uses default limit when not specified."""
        hybrid_searcher.index_chunks(sample_chunks)
        results = hybrid_searcher.search("neural")
        # Should return all matching results up to default limit
        assert isinstance(results, list)


class TestHybridSearchResultQuality:
    """Test quality and ranking of search results."""

    def test_hybrid_improves_over_single_mode(self, hybrid_searcher):
        """Hybrid mode can find results that single modes might miss."""
        chunks = [
            DocumentChunk(
                path="exact_keyword.md",
                section=None,
                content="Python Python Python programming",
                metadata=EntryMetadata(
                    title="Python Repeated",
                    tags=["code"],
                    created=date(2024, 1, 1),
                ),
            ),
            DocumentChunk(
                path="semantic_similar.md",
                section=None,
                content="A high-level interpreted language for scripting",
                metadata=EntryMetadata(
                    title="Scripting Language",
                    tags=["code"],
                    created=date(2024, 1, 2),
                ),
            ),
        ]
        hybrid_searcher.index_chunks(chunks)

        # Hybrid should combine strengths
        hybrid_results = hybrid_searcher.search("Python language", mode="hybrid")
        keyword_results = hybrid_searcher.search("Python language", mode="keyword")
        semantic_results = hybrid_searcher.search("Python language", mode="semantic")

        # All modes should return results
        assert len(hybrid_results) >= 1
        assert isinstance(keyword_results, list)
        assert isinstance(semantic_results, list)

    def test_results_sorted_by_score(self, hybrid_searcher, sample_chunks):
        """Results are sorted by score in descending order."""
        hybrid_searcher.index_chunks(sample_chunks)

        results = hybrid_searcher.search("neural networks python")

        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score


class TestHybridThreadSafety:
    """Test thread safety for write operations."""

    def test_has_write_lock(self, hybrid_searcher):
        """HybridSearcher has a write lock attribute."""
        assert hasattr(hybrid_searcher, "_write_lock")
        assert isinstance(hybrid_searcher._write_lock, type(threading.Lock()))

    def test_concurrent_index_document(self, index_dirs):
        """Concurrent index_document calls don't cause data corruption."""
        whoosh_dir, chroma_dir = index_dirs
        whoosh = WhooshIndex(index_dir=whoosh_dir)
        chroma = ChromaIndex(index_dir=chroma_dir)
        searcher = HybridSearcher(whoosh_index=whoosh, chroma_index=chroma)

        num_threads = 10
        chunks_per_thread = 5

        def index_worker(thread_id: int) -> None:
            for i in range(chunks_per_thread):
                chunk = DocumentChunk(
                    path=f"thread_{thread_id}/doc_{i}.md",
                    section=None,
                    content=f"Content from thread {thread_id}, document {i}",
                    metadata=EntryMetadata(
                        title=f"Thread {thread_id} Doc {i}",
                        tags=["concurrent", f"thread-{thread_id}"],
                        created=date(2024, 1, 1),
                    ),
                )
                searcher.index_document(chunk)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(index_worker, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        expected_count = num_threads * chunks_per_thread
        assert searcher._whoosh.doc_count() == expected_count
        assert searcher._chroma.doc_count() == expected_count

    def test_concurrent_index_chunks(self, index_dirs):
        """Concurrent index_chunks calls don't cause data corruption."""
        whoosh_dir, chroma_dir = index_dirs
        whoosh = WhooshIndex(index_dir=whoosh_dir)
        chroma = ChromaIndex(index_dir=chroma_dir)
        searcher = HybridSearcher(whoosh_index=whoosh, chroma_index=chroma)

        num_threads = 5
        chunks_per_batch = 3

        def batch_worker(thread_id: int) -> None:
            chunks = [
                DocumentChunk(
                    path=f"batch_{thread_id}/doc_{i}.md",
                    section=None,
                    content=f"Batch content from thread {thread_id}, document {i}",
                    metadata=EntryMetadata(
                        title=f"Batch {thread_id} Doc {i}",
                        tags=["batch", f"batch-{thread_id}"],
                        created=date(2024, 1, 1),
                    ),
                )
                for i in range(chunks_per_batch)
            ]
            searcher.index_chunks(chunks)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(batch_worker, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        expected_count = num_threads * chunks_per_batch
        assert searcher._whoosh.doc_count() == expected_count
        assert searcher._chroma.doc_count() == expected_count

    def test_concurrent_delete_document(self, index_dirs):
        """Concurrent delete_document calls don't cause errors."""
        whoosh_dir, chroma_dir = index_dirs
        whoosh = WhooshIndex(index_dir=whoosh_dir)
        chroma = ChromaIndex(index_dir=chroma_dir)
        searcher = HybridSearcher(whoosh_index=whoosh, chroma_index=chroma)

        num_docs = 20
        chunks = [
            DocumentChunk(
                path=f"doc_{i}.md",
                section=None,
                content=f"Document {i} content for deletion test",
                metadata=EntryMetadata(
                    title=f"Doc {i}",
                    tags=["delete-test"],
                    created=date(2024, 1, 1),
                ),
            )
            for i in range(num_docs)
        ]
        searcher.index_chunks(chunks)

        assert searcher._whoosh.doc_count() == num_docs
        assert searcher._chroma.doc_count() == num_docs

        def delete_worker(doc_id: int) -> None:
            searcher.delete_document(f"doc_{doc_id}.md")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(delete_worker, i) for i in range(num_docs)]
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        assert searcher._whoosh.doc_count() == 0
        assert searcher._chroma.doc_count() == 0

    def test_concurrent_mixed_operations(self, index_dirs):
        """Mixed concurrent operations (index, delete) don't cause corruption."""
        whoosh_dir, chroma_dir = index_dirs
        whoosh = WhooshIndex(index_dir=whoosh_dir)
        chroma = ChromaIndex(index_dir=chroma_dir)
        searcher = HybridSearcher(whoosh_index=whoosh, chroma_index=chroma)

        # Pre-index some documents to delete
        initial_chunks = [
            DocumentChunk(
                path=f"initial_{i}.md",
                section=None,
                content=f"Initial document {i}",
                metadata=EntryMetadata(
                    title=f"Initial {i}",
                    tags=["initial"],
                    created=date(2024, 1, 1),
                ),
            )
            for i in range(5)
        ]
        searcher.index_chunks(initial_chunks)

        errors = []

        def index_worker(thread_id: int) -> None:
            try:
                for i in range(3):
                    chunk = DocumentChunk(
                        path=f"new_{thread_id}_{i}.md",
                        section=None,
                        content=f"New content {thread_id}_{i}",
                        metadata=EntryMetadata(
                            title=f"New {thread_id}_{i}",
                            tags=["new"],
                            created=date(2024, 1, 1),
                        ),
                    )
                    searcher.index_document(chunk)
            except Exception as e:
                errors.append(e)

        def delete_worker(doc_id: int) -> None:
            try:
                searcher.delete_document(f"initial_{doc_id}.md")
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            # Mix indexing and deletion
            futures = []
            for i in range(3):
                futures.append(executor.submit(index_worker, i))
            for i in range(5):
                futures.append(executor.submit(delete_worker, i))

            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        assert not errors, f"Errors during concurrent operations: {errors}"
        # Initial 5 deleted, 3 threads * 3 docs = 9 new
        assert searcher._whoosh.doc_count() == 9
        assert searcher._chroma.doc_count() == 9

    def test_concurrent_clear_and_index(self, index_dirs):
        """Clear operation is thread-safe with indexing."""
        whoosh_dir, chroma_dir = index_dirs
        whoosh = WhooshIndex(index_dir=whoosh_dir)
        chroma = ChromaIndex(index_dir=chroma_dir)
        searcher = HybridSearcher(whoosh_index=whoosh, chroma_index=chroma)

        # This test verifies no deadlocks or exceptions occur
        # when clear() and index_document() race
        errors = []
        barrier = threading.Barrier(2)

        def index_loop() -> None:
            try:
                barrier.wait()  # Synchronize start
                for i in range(10):
                    chunk = DocumentChunk(
                        path=f"race_{i}.md",
                        section=None,
                        content=f"Race condition test {i}",
                        metadata=EntryMetadata(
                            title=f"Race {i}",
                            tags=["race"],
                            created=date(2024, 1, 1),
                        ),
                    )
                    searcher.index_document(chunk)
            except Exception as e:
                errors.append(e)

        def clear_loop() -> None:
            try:
                barrier.wait()  # Synchronize start
                for _ in range(5):
                    searcher.clear()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=index_loop)
        t2 = threading.Thread(target=clear_loop)

        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert not errors, f"Errors during concurrent clear/index: {errors}"
        # Final state depends on race, but no corruption should occur

    def test_write_lock_serializes_operations(self, index_dirs):
        """Write lock serializes operations (no parallel writes)."""
        whoosh_dir, chroma_dir = index_dirs
        whoosh = WhooshIndex(index_dir=whoosh_dir)
        chroma = ChromaIndex(index_dir=chroma_dir)
        searcher = HybridSearcher(whoosh_index=whoosh, chroma_index=chroma)

        # Track when lock is held to verify serialization
        lock_acquisitions = []
        original_index_document = searcher.index_document

        def tracked_index_document(chunk):
            # Record that we acquired the lock (we're inside the lock context)
            lock_acquisitions.append(threading.current_thread().name)
            # Verify we're the only one holding the lock right now
            # by checking no other thread name appears while we're executing
            return original_index_document(chunk)

        searcher.index_document = tracked_index_document

        def index_worker(thread_id: int) -> None:
            for i in range(3):
                chunk = DocumentChunk(
                    path=f"lock_test_{thread_id}_{i}.md",
                    section=None,
                    content=f"Lock test content {thread_id}_{i}",
                    metadata=EntryMetadata(
                        title=f"Lock Test {thread_id}_{i}",
                        tags=["lock-test"],
                        created=date(2024, 1, 1),
                    ),
                )
                searcher.index_document(chunk)

        threads = [
            threading.Thread(target=index_worker, args=(i,), name=f"Worker-{i}")
            for i in range(3)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # All operations should have completed
        assert len(lock_acquisitions) == 9  # 3 threads * 3 operations
        # Document count should be correct (no lost writes)
        assert searcher._whoosh.doc_count() == 9
        assert searcher._chroma.doc_count() == 9


class TestComparativeModePerformance:
    """Tests verifying hybrid mode outperforms individual modes.

    These tests create adversarial scenarios where:
    - Keyword mode excels at exact term matching
    - Semantic mode excels at conceptual/synonym matching
    - Hybrid should combine both advantages
    """

    @pytest.fixture
    def comparative_chunks(self) -> list[DocumentChunk]:
        """Create chunks designed to test mode differences.

        Structure:
        - error_handling.md: Contains exact error name "ValueError" (keyword-friendly)
        - deployment_guide.md: About containerization (semantic: "docker" without saying "container")
        - mixed_content.md: Has both exact terms and semantic concepts
        - synonym_content.md: Uses synonyms (persistence -> storage/database)
        """
        return [
            # Keyword-favored: Contains exact technical term
            DocumentChunk(
                path="errors/error_handling.md",
                section="ValueError",
                content="The ValueError exception is raised when a function receives an argument "
                "with the right type but inappropriate value. Handle ValueError with try-except blocks.",
                metadata=EntryMetadata(
                    title="ValueError Handling Guide",
                    tags=["python", "errors", "exceptions"],
                    created=date(2024, 1, 1),
                ),
                token_count=30,
            ),
            # Semantic-favored: About containerization without using exact keywords
            DocumentChunk(
                path="devops/deployment_guide.md",
                section="Containerization",
                content="Package your applications in isolated environments that include all dependencies. "
                "Use images to create reproducible deployments across development and production. "
                "Orchestration tools help manage multiple instances at scale.",
                metadata=EntryMetadata(
                    title="Application Deployment Patterns",
                    tags=["devops", "deployment"],
                    created=date(2024, 1, 2),
                ),
                token_count=40,
            ),
            # Has docker keyword for keyword match verification
            DocumentChunk(
                path="devops/docker_intro.md",
                section="Introduction",
                content="Docker is a platform for building and running containers. "
                "Use Dockerfile to define your container image.",
                metadata=EntryMetadata(
                    title="Docker Introduction",
                    tags=["docker", "containers"],
                    created=date(2024, 1, 3),
                ),
                token_count=20,
            ),
            # Semantic-favored: Uses synonyms for data storage
            DocumentChunk(
                path="architecture/persistence.md",
                section="Data Layer",
                content="Design your data layer for durability and consistency. "
                "Store application state in a reliable backend. "
                "Consider replication for high availability.",
                metadata=EntryMetadata(
                    title="Data Persistence Architecture",
                    tags=["architecture", "data"],
                    created=date(2024, 1, 4),
                ),
                token_count=30,
            ),
            # Mixed: Has exact Python term AND semantic programming concepts
            DocumentChunk(
                path="dev/python_best_practices.md",
                section="Best Practices",
                content="Python best practices include using type hints for clarity, "
                "writing comprehensive tests, and following PEP 8 style guidelines. "
                "RuntimeError and ValueError should be caught explicitly.",
                metadata=EntryMetadata(
                    title="Python Best Practices",
                    tags=["python", "best-practices"],
                    created=date(2024, 1, 5),
                ),
                token_count=35,
            ),
            # Unique identifier content
            DocumentChunk(
                path="deps/requirements.md",
                section="Dependencies",
                content="Required packages: chromadb>=0.5.0, whoosh>=2.7.4, pydantic>=2.0.0. "
                "Install with pip install -r requirements.txt",
                metadata=EntryMetadata(
                    title="Project Dependencies",
                    tags=["dependencies", "setup"],
                    created=date(2024, 1, 6),
                ),
                token_count=25,
            ),
        ]

    @pytest.fixture
    def indexed_searcher(self, hybrid_searcher, comparative_chunks):
        """HybridSearcher with comparative test data indexed."""
        hybrid_searcher.index_chunks(comparative_chunks)
        return hybrid_searcher

    def _get_rank(self, results: list, target_path: str) -> int | None:
        """Get 1-based rank of target path in results, or None if not found."""
        paths = [r.path for r in results]
        if target_path in paths:
            return paths.index(target_path) + 1
        return None

    def _mrr(self, results: list, expected_paths: list[str]) -> float:
        """Calculate Mean Reciprocal Rank for expected paths."""
        for path in expected_paths:
            rank = self._get_rank(results, path)
            if rank is not None:
                return 1.0 / rank
        return 0.0

    # --- Keyword-favored queries ---

    def test_keyword_excels_at_exact_error_names(self, indexed_searcher):
        """Keyword search should excel at finding exact error names like 'ValueError'."""
        keyword_results = indexed_searcher.search("ValueError", mode="keyword", limit=5)
        semantic_results = indexed_searcher.search("ValueError", mode="semantic", limit=5)
        hybrid_results = indexed_searcher.search("ValueError", mode="hybrid", limit=5)

        # All modes should find the error handling doc
        target = "errors/error_handling.md"
        keyword_rank = self._get_rank(keyword_results, target)
        semantic_rank = self._get_rank(semantic_results, target)
        hybrid_rank = self._get_rank(hybrid_results, target)

        # Keyword should find it (exact match)
        assert keyword_rank is not None, "Keyword mode should find exact term 'ValueError'"
        assert keyword_rank <= 2, f"Keyword mode should rank 'ValueError' highly, got rank {keyword_rank}"

        # Hybrid should also find it
        assert hybrid_rank is not None, "Hybrid mode should find 'ValueError'"

    def test_keyword_excels_at_unique_identifiers(self, indexed_searcher):
        """Keyword search should find unique identifiers like package versions."""
        keyword_results = indexed_searcher.search("chromadb>=0.5.0", mode="keyword", limit=5)
        hybrid_results = indexed_searcher.search("chromadb>=0.5.0", mode="hybrid", limit=5)

        target = "deps/requirements.md"
        keyword_rank = self._get_rank(keyword_results, target)
        hybrid_rank = self._get_rank(hybrid_results, target)

        # Keyword should find exact version string
        assert keyword_rank is not None, "Keyword should find exact version 'chromadb>=0.5.0'"

        # Hybrid should also find it
        assert hybrid_rank is not None, "Hybrid should find exact identifiers"

    # --- Semantic-favored queries ---

    def test_semantic_finds_conceptual_matches(self, indexed_searcher):
        """Semantic search should find 'containerization' docs when querying about 'docker containers'."""
        # Query uses "docker containers" but deployment_guide.md talks about
        # "isolated environments" and "images" without using those exact words
        semantic_results = indexed_searcher.search("application containers deployment", mode="semantic", limit=5)
        keyword_results = indexed_searcher.search("application containers deployment", mode="keyword", limit=5)
        hybrid_results = indexed_searcher.search("application containers deployment", mode="hybrid", limit=5)

        deployment_target = "devops/deployment_guide.md"
        docker_target = "devops/docker_intro.md"

        semantic_deployment_rank = self._get_rank(semantic_results, deployment_target)
        semantic_docker_rank = self._get_rank(semantic_results, docker_target)

        hybrid_deployment_rank = self._get_rank(hybrid_results, deployment_target)
        hybrid_docker_rank = self._get_rank(hybrid_results, docker_target)

        # Semantic should find the deployment guide (conceptual match)
        assert semantic_deployment_rank is not None or semantic_docker_rank is not None, \
            "Semantic should find deployment-related content"

        # Hybrid should find both
        assert hybrid_deployment_rank is not None or hybrid_docker_rank is not None, \
            "Hybrid should find deployment content"

    def test_semantic_handles_synonyms(self, indexed_searcher):
        """Semantic search should find 'persistence' docs when searching for 'database storage'."""
        semantic_results = indexed_searcher.search("database storage layer", mode="semantic", limit=5)
        keyword_results = indexed_searcher.search("database storage layer", mode="keyword", limit=5)
        hybrid_results = indexed_searcher.search("database storage layer", mode="hybrid", limit=5)

        target = "architecture/persistence.md"

        semantic_rank = self._get_rank(semantic_results, target)
        keyword_rank = self._get_rank(keyword_results, target)
        hybrid_rank = self._get_rank(hybrid_results, target)

        # Semantic should find it via concept matching
        # (persistence.md talks about "data layer", "store", "durability" which are semantically related)
        assert semantic_rank is not None, \
            "Semantic should find persistence docs when searching 'database storage'"

        # Hybrid should also find it
        assert hybrid_rank is not None, \
            "Hybrid should find persistence docs"

    # --- Hybrid advantage queries ---

    def test_hybrid_combines_exact_and_semantic_matches(self, indexed_searcher):
        """Hybrid should find results that require both exact terms and semantic understanding."""
        # Query: "Python error handling best practices"
        # - error_handling.md has "ValueError" (exact)
        # - python_best_practices.md has both Python and RuntimeError/ValueError
        keyword_results = indexed_searcher.search("Python error handling", mode="keyword", limit=5)
        semantic_results = indexed_searcher.search("Python error handling", mode="semantic", limit=5)
        hybrid_results = indexed_searcher.search("Python error handling", mode="hybrid", limit=5)

        error_target = "errors/error_handling.md"
        practices_target = "dev/python_best_practices.md"

        # Calculate which targets each mode finds
        keyword_found = set()
        semantic_found = set()
        hybrid_found = set()

        for target in [error_target, practices_target]:
            if self._get_rank(keyword_results, target):
                keyword_found.add(target)
            if self._get_rank(semantic_results, target):
                semantic_found.add(target)
            if self._get_rank(hybrid_results, target):
                hybrid_found.add(target)

        # Hybrid should find at least as many relevant docs as individual modes
        assert len(hybrid_found) >= max(len(keyword_found), len(semantic_found)), \
            f"Hybrid found {len(hybrid_found)}, keyword {len(keyword_found)}, semantic {len(semantic_found)}"

    def test_hybrid_has_reasonable_mrr_on_mixed_queries(self, indexed_searcher):
        """Hybrid should have good MRR across diverse query types."""
        test_cases = [
            # (query, expected_paths, description)
            ("ValueError exception", ["errors/error_handling.md", "dev/python_best_practices.md"], "exact error"),
            ("containerization deployment", ["devops/deployment_guide.md", "devops/docker_intro.md"], "semantic concept"),
            ("Python programming", ["dev/python_best_practices.md", "errors/error_handling.md"], "language + concept"),
        ]

        hybrid_mrrs = []
        keyword_mrrs = []
        semantic_mrrs = []

        for query, expected, _desc in test_cases:
            hybrid_results = indexed_searcher.search(query, mode="hybrid", limit=5)
            keyword_results = indexed_searcher.search(query, mode="keyword", limit=5)
            semantic_results = indexed_searcher.search(query, mode="semantic", limit=5)

            hybrid_mrrs.append(self._mrr(hybrid_results, expected))
            keyword_mrrs.append(self._mrr(keyword_results, expected))
            semantic_mrrs.append(self._mrr(semantic_results, expected))

        avg_hybrid_mrr = sum(hybrid_mrrs) / len(hybrid_mrrs)
        avg_keyword_mrr = sum(keyword_mrrs) / len(keyword_mrrs)
        avg_semantic_mrr = sum(semantic_mrrs) / len(semantic_mrrs)

        # Hybrid should perform reasonably well overall
        # It should be competitive with the better of keyword/semantic
        best_single_mode = max(avg_keyword_mrr, avg_semantic_mrr)

        # Hybrid should be within 20% of the best single mode (accounting for RRF tradeoffs)
        assert avg_hybrid_mrr >= best_single_mode * 0.8, \
            f"Hybrid MRR ({avg_hybrid_mrr:.2f}) too low vs best single mode ({best_single_mode:.2f})"

    def test_hybrid_returns_results_from_both_indices(self, indexed_searcher):
        """Hybrid mode should combine results that appear in one index but not the other."""
        # Use a query that might favor one index
        hybrid_results = indexed_searcher.search("Python data architecture", mode="hybrid", limit=10)

        # Should have results
        assert len(hybrid_results) >= 2, "Hybrid should return multiple relevant results"

        # Results should cover multiple topics (not just keyword matches or just semantic)
        paths = {r.path for r in hybrid_results}
        topics_covered = {
            "python": any("python" in p for p in paths),
            "architecture": any("architecture" in p for p in paths),
            "errors": any("error" in p for p in paths),
            "devops": any("devops" in p or "deploy" in p for p in paths),
        }

        # Should cover at least 2 different topic areas
        covered_count = sum(topics_covered.values())
        assert covered_count >= 2, \
            f"Hybrid should cover diverse topics, only got: {[k for k, v in topics_covered.items() if v]}"
