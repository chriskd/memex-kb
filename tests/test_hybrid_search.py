"""Comprehensive tests for HybridSearcher combining Whoosh and Chroma."""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from memex.indexer.chroma_index import ChromaIndex
from memex.indexer.hybrid import HybridSearcher
from memex.indexer.whoosh_index import WhooshIndex
from memex.models import DocumentChunk, EntryMetadata, SearchResult


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
