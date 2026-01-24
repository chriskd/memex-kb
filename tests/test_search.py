"""Comprehensive tests for search functionality in memex indexer.

Tests cover:
1. Search modes: keyword (Whoosh), semantic (Chroma), hybrid (combined)
2. Filters: tags, category, date-based (if applicable)
3. Edge cases: empty query, no results, special characters, long queries
4. Result quality: score validation, sorting, limit parameter

Design philosophy:
- Test behaviors, not implementations
- Mark semantic tests with @pytest.mark.semantic
- Use parametrize for search mode variants
- Target: <5 seconds for non-semantic tests
"""

from datetime import datetime

import pytest

from memex.indexer.hybrid import HybridSearcher
from memex.indexer.whoosh_index import WhooshIndex
from memex.models import DocumentChunk, EntryMetadata


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def whoosh_index(tmp_path) -> WhooshIndex:
    """Create a fresh WhooshIndex for each test."""
    return WhooshIndex(index_dir=tmp_path / "whoosh")


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
                created=datetime(2024, 1, 1),
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
                created=datetime(2024, 1, 2),
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
                created=datetime(2024, 1, 3),
                source_project="dev-docs",
            ),
            token_count=10,
        ),
        DocumentChunk(
            path="dev/testing.md",
            section="unit-tests",
            content="Unit testing is essential for software quality assurance.",
            metadata=EntryMetadata(
                title="Testing Guide",
                tags=["testing", "quality"],
                created=datetime(2024, 1, 4),
            ),
            token_count=8,
        ),
        DocumentChunk(
            path="ops/docker.md",
            section=None,
            content="Docker containers provide isolated runtime environments.",
            metadata=EntryMetadata(
                title="Docker Guide",
                tags=["docker", "containers", "devops"],
                created=datetime(2024, 1, 5),
            ),
            token_count=7,
        ),
    ]


def _make_chunk(
    path: str,
    content: str,
    title: str = "Test",
    tags: list[str] | None = None,
    source_project: str | None = None,
) -> DocumentChunk:
    """Create a DocumentChunk with minimal boilerplate."""
    return DocumentChunk(
        path=path,
        section=None,
        content=content,
        metadata=EntryMetadata(
            title=title,
            tags=tags or ["test"],
            created=datetime(2024, 1, 1),
            source_project=source_project,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Whoosh Keyword Search Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestWhooshKeywordSearch:
    """Test Whoosh-based keyword/BM25 search functionality."""

    def test_keyword_search_finds_exact_match(self, whoosh_index, sample_chunks):
        """Keyword search finds documents with exact term matches."""
        whoosh_index.index_documents(sample_chunks)

        results = whoosh_index.search("Python")
        assert len(results) >= 1
        assert any("python" in r.path.lower() for r in results)

    def test_keyword_search_by_title(self, whoosh_index, sample_chunks):
        """Search matches document titles."""
        whoosh_index.index_documents(sample_chunks)

        results = whoosh_index.search("Neural Networks Basics")
        assert len(results) >= 1
        assert results[0].title == "Neural Networks Basics"

    def test_keyword_search_by_content(self, whoosh_index, sample_chunks):
        """Search matches document content."""
        whoosh_index.index_documents(sample_chunks)

        results = whoosh_index.search("biological neurons brain")
        assert len(results) >= 1
        assert "neural_nets" in results[0].path

    def test_keyword_search_by_tags(self, whoosh_index, sample_chunks):
        """Search matches document tags."""
        whoosh_index.index_documents(sample_chunks)

        results = whoosh_index.search("deep-learning")
        assert len(results) >= 1
        assert "deep-learning" in results[0].tags

    def test_keyword_search_multi_field(self, whoosh_index):
        """Search finds matches across title, content, and tags."""
        chunks = [
            _make_chunk("title_match.md", "Generic content", "Machine Learning Guide"),
            _make_chunk("content_match.md", "This discusses machine learning", "Other"),
            _make_chunk("tag_match.md", "Generic content", "Ref", tags=["machine-learning"]),
        ]
        whoosh_index.index_documents(chunks)

        results = whoosh_index.search("machine learning")
        assert len(results) >= 2  # Should find multiple matches

    def test_keyword_search_returns_normalized_scores(self, whoosh_index, sample_chunks):
        """Search results have scores normalized to 0-1 range."""
        whoosh_index.index_documents(sample_chunks)

        results = whoosh_index.search("neural networks")
        assert len(results) >= 1
        for result in results:
            assert 0.0 <= result.score <= 1.0

    def test_keyword_search_sorted_by_relevance(self, whoosh_index):
        """Results are sorted by score in descending order."""
        chunks = [
            _make_chunk("low.md", "Generic document with other topics"),
            _make_chunk("high.md", "Python Python Python programming language"),
            _make_chunk("medium.md", "Some Python code examples"),
        ]
        whoosh_index.index_documents(chunks)

        results = whoosh_index.search("Python programming")
        assert len(results) >= 2
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score


class TestWhooshEdgeCases:
    """Test Whoosh edge cases and special inputs."""

    def test_empty_query_returns_empty(self, whoosh_index, sample_chunks):
        """Empty query returns empty results."""
        whoosh_index.index_documents(sample_chunks)
        results = whoosh_index.search("")
        assert results == []

    def test_no_results_for_nonexistent_term(self, whoosh_index, sample_chunks):
        """Search for nonexistent term returns empty list."""
        whoosh_index.index_documents(sample_chunks)
        results = whoosh_index.search("xyznonexistent123")
        assert results == []

    def test_special_characters_handled(self, whoosh_index):
        """Search handles special characters gracefully."""
        chunk = _make_chunk(
            "code.md",
            "Code example: def function(x, y): return x + y",
            "Code Examples",
        )
        whoosh_index.index_document(chunk)

        # Special chars should not crash
        results = whoosh_index.search("function(x, y)")
        assert isinstance(results, list)

    def test_very_long_query_handled(self, whoosh_index, sample_chunks):
        """Very long queries are handled without error."""
        whoosh_index.index_documents(sample_chunks)
        long_query = "neural networks " * 50
        results = whoosh_index.search(long_query)
        assert isinstance(results, list)

    def test_invalid_syntax_fallback(self, whoosh_index, sample_chunks):
        """Invalid query syntax falls back gracefully."""
        whoosh_index.index_documents(sample_chunks)
        problematic_queries = ["AND OR NOT", "(((", "****", "]]]]"]
        for query in problematic_queries:
            results = whoosh_index.search(query)
            assert isinstance(results, list)

    def test_unicode_content_searchable(self, whoosh_index):
        """Unicode content can be indexed and searched."""
        chunk = _make_chunk(
            "unicode.md",
            "Unicode test: cafe naive resume",
            "Unicode Test",
        )
        whoosh_index.index_document(chunk)
        results = whoosh_index.search("cafe")
        assert isinstance(results, list)

    def test_search_empty_index_returns_empty(self, whoosh_index):
        """Searching empty index returns empty list."""
        results = whoosh_index.search("anything")
        assert results == []


class TestWhooshLimitParameter:
    """Test the limit parameter behavior."""

    def test_limit_restricts_results(self, whoosh_index):
        """Limit parameter restricts number of results."""
        chunks = [
            _make_chunk(f"doc{i}.md", f"Python programming document {i}")
            for i in range(20)
        ]
        whoosh_index.index_documents(chunks)

        results = whoosh_index.search("Python", limit=5)
        assert len(results) <= 5

    def test_limit_one_returns_single_result(self, whoosh_index, sample_chunks):
        """Limit=1 returns at most one result."""
        whoosh_index.index_documents(sample_chunks)
        results = whoosh_index.search("neural", limit=1)
        assert len(results) <= 1

    def test_limit_larger_than_results(self, whoosh_index, sample_chunks):
        """Limit larger than available results returns all results."""
        whoosh_index.index_documents(sample_chunks)
        results = whoosh_index.search("neural", limit=1000)
        assert len(results) <= len(sample_chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Search Tests (require chromadb/sentence-transformers)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.semantic
class TestSemanticSearch:
    """Test ChromaDB-based semantic search functionality."""

    @pytest.fixture
    def chroma_index(self, tmp_path):
        """Create a fresh ChromaIndex for each test."""
        from memex.indexer.chroma_index import ChromaIndex
        return ChromaIndex(index_dir=tmp_path / "chroma")

    def test_semantic_finds_conceptually_similar(self, chroma_index, sample_chunks):
        """Semantic search finds conceptually related content."""
        chroma_index.index_documents(sample_chunks)

        # Search for related concept without exact keyword match
        results = chroma_index.search("brain-inspired computing", limit=5)
        assert len(results) >= 1
        # Neural networks doc should rank high (conceptually related)
        assert any("neural" in r.path for r in results[:2])

    def test_semantic_finds_synonyms(self, chroma_index):
        """Semantic search finds content via synonyms."""
        chunk = _make_chunk(
            "ml/classification.md",
            "Machine learning algorithms for classification",
            "ML Classification",
            tags=["machine-learning"],
        )
        chroma_index.index_document(chunk)

        # Query using synonyms (no keyword overlap)
        results = chroma_index.search("AI models for categorization", limit=5)
        assert len(results) >= 1
        assert results[0].path == "ml/classification.md"

    def test_semantic_scores_in_valid_range(self, chroma_index, sample_chunks):
        """Semantic search scores are in 0-1 range."""
        chroma_index.index_documents(sample_chunks)

        results = chroma_index.search("programming language", limit=10)
        for result in results:
            assert 0.0 <= result.score <= 1.0

    def test_semantic_unrelated_content_low_score(self, chroma_index):
        """Unrelated content has low semantic similarity score."""
        chunk = _make_chunk(
            "recipes/cake.md",
            "Chocolate cake recipe with frosting. Mix flour, sugar, and eggs.",
            "Chocolate Cake",
            tags=["recipes"],
        )
        chroma_index.index_document(chunk)

        results = chroma_index.search("kubernetes deployment", limit=5)
        if results:
            assert results[0].score < 0.5

    def test_semantic_empty_index_returns_empty(self, chroma_index):
        """Searching empty semantic index returns empty list."""
        results = chroma_index.search("anything", limit=10)
        assert results == []

    def test_semantic_respects_limit(self, chroma_index):
        """Semantic search respects limit parameter."""
        chunks = [
            _make_chunk(f"doc{i}.md", f"Python programming tutorial {i}")
            for i in range(20)
        ]
        chroma_index.index_documents(chunks)

        results = chroma_index.search("Python programming", limit=5)
        assert len(results) == 5

    def test_build_embedding_text_includes_keywords_and_tags(self, chroma_index):
        """Embedding text builder includes content, keywords, and tags."""
        # Test the internal _build_embedding_text method
        result = chroma_index._build_embedding_text(
            content="Main document content",
            keywords=["concept1", "concept2"],
            tags=["tag1", "tag2"],
        )
        assert "Main document content" in result
        assert "Keywords: concept1, concept2" in result
        assert "Tags: tag1, tag2" in result

    def test_build_embedding_text_empty_keywords(self, chroma_index):
        """Embedding text handles empty keywords gracefully."""
        result = chroma_index._build_embedding_text(
            content="Content only",
            keywords=[],
            tags=["tag1"],
        )
        assert "Content only" in result
        assert "Keywords:" not in result
        assert "Tags: tag1" in result

    def test_build_embedding_text_empty_tags(self, chroma_index):
        """Embedding text handles empty tags gracefully."""
        result = chroma_index._build_embedding_text(
            content="Content only",
            keywords=["keyword1"],
            tags=[],
        )
        assert "Content only" in result
        assert "Keywords: keyword1" in result
        assert "Tags:" not in result

    def test_semantic_finds_by_keywords(self, chroma_index):
        """Semantic search finds content via keywords in metadata."""
        # Document content doesn't mention "neural networks" but keywords do
        chunk = _make_chunk(
            "ml/classifier.md",
            "A system for categorizing images into predefined classes.",
            "Image Classifier",
            tags=["ml"],
        )
        # Add keywords to metadata
        chunk.metadata.keywords = ["neural networks", "deep learning", "CNN"]
        chroma_index.index_document(chunk)

        # Query using keyword concept (neural networks)
        results = chroma_index.search("neural networks", limit=5)
        assert len(results) >= 1
        assert results[0].path == "ml/classifier.md"

    def test_semantic_finds_by_tags(self, chroma_index):
        """Semantic search finds content via tags included in embedding."""
        chunk = _make_chunk(
            "devops/ci.md",
            "Automate build and test pipelines for software projects.",
            "CI Pipeline Guide",
            tags=["continuous-integration", "devops", "automation"],
        )
        chroma_index.index_document(chunk)

        # Query using tag concept
        results = chroma_index.search("continuous integration", limit=5)
        assert len(results) >= 1
        assert results[0].path == "devops/ci.md"

    def test_keywords_improve_search_quality(self, chroma_index):
        """Keywords improve semantic search by adding semantic context."""
        # Two documents about similar topics
        chunk_with_keywords = _make_chunk(
            "concepts/attention.md",
            "A mechanism for weighting different parts of input data.",
            "Attention Mechanisms",
            tags=["ml"],
        )
        chunk_with_keywords.metadata.keywords = [
            "transformer",
            "self-attention",
            "NLP",
        ]

        chunk_without_keywords = _make_chunk(
            "concepts/weighting.md",
            "A mechanism for weighting different parts of input data.",
            "Weighting Scheme",
            tags=["ml"],
        )
        # No keywords

        chroma_index.index_documents([chunk_with_keywords, chunk_without_keywords])

        # Search for transformer-related concept
        results = chroma_index.search("transformer attention models", limit=5)
        assert len(results) >= 1
        # Document with relevant keywords should rank higher
        assert results[0].path == "concepts/attention.md"


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Search Tests (require chromadb/sentence-transformers)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.semantic
class TestHybridSearchModes:
    """Test HybridSearcher with different search modes."""

    @pytest.fixture
    def hybrid_searcher(self, tmp_path) -> HybridSearcher:
        """Create HybridSearcher with separate test indices."""
        from memex.indexer.chroma_index import ChromaIndex

        whoosh = WhooshIndex(index_dir=tmp_path / "whoosh")
        chroma = ChromaIndex(index_dir=tmp_path / "chroma")
        return HybridSearcher(whoosh_index=whoosh, chroma_index=chroma)

    def test_keyword_mode_uses_whoosh(self, hybrid_searcher, sample_chunks):
        """Keyword mode uses only Whoosh index."""
        hybrid_searcher.index_chunks(sample_chunks)

        results = hybrid_searcher.search("neural networks", mode="keyword")
        assert len(results) >= 1
        assert any("neural" in r.snippet.lower() for r in results)

    def test_semantic_mode_uses_chroma(self, hybrid_searcher, sample_chunks):
        """Semantic mode uses only Chroma index."""
        hybrid_searcher.index_chunks(sample_chunks)

        results = hybrid_searcher.search("brain-inspired computing", mode="semantic")
        assert len(results) >= 1

    def test_hybrid_mode_combines_both(self, hybrid_searcher, sample_chunks):
        """Hybrid mode combines results from both indices."""
        hybrid_searcher.index_chunks(sample_chunks)

        results = hybrid_searcher.search("neural networks", mode="hybrid")
        assert len(results) >= 1

    def test_default_mode_is_hybrid(self, hybrid_searcher, sample_chunks):
        """Default search mode is hybrid."""
        hybrid_searcher.index_chunks(sample_chunks)

        results = hybrid_searcher.search("neural")
        assert len(results) >= 1


@pytest.mark.semantic
class TestHybridSearchRRF:
    """Test Reciprocal Rank Fusion algorithm in hybrid search."""

    @pytest.fixture
    def hybrid_searcher(self, tmp_path) -> HybridSearcher:
        from memex.indexer.chroma_index import ChromaIndex

        whoosh = WhooshIndex(index_dir=tmp_path / "whoosh")
        chroma = ChromaIndex(index_dir=tmp_path / "chroma")
        return HybridSearcher(whoosh_index=whoosh, chroma_index=chroma)

    def test_rrf_normalizes_scores(self, hybrid_searcher, sample_chunks):
        """RRF produces normalized scores (0-1 range)."""
        hybrid_searcher.index_chunks(sample_chunks)

        results = hybrid_searcher.search("neural networks", mode="hybrid")
        assert len(results) >= 1
        for result in results:
            assert 0.0 <= result.score <= 1.0

    def test_rrf_deduplicates_results(self, hybrid_searcher):
        """RRF deduplicates results appearing in both indices."""
        chunk = _make_chunk(
            "duplicate.md",
            "Machine learning algorithms for data analysis",
            "ML Algorithms",
            tags=["machine-learning"],
        )
        hybrid_searcher.index_chunks([chunk])

        results = hybrid_searcher.search("machine learning", mode="hybrid")
        paths = [r.path for r in results]
        assert paths.count("duplicate.md") <= 1

    def test_hybrid_handles_empty_indices(self, hybrid_searcher):
        """Hybrid search handles empty indices gracefully."""
        results = hybrid_searcher.search("anything")
        assert results == []


@pytest.mark.semantic
class TestHybridDeduplication:
    """Test deduplication of search results by path."""

    @pytest.fixture
    def hybrid_searcher(self, tmp_path) -> HybridSearcher:
        from memex.indexer.chroma_index import ChromaIndex

        whoosh = WhooshIndex(index_dir=tmp_path / "whoosh")
        chroma = ChromaIndex(index_dir=tmp_path / "chroma")
        return HybridSearcher(whoosh_index=whoosh, chroma_index=chroma)

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
                    created=datetime(2024, 1, 1),
                ),
            ),
            DocumentChunk(
                path="multi/doc.md",
                section="main",
                content="Main section with Python programming tutorial",
                metadata=EntryMetadata(
                    title="Multi-section Doc",
                    tags=["test"],
                    created=datetime(2024, 1, 1),
                ),
            ),
        ]
        hybrid_searcher.index_chunks(chunks)

        results = hybrid_searcher.search("Python programming")
        paths = [r.path for r in results]
        assert paths.count("multi/doc.md") == 1

    def test_deduplicate_respects_limit(self, hybrid_searcher):
        """Deduplication respects limit parameter."""
        chunks = [
            _make_chunk(f"doc{i}.md", f"Document {i} about Python programming")
            for i in range(15)
        ]
        hybrid_searcher.index_chunks(chunks)

        results = hybrid_searcher.search("Python", limit=5)
        assert len(results) <= 5
        paths = [r.path for r in results]
        assert len(paths) == len(set(paths))  # All unique


# ─────────────────────────────────────────────────────────────────────────────
# Parametrized Mode Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["keyword"])
def test_search_respects_limit_keyword(tmp_path, sample_chunks, mode):
    """Search limit parameter works for keyword mode."""
    whoosh = WhooshIndex(index_dir=tmp_path / "whoosh")
    whoosh.index_documents(sample_chunks)

    results = whoosh.search("neural", limit=2)
    assert len(results) <= 2


@pytest.mark.semantic
@pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
def test_search_respects_limit_all_modes(tmp_path, sample_chunks, mode):
    """Search limit parameter works for all modes."""
    from memex.indexer.chroma_index import ChromaIndex

    whoosh = WhooshIndex(index_dir=tmp_path / "whoosh")
    chroma = ChromaIndex(index_dir=tmp_path / "chroma")
    searcher = HybridSearcher(whoosh_index=whoosh, chroma_index=chroma)
    searcher.index_chunks(sample_chunks)

    results = searcher.search("neural", limit=2, mode=mode)
    assert len(results) <= 2


@pytest.mark.semantic
@pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
def test_empty_query_all_modes(tmp_path, sample_chunks, mode):
    """Empty query handled correctly in all modes."""
    from memex.indexer.chroma_index import ChromaIndex

    whoosh = WhooshIndex(index_dir=tmp_path / "whoosh")
    chroma = ChromaIndex(index_dir=tmp_path / "chroma")
    searcher = HybridSearcher(whoosh_index=whoosh, chroma_index=chroma)
    searcher.index_chunks(sample_chunks)

    results = searcher.search("", mode=mode)
    assert isinstance(results, list)


@pytest.mark.semantic
@pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
def test_scores_normalized_all_modes(tmp_path, sample_chunks, mode):
    """Scores are normalized to 0-1 in all modes."""
    from memex.indexer.chroma_index import ChromaIndex

    whoosh = WhooshIndex(index_dir=tmp_path / "whoosh")
    chroma = ChromaIndex(index_dir=tmp_path / "chroma")
    searcher = HybridSearcher(whoosh_index=whoosh, chroma_index=chroma)
    searcher.index_chunks(sample_chunks)

    results = searcher.search("programming", mode=mode)
    for result in results:
        assert 0.0 <= result.score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Ranking and Boost Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.semantic
class TestRankingAdjustments:
    """Test tag matching and context-based ranking boosts."""

    @pytest.fixture
    def hybrid_searcher(self, tmp_path) -> HybridSearcher:
        from memex.indexer.chroma_index import ChromaIndex

        whoosh = WhooshIndex(index_dir=tmp_path / "whoosh")
        chroma = ChromaIndex(index_dir=tmp_path / "chroma")
        return HybridSearcher(whoosh_index=whoosh, chroma_index=chroma)

    def test_tag_match_boost(self, hybrid_searcher):
        """Results with matching tags get score boost."""
        chunks = [
            _make_chunk(
                "with_tag.md",
                "Generic content about topics",
                "With Tag",
                tags=["python", "testing"],
            ),
            _make_chunk(
                "without_tag.md",
                "Generic content about topics",
                "Without Tag",
                tags=["other"],
            ),
        ]
        hybrid_searcher.index_chunks(chunks)

        results = hybrid_searcher.search("python testing")
        if len(results) >= 2:
            top_result = results[0]
            assert "python" in top_result.tags or "testing" in top_result.tags

    def test_project_context_boost(self, hybrid_searcher, sample_chunks):
        """Results from current project get score boost."""
        hybrid_searcher.index_chunks(sample_chunks)

        results = hybrid_searcher.search("neural networks", project_context="ai-docs")
        if len(results) >= 1:
            assert results[0].source_project in [None, "ai-docs"]

    def test_boost_renormalizes_scores(self, hybrid_searcher, sample_chunks):
        """After boosting, scores are renormalized to 0-1 range."""
        hybrid_searcher.index_chunks(sample_chunks)

        results = hybrid_searcher.search(
            "neural python",
            project_context="ai-docs",
        )
        for result in results:
            assert 0.0 <= result.score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Result Quality Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestResultQuality:
    """Test result quality and metadata preservation."""

    def test_results_contain_path(self, whoosh_index, sample_chunks):
        """Search results contain document path."""
        whoosh_index.index_documents(sample_chunks)
        results = whoosh_index.search("Python")
        assert all(result.path for result in results)

    def test_results_contain_title(self, whoosh_index, sample_chunks):
        """Search results contain document title."""
        whoosh_index.index_documents(sample_chunks)
        results = whoosh_index.search("Python")
        assert all(result.title for result in results)

    def test_results_contain_snippet(self, whoosh_index, sample_chunks):
        """Search results contain content snippet."""
        whoosh_index.index_documents(sample_chunks)
        results = whoosh_index.search("Python")
        assert all(result.snippet for result in results)

    def test_results_contain_tags(self, whoosh_index, sample_chunks):
        """Search results contain document tags."""
        whoosh_index.index_documents(sample_chunks)
        results = whoosh_index.search("Python")
        # At least some results should have tags
        assert any(result.tags for result in results)

    def test_snippet_is_truncated(self, whoosh_index):
        """Long content is truncated to snippet."""
        chunk = _make_chunk(
            "long.md",
            "This is a very long document. " * 100,
            "Long Doc",
        )
        whoosh_index.index_document(chunk)

        results = whoosh_index.search("document")
        assert len(results) >= 1
        assert len(results[0].snippet) <= 210  # Max length + "..."


# ─────────────────────────────────────────────────────────────────────────────
# Index Operations Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestIndexOperations:
    """Test index management operations."""

    def test_doc_count_after_indexing(self, whoosh_index, sample_chunks):
        """Document count reflects indexed documents."""
        assert whoosh_index.doc_count() == 0
        whoosh_index.index_documents(sample_chunks)
        assert whoosh_index.doc_count() == len(sample_chunks)

    def test_clear_removes_all_documents(self, whoosh_index, sample_chunks):
        """Clear removes all documents from index."""
        whoosh_index.index_documents(sample_chunks)
        assert whoosh_index.doc_count() > 0

        whoosh_index.clear()
        assert whoosh_index.doc_count() == 0

    def test_delete_document_by_path(self, whoosh_index, sample_chunks):
        """Can delete specific document by path."""
        whoosh_index.index_documents(sample_chunks)
        initial_count = whoosh_index.doc_count()

        whoosh_index.delete_document("ai/neural_nets.md")
        assert whoosh_index.doc_count() == initial_count - 1

        # Deleted document should not appear in search
        results = whoosh_index.search("Neural Networks Basics")
        assert not any(r.path == "ai/neural_nets.md" for r in results)

    def test_update_existing_document(self, whoosh_index):
        """Updating a document replaces the old version."""
        chunk1 = _make_chunk("test.md", "Original content", "Original Title")
        whoosh_index.index_document(chunk1)
        assert whoosh_index.doc_count() == 1

        chunk2 = _make_chunk("test.md", "Updated content", "Updated Title")
        whoosh_index.index_document(chunk2)
        assert whoosh_index.doc_count() == 1

        results = whoosh_index.search("Updated")
        assert len(results) == 1
        assert results[0].title == "Updated Title"


@pytest.mark.semantic
class TestHybridIndexOperations:
    """Test hybrid searcher index operations."""

    @pytest.fixture
    def hybrid_searcher(self, tmp_path) -> HybridSearcher:
        from memex.indexer.chroma_index import ChromaIndex

        whoosh = WhooshIndex(index_dir=tmp_path / "whoosh")
        chroma = ChromaIndex(index_dir=tmp_path / "chroma")
        return HybridSearcher(whoosh_index=whoosh, chroma_index=chroma)

    def test_index_document_to_both_indices(self, hybrid_searcher):
        """Indexing a document adds it to both Whoosh and Chroma."""
        chunk = _make_chunk("test.md", "Test content", "Test Doc")
        hybrid_searcher.index_document(chunk)

        assert hybrid_searcher._whoosh.doc_count() == 1
        assert hybrid_searcher._chroma.doc_count() == 1

    def test_delete_from_both_indices(self, hybrid_searcher, sample_chunks):
        """Deleting a document removes it from both indices."""
        hybrid_searcher.index_chunks(sample_chunks)

        hybrid_searcher.delete_document("ai/neural_nets.md")

        assert hybrid_searcher._whoosh.doc_count() == len(sample_chunks) - 1
        assert hybrid_searcher._chroma.doc_count() == len(sample_chunks) - 1

    def test_clear_both_indices(self, hybrid_searcher, sample_chunks):
        """Clear removes all documents from both indices."""
        hybrid_searcher.index_chunks(sample_chunks)
        assert hybrid_searcher._whoosh.doc_count() > 0
        assert hybrid_searcher._chroma.doc_count() > 0

        hybrid_searcher.clear()

        assert hybrid_searcher._whoosh.doc_count() == 0
        assert hybrid_searcher._chroma.doc_count() == 0

    def test_status_reports_counts(self, hybrid_searcher, sample_chunks, tmp_path, monkeypatch):
        """Status reports document counts."""
        kb_root = tmp_path / "kb"
        kb_root.mkdir()
        monkeypatch.setenv("MEMEX_USER_KB_ROOT", str(kb_root))

        hybrid_searcher.index_chunks(sample_chunks)
        status = hybrid_searcher.status()

        assert status.whoosh_docs == len(sample_chunks)
        assert status.chroma_docs == len(sample_chunks)
