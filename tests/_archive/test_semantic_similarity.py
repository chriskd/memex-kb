"""Tests verifying semantic search returns semantically similar content.

These tests validate that the embedding model and semantic search actually
understand meaning, not just match keywords. This is critical because semantic
search could silently fail (return random results) without such validation.
"""

from datetime import date

import numpy as np
import pytest

from memex.indexer.chroma_index import ChromaIndex
from memex.models import DocumentChunk, EntryMetadata

pytestmark = pytest.mark.semantic


@pytest.fixture
def chroma_index(tmp_path) -> ChromaIndex:
    """Create a fresh ChromaIndex for each test."""
    return ChromaIndex(index_dir=tmp_path / "chroma")


def _make_chunk(
    path: str,
    content: str,
    title: str = "Test",
    tags: list[str] | None = None,
) -> DocumentChunk:
    """Create a DocumentChunk with minimal boilerplate."""
    return DocumentChunk(
        path=path,
        section=None,
        content=content,
        metadata=EntryMetadata(
            title=title,
            tags=tags or ["test"],
            created=date(2024, 1, 1),
        ),
    )


class TestSynonymRetrieval:
    """Test that semantic search finds content via synonyms, not just keyword matches."""

    def test_ml_ai_synonym_match(self, chroma_index):
        """'AI models for categorization' should find 'Machine learning algorithms for classification'.

        This tests synonym understanding: ML <-> AI, classification <-> categorization.
        No words overlap between query and content.
        """
        # Index document about ML classification
        ml_doc = _make_chunk(
            path="ml/classification.md",
            content="Machine learning algorithms for classification",
            title="ML Classification",
            tags=["machine-learning"],
        )
        chroma_index.index_document(ml_doc)

        # Query using synonyms (no keyword overlap)
        results = chroma_index.search("AI models for categorization", limit=5)

        # Should find the ML doc despite zero keyword overlap
        assert len(results) >= 1, "Should find at least one result for synonym query"
        assert results[0].path == "ml/classification.md", (
            f"Top result should be ML classification doc, got {results[0].path}"
        )
        # Score should indicate reasonable semantic similarity
        assert results[0].score > 0.3, (
            f"Semantic similarity score should be > 0.3, got {results[0].score:.3f}"
        )

    def test_automobile_car_synonym(self, chroma_index):
        """'automobile repair' should find 'car maintenance' content."""
        car_doc = _make_chunk(
            path="vehicles/maintenance.md",
            content="Guide to car maintenance and fixing vehicle problems",
            title="Car Maintenance",
            tags=["automotive"],
        )
        chroma_index.index_document(car_doc)

        results = chroma_index.search("automobile repair guide", limit=5)

        assert len(results) >= 1
        assert results[0].path == "vehicles/maintenance.md"
        assert results[0].score > 0.3

    def test_purchase_buy_synonym(self, chroma_index):
        """'purchase process' should find 'buying guide' content."""
        buying_doc = _make_chunk(
            path="commerce/buying.md",
            content="Step by step buying guide for customers shopping online",
            title="Buying Guide",
        )
        chroma_index.index_document(buying_doc)

        results = chroma_index.search("customer purchase process", limit=5)

        assert len(results) >= 1
        assert results[0].path == "commerce/buying.md"


class TestConceptualSimilarity:
    """Test that semantic search understands conceptual relationships."""

    def test_docker_containerization_concept(self, chroma_index):
        """'packaging apps in isolated environments' should find Docker documentation.

        This tests conceptual understanding: Docker IS containerization,
        containerization IS about isolated environments and packaging.
        """
        docker_doc = _make_chunk(
            path="devops/docker.md",
            content="How to containerize Python applications with Docker",
            title="Docker Guide",
            tags=["docker", "containers"],
        )
        chroma_index.index_document(docker_doc)

        # Query using conceptual description (no mention of Docker/container)
        results = chroma_index.search("packaging apps in isolated environments", limit=5)

        assert len(results) >= 1, "Should find Docker doc via conceptual query"
        assert results[0].path == "devops/docker.md", (
            f"Expected Docker doc, got {results[0].path}"
        )
        assert results[0].score > 0.25, (
            f"Conceptual match score should be > 0.25, got {results[0].score:.3f}"
        )

    def test_api_endpoint_concept(self, chroma_index):
        """'web service interface' should find 'REST API endpoints' documentation."""
        api_doc = _make_chunk(
            path="dev/api.md",
            content="Designing REST API endpoints for your backend service",
            title="REST API Design",
        )
        chroma_index.index_document(api_doc)

        results = chroma_index.search("web service interface design patterns", limit=5)

        assert len(results) >= 1
        assert results[0].path == "dev/api.md"

    def test_database_persistence_concept(self, chroma_index):
        """'storing data permanently' should find 'database' documentation."""
        db_doc = _make_chunk(
            path="backend/database.md",
            content="Using PostgreSQL databases for persistent data storage",
            title="Database Guide",
        )
        chroma_index.index_document(db_doc)

        results = chroma_index.search("storing data permanently in application", limit=5)

        assert len(results) >= 1
        assert results[0].path == "backend/database.md"


class TestNegativeSemanticMatch:
    """Test that unrelated content does NOT rank highly."""

    def test_unrelated_content_low_score(self, chroma_index):
        """Chocolate cake recipe should NOT rank highly for kubernetes query.

        This validates that semantic search doesn't just return random results.
        """
        # Index obviously unrelated content
        cake_doc = _make_chunk(
            path="recipes/cake.md",
            content="Chocolate cake recipe with frosting. Mix flour, sugar, cocoa powder and eggs.",
            title="Chocolate Cake",
            tags=["recipes", "desserts"],
        )
        chroma_index.index_document(cake_doc)

        # Query for completely unrelated technical topic
        results = chroma_index.search("kubernetes deployment strategies", limit=5)

        # Should still return results (Chroma always returns something)
        # but score should be very low
        if results:
            assert results[0].score < 0.5, (
                f"Unrelated content should have low score (<0.5), got {results[0].score:.3f}"
            )

    def test_gardening_vs_programming(self, chroma_index):
        """Gardening content should score low for programming queries."""
        garden_doc = _make_chunk(
            path="hobbies/gardening.md",
            content="How to plant tomatoes and care for your vegetable garden",
            title="Vegetable Gardening",
            tags=["gardening"],
        )
        chroma_index.index_document(garden_doc)

        results = chroma_index.search("Python async await programming patterns", limit=5)

        if results:
            assert results[0].score < 0.5, (
                f"Gardening should not match programming, score: {results[0].score:.3f}"
            )

    def test_related_content_beats_unrelated(self, chroma_index):
        """Related content should score higher than unrelated content."""
        # Index both related and unrelated content
        chunks = [
            _make_chunk(
                path="cooking/pasta.md",
                content="Italian pasta recipe with tomato sauce and fresh basil",
                title="Pasta Recipe",
            ),
            _make_chunk(
                path="dev/python.md",
                content="Python programming best practices and coding standards",
                title="Python Guide",
            ),
            _make_chunk(
                path="finance/budget.md",
                content="Personal budget planning and expense tracking methods",
                title="Budget Guide",
            ),
        ]
        chroma_index.index_documents(chunks)

        # Query for programming
        results = chroma_index.search("software development coding", limit=5)

        assert len(results) >= 2

        # Python doc should be ranked higher than cooking/finance
        python_result = next((r for r in results if "python" in r.path), None)
        cooking_result = next((r for r in results if "pasta" in r.path), None)
        budget_result = next((r for r in results if "budget" in r.path), None)

        assert python_result is not None, "Should find Python doc"
        if cooking_result:
            assert python_result.score > cooking_result.score, (
                f"Python ({python_result.score:.3f}) should rank higher than cooking ({cooking_result.score:.3f})"
            )
        if budget_result:
            assert python_result.score > budget_result.score, (
                f"Python ({python_result.score:.3f}) should rank higher than budget ({budget_result.score:.3f})"
            )


class TestEmbeddingVectorSanity:
    """Test that embeddings capture semantic meaning via vector arithmetic.

    The classic word2vec test: king - man + woman ≈ queen
    Modern sentence transformers may not produce perfect results,
    but should show directional correctness.
    """

    def test_king_queen_analogy(self, chroma_index):
        """Verify embed(king) - embed(man) + embed(woman) is closer to embed(queen) than random.

        This is the classic word2vec analogy test. While sentence transformers
        are optimized for sentences (not words), they should still show
        directional correctness on this well-known analogy.
        """
        # Get embeddings for the analogy words
        king_emb = np.array(chroma_index._embed(["king"])[0])
        man_emb = np.array(chroma_index._embed(["man"])[0])
        woman_emb = np.array(chroma_index._embed(["woman"])[0])
        queen_emb = np.array(chroma_index._embed(["queen"])[0])

        # Compute analogy vector: king - man + woman
        analogy_vector = king_emb - man_emb + woman_emb

        # Compute cosine similarity to queen
        def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        queen_similarity = cosine_similarity(analogy_vector, queen_emb)

        # Get similarity to an unrelated word for comparison
        random_emb = np.array(chroma_index._embed(["refrigerator"])[0])
        random_similarity = cosine_similarity(analogy_vector, random_emb)

        # The analogy vector should be more similar to queen than to random word
        assert queen_similarity > random_similarity, (
            f"Analogy vector should be closer to 'queen' ({queen_similarity:.3f}) "
            f"than to 'refrigerator' ({random_similarity:.3f})"
        )

        # Queen similarity should be reasonably high (not random)
        assert queen_similarity > 0.3, (
            f"Queen similarity ({queen_similarity:.3f}) should be > 0.3"
        )

    def test_similar_words_have_similar_embeddings(self, chroma_index):
        """Words with similar meanings should have similar embeddings."""
        # Get embeddings for similar words
        happy_emb = np.array(chroma_index._embed(["happy"])[0])
        joyful_emb = np.array(chroma_index._embed(["joyful"])[0])
        sad_emb = np.array(chroma_index._embed(["sad"])[0])

        def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        happy_joyful_sim = cosine_similarity(happy_emb, joyful_emb)
        happy_sad_sim = cosine_similarity(happy_emb, sad_emb)

        # Happy should be more similar to joyful than to sad
        assert happy_joyful_sim > happy_sad_sim, (
            f"'happy' should be more similar to 'joyful' ({happy_joyful_sim:.3f}) "
            f"than to 'sad' ({happy_sad_sim:.3f})"
        )

    def test_sentence_semantic_similarity(self, chroma_index):
        """Semantically similar sentences should have similar embeddings."""
        sent1 = "The cat sat on the mat"
        sent2 = "A feline rested on the rug"
        sent3 = "Stock prices dropped sharply today"

        emb1 = np.array(chroma_index._embed([sent1])[0])
        emb2 = np.array(chroma_index._embed([sent2])[0])
        emb3 = np.array(chroma_index._embed([sent3])[0])

        def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        similar_sim = cosine_similarity(emb1, emb2)
        unrelated_sim = cosine_similarity(emb1, emb3)

        # Cat sentences should be more similar to each other than to stock news
        assert similar_sim > unrelated_sim, (
            f"Similar sentences ({similar_sim:.3f}) should have higher similarity "
            f"than unrelated sentences ({unrelated_sim:.3f})"
        )
        # Similar sentences should have high similarity
        assert similar_sim > 0.5, (
            f"Similar sentences should have similarity > 0.5, got {similar_sim:.3f}"
        )


class TestSemanticSearchQuality:
    """End-to-end tests for semantic search quality with multiple documents."""

    def test_semantic_ranking_order(self, chroma_index):
        """Documents should be ranked by semantic relevance to query."""
        chunks = [
            _make_chunk(
                path="highly_relevant.md",
                content="Deep learning neural networks for image recognition and computer vision",
                title="Deep Learning CV",
            ),
            _make_chunk(
                path="somewhat_relevant.md",
                content="Introduction to artificial intelligence and its applications",
                title="AI Intro",
            ),
            _make_chunk(
                path="barely_relevant.md",
                content="General technology trends and digital transformation",
                title="Tech Trends",
            ),
            _make_chunk(
                path="not_relevant.md",
                content="Cooking recipes and kitchen organization tips",
                title="Kitchen Guide",
            ),
        ]
        chroma_index.index_documents(chunks)

        results = chroma_index.search("convolutional neural networks for images", limit=10)

        # Get ranks of each document
        paths = [r.path for r in results]
        assert "highly_relevant.md" in paths, "Should find highly relevant doc"
        assert "not_relevant.md" in paths, "Kitchen guide should still appear (Chroma returns all)"

        highly_idx = paths.index("highly_relevant.md")
        not_rel_idx = paths.index("not_relevant.md")

        assert highly_idx < not_rel_idx, (
            f"Highly relevant doc (rank {highly_idx + 1}) should rank before "
            f"irrelevant doc (rank {not_rel_idx + 1})"
        )

    def test_finds_paraphrased_content(self, chroma_index):
        """Should find content that says the same thing differently."""
        original = _make_chunk(
            path="auth/security.md",
            content="Implement user authentication with JSON Web Tokens for secure API access",
            title="JWT Auth",
        )
        chroma_index.index_document(original)

        # Query with completely different phrasing
        results = chroma_index.search(
            "protect your web service endpoints using cryptographic tokens",
            limit=5,
        )

        assert len(results) >= 1
        assert results[0].path == "auth/security.md"
        assert results[0].score > 0.3

    def test_multilingual_concept_matching(self, chroma_index):
        """Should match concepts across technical terminology."""
        doc = _make_chunk(
            path="patterns/mvc.md",
            content="Model-View-Controller architecture pattern separates business logic from presentation",
            title="MVC Pattern",
        )
        chroma_index.index_document(doc)

        # Query using alternative terminology
        results = chroma_index.search(
            "software design pattern separating data and display layers",
            limit=5,
        )

        assert len(results) >= 1
        assert results[0].path == "patterns/mvc.md"


class TestEdgeCasesAndBoundaries:
    """Test edge cases, boundary conditions, and error scenarios."""

    def test_empty_index_returns_empty_results(self, chroma_index):
        """Searching an empty index should return empty results, not error."""
        results = chroma_index.search("any query at all", limit=10)

        assert results == [], "Empty index should return empty list"

    def test_empty_query_string(self, chroma_index):
        """Empty query should not crash, should return empty or all results."""
        doc = _make_chunk(
            path="test.md",
            content="Some test content",
            title="Test",
        )
        chroma_index.index_document(doc)

        # Empty string query
        results = chroma_index.search("", limit=10)

        # Should not crash - either returns empty or returns all docs
        assert isinstance(results, list)

    def test_very_long_query(self, chroma_index):
        """Very long queries should work without crashing."""
        doc = _make_chunk(
            path="ai/nlp.md",
            content="Natural language processing and machine learning",
            title="NLP",
        )
        chroma_index.index_document(doc)

        # Create a very long query
        long_query = " ".join(
            [
                "natural language processing machine learning",
                "artificial intelligence deep learning neural networks",
                "transformer models attention mechanisms contextual embeddings",
                "semantic understanding word vectors sentence representations",
            ]
            * 10  # Repeat to make it very long
        )

        results = chroma_index.search(long_query, limit=5)

        # Should find the NLP doc despite query length
        assert len(results) >= 1
        assert results[0].path == "ai/nlp.md"

    def test_very_short_query_vs_long_document(self, chroma_index):
        """Short queries should still match long documents."""
        long_content = (
            "This is a comprehensive guide to web development. "
            "It covers HTML, CSS, JavaScript, React, Node.js, "
            "databases, authentication, deployment, and best practices. " * 50
        )
        doc = _make_chunk(
            path="web/comprehensive.md",
            content=long_content,
            title="Web Development Guide",
        )
        chroma_index.index_document(doc)

        # Very short query
        results = chroma_index.search("React", limit=5)

        assert len(results) >= 1
        assert results[0].path == "web/comprehensive.md"

    def test_special_characters_in_content(self, chroma_index):
        """Content with special characters should be searchable."""
        doc = _make_chunk(
            path="code/regex.md",
            content="Regular expressions: ^[a-zA-Z0-9]+$ matches alphanumeric strings. Use \\d+ for digits.",
            title="Regex Guide",
        )
        chroma_index.index_document(doc)

        results = chroma_index.search("pattern matching alphanumeric", limit=5)

        assert len(results) >= 1
        assert results[0].path == "code/regex.md"

    def test_unicode_content_and_query(self, chroma_index):
        """Unicode content should be searchable with Unicode queries."""
        doc = _make_chunk(
            path="languages/unicode.md",
            content="机器学习和人工智能 (Machine learning and artificial intelligence) 日本語 Español",
            title="Unicode Test",
        )
        chroma_index.index_document(doc)

        # Search with English term that appears in the content
        results = chroma_index.search("artificial intelligence", limit=5)

        assert len(results) >= 1
        assert results[0].path == "languages/unicode.md"

    def test_limit_parameter_respected(self, chroma_index):
        """Search should respect the limit parameter."""
        # Index many documents
        chunks = [
            _make_chunk(
                path=f"doc{i}.md",
                content=f"Python programming tutorial number {i}",
                title=f"Tutorial {i}",
            )
            for i in range(20)
        ]
        chroma_index.index_documents(chunks)

        # Request only 5 results
        results = chroma_index.search("Python programming", limit=5)

        assert len(results) == 5, f"Should return exactly 5 results, got {len(results)}"

    def test_search_with_limit_one(self, chroma_index):
        """Limit=1 should return only the best match."""
        chunks = [
            _make_chunk(
                path="best.md",
                content="Python is a high-level programming language",
                title="Python",
            ),
            _make_chunk(
                path="okay.md",
                content="Programming languages are important",
                title="Languages",
            ),
        ]
        chroma_index.index_documents(chunks)

        results = chroma_index.search("Python programming language", limit=1)

        assert len(results) == 1
        assert results[0].path == "best.md"

    def test_duplicate_content_different_paths(self, chroma_index):
        """Duplicate content at different paths should both be findable."""
        chunks = [
            _make_chunk(
                path="copy1/guide.md",
                content="Complete guide to Docker containerization",
                title="Docker Guide",
            ),
            _make_chunk(
                path="copy2/guide.md",
                content="Complete guide to Docker containerization",
                title="Docker Guide Copy",
            ),
        ]
        chroma_index.index_documents(chunks)

        results = chroma_index.search("Docker containerization", limit=5)

        # Both should be found
        assert len(results) >= 2
        paths = {r.path for r in results}
        assert "copy1/guide.md" in paths
        assert "copy2/guide.md" in paths
        # Scores should be nearly identical for identical content
        scores = [r.score for r in results[:2]]
        assert abs(scores[0] - scores[1]) < 0.01


class TestSemanticRobustness:
    """Test semantic search handles challenging scenarios robustly."""

    def test_query_with_typos_still_matches(self, chroma_index):
        """Minor typos in query should still find relevant content.

        This tests the robustness of embeddings to minor spelling variations.
        """
        doc = _make_chunk(
            path="db/postgresql.md",
            content="PostgreSQL is a powerful relational database system",
            title="PostgreSQL",
        )
        chroma_index.index_document(doc)

        # Query with minor typo
        results = chroma_index.search("powerfull relational databse", limit=5)

        assert len(results) >= 1
        # Should still find it despite typos
        assert results[0].path == "db/postgresql.md"

    def test_semantic_search_with_acronyms(self, chroma_index):
        """Acronyms should match their full forms semantically."""
        doc = _make_chunk(
            path="arch/soa.md",
            content="Service-Oriented Architecture enables modular system design",
            title="SOA",
        )
        chroma_index.index_document(doc)

        # Query with acronym
        results = chroma_index.search("SOA modular systems", limit=5)

        assert len(results) >= 1
        assert results[0].path == "arch/soa.md"

    def test_question_query_matches_statement_content(self, chroma_index):
        """Questions should match relevant declarative content."""
        doc = _make_chunk(
            path="faq/testing.md",
            content="Unit tests verify individual functions work correctly in isolation",
            title="Testing FAQ",
        )
        chroma_index.index_document(doc)

        # Query as a question
        results = chroma_index.search("How do unit tests work?", limit=5)

        assert len(results) >= 1
        assert results[0].path == "faq/testing.md"

    def test_negation_handling(self, chroma_index):
        """Test that semantic search understands context including negations.

        Note: This is a challenging case for embeddings.
        """
        chunks = [
            _make_chunk(
                path="security/good.md",
                content="Always use HTTPS for secure communication and encrypt sensitive data",
                title="Security Best Practices",
            ),
            _make_chunk(
                path="security/bad.md",
                content="Never use HTTP for sensitive data transmission without encryption",
                title="Security Anti-patterns",
            ),
        ]
        chroma_index.index_documents(chunks)

        # Query for best practices
        results = chroma_index.search("how to secure data transmission", limit=5)

        assert len(results) >= 2
        # Both should be relevant, but we can't easily test that "good" ranks higher
        # since both documents are about the same security topic

    def test_multi_word_technical_terms(self, chroma_index):
        """Multi-word technical terms should be understood as units."""
        doc = _make_chunk(
            path="arch/microservices.md",
            content="Event-driven microservices architecture with asynchronous message passing",
            title="Microservices",
        )
        chroma_index.index_document(doc)

        results = chroma_index.search("event driven architecture", limit=5)

        assert len(results) >= 1
        assert results[0].path == "arch/microservices.md"


class TestIndexingBehavior:
    """Test document indexing behavior and optimizations."""

    def test_reindexing_same_content_is_idempotent(self, chroma_index):
        """Re-indexing unchanged content should not affect search results."""
        doc = _make_chunk(
            path="test.md",
            content="Test content for idempotency check",
            title="Test",
        )

        # Index the same document twice
        chroma_index.index_document(doc)
        chroma_index.index_document(doc)

        results = chroma_index.search("idempotency check", limit=5)

        # Should appear only once, not duplicated
        assert len(results) == 1
        assert results[0].path == "test.md"

    def test_updating_document_content(self, chroma_index):
        """Updating document content should reflect in search results."""
        # Index initial version
        doc_v1 = _make_chunk(
            path="evolving.md",
            content="This document is about cooking recipes and baking",
            title="Evolving Doc",
        )
        chroma_index.index_document(doc_v1)

        # Search finds cooking content
        results_v1 = chroma_index.search("programming", limit=5)
        cooking_score_v1 = results_v1[0].score if results_v1 else 0.0

        # Update to programming content
        doc_v2 = _make_chunk(
            path="evolving.md",
            content="This document is about Python programming and software development",
            title="Evolving Doc",
        )
        chroma_index.index_document(doc_v2)

        # Now programming query should match much better
        results_v2 = chroma_index.search("programming", limit=5)

        assert len(results_v2) >= 1
        assert results_v2[0].path == "evolving.md"
        # Score should be higher after update
        assert results_v2[0].score > cooking_score_v1

    def test_batch_indexing_multiple_documents(self, chroma_index):
        """Batch indexing should index all documents correctly."""
        chunks = [
            _make_chunk(
                path=f"batch/doc{i}.md",
                content=f"Document {i} about topic {i % 3}",
                title=f"Doc {i}",
            )
            for i in range(10)
        ]

        chroma_index.index_documents(chunks)

        # Verify all documents are indexed
        assert chroma_index.doc_count() == 10

        # Search should find relevant docs
        results = chroma_index.search("topic 1", limit=5)
        assert len(results) > 0
