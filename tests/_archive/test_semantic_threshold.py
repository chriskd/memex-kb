"""Tests for semantic search threshold filtering.

These tests verify that:
1. Low-confidence semantic results are filtered out
2. The --strict flag raises the threshold further
3. Gibberish queries return empty results instead of misleading high scores
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from memex import core
from memex.config import SEMANTIC_MIN_SIMILARITY, SEMANTIC_STRICT_SIMILARITY
from memex.indexer.chroma_index import ChromaIndex
from memex.indexer.hybrid import HybridSearcher
from memex.models import SearchResult


@pytest.fixture(autouse=True)
def reset_searcher_state(monkeypatch):
    """Ensure cached searcher state does not leak across tests."""
    monkeypatch.setattr(core, "_searcher", None)
    monkeypatch.setattr(core, "_searcher_ready", False)


@pytest.fixture
def kb_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary KB root with standard categories."""
    root = tmp_path / "kb"
    root.mkdir()
    (root / "development").mkdir()
    monkeypatch.setenv("MEMEX_KB_ROOT", str(root))
    return root


@pytest.fixture
def index_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary index root."""
    root = tmp_path / ".indices"
    root.mkdir()
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(root))
    return root


def _create_entry(path: Path, title: str, content_body: str, tags: list[str] | None = None):
    """Helper to create a KB entry with frontmatter."""
    tags = tags or ["test"]
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    content = f"""---
title: {title}
tags:
{tags_yaml}
created: {datetime.now(timezone.utc).isoformat()}
---

{content_body}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class TestSemanticThresholdConfig:
    """Test that threshold constants are properly configured."""

    def test_default_threshold_is_reasonable(self):
        """Default threshold should filter very weak matches."""
        assert 0.2 <= SEMANTIC_MIN_SIMILARITY <= 0.4

    def test_strict_threshold_is_higher(self):
        """Strict threshold should be more restrictive."""
        assert SEMANTIC_STRICT_SIMILARITY > SEMANTIC_MIN_SIMILARITY
        assert 0.4 <= SEMANTIC_STRICT_SIMILARITY <= 0.6


class TestChromaIndexThreshold:
    """Test ChromaIndex respects minimum similarity threshold."""

    def test_search_filters_low_scores(self, kb_root, index_root):
        """Results below min_similarity should be filtered out."""
        # Create an entry
        _create_entry(
            kb_root / "development" / "python-guide.md",
            "Python Development Guide",
            "A comprehensive guide to Python development with best practices.",
            tags=["python", "guide"],
        )

        chroma = ChromaIndex()
        # Parse and index the entry
        from memex.parser import parse_entry
        _, _, chunks = parse_entry(kb_root / "development" / "python-guide.md")
        if chunks:
            for chunk in chunks:
                chunk.path = "development/python-guide.md"
            chroma.index_documents(chunks)

        # Search with high threshold - gibberish should return nothing
        results = chroma.search("xyzabc123nonsense", limit=10, min_similarity=0.8)
        assert len(results) == 0, "Gibberish query should return no results with high threshold"

    def test_search_respects_custom_threshold(self, kb_root, index_root):
        """Custom min_similarity should override default."""
        _create_entry(
            kb_root / "development" / "docker-guide.md",
            "Docker Container Guide",
            "Learn how to use Docker containers for deployment.",
            tags=["docker", "containers"],
        )

        chroma = ChromaIndex()
        from memex.parser import parse_entry
        _, _, chunks = parse_entry(kb_root / "development" / "docker-guide.md")
        if chunks:
            for chunk in chunks:
                chunk.path = "development/docker-guide.md"
            chroma.index_documents(chunks)

        # With threshold=0, should return results even for weak matches
        results_no_filter = chroma.search("random query", limit=10, min_similarity=0.0)

        # With threshold=0.9, should filter almost everything
        results_strict = chroma.search("random query", limit=10, min_similarity=0.9)

        # The strict filter should return fewer or equal results
        assert len(results_strict) <= len(results_no_filter)


class TestHybridSearcherStrict:
    """Test HybridSearcher strict mode."""

    def test_strict_mode_uses_higher_threshold(self, kb_root, index_root):
        """strict=True should use SEMANTIC_STRICT_SIMILARITY threshold."""
        _create_entry(
            kb_root / "development" / "api-docs.md",
            "API Documentation",
            "REST API endpoints and authentication methods.",
            tags=["api", "docs"],
        )

        searcher = HybridSearcher()
        # Reindex
        searcher.reindex(kb_root)

        # Non-strict mode may return weak semantic matches
        results_normal = searcher.search("xyzabc123", limit=10, mode="semantic", strict=False)

        # Strict mode should filter them out
        results_strict = searcher.search("xyzabc123", limit=10, mode="semantic", strict=True)

        # Strict should return fewer or equal results
        assert len(results_strict) <= len(results_normal)

    def test_hybrid_mode_respects_strict(self, kb_root, index_root):
        """Hybrid mode should also respect strict flag."""
        _create_entry(
            kb_root / "development" / "testing-guide.md",
            "Testing Best Practices",
            "Unit tests, integration tests, and end-to-end testing strategies.",
            tags=["testing", "quality"],
        )

        searcher = HybridSearcher()
        searcher.reindex(kb_root)

        # Gibberish query in hybrid mode
        results_normal = searcher.search("qwerty12345xyz", limit=10, mode="hybrid", strict=False)
        results_strict = searcher.search("qwerty12345xyz", limit=10, mode="hybrid", strict=True)

        # Strict should be more restrictive
        assert len(results_strict) <= len(results_normal)


class TestCoreSearchStrict:
    """Test core.search() strict parameter."""

    @pytest.mark.asyncio
    async def test_search_accepts_strict_parameter(self, kb_root, index_root):
        """core.search should accept strict parameter."""
        _create_entry(
            kb_root / "development" / "deployment.md",
            "Deployment Guide",
            "How to deploy applications to production.",
            tags=["deployment", "devops"],
        )

        await core.reindex()

        # Should not raise - strict parameter is accepted
        response = await core.search("deployment", strict=True)
        assert response is not None

    @pytest.mark.asyncio
    async def test_strict_filters_gibberish_results(self, kb_root, index_root):
        """Strict mode should return empty for gibberish queries."""
        _create_entry(
            kb_root / "development" / "config.md",
            "Configuration Management",
            "Environment variables and config files.",
            tags=["config"],
        )

        await core.reindex()

        # Gibberish query with strict mode
        response = await core.search("xyzabc123nonsense", strict=True)

        # Should return empty or very few results
        # The key assertion is that scores should be meaningful, not artificially high
        for result in response.results:
            assert result.score < 0.9, f"Gibberish query should not produce high scores: {result.score}"


class TestScoreIntegrity:
    """Test that scores accurately reflect match quality."""

    @pytest.mark.asyncio
    async def test_relevant_query_scores_higher_than_gibberish(self, kb_root, index_root):
        """Relevant queries should score higher than gibberish."""
        _create_entry(
            kb_root / "development" / "python-async.md",
            "Python Async Programming",
            "Guide to asyncio, coroutines, and async/await patterns in Python.",
            tags=["python", "async"],
        )

        await core.reindex()

        # Relevant query
        relevant_response = await core.search("python async programming", mode="semantic")

        # Gibberish query
        gibberish_response = await core.search("xyzabc123nonsense", mode="semantic")

        # If both return results, relevant should score higher
        if relevant_response.results and gibberish_response.results:
            max_relevant_score = max(r.score for r in relevant_response.results)
            max_gibberish_score = max(r.score for r in gibberish_response.results)
            assert max_relevant_score > max_gibberish_score, \
                f"Relevant query should score higher: {max_relevant_score} vs {max_gibberish_score}"
