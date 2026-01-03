"""Tests for token counting in search results."""

from datetime import date
from pathlib import Path

import pytest
import tiktoken

from memex.indexer.chroma_index import ChromaIndex
from memex.indexer.hybrid import HybridSearcher
from memex.indexer.whoosh_index import WhooshIndex
from memex.models import DocumentChunk, EntryMetadata
from memex.parser.markdown import _get_token_count, parse_entry


class TestTokenCountFunction:
    """Tests for the _get_token_count utility function."""

    def test_basic_token_count(self):
        """Verify token counting matches tiktoken directly."""
        enc = tiktoken.get_encoding("cl100k_base")

        test_cases = [
            "Hello world",
            "The quick brown fox jumps over the lazy dog.",
            "def foo(): return 42",
            "",  # Empty string
            "a" * 1000,  # Long string
        ]

        for text in test_cases:
            expected = len(enc.encode(text))
            actual = _get_token_count(text)
            assert actual == expected, f"Mismatch for {text!r}: expected {expected}, got {actual}"

    def test_encoder_caching(self):
        """Verify encoder is cached (doesn't raise on repeated calls)."""
        # Call multiple times to ensure caching works
        for _ in range(10):
            count = _get_token_count("test string")
            assert count == 2


class TestParserTokenCounts:
    """Tests for token counts in parsed document chunks."""

    @pytest.fixture
    def sample_md(self, tmp_path) -> Path:
        """Create a sample markdown file."""
        md_file = tmp_path / "sample.md"
        md_file.write_text("""---
title: Sample Document
tags:
  - test
created: 2024-01-01
---

This is the intro section with some content.

## First Section

Content of the first section.

## Second Section

Content of the second section with more text.
""")
        return md_file

    def test_chunks_have_token_counts(self, sample_md):
        """Verify all chunks have token_count set."""
        metadata, content, chunks = parse_entry(sample_md)

        assert len(chunks) == 3  # intro + 2 sections
        for chunk in chunks:
            assert chunk.token_count is not None
            assert chunk.token_count > 0

    def test_token_counts_are_accurate(self, sample_md):
        """Verify token counts match content length."""
        enc = tiktoken.get_encoding("cl100k_base")
        metadata, content, chunks = parse_entry(sample_md)

        for chunk in chunks:
            expected = len(enc.encode(chunk.content))
            assert chunk.token_count == expected


class TestWhooshTokenCounts:
    """Tests for token counts in Whoosh index."""

    @pytest.fixture
    def whoosh_index(self, tmp_path) -> WhooshIndex:
        """Create a temporary Whoosh index."""
        return WhooshIndex(index_dir=tmp_path / "whoosh")

    @pytest.fixture
    def sample_chunk(self) -> DocumentChunk:
        """Create a sample document chunk."""
        return DocumentChunk(
            path="test/doc.md",
            section="Overview",
            content="This is test content for indexing.",
            metadata=EntryMetadata(
                title="Test Doc",
                tags=["test"],
                created=date(2024, 1, 1),
            ),
            token_count=7,  # Pre-calculated
        )

    def test_index_stores_token_count(self, whoosh_index, sample_chunk):
        """Verify token count is stored and retrieved from index."""
        whoosh_index.index_document(sample_chunk)

        results = whoosh_index.search("test content", limit=1)

        assert len(results) == 1
        assert results[0].token_count == 7

    def test_batch_index_stores_token_counts(self, whoosh_index):
        """Verify batch indexing preserves token counts."""
        chunks = [
            DocumentChunk(
                path=f"test/doc{i}.md",
                section=None,
                content=f"Content {i}",
                metadata=EntryMetadata(
                    title=f"Doc {i}",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
                token_count=10 + i,
            )
            for i in range(3)
        ]

        whoosh_index.index_documents(chunks)
        results = whoosh_index.search("Content", limit=10)

        assert len(results) == 3
        token_counts = {r.token_count for r in results}
        assert token_counts == {10, 11, 12}


@pytest.mark.semantic
class TestChromaTokenCounts:
    """Tests for token counts in ChromaDB index."""

    @pytest.fixture
    def chroma_index(self, tmp_path) -> ChromaIndex:
        """Create a temporary Chroma index."""
        return ChromaIndex(index_dir=tmp_path / "chroma")

    @pytest.fixture
    def sample_chunk(self) -> DocumentChunk:
        """Create a sample document chunk."""
        return DocumentChunk(
            path="test/doc.md",
            section="Overview",
            content="This is test content for semantic search indexing.",
            metadata=EntryMetadata(
                title="Test Doc",
                tags=["test"],
                created=date(2024, 1, 1),
            ),
            token_count=9,
        )

    def test_index_stores_token_count(self, chroma_index, sample_chunk):
        """Verify token count is stored and retrieved from Chroma."""
        chroma_index.index_document(sample_chunk)

        results = chroma_index.search("semantic search", limit=1)

        assert len(results) == 1
        assert results[0].token_count == 9

    def test_batch_index_stores_token_counts(self, chroma_index):
        """Verify batch indexing preserves token counts."""
        chunks = [
            DocumentChunk(
                path=f"test/doc{i}.md",
                section=None,
                content=f"Unique searchable content number {i}",
                metadata=EntryMetadata(
                    title=f"Doc {i}",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
                token_count=20 + i,
            )
            for i in range(3)
        ]

        chroma_index.index_documents(chunks)
        results = chroma_index.search("searchable content", limit=10)

        assert len(results) == 3
        token_counts = {r.token_count for r in results}
        assert token_counts == {20, 21, 22}


@pytest.mark.semantic
class TestHybridSearchTokenCounts:
    """Tests for token counts through hybrid search."""

    @pytest.fixture
    def hybrid_searcher(self, tmp_path) -> HybridSearcher:
        """Create a hybrid searcher with temp indices."""
        whoosh = WhooshIndex(index_dir=tmp_path / "whoosh")
        chroma = ChromaIndex(index_dir=tmp_path / "chroma")
        return HybridSearcher(whoosh_index=whoosh, chroma_index=chroma)

    def test_hybrid_search_preserves_token_count(self, hybrid_searcher):
        """Verify token counts survive hybrid search RRF merge."""
        chunk = DocumentChunk(
            path="test/doc.md",
            section=None,
            content="Python programming guide for developers",
            metadata=EntryMetadata(
                title="Python Guide",
                tags=["python"],
                created=date(2024, 1, 1),
            ),
            token_count=42,
        )

        hybrid_searcher.index_document(chunk)
        results = hybrid_searcher.search("python programming", limit=1)

        assert len(results) == 1
        assert results[0].token_count == 42

    def test_deduplication_preserves_highest_scoring_token_count(self, hybrid_searcher):
        """When deduplicating by path, token_count from best chunk is kept."""
        chunks = [
            DocumentChunk(
                path="test/doc.md",
                section="Intro",
                content="Introduction to the topic",
                metadata=EntryMetadata(
                    title="Test Doc",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
                token_count=100,
            ),
            DocumentChunk(
                path="test/doc.md",
                section="Details",
                content="Detailed explanation with more specific keywords for search",
                metadata=EntryMetadata(
                    title="Test Doc",
                    tags=["test"],
                    created=date(2024, 1, 1),
                ),
                token_count=200,
            ),
        ]

        hybrid_searcher.index_chunks(chunks)

        # Search for something in the second chunk
        results = hybrid_searcher.search("specific keywords search", limit=1)

        assert len(results) == 1
        # Should get the higher-scoring chunk's token count
        assert results[0].token_count in (100, 200)
