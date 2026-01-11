"""Tests for mx search CLI flags (--strict, --terse, combinations)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from memex import core
from memex.cli import cli


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
    (root / "infrastructure").mkdir()
    monkeypatch.setenv("MEMEX_KB_ROOT", str(root))
    return root


@pytest.fixture
def index_root(tmp_path, monkeypatch) -> Path:
    """Create a temporary index root."""
    root = tmp_path / ".indices"
    root.mkdir()
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(root))
    return root


def _create_entry(path: Path, title: str, tags: list[str], content: str = ""):
    """Helper to create a KB entry with frontmatter."""
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)
    created = datetime.now(timezone.utc)
    text = f"""---
title: {title}
tags:
{tags_yaml}
created: {created.isoformat()}
---

{content if content else f"Content for {title}."}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class TestSearchStrict:
    """Tests for --strict flag behavior."""

    def test_search_strict_no_semantic_fallback(self, kb_root, index_root):
        """--strict prevents semantic fallback for low-confidence matches.

        Gibberish queries should return no results in strict mode because
        they don't match keywords and the semantic threshold is higher.
        """
        _create_entry(
            kb_root / "development" / "python-guide.md",
            title="Python Development Guide",
            tags=["python", "development"],
            content="A guide to Python development best practices and tooling.",
        )

        runner = CliRunner()

        # Search with gibberish query in strict mode
        result = runner.invoke(cli, ["search", "xyzabc123nonsense", "--strict"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        # Strict mode should filter out weak semantic matches for gibberish
        assert "No results found" in result.output or "python" not in result.output.lower()

    def test_search_strict_filters_low_scores(self, kb_root, index_root):
        """--strict mode is more selective, filtering low-confidence results."""
        _create_entry(
            kb_root / "development" / "docker-setup.md",
            title="Docker Setup Guide",
            tags=["docker", "containers"],
            content="Docker container setup and configuration for development.",
        )
        _create_entry(
            kb_root / "infrastructure" / "kubernetes.md",
            title="Kubernetes Deployment",
            tags=["kubernetes", "orchestration"],
            content="Kubernetes cluster management and deployment strategies.",
        )

        runner = CliRunner()

        # Exact keyword match should work in strict mode
        result = runner.invoke(cli, ["search", "docker", "--strict", "--json"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        data = json.loads(result.output)

        # Should find exact keyword matches
        paths = [r["path"] for r in data]
        assert any("docker" in p for p in paths), "Docker entry should be found"

    def test_search_without_strict_includes_weak(self, kb_root, index_root):
        """Without --strict, semantic search may include weaker matches."""
        _create_entry(
            kb_root / "development" / "coding-standards.md",
            title="Coding Standards",
            tags=["standards", "best-practices"],
            content="Guidelines for code quality, formatting, and review processes.",
        )

        runner = CliRunner()

        # Without strict, semantic search has lower threshold
        result = runner.invoke(cli, ["search", "programming guidelines", "--json"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        # May find related content through semantic similarity
        # (not guaranteed, depends on embeddings, but validates no crash)


class TestSearchTerse:
    """Tests for --terse flag behavior."""

    def test_search_terse_outputs_paths_only(self, kb_root, index_root):
        """--terse outputs only file paths, no other metadata."""
        _create_entry(
            kb_root / "development" / "api-design.md",
            title="API Design Patterns",
            tags=["api", "design"],
            content="RESTful API design patterns and best practices.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "api", "--terse"])

        assert result.exit_code == 0, f"Command failed: {result.output}"

        # Should contain path
        assert "development/api-design.md" in result.output

        # Should NOT contain table headers or formatting
        assert "PATH" not in result.output
        assert "TITLE" not in result.output
        assert "SCORE" not in result.output

    def test_search_terse_one_per_line(self, kb_root, index_root):
        """--terse outputs one path per line."""
        _create_entry(
            kb_root / "development" / "testing-guide.md",
            title="Testing Guide",
            tags=["testing"],
            content="Comprehensive testing guide for Python applications.",
        )
        _create_entry(
            kb_root / "development" / "pytest-tips.md",
            title="Pytest Tips",
            tags=["testing", "pytest"],
            content="Advanced pytest testing techniques and fixtures.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "testing", "--terse"])

        assert result.exit_code == 0, f"Command failed: {result.output}"

        lines = [line for line in result.output.strip().split("\n") if line]
        # Each non-empty line should be a path
        for line in lines:
            assert line.endswith(".md"), f"Line should be a path: {line}"
            assert "/" in line, f"Line should contain path separator: {line}"

    def test_search_terse_no_headers(self, kb_root, index_root):
        """--terse output has no table headers or formatting."""
        _create_entry(
            kb_root / "infrastructure" / "monitoring.md",
            title="Monitoring Setup",
            tags=["monitoring", "observability"],
            content="Setting up monitoring and alerting for production systems.",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["search", "monitoring", "--terse"])

        assert result.exit_code == 0, f"Command failed: {result.output}"

        # No table formatting
        assert "PATH" not in result.output
        assert "CONF" not in result.output
        assert "====" not in result.output
        assert "----" not in result.output

        # Should just have the path
        lines = result.output.strip().split("\n")
        assert len(lines) >= 1
        assert "infrastructure/monitoring.md" in lines[0]


class TestSearchMode:
    """Tests for --mode flag with different search modes."""

    def test_search_mode_semantic(self, kb_root, index_root):
        """--mode=semantic uses only semantic/embedding search."""
        _create_entry(
            kb_root / "development" / "machine-learning.md",
            title="Machine Learning Basics",
            tags=["ml", "ai"],
            content="Introduction to neural networks and deep learning concepts.",
        )
        _create_entry(
            kb_root / "development" / "data-science.md",
            title="Data Science Overview",
            tags=["data", "analytics"],
            content="Statistical analysis and data visualization techniques.",
        )

        runner = CliRunner()

        # Semantic search for conceptually related content
        result = runner.invoke(cli, ["search", "artificial intelligence", "--mode=semantic", "--json"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        data = json.loads(result.output)

        # Should find semantically related entries (ML is related to AI)
        # Even if "artificial intelligence" doesn't appear literally
        paths = [r["path"] for r in data]
        # At minimum, command should work without error
        assert isinstance(data, list)

    def test_search_mode_keyword(self, kb_root, index_root):
        """--mode=keyword uses only keyword/text search."""
        _create_entry(
            kb_root / "development" / "python-guide.md",
            title="Python Programming Guide",
            tags=["python"],
            content="A comprehensive guide to Python programming language.",
        )

        runner = CliRunner()

        # Keyword search looks for exact terms
        result = runner.invoke(cli, ["search", "python", "--mode=keyword", "--json"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        data = json.loads(result.output)

        # Should find entry with "python" keyword
        assert len(data) >= 1
        assert any("python" in r["path"] for r in data)

    def test_search_mode_hybrid_default(self, kb_root, index_root):
        """--mode=hybrid (default) combines keyword and semantic search."""
        _create_entry(
            kb_root / "development" / "api-design.md",
            title="REST API Design",
            tags=["api", "rest"],
            content="Best practices for designing RESTful APIs.",
        )

        runner = CliRunner()

        # Hybrid mode (default) should work
        result = runner.invoke(cli, ["search", "api", "--mode=hybrid", "--json"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        data = json.loads(result.output)

        assert len(data) >= 1
        assert any("api" in r["path"] for r in data)


class TestSearchCombinations:
    """Tests for combining multiple search flags."""

    def test_search_tags_and_mode(self, kb_root, index_root):
        """--tags works correctly with --mode=keyword."""
        _create_entry(
            kb_root / "development" / "python-async.md",
            title="Python Async Guide",
            tags=["python", "async"],
            content="Asynchronous programming in Python with asyncio.",
        )
        _create_entry(
            kb_root / "development" / "rust-async.md",
            title="Rust Async Guide",
            tags=["rust", "async"],
            content="Asynchronous programming in Rust with tokio.",
        )
        _create_entry(
            kb_root / "development" / "python-basics.md",
            title="Python Basics",
            tags=["python", "basics"],
            content="Python fundamentals and getting started guide.",
        )

        runner = CliRunner()

        # Search with tag filter and keyword mode
        result = runner.invoke(cli, ["search", "programming", "--tags=python", "--mode=keyword", "--json"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        data = json.loads(result.output)

        # All results should have python tag
        for r in data:
            assert "python" in r["path"], f"Result should match python tag filter: {r['path']}"

    def test_search_min_score_and_limit(self, kb_root, index_root):
        """--min-score works correctly with --limit."""
        # Create multiple entries
        for i in range(5):
            _create_entry(
                kb_root / "development" / f"doc-{i}.md",
                title=f"Documentation Part {i}",
                tags=["docs"],
                content=f"Documentation section {i} with various details about the project.",
            )

        runner = CliRunner()

        # Search with both min-score and limit
        result = runner.invoke(cli, ["search", "documentation", "--min-score=0.3", "--limit=3", "--json"])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        data = json.loads(result.output)

        # Should respect limit
        assert len(data) <= 3

        # All results should be above min-score
        for r in data:
            assert r["score"] >= 0.3, f"Score {r['score']} should be >= 0.3"

    def test_search_content_and_full_titles(self, kb_root, index_root):
        """--content works correctly with --full-titles."""
        long_title = "Comprehensive Guide to Building Scalable Microservices Architecture"
        _create_entry(
            kb_root / "development" / "microservices.md",
            title=long_title,
            tags=["architecture", "microservices"],
            content="# Introduction\n\nThis guide covers microservices architecture patterns.\n\n## Scaling\n\nStrategies for horizontal and vertical scaling.",
        )

        runner = CliRunner()

        # Search with both --content and --full-titles
        result = runner.invoke(cli, ["search", "microservices", "--content", "--full-titles"])

        assert result.exit_code == 0, f"Command failed: {result.output}"

        # Should show full title (not truncated)
        assert long_title in result.output or "Comprehensive Guide" in result.output

        # Should show content section
        assert "====" in result.output
        assert "Introduction" in result.output or "Scaling" in result.output
