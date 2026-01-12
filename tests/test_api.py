"""Tests for webapp/api.py REST API endpoints.

Coverage:
- Search endpoint (GET /api/search)
- Entry endpoint (GET /api/entries/{path})
- Tree endpoint (GET /api/tree)
- Graph endpoint (GET /api/graph)
- Stats endpoint (GET /api/stats)
- Tags endpoint (GET /api/tags)
- Recent endpoint (GET /api/recent)
- Root endpoint (GET /)

Testing philosophy:
- Use FastAPI TestClient for HTTP layer testing
- Focus on status codes, response structure, error handling
- Mock heavy dependencies (searcher, file watcher)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from memex.webapp.api import app


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_kb(tmp_path) -> Path:
    """Create a sample KB directory with test entries."""
    kb_root = tmp_path / "kb"
    kb_root.mkdir()

    # Create directory structure
    (kb_root / "guides").mkdir()
    (kb_root / "reference").mkdir()

    # Basic entry at root
    (kb_root / "intro.md").write_text("""---
title: Introduction
tags: [getting-started, basics]
created: 2024-01-01
---

# Introduction

Welcome to the knowledge base.

See also [[guides/setup.md|Setup Guide]].
""")

    # Entry in subdirectory
    (kb_root / "guides" / "setup.md").write_text("""---
title: Setup Guide
tags: [setup, guides]
created: 2024-01-02
updated: 2024-01-15
---

# Setup Guide

How to set up the system.

Links to [[intro.md]] and [[reference/api.md]].
""")

    # Another entry in subdirectory
    (kb_root / "reference" / "api.md").write_text("""---
title: API Reference
tags: [api, reference]
created: 2024-01-03
---

# API Reference

API documentation here.
""")

    return kb_root


@pytest.fixture
def mock_searcher():
    """Create a mock HybridSearcher."""
    searcher = MagicMock()
    searcher.search.return_value = []
    searcher.status.return_value = MagicMock(kb_files=0, whoosh_docs=0, chroma_docs=0)
    return searcher


@pytest.fixture
def client(sample_kb, mock_searcher, monkeypatch):
    """Create TestClient with mocked dependencies."""
    import memex.webapp.api as api_module

    api_module._searcher = mock_searcher
    monkeypatch.setenv("MEMEX_KB_ROOT", str(sample_kb))

    return TestClient(app, raise_server_exceptions=False)


# =============================================================================
# Search Endpoint Tests (/api/search)
# =============================================================================


class TestSearchEndpoint:
    """Tests for /api/search endpoint."""

    def test_search_requires_query_parameter(self, client):
        """Search without query returns 422."""
        response = client.get("/api/search")
        assert response.status_code == 422

    def test_search_rejects_empty_query(self, client):
        """Empty query string returns 422."""
        response = client.get("/api/search?q=")
        assert response.status_code == 422

    def test_search_returns_results_structure(self, client, mock_searcher):
        """Search returns proper JSON structure."""
        from memex.models import SearchResult

        mock_searcher.search.return_value = [
            SearchResult(
                path="intro.md",
                title="Introduction",
                snippet="Welcome to the knowledge base.",
                score=0.95,
                tags=["getting-started"],
            )
        ]

        response = client.get("/api/search?q=introduction")
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert "total" in data
        assert data["total"] == 1
        assert data["results"][0]["title"] == "Introduction"

    def test_search_respects_limit_parameter(self, client, mock_searcher):
        """Search passes limit to searcher."""
        mock_searcher.search.return_value = []

        client.get("/api/search?q=test&limit=5")
        mock_searcher.search.assert_called_with("test", limit=5, mode="hybrid")

    def test_search_respects_mode_parameter(self, client, mock_searcher):
        """Search passes mode to searcher."""
        mock_searcher.search.return_value = []

        client.get("/api/search?q=test&mode=semantic")
        mock_searcher.search.assert_called_with("test", limit=20, mode="semantic")

    def test_search_rejects_invalid_mode(self, client):
        """Invalid search mode returns 422."""
        response = client.get("/api/search?q=test&mode=invalid")
        assert response.status_code == 422

    def test_search_validates_limit_bounds(self, client):
        """Limit outside 1-100 range returns 422."""
        assert client.get("/api/search?q=test&limit=0").status_code == 422
        assert client.get("/api/search?q=test&limit=101").status_code == 422


# =============================================================================
# Entry Endpoint Tests (/api/entries/{path})
# =============================================================================


class TestEntryEndpoint:
    """Tests for /api/entries/{path} endpoint."""

    def test_get_entry_success(self, client):
        """Retrieve entry by path returns 200."""
        response = client.get("/api/entries/intro.md")
        assert response.status_code == 200

        entry = response.json()
        assert entry["path"] == "intro.md"
        assert entry["title"] == "Introduction"
        assert "getting-started" in entry["tags"]

    def test_get_entry_auto_adds_md_extension(self, client):
        """Path without .md extension is handled."""
        response = client.get("/api/entries/intro")
        assert response.status_code == 200
        assert response.json()["title"] == "Introduction"

    def test_get_entry_nested_path(self, client):
        """Retrieve entry from subdirectory."""
        response = client.get("/api/entries/guides/setup.md")
        assert response.status_code == 200
        assert response.json()["title"] == "Setup Guide"

    def test_get_entry_includes_required_fields(self, client):
        """Entry response has all required fields."""
        response = client.get("/api/entries/intro.md")
        entry = response.json()

        required_fields = ["path", "title", "content", "content_html", "tags",
                          "created", "updated", "links", "backlinks"]
        for field in required_fields:
            assert field in entry, f"Missing field: {field}"

    def test_get_entry_not_found(self, client):
        """Missing entry returns 404."""
        response = client.get("/api/entries/nonexistent.md")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_entry_parse_error(self, sample_kb, client):
        """Unparseable entry returns 500."""
        (sample_kb / "broken.md").write_text("---\n!!invalid yaml\n---\n")

        response = client.get("/api/entries/broken.md")
        assert response.status_code == 500


# =============================================================================
# Tree Endpoint Tests (/api/tree)
# =============================================================================


class TestTreeEndpoint:
    """Tests for /api/tree endpoint."""

    def test_tree_returns_directory_structure(self, client):
        """Tree endpoint returns list of nodes."""
        response = client.get("/api/tree")
        assert response.status_code == 200

        tree = response.json()
        assert isinstance(tree, list)

        names = {node["name"] for node in tree}
        assert "guides" in names
        assert "intro.md" in names

    def test_tree_nodes_have_required_fields(self, client):
        """Tree nodes have name, type, path fields."""
        response = client.get("/api/tree")
        tree = response.json()

        for node in tree:
            assert "name" in node
            assert "type" in node
            assert "path" in node

    def test_tree_excludes_hidden_files(self, sample_kb, client):
        """Hidden files and _ prefixed files are excluded."""
        (sample_kb / ".hidden.md").write_text("---\ntitle: Hidden\ntags: [test]\ncreated: 2024-01-01\n---\n")
        (sample_kb / "_private.md").write_text("---\ntitle: Private\ntags: [test]\ncreated: 2024-01-01\n---\n")

        response = client.get("/api/tree")
        names = {n["name"] for n in response.json()}

        assert ".hidden.md" not in names
        assert "_private.md" not in names


# =============================================================================
# Graph Endpoint Tests (/api/graph)
# =============================================================================


class TestGraphEndpoint:
    """Tests for /api/graph endpoint."""

    def test_graph_returns_nodes_and_edges(self, client):
        """Graph has nodes and edges arrays."""
        response = client.get("/api/graph")
        assert response.status_code == 200

        graph = response.json()
        assert "nodes" in graph
        assert "edges" in graph
        assert isinstance(graph["nodes"], list)
        assert isinstance(graph["edges"], list)

    def test_graph_nodes_have_required_fields(self, client):
        """Graph nodes have id, label, path, tags, group."""
        response = client.get("/api/graph")
        graph = response.json()

        for node in graph["nodes"]:
            assert "id" in node
            assert "label" in node
            assert "path" in node
            assert "tags" in node
            assert "group" in node

    def test_graph_edges_reference_valid_nodes(self, client):
        """Graph edges connect existing node ids."""
        response = client.get("/api/graph")
        graph = response.json()

        node_ids = {n["id"] for n in graph["nodes"]}
        for edge in graph["edges"]:
            assert edge["source"] in node_ids
            assert edge["target"] in node_ids


# =============================================================================
# Stats Endpoint Tests (/api/stats)
# =============================================================================


class TestStatsEndpoint:
    """Tests for /api/stats endpoint."""

    def test_stats_returns_required_fields(self, client):
        """Stats has all required fields."""
        response = client.get("/api/stats")
        assert response.status_code == 200

        stats = response.json()
        assert "total_entries" in stats
        assert "total_tags" in stats
        assert "total_links" in stats
        assert "categories" in stats
        assert "recent_entries" in stats

    def test_stats_counts_entries_correctly(self, client):
        """Stats counts entries accurately."""
        response = client.get("/api/stats")
        stats = response.json()

        # We have 3 entries: intro, setup, api
        assert stats["total_entries"] == 3


# =============================================================================
# Tags Endpoint Tests (/api/tags)
# =============================================================================


class TestTagsEndpoint:
    """Tests for /api/tags endpoint."""

    def test_tags_returns_list(self, client):
        """Tags endpoint returns list of tag objects."""
        response = client.get("/api/tags")
        assert response.status_code == 200

        tags = response.json()
        assert isinstance(tags, list)

    def test_tags_have_tag_and_count(self, client):
        """Each tag has tag name and count."""
        response = client.get("/api/tags")
        tags = response.json()

        for tag in tags:
            assert "tag" in tag
            assert "count" in tag


# =============================================================================
# Recent Endpoint Tests (/api/recent)
# =============================================================================


class TestRecentEndpoint:
    """Tests for /api/recent endpoint."""

    def test_recent_returns_list(self, client):
        """Recent endpoint returns list of entries."""
        response = client.get("/api/recent")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_recent_respects_limit(self, client):
        """Recent respects limit parameter."""
        response = client.get("/api/recent?limit=1")
        entries = response.json()
        assert len(entries) == 1

    def test_recent_entries_have_metadata(self, client):
        """Recent entries include path, title, tags."""
        response = client.get("/api/recent")
        for entry in response.json():
            assert "path" in entry
            assert "title" in entry
            assert "tags" in entry


# =============================================================================
# Root Endpoint Tests (/)
# =============================================================================


class TestRootEndpoint:
    """Tests for / root endpoint."""

    def test_root_returns_200(self, client):
        """Root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_kb_returns_empty_results(self, tmp_path, mock_searcher, monkeypatch):
        """Empty KB returns empty tree and zero stats."""
        import memex.webapp.api as api_module

        empty_kb = tmp_path / "empty"
        empty_kb.mkdir()

        api_module._searcher = mock_searcher
        monkeypatch.setenv("MEMEX_KB_ROOT", str(empty_kb))

        client = TestClient(app, raise_server_exceptions=False)

        assert client.get("/api/tree").json() == []
        assert client.get("/api/stats").json()["total_entries"] == 0

    def test_url_encoded_path(self, sample_kb, client):
        """URL-encoded paths are handled correctly."""
        (sample_kb / "file with spaces.md").write_text("""---
title: Spaces
tags: [test]
created: 2024-01-01
---

# Spaces
""")

        response = client.get("/api/entries/file%20with%20spaces.md")
        assert response.status_code == 200

    def test_deeply_nested_path(self, sample_kb, client):
        """Deeply nested paths are handled."""
        deep = sample_kb / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "deep.md").write_text("""---
title: Deep
tags: [test]
created: 2024-01-01
---

# Deep
""")

        response = client.get("/api/entries/a/b/c/deep.md")
        assert response.status_code == 200
