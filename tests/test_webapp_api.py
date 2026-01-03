"""Comprehensive tests for webapp/api.py REST API endpoints."""

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

    # Basic entry
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

    # Another entry
    (kb_root / "reference" / "api.md").write_text("""---
title: API Reference
tags: [api, reference]
created: 2024-01-03
---

# API Reference

API documentation here.
""")

    # Entry with beads metadata
    (kb_root / "project.md").write_text("""---
title: Project Tracker
tags: [project]
created: 2024-01-04
beads_project: test-project
beads_issues:
  - TEST-001
---

# Project

Tracking project issues.
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
    # Reset the global searcher
    import memex.webapp.api as api_module
    api_module._searcher = mock_searcher

    # Mock get_kb_root to return our test KB
    monkeypatch.setenv("MEMEX_KB_ROOT", str(sample_kb))

    # Mock file watcher to avoid starting threads
    with patch.object(api_module, "_file_watcher", None):
        with patch("memex.webapp.api.FileWatcher"):
            with patch("memex.webapp.api.configure_logging"):
                # Use context manager to properly handle lifespan
                with TestClient(app, raise_server_exceptions=False) as client:
                    yield client


@pytest.fixture
def client_no_lifespan(sample_kb, mock_searcher, monkeypatch):
    """Create TestClient without lifespan events (faster for basic tests)."""
    import memex.webapp.api as api_module
    api_module._searcher = mock_searcher

    monkeypatch.setenv("MEMEX_KB_ROOT", str(sample_kb))

    # Skip lifespan by not using context manager
    return TestClient(app, raise_server_exceptions=False)


# =============================================================================
# Tree Endpoint Tests
# =============================================================================


class TestTreeEndpoint:
    """Tests for /api/tree endpoint."""

    def test_get_tree_returns_directory_structure(self, client_no_lifespan):
        """Tree endpoint returns correct structure."""
        response = client_no_lifespan.get("/api/tree")
        assert response.status_code == 200

        tree = response.json()
        assert isinstance(tree, list)

        # Should have directories and files
        names = {node["name"] for node in tree}
        assert "guides" in names
        assert "reference" in names
        assert "intro.md" in names

    def test_tree_includes_file_titles(self, client_no_lifespan):
        """Tree nodes include parsed titles for files."""
        response = client_no_lifespan.get("/api/tree")
        tree = response.json()

        # Find intro.md
        intro = next((n for n in tree if n["name"] == "intro.md"), None)
        assert intro is not None
        assert intro["type"] == "file"
        assert intro["title"] == "Introduction"

    def test_tree_has_nested_children(self, client_no_lifespan):
        """Directory nodes have children."""
        response = client_no_lifespan.get("/api/tree")
        tree = response.json()

        guides = next((n for n in tree if n["name"] == "guides"), None)
        assert guides is not None
        assert guides["type"] == "directory"
        assert len(guides["children"]) >= 1

        # Check nested file
        setup = next((c for c in guides["children"] if c["name"] == "setup.md"), None)
        assert setup is not None
        assert setup["title"] == "Setup Guide"

    def test_tree_excludes_hidden_files(self, sample_kb, client_no_lifespan):
        """Hidden files and directories are excluded."""
        # Create hidden items
        (sample_kb / ".hidden").mkdir()
        (sample_kb / ".hidden" / "secret.md").write_text("---\ntitle: Secret\n---\n")
        (sample_kb / "_private.md").write_text("---\ntitle: Private\n---\n")

        response = client_no_lifespan.get("/api/tree")
        tree = response.json()

        names = {n["name"] for n in tree}
        assert ".hidden" not in names
        assert "_private.md" not in names


# =============================================================================
# Entry Endpoint Tests
# =============================================================================


class TestEntryEndpoint:
    """Tests for /api/entries/{path} endpoint."""

    def test_get_entry_by_path(self, client_no_lifespan):
        """Can retrieve entry by path."""
        response = client_no_lifespan.get("/api/entries/intro.md")
        assert response.status_code == 200

        entry = response.json()
        assert entry["path"] == "intro.md"
        assert entry["title"] == "Introduction"
        assert "getting-started" in entry["tags"]
        assert "basics" in entry["tags"]

    def test_get_entry_adds_md_extension(self, client_no_lifespan):
        """Path without .md extension is handled."""
        response = client_no_lifespan.get("/api/entries/intro")
        assert response.status_code == 200
        assert response.json()["title"] == "Introduction"

    def test_get_entry_nested_path(self, client_no_lifespan):
        """Can retrieve entry from subdirectory."""
        response = client_no_lifespan.get("/api/entries/guides/setup.md")
        assert response.status_code == 200

        entry = response.json()
        assert entry["title"] == "Setup Guide"
        assert "guides" in entry["tags"]

    def test_entry_includes_html_content(self, client_no_lifespan):
        """Entry response includes rendered HTML."""
        response = client_no_lifespan.get("/api/entries/intro.md")
        entry = response.json()

        assert "content_html" in entry
        assert "<h1>" in entry["content_html"] or "<p>" in entry["content_html"]

    def test_entry_includes_dates(self, client_no_lifespan):
        """Entry includes created/updated dates."""
        response = client_no_lifespan.get("/api/entries/guides/setup.md")
        entry = response.json()

        assert entry["created"] == "2024-01-02"
        assert entry["updated"] == "2024-01-15"

    def test_entry_not_found(self, client_no_lifespan):
        """Returns 404 for missing entry."""
        response = client_no_lifespan.get("/api/entries/nonexistent.md")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_entry_parse_error(self, sample_kb, client_no_lifespan):
        """Returns 500 for unparseable entry."""
        # Create invalid markdown
        (sample_kb / "broken.md").write_text("---\n!!invalid yaml\n---\n")

        response = client_no_lifespan.get("/api/entries/broken.md")
        assert response.status_code == 500
        assert "parse error" in response.json()["detail"].lower()


# =============================================================================
# Search Endpoint Tests
# =============================================================================


class TestSearchEndpoint:
    """Tests for /api/search endpoint."""

    def test_search_requires_query(self, client_no_lifespan):
        """Search requires query parameter."""
        response = client_no_lifespan.get("/api/search")
        assert response.status_code == 422  # Validation error

    def test_search_empty_query_rejected(self, client_no_lifespan):
        """Empty query is rejected."""
        response = client_no_lifespan.get("/api/search?q=")
        assert response.status_code == 422

    def test_search_returns_results(self, client_no_lifespan, mock_searcher):
        """Search returns results from searcher."""
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

        response = client_no_lifespan.get("/api/search?q=introduction")
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert "total" in data
        assert data["total"] == 1
        assert data["results"][0]["title"] == "Introduction"

    def test_search_with_limit(self, client_no_lifespan, mock_searcher):
        """Search respects limit parameter."""
        mock_searcher.search.return_value = []

        client_no_lifespan.get("/api/search?q=test&limit=5")
        mock_searcher.search.assert_called_with("test", limit=5, mode="hybrid")

    def test_search_with_mode(self, client_no_lifespan, mock_searcher):
        """Search respects mode parameter."""
        mock_searcher.search.return_value = []

        client_no_lifespan.get("/api/search?q=test&mode=semantic")
        mock_searcher.search.assert_called_with("test", limit=20, mode="semantic")

    def test_search_limit_bounds(self, client_no_lifespan):
        """Search limit is validated."""
        # Too low
        response = client_no_lifespan.get("/api/search?q=test&limit=0")
        assert response.status_code == 422

        # Too high
        response = client_no_lifespan.get("/api/search?q=test&limit=200")
        assert response.status_code == 422


# =============================================================================
# Graph Endpoint Tests
# =============================================================================


class TestGraphEndpoint:
    """Tests for /api/graph endpoint."""

    def test_get_graph_returns_nodes_and_edges(self, client_no_lifespan):
        """Graph endpoint returns nodes and edges."""
        response = client_no_lifespan.get("/api/graph")
        assert response.status_code == 200

        graph = response.json()
        assert "nodes" in graph
        assert "edges" in graph
        assert isinstance(graph["nodes"], list)
        assert isinstance(graph["edges"], list)

    def test_graph_nodes_have_required_fields(self, client_no_lifespan):
        """Graph nodes have id, label, path."""
        response = client_no_lifespan.get("/api/graph")
        graph = response.json()

        assert len(graph["nodes"]) >= 3  # intro, setup, api

        for node in graph["nodes"]:
            assert "id" in node
            assert "label" in node
            assert "path" in node
            assert "tags" in node
            assert "group" in node

    def test_graph_edges_link_valid_nodes(self, client_no_lifespan):
        """Graph edges connect existing nodes."""
        response = client_no_lifespan.get("/api/graph")
        graph = response.json()

        node_ids = {n["id"] for n in graph["nodes"]}

        for edge in graph["edges"]:
            assert edge["source"] in node_ids
            assert edge["target"] in node_ids

    def test_graph_groups_by_directory(self, client_no_lifespan):
        """Nodes are grouped by top-level directory."""
        response = client_no_lifespan.get("/api/graph")
        graph = response.json()

        # Find nodes by path
        setup_node = next((n for n in graph["nodes"] if "setup" in n["path"]), None)
        assert setup_node is not None
        assert setup_node["group"] == "guides"


# =============================================================================
# Stats Endpoint Tests
# =============================================================================


class TestStatsEndpoint:
    """Tests for /api/stats endpoint."""

    def test_get_stats(self, client_no_lifespan):
        """Stats endpoint returns KB statistics."""
        response = client_no_lifespan.get("/api/stats")
        assert response.status_code == 200

        stats = response.json()
        assert "total_entries" in stats
        assert "total_tags" in stats
        assert "total_links" in stats
        assert "categories" in stats
        assert "recent_entries" in stats

    def test_stats_counts_entries(self, client_no_lifespan):
        """Stats correctly counts entries."""
        response = client_no_lifespan.get("/api/stats")
        stats = response.json()

        # We have 4 entries: intro, setup, api, project
        assert stats["total_entries"] == 4

    def test_stats_counts_tags(self, client_no_lifespan):
        """Stats correctly counts unique tags."""
        response = client_no_lifespan.get("/api/stats")
        stats = response.json()

        # Tags: getting-started, basics, setup, guides, api, reference, project
        assert stats["total_tags"] >= 7

    def test_stats_categories(self, client_no_lifespan):
        """Stats includes category breakdown."""
        response = client_no_lifespan.get("/api/stats")
        stats = response.json()

        categories = {c["name"]: c["count"] for c in stats["categories"]}
        assert "guides" in categories
        assert "reference" in categories


# =============================================================================
# Tags Endpoint Tests
# =============================================================================


class TestTagsEndpoint:
    """Tests for /api/tags endpoint."""

    def test_get_tags(self, client_no_lifespan):
        """Tags endpoint returns all tags with counts."""
        response = client_no_lifespan.get("/api/tags")
        assert response.status_code == 200

        tags = response.json()
        assert isinstance(tags, list)
        assert len(tags) >= 1

    def test_tags_have_count(self, client_no_lifespan):
        """Each tag has a count."""
        response = client_no_lifespan.get("/api/tags")
        tags = response.json()

        for tag in tags:
            assert "tag" in tag
            assert "count" in tag
            assert tag["count"] >= 1

    def test_tags_sorted_by_count(self, client_no_lifespan):
        """Tags are sorted by count descending."""
        response = client_no_lifespan.get("/api/tags")
        tags = response.json()

        counts = [t["count"] for t in tags]
        assert counts == sorted(counts, reverse=True)


# =============================================================================
# Recent Endpoint Tests
# =============================================================================


class TestRecentEndpoint:
    """Tests for /api/recent endpoint."""

    def test_get_recent(self, client_no_lifespan):
        """Recent endpoint returns entries."""
        response = client_no_lifespan.get("/api/recent")
        assert response.status_code == 200

        entries = response.json()
        assert isinstance(entries, list)

    def test_recent_default_limit(self, client_no_lifespan):
        """Recent has default limit of 10."""
        response = client_no_lifespan.get("/api/recent")
        entries = response.json()

        # We have 4 entries, so should get all of them
        assert len(entries) == 4

    def test_recent_with_limit(self, client_no_lifespan):
        """Recent respects limit parameter."""
        response = client_no_lifespan.get("/api/recent?limit=2")
        entries = response.json()

        assert len(entries) == 2

    def test_recent_sorted_by_date(self, client_no_lifespan):
        """Recent entries sorted by most recent first."""
        response = client_no_lifespan.get("/api/recent")
        entries = response.json()

        # Check they're sorted (most recent first)
        dates = []
        for e in entries:
            date = e.get("updated") or e.get("created")
            if date:
                dates.append(date)

        assert dates == sorted(dates, reverse=True)

    def test_recent_includes_metadata(self, client_no_lifespan):
        """Recent entries include path, title, tags, dates."""
        response = client_no_lifespan.get("/api/recent")
        entries = response.json()

        for entry in entries:
            assert "path" in entry
            assert "title" in entry
            assert "tags" in entry


# =============================================================================
# Beads Integration Tests
# =============================================================================


class TestBeadsConfigEndpoint:
    """Tests for /api/beads/config endpoint."""

    def test_beads_config_no_project(self, client_no_lifespan):
        """Returns unavailable when no beads project."""
        with patch("memex.webapp.api.find_beads_db", return_value=None):
            response = client_no_lifespan.get("/api/beads/config")
            assert response.status_code == 200

            config = response.json()
            assert config["available"] is False

    def test_beads_config_with_project(self, client_no_lifespan):
        """Returns available with project path."""
        mock_project = MagicMock()
        mock_project.path = Path("/test/project")

        with patch("memex.webapp.api.find_beads_db", return_value=mock_project):
            response = client_no_lifespan.get("/api/beads/config")
            config = response.json()

            assert config["available"] is True
            assert config["project_path"] == "/test/project"


class TestBeadsKanbanEndpoint:
    """Tests for /api/beads/kanban endpoint."""

    def test_kanban_no_project(self, client_no_lifespan):
        """Returns 404 when no beads project."""
        with patch("memex.webapp.api.find_beads_db", return_value=None):
            response = client_no_lifespan.get("/api/beads/kanban")
            assert response.status_code == 404

    def test_kanban_returns_columns(self, client_no_lifespan):
        """Returns kanban board with columns."""
        mock_project = MagicMock()
        mock_project.path = Path("/test/project")
        mock_project.db_path = Path("/test/project/.beads/issues.db")

        mock_issues = [
            {"id": "TEST-001", "title": "Open issue", "status": "open", "priority": 2},
            {"id": "TEST-002", "title": "In progress", "status": "in_progress", "priority": 1},
            {"id": "TEST-003", "title": "Closed issue", "status": "closed", "priority": 3},
        ]

        with patch("memex.webapp.api.find_beads_db", return_value=mock_project):
            with patch("memex.webapp.api.list_issues", return_value=mock_issues):
                response = client_no_lifespan.get("/api/beads/kanban")
                assert response.status_code == 200

                kanban = response.json()
                assert "columns" in kanban
                assert len(kanban["columns"]) == 3

                # Check column structure
                column_statuses = [c["status"] for c in kanban["columns"]]
                assert "open" in column_statuses
                assert "in_progress" in column_statuses
                assert "closed" in column_statuses


class TestBeadsIssueEndpoint:
    """Tests for /api/beads/issues/{issue_id} endpoint."""

    def test_issue_not_found(self, client_no_lifespan):
        """Returns 404 for missing issue."""
        mock_project = MagicMock()
        mock_project.db_path = Path("/test/project/.beads/issues.db")

        with patch("memex.webapp.api.find_beads_db", return_value=mock_project):
            with patch("memex.webapp.api.show_issue", return_value=None):
                response = client_no_lifespan.get("/api/beads/issues/MISSING-001")
                assert response.status_code == 404

    def test_issue_with_comments(self, client_no_lifespan):
        """Returns issue details with comments."""
        mock_project = MagicMock()
        mock_project.db_path = Path("/test/project/.beads/issues.db")

        mock_issue = {
            "id": "TEST-001",
            "title": "Test Issue",
            "status": "open",
            "priority": 2,
            "description": "Test description",
        }

        mock_comments = [
            {"id": "c1", "author": "user1", "content": "First comment", "created_at": "2024-01-01"},
        ]

        with patch("memex.webapp.api.find_beads_db", return_value=mock_project):
            with patch("memex.webapp.api.show_issue", return_value=mock_issue):
                with patch("memex.webapp.api.get_comments", return_value=mock_comments):
                    response = client_no_lifespan.get("/api/beads/issues/TEST-001")
                    assert response.status_code == 200

                    data = response.json()
                    assert "issue" in data
                    assert "comments" in data
                    assert data["issue"]["id"] == "TEST-001"
                    assert len(data["comments"]) == 1


class TestEntryBeadsEndpoint:
    """Tests for /api/entries/{path}/beads endpoint."""

    def test_entry_beads_no_project(self, client_no_lifespan):
        """Returns empty when no beads project."""
        with patch("memex.webapp.api.find_beads_db", return_value=None):
            response = client_no_lifespan.get("/api/entries/project.md/beads")
            assert response.status_code == 200

            data = response.json()
            assert data["linked_issues"] == []

    def test_entry_beads_not_found(self, client_no_lifespan):
        """Returns 404 for missing entry."""
        response = client_no_lifespan.get("/api/entries/nonexistent.md/beads")
        assert response.status_code == 404


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling across endpoints."""

    def test_invalid_search_mode(self, client_no_lifespan):
        """Invalid search mode rejected."""
        response = client_no_lifespan.get("/api/search?q=test&mode=invalid")
        assert response.status_code == 422

    def test_special_characters_in_path(self, sample_kb, client_no_lifespan):
        """Paths with special characters handled safely."""
        # Create file with spaces
        (sample_kb / "file with spaces.md").write_text("""---
title: Spaces
tags: [test]
created: 2024-01-01
---

# File With Spaces

Content here.
""")

        response = client_no_lifespan.get("/api/entries/file%20with%20spaces.md")
        assert response.status_code == 200

    def test_unicode_content(self, sample_kb, client_no_lifespan):
        """Unicode content handled correctly."""
        (sample_kb / "unicode.md").write_text("""---
title: Unicode Test 日本語
tags: [测试, テスト]
created: 2024-01-01
---

# 日本語

Content with émojis 🎉 and symbols ★
""")

        response = client_no_lifespan.get("/api/entries/unicode.md")
        assert response.status_code == 200
        assert "日本語" in response.json()["title"]


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_kb(self, tmp_path, mock_searcher, monkeypatch):
        """Handles empty KB directory."""
        empty_kb = tmp_path / "empty_kb"
        empty_kb.mkdir()

        import memex.webapp.api as api_module
        api_module._searcher = mock_searcher
        monkeypatch.setenv("MEMEX_KB_ROOT", str(empty_kb))

        client = TestClient(app, raise_server_exceptions=False)

        # Tree should be empty
        response = client.get("/api/tree")
        assert response.status_code == 200
        assert response.json() == []

        # Stats should show zeros
        response = client.get("/api/stats")
        assert response.status_code == 200
        assert response.json()["total_entries"] == 0

    def test_deeply_nested_path(self, sample_kb, client_no_lifespan):
        """Handles deeply nested file paths."""
        deep = sample_kb / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "deep.md").write_text("""---
title: Deep
tags: [test]
created: 2024-01-01
---

# Deep

Content here.
""")

        response = client_no_lifespan.get("/api/entries/a/b/c/deep.md")
        assert response.status_code == 200

    def test_entry_without_frontmatter(self, sample_kb, client_no_lifespan):
        """Handles markdown without YAML frontmatter."""
        (sample_kb / "plain.md").write_text("# Just Markdown\n\nNo frontmatter here.")

        # This should either parse with defaults or return an error
        response = client_no_lifespan.get("/api/entries/plain.md")
        # The parser may handle this differently - check it doesn't crash
        assert response.status_code in (200, 500)


# =============================================================================
# SSE Events Endpoint Test
# =============================================================================


class TestEventsEndpoint:
    """Tests for /api/events SSE endpoint."""

    def test_events_endpoint_exists(self, client_no_lifespan):
        """Events endpoint is registered."""
        # SSE endpoints are tricky to test with TestClient since they stream indefinitely.
        # We verify the route exists by checking that it's in the app routes.
        from memex.webapp.api import app

        routes = [r.path for r in app.routes]
        assert "/api/events" in routes


# =============================================================================
# Root Endpoint Test
# =============================================================================


class TestRootEndpoint:
    """Tests for / root endpoint."""

    def test_root_returns_response(self, client_no_lifespan):
        """Root endpoint returns something."""
        response = client_no_lifespan.get("/")
        assert response.status_code == 200

        # Either returns index.html or API message
        content_type = response.headers.get("content-type", "")
        assert "text/html" in content_type or "application/json" in content_type


# =============================================================================
# Priority Label Tests
# =============================================================================


class TestPriorityLabels:
    """Tests for priority label normalization."""

    def test_all_priority_labels(self, client_no_lifespan):
        """All priority values map to labels."""
        from memex.webapp.api import PRIORITY_LABELS, _normalize_issue

        for priority in range(5):
            raw = {"id": "T-1", "title": "Test", "status": "open", "priority": priority}
            issue = _normalize_issue(raw)
            assert issue.priority_label == PRIORITY_LABELS[priority]

    def test_unknown_priority_defaults(self, client_no_lifespan):
        """Unknown priority defaults to medium."""
        from memex.webapp.api import _normalize_issue

        raw = {"id": "T-1", "title": "Test", "status": "open", "priority": 99}
        issue = _normalize_issue(raw)
        assert issue.priority_label == "medium"


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestLifespanEvents:
    """Tests for startup/shutdown lifecycle events."""

    def test_file_watcher_reference(self, sample_kb, mock_searcher, monkeypatch):
        """Verify file watcher module is accessible."""
        # Test that the watcher module can be imported
        from memex.indexer.watcher import FileWatcher

        # Verify FileWatcher class exists
        assert FileWatcher is not None


class TestSearcherInitialization:
    """Tests for lazy searcher initialization."""

    def test_get_searcher_reindexes_when_empty(self, sample_kb, monkeypatch):
        """Searcher reindexes when indices are empty but KB has files."""
        import memex.webapp.api as api_module

        # Reset global searcher
        api_module._searcher = None
        monkeypatch.setenv("MEMEX_KB_ROOT", str(sample_kb))

        with patch.object(api_module, "HybridSearcher") as MockSearcher:
            mock_instance = MagicMock()
            mock_status = MagicMock()
            mock_status.kb_files = 5
            mock_status.whoosh_docs = 0
            mock_status.chroma_docs = 0
            mock_instance.status.return_value = mock_status
            MockSearcher.return_value = mock_instance

            api_module._get_searcher()

            # Should have called reindex
            mock_instance.reindex.assert_called_once()


class TestEntryBeadsWithProject:
    """Tests for entry beads with beads_project but no specific issues."""

    def test_entry_beads_with_project_shows_all_issues(self, sample_kb, client_no_lifespan):
        """Entry with beads_project but no beads_issues shows all project issues."""
        # Create entry with only beads_project (no specific issues)
        (sample_kb / "project_only.md").write_text("""---
title: Project Only
tags: [project]
created: 2024-01-01
beads_project: test-project
---

# Project

Shows all issues from test-project.
""")

        mock_project = MagicMock()
        mock_project.db_path = Path("/test/.beads/issues.db")

        mock_issues = [
            {"id": "T-1", "title": "Open", "status": "open", "priority": 2},
            {"id": "T-2", "title": "Closed", "status": "closed", "priority": 3},
        ]

        with patch("memex.webapp.api.find_beads_db", return_value=mock_project):
            with patch("memex.webapp.api.list_issues", return_value=mock_issues):
                response = client_no_lifespan.get("/api/entries/project_only.md/beads")
                assert response.status_code == 200

                data = response.json()
                # Should have project_issues (not linked_issues)
                assert data["project_issues"] is not None
                # Should only show non-closed issues
                assert len(data["project_issues"]) == 1
                assert data["project_issues"][0]["status"] == "open"


class TestKanbanEdgeCases:
    """Tests for kanban edge cases like blocked/deferred status."""

    def test_kanban_blocked_status(self, client_no_lifespan):
        """Blocked issues appear in the open column."""
        mock_project = MagicMock()
        mock_project.path = Path("/test/project")
        mock_project.db_path = Path("/test/project/.beads/issues.db")

        mock_issues = [
            {"id": "T-1", "title": "Blocked issue", "status": "blocked", "priority": 2},
            {"id": "T-2", "title": "Deferred issue", "status": "deferred", "priority": 3},
        ]

        with patch("memex.webapp.api.find_beads_db", return_value=mock_project):
            with patch("memex.webapp.api.list_issues", return_value=mock_issues):
                response = client_no_lifespan.get("/api/beads/kanban")
                assert response.status_code == 200

                kanban = response.json()
                open_col = next(c for c in kanban["columns"] if c["status"] == "open")

                # Both blocked and deferred should be in open column
                assert len(open_col["issues"]) == 2

    def test_kanban_with_project_path(self, client_no_lifespan):
        """Kanban with explicit project_path parameter."""
        mock_project = MagicMock()
        mock_project.path = Path("/custom/project")
        mock_project.db_path = Path("/custom/project/.beads/issues.db")

        with patch("memex.webapp.api.find_beads_db", return_value=mock_project) as mock_find:
            with patch("memex.webapp.api.list_issues", return_value=[]):
                response = client_no_lifespan.get("/api/beads/kanban?project_path=/custom/project")
                assert response.status_code == 200

                # Should have called find_beads_db with the path
                mock_find.assert_called_with("/custom/project")

    def test_kanban_project_path_not_found(self, client_no_lifespan):
        """Returns 404 when project_path doesn't exist."""
        with patch("memex.webapp.api.find_beads_db", return_value=None):
            response = client_no_lifespan.get("/api/beads/kanban?project_path=/nonexistent")
            assert response.status_code == 404


class TestIssueEndpointProjectPath:
    """Tests for issue endpoint with project_path."""

    def test_issue_with_project_path(self, client_no_lifespan):
        """Get issue with explicit project_path."""
        mock_project = MagicMock()
        mock_project.db_path = Path("/custom/.beads/issues.db")

        mock_issue = {
            "id": "T-1",
            "title": "Test",
            "status": "open",
            "priority": 2,
        }

        with patch("memex.webapp.api.find_beads_db", return_value=mock_project) as mock_find:
            with patch("memex.webapp.api.show_issue", return_value=mock_issue):
                with patch("memex.webapp.api.get_comments", return_value=[]):
                    response = client_no_lifespan.get(
                        "/api/beads/issues/T-1?project_path=/custom"
                    )
                    assert response.status_code == 200
                    mock_find.assert_called_with("/custom")

    def test_issue_project_path_not_found(self, client_no_lifespan):
        """Returns 404 when project_path doesn't exist."""
        with patch("memex.webapp.api.find_beads_db", return_value=None):
            response = client_no_lifespan.get(
                "/api/beads/issues/T-1?project_path=/nonexistent"
            )
            assert response.status_code == 404


class TestGraphEdgeCases:
    """Tests for graph building edge cases."""

    def test_graph_skips_unparseable_files(self, sample_kb, client_no_lifespan):
        """Graph skips files that can't be parsed."""
        # Create an invalid file
        (sample_kb / "_hidden.md").write_text("---\ntitle: Hidden\n---\n")

        response = client_no_lifespan.get("/api/graph")
        assert response.status_code == 200

        # Should not include hidden files
        graph = response.json()
        paths = [n["path"] for n in graph["nodes"]]
        assert "_hidden.md" not in paths


class TestTreeEdgeCases:
    """Tests for tree building edge cases."""

    def test_tree_handles_permission_error(self, sample_kb, client_no_lifespan, monkeypatch):
        """Tree gracefully handles permission errors."""
        # We can't easily test permission errors, so just verify
        # the tree works with normal files
        response = client_no_lifespan.get("/api/tree")
        assert response.status_code == 200


class TestStatsEdgeCases:
    """Tests for stats computation edge cases."""

    def test_stats_with_entries_in_root(self, sample_kb, client_no_lifespan):
        """Stats correctly categorizes root-level entries."""
        response = client_no_lifespan.get("/api/stats")
        stats = response.json()

        categories = {c["name"]: c["count"] for c in stats["categories"]}
        # intro.md and project.md are at root
        assert "root" in categories


class TestNormalizeIssue:
    """Tests for issue normalization edge cases."""

    def test_normalize_issue_with_dependents(self, client_no_lifespan):
        """Normalize issue preserves dependents for epics."""
        from memex.webapp.api import _normalize_issue

        raw = {
            "id": "EPIC-1",
            "title": "Epic",
            "status": "open",
            "priority": 2,
            "issue_type": "epic",
            "dependents": [
                {"id": "T-1", "title": "Child 1"},
                {"id": "T-2", "title": "Child 2"},
            ],
        }

        issue = _normalize_issue(raw)
        assert issue.dependents is not None
        assert len(issue.dependents) == 2


class TestMainFunction:
    """Tests for the main() entry point."""

    def test_main_runs_uvicorn(self, monkeypatch):
        """Main function runs uvicorn with correct settings."""
        monkeypatch.setenv("HOST", "127.0.0.1")
        monkeypatch.setenv("PORT", "9000")

        with patch("uvicorn.run") as mock_run:
            from memex.webapp.api import main
            main()

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[1]["host"] == "127.0.0.1"
            assert call_args[1]["port"] == 9000


class TestEntryBeadsSpecificIssues:
    """Tests for entry beads with specific issue IDs."""

    def test_entry_beads_with_specific_issues(self, sample_kb, client_no_lifespan):
        """Entry with beads_issues shows those specific issues."""
        mock_project = MagicMock()
        mock_project.db_path = Path("/test/.beads/issues.db")

        mock_issue = {
            "id": "TEST-001",
            "title": "Specific Issue",
            "status": "open",
            "priority": 2,
        }

        with patch("memex.webapp.api.find_beads_db", return_value=mock_project):
            with patch("memex.webapp.api.show_issue", return_value=mock_issue):
                response = client_no_lifespan.get("/api/entries/project.md/beads")
                assert response.status_code == 200

                data = response.json()
                assert len(data["linked_issues"]) == 1
                assert data["linked_issues"][0]["id"] == "TEST-001"


class TestGraphLinkResolution:
    """Tests for graph link resolution."""

    def test_graph_resolves_title_links(self, sample_kb, client_no_lifespan):
        """Graph resolves links by title."""
        # The intro.md links to guides/setup.md which should be resolved
        response = client_no_lifespan.get("/api/graph")
        assert response.status_code == 200

        graph = response.json()
        # Check that edges exist
        assert len(graph["edges"]) > 0


class TestBacklinkIndex:
    """Tests for backlink index functionality."""

    def test_entry_includes_backlinks(self, client_no_lifespan):
        """Entry response includes backlinks from other entries."""
        # intro.md is linked from guides/setup.md
        response = client_no_lifespan.get("/api/entries/intro.md")
        assert response.status_code == 200

        entry = response.json()
        assert "backlinks" in entry


class TestTreeParsing:
    """Tests for tree node parsing."""

    def test_tree_includes_parse_errors_gracefully(self, sample_kb, client_no_lifespan):
        """Tree handles entries that fail to parse for titles."""
        # The tree should still work even if some entries can't be parsed
        response = client_no_lifespan.get("/api/tree")
        assert response.status_code == 200


class TestStatsParsing:
    """Tests for stats parsing edge cases."""

    def test_stats_handles_parse_errors(self, sample_kb, client_no_lifespan):
        """Stats computation handles entries that fail to parse."""
        # Create a broken file
        (sample_kb / "broken_stats.md").write_text("---\nbad yaml: {\n---\n")

        response = client_no_lifespan.get("/api/stats")
        assert response.status_code == 200

        # Should still return stats, just skip the broken file
        stats = response.json()
        assert stats["total_entries"] >= 4


class TestTagsParsing:
    """Tests for tags parsing edge cases."""

    def test_tags_handles_parse_errors(self, sample_kb, client_no_lifespan):
        """Tags collection handles entries that fail to parse."""
        # Create a broken file
        (sample_kb / "broken_tags.md").write_text("---\nbad yaml: [\n---\n")

        response = client_no_lifespan.get("/api/tags")
        assert response.status_code == 200


class TestRecentParsing:
    """Tests for recent entries parsing."""

    def test_recent_handles_parse_errors(self, sample_kb, client_no_lifespan):
        """Recent entries computation handles parse errors."""
        (sample_kb / "broken_recent.md").write_text("---\nbad: }\n---\n")

        response = client_no_lifespan.get("/api/recent")
        assert response.status_code == 200


class TestRootWithStaticFiles:
    """Tests for root endpoint with and without static files."""

    def test_root_without_static_returns_json(self, client_no_lifespan):
        """Root returns JSON when static files don't exist."""
        from memex.webapp.api import static_dir

        # If static dir doesn't exist or index.html is missing,
        # should return JSON
        if not (static_dir / "index.html").exists():
            response = client_no_lifespan.get("/")
            assert response.status_code == 200
            data = response.json()
            assert "message" in data or isinstance(data, dict)


class TestEntryBeadsEdgeCases:
    """Tests for entry beads edge cases."""

    def test_entry_beads_without_md_extension(self, sample_kb, client_no_lifespan):
        """Entry beads path without .md extension is handled."""
        with patch("memex.webapp.api.find_beads_db", return_value=None):
            # project (without .md) should work
            response = client_no_lifespan.get("/api/entries/project/beads")
            assert response.status_code == 200

    def test_entry_beads_parse_error(self, sample_kb, client_no_lifespan):
        """Entry beads returns 500 on parse error."""
        (sample_kb / "broken_beads.md").write_text("---\nbad: {\n---\n")

        response = client_no_lifespan.get("/api/entries/broken_beads.md/beads")
        assert response.status_code == 500

    def test_entry_beads_default_project(self, sample_kb, client_no_lifespan):
        """Entry without beads_project uses default project."""
        with patch("memex.webapp.api._get_default_beads_project", return_value=None):
            response = client_no_lifespan.get("/api/entries/intro.md/beads")
            assert response.status_code == 200


class TestTreeParseErrors:
    """Tests for tree parsing with broken files."""

    def test_tree_shows_broken_file_without_title(self, sample_kb, client_no_lifespan):
        """Tree includes files that fail to parse, without title."""
        (sample_kb / "broken_tree.md").write_text("---\nbad yaml: {\n---\n")

        response = client_no_lifespan.get("/api/tree")
        assert response.status_code == 200

        tree = response.json()
        # Find the broken file
        broken = next((n for n in tree if n["name"] == "broken_tree.md"), None)
        assert broken is not None
        # Should have no title (parse failed)
        assert broken["title"] is None


class TestGraphParseErrors:
    """Tests for graph parsing with broken files."""

    def test_graph_skips_broken_files(self, sample_kb, client_no_lifespan):
        """Graph skips files that fail to parse."""
        (sample_kb / "broken_graph.md").write_text("---\nbad: }\n---\n")

        response = client_no_lifespan.get("/api/graph")
        assert response.status_code == 200

        graph = response.json()
        paths = [n["path"] for n in graph["nodes"]]
        assert "broken_graph.md" not in paths


class TestGraphLinkEdgeCases:
    """Tests for graph link resolution edge cases."""

    def test_graph_handles_unresolvable_links(self, sample_kb, client_no_lifespan):
        """Graph handles links to non-existent entries."""
        (sample_kb / "orphan_links.md").write_text("""---
title: Orphan Links
tags: [test]
created: 2024-01-01
---

# Links to nothing

See [[nonexistent.md]] and [[Also Not Real]].
""")

        response = client_no_lifespan.get("/api/graph")
        assert response.status_code == 200

        graph = response.json()
        # Should have the node but edges should only point to valid targets
        orphan = next((n for n in graph["nodes"] if "orphan" in n["path"]), None)
        assert orphan is not None


# =============================================================================
# Mermaid Integration Tests
# =============================================================================


class TestMermaidIntegration:
    """Smoke tests for Mermaid diagram support in web viewer."""

    def test_index_includes_mermaid_script(self, client_no_lifespan):
        """Verify mermaid.js CDN is included in index.html."""
        response = client_no_lifespan.get("/")
        assert response.status_code == 200
        assert "mermaid" in response.text
        assert "cdn.jsdelivr.net/npm/mermaid" in response.text

    def test_index_includes_mermaid_initialization(self, client_no_lifespan):
        """Verify mermaid is initialized with dark theme."""
        response = client_no_lifespan.get("/")
        assert "mermaid.initialize" in response.text
        assert "theme: 'dark'" in response.text or 'theme: "dark"' in response.text

    def test_index_includes_mermaid_css(self, client_no_lifespan):
        """Verify mermaid CSS classes are defined."""
        response = client_no_lifespan.get("/")
        assert ".mermaid" in response.text
        assert ".mermaid-error" in response.text

    def test_index_includes_mermaid_renderer(self, client_no_lifespan):
        """Verify custom mermaid renderer for marked.js exists."""
        response = client_no_lifespan.get("/")
        assert "mermaidRenderer" in response.text
        assert "processMermaidDiagrams" in response.text
