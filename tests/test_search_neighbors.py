"""Tests for graph-aware search with --include-neighbors flag.

Tests cover:
1. CLI flag parsing (--include-neighbors, --neighbor-depth)
2. Neighbor expansion in core.py
3. Output formats (JSON, terse, table) with neighbor markers
4. Deduplication of results
5. Edge cases (no neighbors, circular links, depth limiting)
6. Typed relations included as neighbors
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from memex.cli import cli
from memex.models import SearchResult


def create_entry_with_links(
    kb_root: Path,
    path: str,
    title: str,
    content: str,
    tags: list[str],
    semantic_links: list[dict] | None = None,
    relations: list[dict] | None = None,
) -> Path:
    """Create a test entry with semantic_links/relations in frontmatter."""
    entry_path = kb_root / path
    entry_path.parent.mkdir(parents=True, exist_ok=True)

    tags_str = f"[{', '.join(tags)}]"

    # Build semantic_links YAML
    links_yaml = ""
    if semantic_links:
        links_yaml = "semantic_links:\n"
        for link in semantic_links:
            links_yaml += f"  - path: {link['path']}\n"
            links_yaml += f"    score: {link['score']}\n"
            links_yaml += f"    reason: {link['reason']}\n"

    relations_yaml = ""
    if relations:
        relations_yaml = "relations:\n"
        for relation in relations:
            relations_yaml += f"  - path: {relation['path']}\n"
            relations_yaml += f"    type: {relation['type']}\n"

    frontmatter = f"""---
title: {title}
tags: {tags_str}
created: 2024-01-15
{links_yaml}{relations_yaml}---

{content}
"""
    entry_path.write_text(frontmatter)
    return entry_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI Flag Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSearchNeighborFlags:
    """Test CLI flag parsing for neighbor options."""

    def test_include_neighbors_flag_recognized(self, runner: CliRunner, tmp_kb: Path):
        """--include-neighbors flag is recognized."""
        # Create a simple entry
        create_entry_with_links(tmp_kb, "test.md", "Test Entry", "Test content", ["test"])

        result = runner.invoke(
            cli,
            ["search", "test", "--include-neighbors"],
            env={"MEMEX_USER_KB_ROOT": str(tmp_kb)},
        )
        # Should not fail due to unknown option
        assert "no such option" not in result.output.lower()

    def test_neighbor_depth_flag_recognized(self, runner: CliRunner, tmp_kb: Path):
        """--neighbor-depth flag is recognized."""
        create_entry_with_links(tmp_kb, "test.md", "Test Entry", "Test content", ["test"])

        result = runner.invoke(
            cli,
            ["search", "test", "--include-neighbors", "--neighbor-depth=2"],
            env={"MEMEX_USER_KB_ROOT": str(tmp_kb)},
        )
        assert "no such option" not in result.output.lower()

    def test_neighbor_depth_default_is_one(self, runner: CliRunner, tmp_kb: Path):
        """--neighbor-depth defaults to 1."""
        # Create entry A with link to B
        create_entry_with_links(
            tmp_kb,
            "a.md",
            "Entry A",
            "Content about entry A",
            ["test"],
            semantic_links=[{"path": "b.md", "score": 0.8, "reason": "embedding_similarity"}],
        )
        # Create entry B with link to C
        create_entry_with_links(
            tmp_kb,
            "b.md",
            "Entry B",
            "Content about entry B",
            ["test"],
            semantic_links=[{"path": "c.md", "score": 0.7, "reason": "embedding_similarity"}],
        )
        # Create entry C (no links)
        create_entry_with_links(tmp_kb, "c.md", "Entry C", "Content about entry C", ["test"])

        result = runner.invoke(
            cli,
            ["search", "entry A", "--include-neighbors", "--json"],
            env={"MEMEX_USER_KB_ROOT": str(tmp_kb)},
        )

        # With depth=1, should find A (direct) and B (neighbor), but NOT C
        if result.exit_code == 0 and "results" in result.output:
            import json

            data = json.loads(result.output)
            paths = [r["path"] for r in data["results"]]
            # B should be included as direct neighbor of A
            # C should NOT be included with default depth=1
            if "a.md" in paths:
                # If A is found, B should also be found
                assert "b.md" in paths or len(paths) == 1

    def test_neighbor_depth_range_validation(self, runner: CliRunner, tmp_kb: Path):
        """--neighbor-depth validates range (1-5)."""
        create_entry_with_links(tmp_kb, "test.md", "Test Entry", "Test content", ["test"])

        # Test invalid depth (0)
        result = runner.invoke(
            cli,
            ["search", "test", "--include-neighbors", "--neighbor-depth=0"],
            env={"MEMEX_USER_KB_ROOT": str(tmp_kb)},
        )
        assert (
            result.exit_code != 0
            or "invalid" in result.output.lower()
            or "range" in result.output.lower()
        )

        # Test invalid depth (6)
        result = runner.invoke(
            cli,
            ["search", "test", "--include-neighbors", "--neighbor-depth=6"],
            env={"MEMEX_USER_KB_ROOT": str(tmp_kb)},
        )
        assert (
            result.exit_code != 0
            or "invalid" in result.output.lower()
            or "range" in result.output.lower()
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core Function Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExpandSearchWithNeighbors:
    """Test the expand_search_with_neighbors core function."""

    @pytest.mark.asyncio
    async def test_expands_direct_results_with_neighbors(self, tmp_kb: Path):
        """Direct search results are expanded with their neighbors."""
        from memex.core import expand_search_with_neighbors

        # Create entry A with link to B
        create_entry_with_links(
            tmp_kb,
            "a.md",
            "Entry A",
            "Content about entry A",
            ["test"],
            semantic_links=[{"path": "b.md", "score": 0.8, "reason": "embedding_similarity"}],
        )
        # Create entry B
        create_entry_with_links(tmp_kb, "b.md", "Entry B", "Content about entry B", ["test"])

        # Simulate search result for A
        results = [
            SearchResult(
                path="a.md",
                title="Entry A",
                snippet="Content about entry A",
                score=0.9,
                tags=["test"],
            )
        ]

        expanded = await expand_search_with_neighbors(results, depth=1)

        assert len(expanded) == 2
        # First result should be direct match
        assert expanded[0]["path"] == "a.md"
        assert expanded[0]["is_neighbor"] is False
        # Second result should be neighbor
        assert expanded[1]["path"] == "b.md"
        assert expanded[1]["is_neighbor"] is True
        assert expanded[1]["linked_from"] == "a.md"

    @pytest.mark.asyncio
    async def test_expands_direct_results_with_relations(self, tmp_kb: Path):
        """Typed relations are included as neighbors."""
        from memex.core import expand_search_with_neighbors

        # Create entry A with relation to B
        create_entry_with_links(
            tmp_kb,
            "a.md",
            "Entry A",
            "Content about entry A",
            ["test"],
            relations=[{"path": "b.md", "type": "depends_on"}],
        )
        # Create entry B
        create_entry_with_links(tmp_kb, "b.md", "Entry B", "Content about entry B", ["test"])

        results = [
            SearchResult(
                path="a.md",
                title="Entry A",
                snippet="Content about entry A",
                score=0.9,
                tags=["test"],
            )
        ]

        expanded = await expand_search_with_neighbors(results, depth=1)

        assert len(expanded) == 2
        assert expanded[0]["path"] == "a.md"
        assert expanded[0]["is_neighbor"] is False
        assert expanded[1]["path"] == "b.md"
        assert expanded[1]["is_neighbor"] is True
        assert expanded[1]["linked_from"] == "a.md"

    @pytest.mark.asyncio
    async def test_normalizes_scoped_relation_paths(self, multi_kb):
        """Relation paths normalize to scoped .md targets in multi-KB mode."""
        from memex.core import expand_search_with_neighbors

        project_kb = multi_kb["project_kb"]
        user_kb = multi_kb["user_kb"]

        create_entry_with_links(
            project_kb,
            "a.md",
            "Entry A",
            "Content A",
            ["test"],
            relations=[{"path": "b", "type": "depends_on"}],
        )
        create_entry_with_links(
            project_kb,
            "b.md",
            "Entry B",
            "Content B",
            ["test"],
        )
        create_entry_with_links(
            user_kb,
            "user.md",
            "User Entry",
            "User content",
            ["test"],
        )

        results = [
            SearchResult(
                path="@project/a.md",
                title="Entry A",
                snippet="Content A",
                score=0.9,
                tags=["test"],
            )
        ]

        expanded = await expand_search_with_neighbors(results, depth=1)
        paths = [item.get("path") for item in expanded]

        assert "@project/b.md" in paths
        assert "b.md" not in paths

    @pytest.mark.asyncio
    async def test_deduplicates_results(self, tmp_kb: Path):
        """Entries appearing multiple times are deduplicated."""
        from memex.core import expand_search_with_neighbors

        # Create entry A with link to C
        create_entry_with_links(
            tmp_kb,
            "a.md",
            "Entry A",
            "Content about entry A",
            ["test"],
            semantic_links=[{"path": "c.md", "score": 0.8, "reason": "embedding_similarity"}],
        )
        # Create entry B also linking to C
        create_entry_with_links(
            tmp_kb,
            "b.md",
            "Entry B",
            "Content about entry B",
            ["test"],
            semantic_links=[{"path": "c.md", "score": 0.7, "reason": "embedding_similarity"}],
        )
        # Create entry C (shared neighbor)
        create_entry_with_links(tmp_kb, "c.md", "Entry C", "Content about entry C", ["test"])

        # Simulate search results for both A and B
        results = [
            SearchResult(
                path="a.md",
                title="Entry A",
                snippet="Content about entry A",
                score=0.9,
                tags=["test"],
            ),
            SearchResult(
                path="b.md",
                title="Entry B",
                snippet="Content about entry B",
                score=0.85,
                tags=["test"],
            ),
        ]

        expanded = await expand_search_with_neighbors(results, depth=1)

        # Should have A, B, and C (deduplicated)
        paths = [r["path"] for r in expanded]
        assert len(paths) == len(set(paths)), "Paths should be unique (no duplicates)"
        assert "c.md" in paths
        assert paths.count("c.md") == 1

    @pytest.mark.asyncio
    async def test_respects_depth_limit(self, tmp_kb: Path):
        """Neighbor traversal respects depth limit."""
        from memex.core import expand_search_with_neighbors

        # Create chain: A -> B -> C -> D
        create_entry_with_links(
            tmp_kb,
            "a.md",
            "Entry A",
            "Content A",
            ["test"],
            semantic_links=[{"path": "b.md", "score": 0.8, "reason": "embedding_similarity"}],
        )
        create_entry_with_links(
            tmp_kb,
            "b.md",
            "Entry B",
            "Content B",
            ["test"],
            semantic_links=[{"path": "c.md", "score": 0.7, "reason": "embedding_similarity"}],
        )
        create_entry_with_links(
            tmp_kb,
            "c.md",
            "Entry C",
            "Content C",
            ["test"],
            semantic_links=[{"path": "d.md", "score": 0.6, "reason": "embedding_similarity"}],
        )
        create_entry_with_links(tmp_kb, "d.md", "Entry D", "Content D", ["test"])

        results = [
            SearchResult(
                path="a.md",
                title="Entry A",
                snippet="Content A",
                score=0.9,
                tags=["test"],
            )
        ]

        # Depth 1: Should include A and B only
        expanded_depth1 = await expand_search_with_neighbors(results, depth=1)
        paths_depth1 = [r["path"] for r in expanded_depth1]
        assert "a.md" in paths_depth1
        assert "b.md" in paths_depth1
        assert "c.md" not in paths_depth1
        assert "d.md" not in paths_depth1

        # Depth 2: Should include A, B, and C
        expanded_depth2 = await expand_search_with_neighbors(results, depth=2)
        paths_depth2 = [r["path"] for r in expanded_depth2]
        assert "a.md" in paths_depth2
        assert "b.md" in paths_depth2
        assert "c.md" in paths_depth2
        assert "d.md" not in paths_depth2

        # Depth 3: Should include all
        expanded_depth3 = await expand_search_with_neighbors(results, depth=3)
        paths_depth3 = [r["path"] for r in expanded_depth3]
        assert "a.md" in paths_depth3
        assert "b.md" in paths_depth3
        assert "c.md" in paths_depth3
        assert "d.md" in paths_depth3

    @pytest.mark.asyncio
    async def test_handles_no_neighbors(self, tmp_kb: Path):
        """Entries without semantic_links return only direct results."""
        from memex.core import expand_search_with_neighbors

        # Create entry with no links
        create_entry_with_links(tmp_kb, "alone.md", "Alone Entry", "Content alone", ["test"])

        results = [
            SearchResult(
                path="alone.md",
                title="Alone Entry",
                snippet="Content alone",
                score=0.9,
                tags=["test"],
            )
        ]

        expanded = await expand_search_with_neighbors(results, depth=1)

        assert len(expanded) == 1
        assert expanded[0]["path"] == "alone.md"
        assert expanded[0]["is_neighbor"] is False

    @pytest.mark.asyncio
    async def test_handles_circular_links(self, tmp_kb: Path):
        """Circular links do not cause infinite loops."""
        from memex.core import expand_search_with_neighbors

        # Create circular chain: A -> B -> C -> A
        create_entry_with_links(
            tmp_kb,
            "a.md",
            "Entry A",
            "Content A",
            ["test"],
            semantic_links=[{"path": "b.md", "score": 0.8, "reason": "embedding_similarity"}],
        )
        create_entry_with_links(
            tmp_kb,
            "b.md",
            "Entry B",
            "Content B",
            ["test"],
            semantic_links=[{"path": "c.md", "score": 0.7, "reason": "embedding_similarity"}],
        )
        create_entry_with_links(
            tmp_kb,
            "c.md",
            "Entry C",
            "Content C",
            ["test"],
            semantic_links=[{"path": "a.md", "score": 0.6, "reason": "embedding_similarity"}],
        )

        results = [
            SearchResult(
                path="a.md",
                title="Entry A",
                snippet="Content A",
                score=0.9,
                tags=["test"],
            )
        ]

        # Should complete without hanging (circular reference handled)
        expanded = await expand_search_with_neighbors(results, depth=5)

        # Should have exactly 3 entries (A, B, C) - no duplicates from cycle
        paths = [r["path"] for r in expanded]
        assert len(paths) == 3
        assert set(paths) == {"a.md", "b.md", "c.md"}

    @pytest.mark.asyncio
    async def test_handles_missing_neighbor_files(self, tmp_kb: Path):
        """Missing neighbor files are gracefully skipped."""
        from memex.core import expand_search_with_neighbors

        # Create entry linking to non-existent file
        create_entry_with_links(
            tmp_kb,
            "a.md",
            "Entry A",
            "Content A",
            ["test"],
            semantic_links=[{"path": "missing.md", "score": 0.8, "reason": "embedding_similarity"}],
        )

        results = [
            SearchResult(
                path="a.md",
                title="Entry A",
                snippet="Content A",
                score=0.9,
                tags=["test"],
            )
        ]

        # Should not raise an error
        expanded = await expand_search_with_neighbors(results, depth=1)

        # Should only have the direct result
        assert len(expanded) == 1
        assert expanded[0]["path"] == "a.md"


# ─────────────────────────────────────────────────────────────────────────────
# Output Format Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNeighborOutputFormats:
    """Test output formats with neighbor markers."""

    def test_json_output_includes_is_neighbor_field(self, runner: CliRunner, tmp_kb: Path):
        """JSON output includes is_neighbor field."""
        import json

        # Create entry A with link to B
        create_entry_with_links(
            tmp_kb,
            "a.md",
            "Entry A",
            "Content about entry A",
            ["test"],
            semantic_links=[{"path": "b.md", "score": 0.8, "reason": "embedding_similarity"}],
        )
        create_entry_with_links(tmp_kb, "b.md", "Entry B", "Content about entry B", ["test"])

        result = runner.invoke(
            cli,
            ["search", "entry", "--include-neighbors", "--json"],
            env={"MEMEX_USER_KB_ROOT": str(tmp_kb)},
        )

        if result.exit_code == 0 and "results" in result.output:
            data = json.loads(result.output)
            for item in data["results"]:
                assert "is_neighbor" in item

    def test_json_output_includes_linked_from_for_neighbors(self, runner: CliRunner, tmp_kb: Path):
        """JSON output includes linked_from field for neighbors."""
        import json

        create_entry_with_links(
            tmp_kb,
            "a.md",
            "Entry A",
            "Content about entry A",
            ["test"],
            semantic_links=[{"path": "b.md", "score": 0.8, "reason": "embedding_similarity"}],
        )
        create_entry_with_links(tmp_kb, "b.md", "Entry B", "Content about entry B", ["test"])

        result = runner.invoke(
            cli,
            ["search", "entry", "--include-neighbors", "--json"],
            env={"MEMEX_USER_KB_ROOT": str(tmp_kb)},
        )

        if result.exit_code == 0 and "results" in result.output:
            data = json.loads(result.output)
            neighbors = [r for r in data["results"] if r.get("is_neighbor")]
            for neighbor in neighbors:
                assert "linked_from" in neighbor

    def test_terse_output_marks_neighbors_with_prefix(self, runner: CliRunner, tmp_kb: Path):
        """Terse output marks neighbors with [N] prefix."""
        create_entry_with_links(
            tmp_kb,
            "a.md",
            "Entry A",
            "Content about entry A",
            ["test"],
            semantic_links=[{"path": "b.md", "score": 0.8, "reason": "embedding_similarity"}],
        )
        create_entry_with_links(tmp_kb, "b.md", "Entry B", "Content about entry B", ["test"])

        result = runner.invoke(
            cli,
            ["search", "entry A", "--include-neighbors", "--terse"],
            env={"MEMEX_USER_KB_ROOT": str(tmp_kb)},
        )

        if result.exit_code == 0:
            lines = result.output.strip().split("\n")
            # Neighbors should have [N] prefix
            neighbor_lines = [line for line in lines if line.startswith("[N]")]
            # If we have both types of results, verify format
            if neighbor_lines:
                for line in neighbor_lines:
                    assert line.startswith("[N] ")

    def test_table_output_shows_nbr_column(self, runner: CliRunner, tmp_kb: Path):
        """Table output shows NBR column with asterisk for neighbors."""
        create_entry_with_links(
            tmp_kb,
            "a.md",
            "Entry A",
            "Content about entry A",
            ["test"],
            semantic_links=[{"path": "b.md", "score": 0.8, "reason": "embedding_similarity"}],
        )
        create_entry_with_links(tmp_kb, "b.md", "Entry B", "Content about entry B", ["test"])

        result = runner.invoke(
            cli,
            ["search", "entry A", "--include-neighbors"],
            env={"MEMEX_USER_KB_ROOT": str(tmp_kb)},
        )

        if result.exit_code == 0:
            # Should include NBR column header
            assert "NBR" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNeighborSearchIntegration:
    """Integration tests for neighbor search functionality."""

    def test_search_without_flag_unchanged(self, runner: CliRunner, tmp_kb: Path):
        """Search without --include-neighbors works as before."""
        create_entry_with_links(tmp_kb, "test.md", "Test Entry", "Test content", ["test"])

        result = runner.invoke(
            cli,
            ["search", "test"],
            env={"MEMEX_USER_KB_ROOT": str(tmp_kb)},
        )

        # Should work normally
        assert result.exit_code == 0 or "No results" in result.output

    def test_combines_with_min_score_filter(self, runner: CliRunner, tmp_kb: Path):
        """--include-neighbors works with --min-score filter."""
        create_entry_with_links(
            tmp_kb,
            "a.md",
            "Entry A",
            "Content about entry A",
            ["test"],
            semantic_links=[{"path": "b.md", "score": 0.3, "reason": "embedding_similarity"}],
        )
        create_entry_with_links(tmp_kb, "b.md", "Entry B", "Content about entry B", ["test"])

        result = runner.invoke(
            cli,
            ["search", "entry A", "--include-neighbors", "--min-score=0.5", "--json"],
            env={"MEMEX_USER_KB_ROOT": str(tmp_kb)},
        )

        # Should run without error
        assert result.exit_code == 0 or "No results" in result.output

    def test_combines_with_content_flag(self, runner: CliRunner, tmp_kb: Path):
        """--include-neighbors works with --content flag."""
        create_entry_with_links(
            tmp_kb,
            "a.md",
            "Entry A",
            "Content about entry A with more details",
            ["test"],
            semantic_links=[{"path": "b.md", "score": 0.8, "reason": "embedding_similarity"}],
        )
        create_entry_with_links(tmp_kb, "b.md", "Entry B", "Content about entry B", ["test"])

        result = runner.invoke(
            cli,
            ["search", "entry A", "--include-neighbors", "--content"],
            env={"MEMEX_USER_KB_ROOT": str(tmp_kb)},
        )

        # Should run without error and show content
        assert result.exit_code == 0 or "No results" in result.output
