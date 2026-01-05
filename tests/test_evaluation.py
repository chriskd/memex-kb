"""Tests for search quality evaluation."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from memex.evaluation import (
    EVAL_QUERIES,
    EvalQuery,
    EvalSet,
    generate_eval_set,
    run_eval_set,
    run_quality_checks,
)
from memex.models import SearchResult


def _make_result(path: str, score: float = 1.0) -> SearchResult:
    """Create a minimal SearchResult for testing."""
    return SearchResult(path=path, title="Test", snippet="...", score=score, tags=[])


class TestRunQualityChecks:
    """Tests for run_quality_checks function."""

    def test_all_queries_fail_when_no_matches(self):
        """When searcher returns unrelated docs, all queries fail."""
        searcher = Mock()
        searcher.search.return_value = [
            _make_result("unrelated/doc1.md"),
            _make_result("unrelated/doc2.md"),
        ]

        report = run_quality_checks(searcher, limit=5, cutoff=3)

        assert report.accuracy == 0.0
        assert report.total_queries == len(EVAL_QUERIES)
        assert all(not d.found for d in report.details)
        assert all(d.best_rank is None for d in report.details)

    def test_query_succeeds_when_expected_doc_in_top_results(self):
        """Query passes when expected doc is within cutoff."""
        searcher = Mock()

        # Return the correct expected doc for each query
        def mock_search(query, limit, mode):
            for case in EVAL_QUERIES:
                if case["query"] == query:
                    return [_make_result(case["expected"][0]), _make_result("other/doc.md")]
            return []

        searcher.search.side_effect = mock_search

        report = run_quality_checks(searcher, limit=5, cutoff=3)

        # All queries should pass with expected doc at rank 1
        assert report.accuracy == 1.0
        assert all(d.found for d in report.details)
        assert all(d.best_rank == 1 for d in report.details)

    def test_query_fails_when_expected_doc_outside_cutoff(self):
        """Query fails when expected doc is beyond cutoff threshold."""
        searcher = Mock()

        # Put expected doc at rank 4, but cutoff is 3
        def mock_search(query, limit, mode):
            for case in EVAL_QUERIES:
                if case["query"] == query:
                    return [
                        _make_result("other/doc1.md"),
                        _make_result("other/doc2.md"),
                        _make_result("other/doc3.md"),
                        _make_result(case["expected"][0]),  # rank 4
                    ]
            return []

        searcher.search.side_effect = mock_search

        report = run_quality_checks(searcher, limit=5, cutoff=3)

        # Doc found but outside cutoff - should fail
        assert report.accuracy == 0.0
        for detail in report.details:
            assert not detail.found
            assert detail.best_rank == 4  # Found at rank 4

    def test_best_rank_tracks_highest_ranked_expected_doc(self):
        """When multiple expected docs match, best_rank is the lowest."""
        searcher = Mock()

        # Simulate a query with multiple expected paths
        def mock_search(query, limit, mode):
            # Return docs at various ranks
            return [
                _make_result("other/doc.md"),  # rank 1
                _make_result("development/python-tooling.md"),  # rank 2
                _make_result("devops/deployment.md"),  # rank 3
            ]

        searcher.search.side_effect = mock_search

        report = run_quality_checks(searcher, limit=5, cutoff=3)

        # Check the first query (python tooling) - expected at rank 2
        first_detail = report.details[0]
        assert first_detail.found is True
        assert first_detail.best_rank == 2

    def test_partial_success_calculates_correct_accuracy(self):
        """Accuracy reflects proportion of successful queries."""
        searcher = Mock()
        call_count = [0]

        def mock_search(query, limit, mode):
            call_count[0] += 1
            # Make first query succeed, rest fail
            if call_count[0] == 1:
                return [_make_result(EVAL_QUERIES[0]["expected"][0])]
            return [_make_result("unrelated/doc.md")]

        searcher.search.side_effect = mock_search

        report = run_quality_checks(searcher, limit=5, cutoff=3)

        # Only 1 of N queries succeeded
        expected_accuracy = 1 / len(EVAL_QUERIES)
        assert report.accuracy == expected_accuracy
        assert report.details[0].found is True
        assert all(not d.found for d in report.details[1:])

    def test_custom_limit_passed_to_searcher(self):
        """The limit parameter is passed to searcher.search."""
        searcher = Mock()
        searcher.search.return_value = []

        run_quality_checks(searcher, limit=10, cutoff=3)

        for call in searcher.search.call_args_list:
            assert call.kwargs["limit"] == 10

    def test_uses_hybrid_mode(self):
        """Search is performed in hybrid mode."""
        searcher = Mock()
        searcher.search.return_value = []

        run_quality_checks(searcher, limit=5, cutoff=3)

        for call in searcher.search.call_args_list:
            assert call.kwargs["mode"] == "hybrid"

    def test_details_contain_query_info(self):
        """Each detail contains the query and expected paths."""
        searcher = Mock()
        searcher.search.return_value = [_make_result("some/doc.md")]

        report = run_quality_checks(searcher, limit=5, cutoff=3)

        for i, detail in enumerate(report.details):
            assert detail.query == EVAL_QUERIES[i]["query"]
            assert detail.expected == EVAL_QUERIES[i]["expected"]
            assert detail.hits == ["some/doc.md"]

    def test_empty_eval_queries_returns_perfect_accuracy(self, monkeypatch):
        """Edge case: empty query set returns 1.0 accuracy."""
        monkeypatch.setattr("memex.evaluation.EVAL_QUERIES", [])
        searcher = Mock()

        report = run_quality_checks(searcher, limit=5, cutoff=3)

        assert report.accuracy == 1.0
        assert report.total_queries == 0
        assert report.details == []


class TestGenerateEvalSet:
    """Tests for generate_eval_set function."""

    def test_generates_title_queries(self, tmp_path):
        """Entries with titles generate title queries."""
        entry = tmp_path / "test-entry.md"
        entry.write_text(
            """---
title: My Test Entry
tags: [testing]
created: 2024-01-01
---
Content here.
"""
        )

        eval_set = generate_eval_set(tmp_path, include_titles=True, include_aliases=False)

        assert eval_set.title_queries == 1
        assert len(eval_set.queries) == 1
        q = eval_set.queries[0]
        assert q.query == "My Test Entry"
        assert q.expected == ["test-entry.md"]
        assert q.query_type == "title"
        assert q.source_entry == "test-entry.md"

    def test_generates_alias_queries(self, tmp_path):
        """Entries with aliases generate alias queries."""
        entry = tmp_path / "entry.md"
        entry.write_text(
            """---
title: Main Title
tags: [test]
created: 2024-01-01
aliases: [Alias One, Alias Two]
---
Content.
"""
        )

        eval_set = generate_eval_set(tmp_path, include_titles=False, include_aliases=True)

        assert eval_set.alias_queries == 2
        assert len(eval_set.queries) == 2
        aliases = {q.query for q in eval_set.queries}
        assert aliases == {"Alias One", "Alias Two"}
        for q in eval_set.queries:
            assert q.query_type == "alias"
            assert q.expected == ["entry.md"]

    def test_generates_tag_queries(self, tmp_path):
        """Tag queries expect entries with that tag."""
        # Create two entries with overlapping tags
        (tmp_path / "entry1.md").write_text(
            """---
title: Entry One
tags: [python, testing]
created: 2024-01-01
---
Content.
"""
        )
        (tmp_path / "entry2.md").write_text(
            """---
title: Entry Two
tags: [python, deployment]
created: 2024-01-01
---
Content.
"""
        )

        eval_set = generate_eval_set(
            tmp_path, include_titles=False, include_aliases=False, include_tags=True
        )

        assert eval_set.tag_queries == 3  # python, testing, deployment
        tag_queries = {q.query: q.expected for q in eval_set.queries}
        assert "python" in tag_queries
        assert set(tag_queries["python"]) == {"entry1.md", "entry2.md"}
        assert set(tag_queries["testing"]) == {"entry1.md"}
        assert set(tag_queries["deployment"]) == {"entry2.md"}

    def test_skips_hidden_directories(self, tmp_path):
        """Entries in hidden directories are skipped."""
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.md").write_text(
            """---
title: Secret Entry
tags: [hidden]
created: 2024-01-01
---
"""
        )
        visible = tmp_path / "visible.md"
        visible.write_text(
            """---
title: Visible Entry
tags: [visible]
created: 2024-01-01
---
"""
        )

        eval_set = generate_eval_set(tmp_path)

        assert eval_set.title_queries == 1
        assert eval_set.queries[0].query == "Visible Entry"

    def test_skips_invalid_entries(self, tmp_path):
        """Invalid entries are skipped and counted."""
        # Valid entry
        (tmp_path / "valid.md").write_text(
            """---
title: Valid
tags: [test]
created: 2024-01-01
---
Content.
"""
        )
        # Invalid entry (missing frontmatter)
        (tmp_path / "invalid.md").write_text("No frontmatter here")

        eval_set = generate_eval_set(tmp_path)

        assert eval_set.title_queries == 1
        assert eval_set.skipped_entries == 1

    def test_respects_max_entries(self, tmp_path):
        """max_entries limits the number of processed entries."""
        for i in range(5):
            (tmp_path / f"entry{i}.md").write_text(
                f"""---
title: Entry {i}
tags: [test]
created: 2024-01-01
---
"""
            )

        eval_set = generate_eval_set(tmp_path, max_entries=2)

        # Should have stopped after 2 entries
        assert eval_set.title_queries == 2

    def test_nested_directories(self, tmp_path):
        """Entries in nested directories are found."""
        subdir = tmp_path / "category" / "subcategory"
        subdir.mkdir(parents=True)
        (subdir / "nested.md").write_text(
            """---
title: Nested Entry
tags: [nested]
created: 2024-01-01
---
"""
        )

        eval_set = generate_eval_set(tmp_path)

        assert eval_set.title_queries == 1
        assert eval_set.queries[0].expected == ["category/subcategory/nested.md"]

    def test_empty_kb_returns_empty_eval_set(self, tmp_path):
        """Empty KB returns empty eval set."""
        eval_set = generate_eval_set(tmp_path)

        assert eval_set.queries == []
        assert eval_set.title_queries == 0
        assert eval_set.alias_queries == 0
        assert eval_set.tag_queries == 0


class TestRunEvalSet:
    """Tests for run_eval_set function."""

    def test_calculates_per_type_accuracy(self):
        """Accuracy is calculated separately for each query type."""
        searcher = Mock()

        # Make title queries succeed, alias queries fail
        def mock_search(query, limit, mode):
            if query.startswith("Title"):
                return [_make_result("entry.md")]  # Success
            return [_make_result("wrong.md")]  # Fail

        searcher.search.side_effect = mock_search

        eval_set = EvalSet(
            queries=[
                EvalQuery(query="Title Query", expected=["entry.md"], query_type="title"),
                EvalQuery(query="Alias Query", expected=["entry.md"], query_type="alias"),
            ],
            title_queries=1,
            alias_queries=1,
        )

        report = run_eval_set(searcher, eval_set)

        assert report.title_accuracy == 1.0
        assert report.alias_accuracy == 0.0
        assert report.overall_accuracy == 0.5
        assert report.title_queries == 1
        assert report.alias_queries == 1

    def test_tag_query_any_match_succeeds(self):
        """Tag queries succeed if any expected entry is found."""
        searcher = Mock()
        searcher.search.return_value = [_make_result("entry2.md")]

        eval_set = EvalSet(
            queries=[
                EvalQuery(
                    query="python",
                    expected=["entry1.md", "entry2.md", "entry3.md"],
                    query_type="tag",
                ),
            ],
            tag_queries=1,
        )

        report = run_eval_set(searcher, eval_set)

        assert report.tag_accuracy == 1.0
        assert report.details[0].found is True

    def test_empty_eval_set_returns_perfect_scores(self):
        """Empty eval set returns 1.0 accuracy (no failures)."""
        searcher = Mock()
        eval_set = EvalSet()

        report = run_eval_set(searcher, eval_set)

        assert report.overall_accuracy == 1.0
        assert report.title_accuracy == 1.0
        assert report.alias_accuracy == 1.0
        assert report.tag_accuracy == 1.0
        assert report.total_queries == 0

    def test_respects_cutoff_threshold(self):
        """Results beyond cutoff threshold are not counted as found."""
        searcher = Mock()
        # Expected at rank 4, cutoff is 3
        searcher.search.return_value = [
            _make_result("other1.md"),
            _make_result("other2.md"),
            _make_result("other3.md"),
            _make_result("expected.md"),
        ]

        eval_set = EvalSet(
            queries=[
                EvalQuery(query="test", expected=["expected.md"], query_type="title"),
            ],
            title_queries=1,
        )

        report = run_eval_set(searcher, eval_set, cutoff=3)

        assert report.overall_accuracy == 0.0
        assert report.details[0].best_rank == 4
        assert report.details[0].found is False

    def test_uses_hybrid_mode(self):
        """Searches are performed in hybrid mode."""
        searcher = Mock()
        searcher.search.return_value = []

        eval_set = EvalSet(
            queries=[EvalQuery(query="test", expected=["entry.md"], query_type="title")],
            title_queries=1,
        )

        run_eval_set(searcher, eval_set)

        searcher.search.assert_called_with("test", limit=5, mode="hybrid")

    def test_preserves_skipped_entries_count(self):
        """Skipped entries count is preserved from eval set."""
        searcher = Mock()
        searcher.search.return_value = []

        eval_set = EvalSet(skipped_entries=5)

        report = run_eval_set(searcher, eval_set)

        assert report.skipped_entries == 5
