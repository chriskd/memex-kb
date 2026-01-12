"""Tests for search quality evaluation."""

from unittest.mock import Mock


from memex.evaluation import (
    EVAL_QUERIES,
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
