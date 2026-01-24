"""Search quality evaluation helpers."""

from __future__ import annotations

from collections.abc import Sequence

from .indexer import HybridSearcher
from .models import QualityDetail, QualityReport

EVAL_QUERIES: Sequence[dict] = (
    {
        "query": "python tooling",
        "expected": ["development/python-tooling.md"],
    },
    {
        "query": "dokploy deployment",
        "expected": ["devops/deployment.md"],
    },
    {
        "query": "dockerfile uv",
        "expected": ["devops/docker-patterns.md"],
    },
    {
        "query": "devcontainer setup",
        "expected": ["infrastructure/devcontainers.md"],
    },
    {
        "query": "dns troubleshooting",
        "expected": ["troubleshooting/dns-resolution-issues.md"],
    },
)


def run_quality_checks(searcher: HybridSearcher, limit: int = 5, cutoff: int = 3) -> QualityReport:
    """Evaluate search accuracy against a fixed query set."""

    details: list[QualityDetail] = []
    successes = 0

    for case in EVAL_QUERIES:
        query = case["query"]
        expected = case["expected"]

        results = searcher.search(query, limit=limit, mode="hybrid")
        result_paths = [res.path for res in results]

        best_rank: int | None = None
        found = False

        for exp in expected:
            if exp in result_paths:
                rank = result_paths.index(exp) + 1
                best_rank = rank if best_rank is None else min(best_rank, rank)
                if rank <= cutoff:
                    found = True

        if found:
            successes += 1

        details.append(
            QualityDetail(
                query=query,
                expected=expected,
                hits=result_paths,
                found=found,
                best_rank=best_rank,
            )
        )

    total = len(EVAL_QUERIES)
    accuracy = successes / total if total else 1.0

    return QualityReport(accuracy=accuracy, total_queries=total, details=details)
