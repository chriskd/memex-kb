"""Search quality evaluation helpers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .indexer import HybridSearcher
from .models import QualityDetail, QualityReport
from .parser import ParseError, parse_entry

logger = logging.getLogger(__name__)

# Legacy hardcoded queries - kept for backwards compatibility
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


@dataclass
class EvalQuery:
    """A dynamically generated evaluation query."""

    query: str
    expected: list[str]
    query_type: Literal["title", "alias", "tag"]
    source_entry: str | None = None  # Path to entry that generated this query


@dataclass
class EvalSet:
    """A set of evaluation queries generated from KB content."""

    queries: list[EvalQuery] = field(default_factory=list)
    title_queries: int = 0
    alias_queries: int = 0
    tag_queries: int = 0
    skipped_entries: int = 0


def generate_eval_set(
    kb_root: Path,
    *,
    include_titles: bool = True,
    include_aliases: bool = True,
    include_tags: bool = False,  # Tags off by default - can produce many queries
    max_entries: int | None = None,
) -> EvalSet:
    """Generate evaluation queries dynamically from KB content.

    Args:
        kb_root: Root directory of the knowledge base.
        include_titles: Generate queries from entry titles.
        include_aliases: Generate queries from entry aliases.
        include_tags: Generate queries from entry tags.
        max_entries: Maximum number of entries to process (for large KBs).

    Returns:
        EvalSet containing generated queries and statistics.

    Note:
        Title queries expect the entry to appear in search results.
        Alias queries expect the aliased entry to appear.
        Tag queries are more permissive - any entry with that tag is valid.
    """
    eval_set = EvalSet()
    entries_processed = 0

    for md_file in kb_root.rglob("*.md"):
        # Skip hidden files and special directories
        if any(part.startswith((".", "_")) for part in md_file.parts):
            continue

        if max_entries and entries_processed >= max_entries:
            break

        try:
            metadata, _, _ = parse_entry(md_file)
        except ParseError as e:
            logger.debug("Skipping %s: %s", md_file, e)
            eval_set.skipped_entries += 1
            continue

        # Convert to relative path for matching search results
        rel_path = str(md_file.relative_to(kb_root))

        # Title-based queries
        if include_titles and metadata.title:
            eval_set.queries.append(
                EvalQuery(
                    query=metadata.title,
                    expected=[rel_path],
                    query_type="title",
                    source_entry=rel_path,
                )
            )
            eval_set.title_queries += 1

        # Alias-based queries
        if include_aliases and metadata.aliases:
            for alias in metadata.aliases:
                eval_set.queries.append(
                    EvalQuery(
                        query=alias,
                        expected=[rel_path],
                        query_type="alias",
                        source_entry=rel_path,
                    )
                )
                eval_set.alias_queries += 1

        entries_processed += 1

    # Tag-based queries are handled separately since they need all entries first
    if include_tags:
        tag_to_entries: dict[str, list[str]] = {}
        for md_file in kb_root.rglob("*.md"):
            if any(part.startswith((".", "_")) for part in md_file.parts):
                continue
            try:
                metadata, _, _ = parse_entry(md_file)
                rel_path = str(md_file.relative_to(kb_root))
                for tag in metadata.tags:
                    tag_to_entries.setdefault(tag, []).append(rel_path)
            except ParseError:
                continue

        for tag, entries in tag_to_entries.items():
            eval_set.queries.append(
                EvalQuery(
                    query=tag,
                    expected=entries,
                    query_type="tag",
                    source_entry=None,
                )
            )
            eval_set.tag_queries += 1

    return eval_set


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


@dataclass
class EvalReport:
    """Extended evaluation report with per-type metrics."""

    overall_accuracy: float
    title_accuracy: float
    alias_accuracy: float
    tag_accuracy: float
    total_queries: int
    title_queries: int
    alias_queries: int
    tag_queries: int
    skipped_entries: int
    details: list[QualityDetail]


def run_eval_set(
    searcher: HybridSearcher,
    eval_set: EvalSet,
    *,
    limit: int = 5,
    cutoff: int = 3,
) -> EvalReport:
    """Run evaluation against a generated eval set.

    Args:
        searcher: The HybridSearcher to evaluate.
        eval_set: Generated evaluation queries.
        limit: Number of results to fetch per query.
        cutoff: Rank threshold for success (entry must appear within top N).

    Returns:
        EvalReport with per-type accuracy breakdowns.
    """
    details: list[QualityDetail] = []
    successes_by_type = {"title": 0, "alias": 0, "tag": 0}
    counts_by_type = {"title": 0, "alias": 0, "tag": 0}

    for eval_query in eval_set.queries:
        query = eval_query.query
        expected = eval_query.expected
        query_type = eval_query.query_type

        counts_by_type[query_type] += 1

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
            successes_by_type[query_type] += 1

        details.append(
            QualityDetail(
                query=query,
                expected=expected,
                hits=result_paths,
                found=found,
                best_rank=best_rank,
            )
        )

    # Calculate per-type accuracies
    def safe_div(num: int, denom: int) -> float:
        return num / denom if denom > 0 else 1.0

    total = len(eval_set.queries)
    total_successes = sum(successes_by_type.values())

    return EvalReport(
        overall_accuracy=safe_div(total_successes, total),
        title_accuracy=safe_div(successes_by_type["title"], counts_by_type["title"]),
        alias_accuracy=safe_div(successes_by_type["alias"], counts_by_type["alias"]),
        tag_accuracy=safe_div(successes_by_type["tag"], counts_by_type["tag"]),
        total_queries=total,
        title_queries=counts_by_type["title"],
        alias_queries=counts_by_type["alias"],
        tag_queries=counts_by_type["tag"],
        skipped_entries=eval_set.skipped_entries,
        details=details,
    )


def evaluate_kb(
    kb_root: Path,
    searcher: HybridSearcher,
    *,
    include_titles: bool = True,
    include_aliases: bool = True,
    include_tags: bool = False,
    max_entries: int | None = None,
    limit: int = 5,
    cutoff: int = 3,
) -> EvalReport:
    """Generate eval set from KB and run evaluation in one step.

    This is the main entry point for KB-agnostic search evaluation.

    Args:
        kb_root: Root directory of the knowledge base.
        searcher: The HybridSearcher to evaluate.
        include_titles: Generate queries from entry titles.
        include_aliases: Generate queries from entry aliases.
        include_tags: Generate queries from entry tags.
        max_entries: Maximum number of entries to process.
        limit: Number of results to fetch per query.
        cutoff: Rank threshold for success.

    Returns:
        EvalReport with accuracy metrics.

    Example:
        >>> report = evaluate_kb(Path("/path/to/kb"), searcher)
        >>> print(f"Title recall: {report.title_accuracy:.0%}")
        >>> print(f"Alias recall: {report.alias_accuracy:.0%}")
    """
    eval_set = generate_eval_set(
        kb_root,
        include_titles=include_titles,
        include_aliases=include_aliases,
        include_tags=include_tags,
        max_entries=max_entries,
    )
    return run_eval_set(searcher, eval_set, limit=limit, cutoff=cutoff)
