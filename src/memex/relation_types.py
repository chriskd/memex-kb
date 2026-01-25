"""Canonical relation types and helpers for linting."""

from __future__ import annotations

import re


CANONICAL_RELATION_TYPES: dict[str, str] = {
    "depends_on": "A requires B to function or make progress.",
    "implements": "A implements the spec, API, or plan described by B.",
    "extends": "A builds on or extends B with additional behavior.",
    "documents": "A documents or explains B (guides, readmes, specs).",
    "references": "A cites B for supporting detail or context.",
    "blocks": "A blocks B from proceeding (directional).",
    "related": "A is generally related to B when no stronger type fits.",
}

RELATION_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def normalize_relation_type(value: str) -> str:
    """Normalize a relation type to snake_case for comparison."""
    cleaned = value.strip().lower()
    cleaned = re.sub(r"[\s\-]+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_]", "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned
