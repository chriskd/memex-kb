"""Search index generation for client-side Lunr.js search.

Generates a JSON file containing document corpus and metadata
for use with Lunr.js client-side search.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generator import EntryData


def build_search_index(entries: dict[str, EntryData]) -> str:
    """Build Lunr.js compatible search index.

    Generates a JSON file containing:
    1. Document corpus for Lunr.js indexing (id, title, tags, content)
    2. Document metadata for displaying results (title, tags, path)

    Args:
        entries: Dict of path -> EntryData

    Returns:
        JSON string with search index data
    """
    documents = []
    metadata = {}

    for path, entry in entries.items():
        # Document for indexing - strip HTML for plain text search
        documents.append({
            "id": path,
            "title": entry.title,
            "tags": " ".join(entry.tags),
            "content": _strip_html(entry.html_content)[:10000],  # Limit content size
        })

        # Metadata for result display
        metadata[path] = {
            "title": entry.title,
            "tags": entry.tags,
            "path": f"{path}.html",
        }

    return json.dumps({
        "documents": documents,
        "metadata": metadata,
    }, ensure_ascii=False, indent=2)


def _strip_html(html_content: str) -> str:
    """Strip HTML tags for plain text indexing.

    Args:
        html_content: HTML string

    Returns:
        Plain text with HTML tags removed and whitespace normalized
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", html_content)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
