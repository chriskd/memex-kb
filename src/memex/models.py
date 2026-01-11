"""Pydantic models for the knowledge base."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class EntryMetadata(BaseModel):
    """Frontmatter metadata for a KB entry."""

    title: str
    tags: list[str] = Field(min_length=1)
    created: datetime
    updated: datetime | None = None
    contributors: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    status: Literal["draft", "published", "archived"] = "published"
    source_project: str | None = None  # Project where entry was created
    edit_sources: list[str] = Field(default_factory=list)  # Projects that edited this
    # Breadcrumb metadata - agent/LLM provenance
    model: str | None = None  # LLM model that created/last updated the entry
    git_branch: str | None = None  # Git branch during creation
    last_edited_by: str | None = None  # Last contributor identity (agent or human)
    # Beads integration
    beads_issues: list[str] = Field(default_factory=list)  # e.g., ["project-id1", "project-id2"]
    beads_project: str | None = None  # Links to all issues in a beads project


class DocumentChunk(BaseModel):
    """A chunk of a document for indexing."""

    path: str
    section: str | None = None
    content: str
    metadata: EntryMetadata
    token_count: int | None = None


class SearchResult(BaseModel):
    """A search result."""

    path: str
    title: str
    snippet: str
    score: float
    tags: list[str] = Field(default_factory=list)
    section: str | None = None
    created: datetime | None = None
    updated: datetime | None = None
    token_count: int = 0
    content: str | None = None  # Full document content when requested
    source_project: str | None = None  # Project that created this entry


class SearchResponse(BaseModel):
    """Response wrapper for search results with optional warnings."""

    results: list[SearchResult]
    warnings: list[str] = Field(default_factory=list)


class KBEntry(BaseModel):
    """A full KB entry."""

    path: str
    metadata: EntryMetadata
    content: str
    links: list[str] = Field(default_factory=list)
    backlinks: list[str] = Field(default_factory=list)


class IndexStatus(BaseModel):
    """Status of the search indices."""

    whoosh_docs: int
    chroma_docs: int
    last_indexed: str | None
    kb_files: int


class QualityDetail(BaseModel):
    """Per-query evaluation result."""

    query: str
    expected: list[str]
    hits: list[str]
    found: bool
    best_rank: int | None = None


class QualityReport(BaseModel):
    """Aggregate quality report for search accuracy."""

    accuracy: float
    total_queries: int
    details: list[QualityDetail] = Field(default_factory=list)


class ViewStats(BaseModel):
    """View statistics for a KB entry."""

    total_views: int = 0
    last_viewed: datetime | None = None
    views_by_day: dict[str, int] = Field(default_factory=dict)  # ISO date -> count
