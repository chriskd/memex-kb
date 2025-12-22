"""Pydantic models for the knowledge base."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class EntryMetadata(BaseModel):
    """Frontmatter metadata for a KB entry."""

    title: str
    tags: list[str] = Field(min_length=1)
    created: date
    updated: date | None = None
    contributors: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    status: Literal["draft", "published", "archived"] = "published"
    source_project: str | None = None  # Project where entry was created
    edit_sources: list[str] = Field(default_factory=list)  # Projects that edited this
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
    created: date | None = None
    updated: date | None = None
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


# Beads integration models


class BeadsIssue(BaseModel):
    """A beads issue for display."""

    id: str
    title: str
    description: str | None = None
    status: Literal["open", "in_progress", "closed"]
    priority: int  # 0-4, where 0 is highest (critical)
    issue_type: str  # task, feature, bug, chore, epic
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    close_reason: str | None = None
    dependency_count: int = 0
    dependent_count: int = 0


class BeadsKanbanColumn(BaseModel):
    """A kanban column with issues."""

    status: str
    label: str
    issues: list[BeadsIssue]


class BeadsKanbanData(BaseModel):
    """Full kanban board data."""

    project: str
    columns: list[BeadsKanbanColumn]
    total_issues: int
