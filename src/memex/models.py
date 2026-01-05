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
    created: date | None = None
    updated: date | None = None
    token_count: int = 0
    content: str | None = None  # Full document content when requested
    source_project: str | None = None  # Project that created this entry


class SearchSuggestion(BaseModel):
    """A search suggestion when results are sparse."""

    query: str  # The suggested query
    reason: str  # Why this was suggested (e.g., "similar spelling", "related tag")


class SearchResponse(BaseModel):
    """Response wrapper for search results with optional warnings."""

    results: list[SearchResult]
    warnings: list[str] = Field(default_factory=list)
    suggestions: list[SearchSuggestion] = Field(default_factory=list)


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


class PotentialDuplicate(BaseModel):
    """A potential duplicate entry detected before creation."""

    path: str  # Path to the existing entry
    title: str  # Title of the existing entry
    score: float  # Semantic similarity score (0-1)
    tags: list[str] = Field(default_factory=list)  # Tags for context


class AddEntryResponse(BaseModel):
    """Response from add_entry including potential duplicates."""

    path: str  # Path where entry was created (or would be created)
    created: bool  # Whether the entry was actually created
    suggested_links: list[dict] = Field(default_factory=list)
    suggested_tags: list[dict] = Field(default_factory=list)
    potential_duplicates: list[PotentialDuplicate] = Field(default_factory=list)
    warning: str | None = None  # Warning message if duplicates detected


class AddEntryPreview(BaseModel):
    """Preview data for add_entry without creating a file."""

    path: str  # Relative path where entry would be created
    absolute_path: str  # Absolute path on disk
    frontmatter: str  # Generated YAML frontmatter
    content: str  # Final content (including related links if provided)
    potential_duplicates: list[PotentialDuplicate] = Field(default_factory=list)
    warning: str | None = None  # Warning message if duplicates detected


class SearchHistoryEntry(BaseModel):
    """A recorded search query."""

    query: str  # The search query string
    timestamp: datetime  # When the search was executed
    result_count: int = 0  # Number of results returned
    mode: str = "hybrid"  # Search mode used (hybrid, keyword, semantic)
    tags: list[str] = Field(default_factory=list)  # Tag filters applied


class UpsertMatch(BaseModel):
    """A potential match for upsert title search."""

    path: str  # Path to the entry
    title: str  # Entry title
    score: float  # Confidence score (0-1)
    match_type: str  # How matched: 'exact_title', 'alias', 'fuzzy'


class UpsertResult(BaseModel):
    """Result of upsert operation."""

    path: str  # Path to the entry (created or updated)
    action: Literal["created", "appended", "no_change"]  # What action was taken
    title: str  # Title of the entry
    matched_by: str | None = None  # How matched: 'exact_title', 'alias', 'fuzzy', None if created
    match_score: float | None = None  # Confidence score if matched


class SessionLogResult(BaseModel):
    """Result of session-log operation."""

    path: str  # Path to the session entry
    action: Literal["appended", "created"]  # What action was taken
    project: str | None = None  # Detected project name
    context_source: str | None = None  # How context was determined
