"""Pydantic models for the knowledge base."""

from datetime import date
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


class DocumentChunk(BaseModel):
    """A chunk of a document for indexing."""

    path: str
    section: str | None = None
    content: str
    metadata: EntryMetadata


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
