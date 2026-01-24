"""Pydantic models for the knowledge base."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class EvolutionRecord(BaseModel):
    """A record of how an entry was evolved based on a related entry."""

    timestamp: datetime  # When evolution occurred
    trigger_entry: str  # Path of entry that triggered evolution
    previous_keywords: list[str] = Field(default_factory=list)  # Keywords before evolution
    new_keywords: list[str] = Field(default_factory=list)  # Keywords after evolution
    previous_description: str | None = None  # Description before (if changed)
    new_description: str | None = None  # Description after (if changed)


class NeighborUpdate(BaseModel):
    """Update suggestion for a single neighbor entry during evolution.

    Represents the LLM's suggested changes for one neighbor entry.
    Part of the EvolutionDecision response structure (A-Mem parity).
    """

    path: str  # Path to the neighbor entry
    new_keywords: list[str] = Field(default_factory=list)  # Updated keyword list (replaces existing)
    new_context: str = ""  # Updated context/description (one sentence)
    relationship: str = ""  # One-sentence description of relationship to new entry


class EvolutionDecision(BaseModel):
    """LLM decision about whether and how to evolve neighbors.

    Mirrors A-Mem's evolution response structure where the LLM explicitly
    decides whether evolution should happen, not just the score threshold.

    Fields:
        should_evolve: LLM's explicit decision (not just score threshold)
        actions: What to do - 'update_keywords', 'update_context', 'add_links'
        neighbor_updates: Per-neighbor update suggestions
        suggested_connections: New link paths to add (for 'add_links' action)
    """

    should_evolve: bool  # LLM decision: should this entry trigger evolution?
    actions: list[str] = Field(default_factory=list)  # Actions: update_keywords, update_context, add_links
    neighbor_updates: list[NeighborUpdate] = Field(default_factory=list)  # Per-neighbor updates
    suggested_connections: list[str] = Field(default_factory=list)  # New links to add


class SemanticLink(BaseModel):
    """A computed semantic relationship to another entry."""

    path: str  # Target entry path
    score: float  # Similarity score (0-1)
    reason: str  # How discovered: 'embedding_similarity' | 'shared_tags' | 'bidirectional'


class RelationLink(BaseModel):
    """A typed relation to another entry (manual, not A-Mem)."""

    path: str  # Target entry path
    type: str  # Relation type (e.g., "implements", "depends_on")


class EntryMetadata(BaseModel):
    """Frontmatter metadata for a KB entry."""

    title: str
    description: str | None = None  # One-line summary for search results
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
    # A-Mem semantic linking
    keywords: list[str] = Field(default_factory=list)  # LLM-extracted key concepts
    semantic_links: list[SemanticLink] = Field(default_factory=list)  # Computed relationships
    # Typed relations (manual, non-A-Mem)
    relations: list[RelationLink] = Field(default_factory=list)
    evolution_history: list[EvolutionRecord] = Field(default_factory=list)  # How entry evolved


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
    kb_scope: str | None = None  # KB scope: "project", "user", or None for single-KB


class SearchResponse(BaseModel):
    """Response wrapper for search results with optional warnings."""

    results: list[SearchResult]
    warnings: list[str] = Field(default_factory=list)


class RelationNode(BaseModel):
    """A node in the relations graph."""

    path: str
    title: str
    tags: list[str] = Field(default_factory=list)
    scope: str | None = None


class RelationEdge(BaseModel):
    """A directed edge in the relations graph."""

    source: str
    target: str
    origin: Literal["wikilink", "relations"]
    relation_type: str | None = None
    score: float | None = None


class RelationsGraph(BaseModel):
    """Unified relations graph across wikilinks and frontmatter edges."""

    nodes: dict[str, RelationNode] = Field(default_factory=dict)
    edges: list[RelationEdge] = Field(default_factory=list)


class RelationsQueryResult(BaseModel):
    """Subgraph returned from a relations graph query."""

    root: str
    depth: int
    nodes: list[RelationNode] = Field(default_factory=list)
    edges: list[RelationEdge] = Field(default_factory=list)


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


class SearchSuggestion(BaseModel):
    """A search suggestion when results are sparse."""

    query: str  # The suggested query
    reason: str  # Why this was suggested (e.g., "similar spelling", "related tag")


class PotentialDuplicate(BaseModel):
    """A potential duplicate entry detected before creation."""

    path: str  # Path to the existing entry
    title: str  # Title of the existing entry
    score: float  # Semantic similarity score (0-1)
    tags: list[str] = Field(default_factory=list)  # Tags for context


class AddEntryPreview(BaseModel):
    """Preview data for add_entry without creating a file."""

    path: str  # Relative path where entry would be created
    absolute_path: str  # Absolute path on disk
    frontmatter: str  # Generated YAML frontmatter
    content: str  # Final content (including related links if provided)
    potential_duplicates: list[PotentialDuplicate] = Field(default_factory=list)
    warning: str | None = None  # Warning message if duplicates detected


class UpsertMatch(BaseModel):
    """A potential match for upsert title search."""

    path: str  # Path to the entry
    title: str  # Entry title
    score: float  # Confidence score (0-1)
    match_type: str  # How matched: 'exact_title', 'alias', 'fuzzy'


class UpsertResult(BaseModel):
    """Result of upsert operation."""

    path: str  # Path to the entry (created or updated)
    action: Literal["created", "appended", "replaced"]  # What action was taken
    title: str  # Title of the entry
    matched_by: str | None = None  # How matched: 'exact_title', 'alias', 'fuzzy', None if created
    match_score: float | None = None  # Confidence score if matched


class SearchHistoryEntry(BaseModel):
    """A recorded search query."""

    query: str  # The search query string
    timestamp: datetime  # When the search was executed
    result_count: int = 0  # Number of results returned
    mode: str = "hybrid"  # Search mode used (hybrid, keyword, semantic)
    tags: list[str] = Field(default_factory=list)  # Tag filters applied


class BatchOperationResult(BaseModel):
    """Result of a single batch operation."""

    index: int  # Index in the batch
    command: str  # The command that was executed
    success: bool  # Whether it succeeded
    result: dict | None = None  # Result if successful
    error: str | None = None  # Error message if failed
    error_code: str | None = None  # Error code if failed


class BatchResponse(BaseModel):
    """Response from batch command execution."""

    total: int  # Total commands processed
    succeeded: int  # Number that succeeded
    failed: int  # Number that failed
    results: list[BatchOperationResult]  # Per-operation results


class IngestResult(BaseModel):
    """Result of ingesting a markdown file into the KB."""

    path: str  # Relative path within the KB
    absolute_path: str  # Absolute path on disk
    moved: bool  # Whether the file was moved from external location
    frontmatter_added: bool  # Whether frontmatter was prepended
    original_path: str | None = None  # Original path if moved
    title: str  # Entry title (extracted or provided)
    tags: list[str]  # Tags applied
    suggested_tags: list[dict] = Field(default_factory=list)  # Tag suggestions


class InitInventoryEntry(BaseModel):
    """An entry in the a-mem-init inventory."""

    path: str  # Relative path to entry (with @scope/ prefix in multi-KB mode)
    title: str  # Entry title
    created: datetime  # Creation timestamp (for chronological ordering)
    has_keywords: bool  # Whether entry has keywords
    keyword_status: Literal["ready", "skipped", "needs_llm"]  # Processing status
    keywords: list[str] = Field(default_factory=list)  # Existing keywords if any
    absolute_path: str | None = None  # Absolute path for file operations


class InitInventoryResult(BaseModel):
    """Result of a-mem-init inventory phase."""

    entries: list[InitInventoryEntry]  # Entries sorted by created (oldest first)
    total_count: int  # Total entries found
    with_keywords: int  # Entries with keywords
    missing_keywords: int  # Entries without keywords
    skipped_count: int  # Entries skipped (missing keywords in skip mode)
    needs_llm_count: int  # Entries queued for LLM keyword extraction
    missing_keyword_mode: Literal["error", "skip", "llm"]  # Mode used
    errors: list[str] = Field(default_factory=list)  # Entries that caused errors


class KeywordExtractionEntry(BaseModel):
    """Result of keyword extraction for a single entry."""

    path: str  # Entry path
    title: str  # Entry title
    keywords: list[str]  # Extracted keywords
    success: bool  # Whether extraction succeeded
    error: str | None = None  # Error message if failed


class KeywordExtractionPhaseResult(BaseModel):
    """Result of Phase 2: LLM keyword extraction."""

    entries_processed: int  # Total entries processed
    entries_updated: int  # Entries successfully updated with keywords
    entries_failed: int  # Entries that failed extraction
    results: list[KeywordExtractionEntry] = Field(default_factory=list)  # Per-entry results
    errors: list[str] = Field(default_factory=list)  # General errors


class LinkingPhaseEntry(BaseModel):
    """Result of linking for a single entry."""

    path: str  # Entry path
    title: str  # Entry title
    links_created: int  # Number of forward links created
    evolution_items_queued: int  # Number of evolution queue items
    neighbors: list[str] = Field(default_factory=list)  # Paths of linked neighbors


class LinkingPhaseResult(BaseModel):
    """Result of Phase 3: Semantic Linking."""

    entries_processed: int  # Total entries processed
    entries_linked: int  # Entries that got at least one link
    entries_skipped: int  # Entries skipped (missing keywords, etc.)
    total_links_created: int  # Total bidirectional link pairs created
    total_evolution_items: int  # Total items queued for evolution
    results: list[LinkingPhaseEntry] = Field(default_factory=list)  # Per-entry results
    errors: list[str] = Field(default_factory=list)  # General errors
