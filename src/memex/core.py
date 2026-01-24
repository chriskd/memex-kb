"""Core business logic for memex.

This module contains the pure business logic, separated from MCP protocol concerns.
Both the MCP server (server.py) and CLI (cli.py) import from here.

Design principles:
- All functions are async for consistency with MCP server
- No MCP decorators or protocol-specific code
- Lazy initialization of expensive resources (searcher, embeddings)
"""

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)


def _ensure_aware(dt: datetime | None) -> datetime | None:
    """Ensure a datetime is timezone-aware, assuming UTC for naive datetimes.

    Args:
        dt: A datetime that may be naive or aware, or None.

    Returns:
        Timezone-aware datetime (with UTC if originally naive), or None.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt

from .backlinks_cache import ensure_backlink_cache, rebuild_backlink_cache
from .config import (
    AMEM_STRICT_ERROR_MESSAGE,
    AMEMStrictError,
    DUPLICATE_DETECTION_LIMIT,
    DUPLICATE_DETECTION_MIN_SCORE,
    LINK_SUGGESTION_MIN_SCORE,
    MAX_CONTENT_RESULTS,
    SEMANTIC_LINK_ENABLED,
    SEMANTIC_LINK_K,
    SEMANTIC_LINK_MIN_SCORE,
    SIMILAR_ENTRY_TAG_WEIGHT,
    TAG_SUGGESTION_MIN_SCORE,
    get_kb_root,
    get_kb_root_by_scope,
    get_memory_evolution_config,
    is_amem_strict_enabled,
)
from .context import KBContext, get_kb_context, get_kbconfig
from .evaluation import run_quality_checks
from .frontmatter import build_frontmatter, create_new_metadata, update_metadata_for_edit
from .indexer import HybridSearcher
from .models import (
    AddEntryPreview,
    DocumentChunk,
    EvolutionRecord,
    IndexStatus,
    IngestResult,
    InitInventoryEntry,
    InitInventoryResult,
    KBEntry,
    KeywordExtractionEntry,
    KeywordExtractionPhaseResult,
    LinkingPhaseEntry,
    LinkingPhaseResult,
    PotentialDuplicate,
    QualityReport,
    SearchResponse,
    SearchResult,
    SemanticLink,
    UpsertMatch,
)
from .parser import ParseError, extract_links, parse_entry, update_links_batch

# ─────────────────────────────────────────────────────────────────────────────
# Module-level state (lazy initialization)
# ─────────────────────────────────────────────────────────────────────────────

_searcher: HybridSearcher | None = None
_searcher_ready = False


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────


def get_current_project() -> str | None:
    """Get the current project name from git remote or working directory.

    Returns:
        Project name (e.g., "memex-ansible") or None if unavailable.
    """
    cwd = Path.cwd()

    # Try to get project name from git remote
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            remote_url = result.stdout.strip()

            # Handle SSH format: git@github.com:user/repo.git
            ssh_match = re.search(r":([^/]+/[^/]+?)(?:\.git)?$", remote_url)
            if ssh_match:
                return ssh_match.group(1).split("/")[-1]

            # Handle HTTPS format: https://github.com/user/repo.git
            https_match = re.search(r"/([^/]+?)(?:\.git)?$", remote_url)
            if https_match:
                return https_match.group(1)
    except (subprocess.TimeoutExpired, OSError) as e:
        log.debug("Could not get git remote URL: %s", e)

    # Fallback to directory name
    return cwd.name


def get_current_contributor() -> str | None:
    """Get the current contributor identity from git config or environment.

    Returns:
        Contributor string like "Name <email>" or "Name", or None if unavailable.
    """
    # Try git config first
    try:
        name_result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        email_result = subprocess.run(
            ["git", "config", "user.email"],
            capture_output=True,
            text=True,
            timeout=2,
        )

        name = name_result.stdout.strip() if name_result.returncode == 0 else None
        email = email_result.stdout.strip() if email_result.returncode == 0 else None

        if name and email:
            return f"{name} <{email}>"
        elif name:
            return name
    except (subprocess.TimeoutExpired, OSError) as e:
        log.debug("Could not get git user config: %s", e)

    # Fall back to environment variables
    name = os.environ.get("GIT_AUTHOR_NAME") or os.environ.get("USER")
    email = os.environ.get("GIT_AUTHOR_EMAIL")

    if name and email:
        return f"{name} <{email}>"
    elif name:
        return name

    return None


def get_llm_model() -> str | None:
    """Get the current LLM model from environment variables.

    Checks multiple env vars that might contain model info:
    - LLM_MODEL: Generic model identifier (preferred)
    - ANTHROPIC_MODEL: Anthropic-specific
    - CLAUDE_MODEL: Claude-specific
    - OPENAI_MODEL: OpenAI-specific

    Returns:
        Model identifier string or None if unavailable.
    """
    for var in ("LLM_MODEL", "ANTHROPIC_MODEL", "CLAUDE_MODEL", "OPENAI_MODEL"):
        model = os.environ.get(var)
        if model:
            return model
    return None


def get_git_branch() -> str | None:
    """Get the current git branch name.

    Returns:
        Branch name (e.g., "main", "feat/search") or None if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            return branch if branch else None
    except (subprocess.TimeoutExpired, OSError) as e:
        log.debug("Could not get git branch: %s", e)
    return None


def get_actor_identity() -> str | None:
    """Get the current actor identity (agent or human).

    Returns:
        Actor identifier like 'claude-opus' (for agents) or 'chris' (for humans).
    """
    # Check for agent identity first
    if actor := os.environ.get("BD_ACTOR"):
        return actor
    # Fall back to user
    if user := os.environ.get("USER"):
        return user
    return None


def _maybe_initialize_searcher(searcher: HybridSearcher) -> None:
    """Ensure search indices are populated before first query."""
    global _searcher_ready
    if _searcher_ready:
        return

    status = searcher.status()
    if status.kb_files > 0 and (status.whoosh_docs == 0 or status.chroma_docs == 0):
        kb_root = get_kb_root()
        if kb_root.exists():
            searcher.reindex(kb_root)

    _searcher_ready = True


def get_searcher() -> HybridSearcher:
    """Get the HybridSearcher, initializing lazily."""
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher()
    _maybe_initialize_searcher(_searcher)
    return _searcher


def slugify(title: str) -> str:
    """Convert title to URL-friendly slug (lowercase, hyphens, alphanumeric only)."""
    slug = title.lower()
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def get_valid_categories() -> list[str]:
    """Get list of valid category directories."""
    kb_root = get_kb_root()
    if not kb_root.exists():
        return []
    return [
        d.name
        for d in kb_root.iterdir()
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
    ]


def relative_kb_path(kb_root: Path, file_path: Path) -> str:
    """Return file path relative to KB root."""
    return str(file_path.relative_to(kb_root))


def normalize_chunks(chunks: list[DocumentChunk], relative_path: str) -> list[DocumentChunk]:
    """Ensure all chunk paths share the relative file path."""
    return [chunk.model_copy(update={"path": relative_path}) for chunk in chunks]


def apply_section_updates(content: str, updates: dict[str, str]) -> str:
    """Apply section-level updates to markdown content."""

    updated = content
    for section, replacement in updates.items():
        clean_section = section.strip()
        clean_replacement = replacement.strip()
        if not clean_section or not clean_replacement:
            continue

        pattern = re.compile(
            rf"(^##\s+{re.escape(clean_section)}\s*$)(.*?)(?=^##\s+|\Z)",
            re.MULTILINE | re.DOTALL,
        )

        if pattern.search(updated):
            updated = pattern.sub(
                f"## {clean_section}\n\n{clean_replacement}\n\n", updated, count=1
            )
        else:
            updated = updated.rstrip() + f"\n\n## {clean_section}\n\n{clean_replacement}\n"

    return updated


def get_backlink_index() -> dict[str, list[str]]:
    """Return cached backlink index, refreshing if files changed."""
    kb_root = get_kb_root()
    return ensure_backlink_cache(kb_root)


def validate_nested_path(path: str) -> tuple[Path, str]:
    """Validate a nested path within the KB.

    Args:
        path: Relative path like "development/python/file.md" or "development/python/"

    Returns:
        Tuple of (absolute_path, normalized_relative_path)

    Raises:
        ValueError: If path is invalid or outside KB.
    """
    kb_root = get_kb_root()

    # Security: prevent path traversal
    normalized = Path(path).as_posix()
    if ".." in normalized or normalized.startswith("/"):
        raise ValueError(f"Invalid path: {path}")

    # Check for hidden/special path components
    for part in Path(path).parts:
        if part.startswith(".") or part.startswith("_"):
            raise ValueError(f"Invalid path component: {part}")

    abs_path = (kb_root / path).resolve()

    # Ensure path is within KB root
    try:
        abs_path.relative_to(kb_root.resolve())
    except ValueError:
        raise ValueError(f"Path escapes knowledge base: {path}")

    return abs_path, normalized


def directory_exists(directory: str) -> bool:
    """Check if a directory path exists within the KB."""
    kb_root = get_kb_root()
    dir_path = kb_root / directory
    return dir_path.exists() and dir_path.is_dir()


def get_parent_category(path: str) -> str:
    """Extract top-level category from a nested path.

    Args:
        path: "development/python/frameworks/file.md"

    Returns:
        "development"
    """
    parts = Path(path).parts
    if not parts:
        raise ValueError("Empty path")
    return parts[0]


def hydrate_content(results: list[SearchResult]) -> list[SearchResult]:
    """Add full document content to search results.

    Reads content from disk using parse_entry() for each result.

    Args:
        results: Search results to hydrate.

    Returns:
        New list of SearchResult with content populated.
    """
    if not results:
        return results

    kb_root = get_kb_root()
    hydrated = []

    for result in results:
        file_path = kb_root / result.path
        content = None

        if file_path.exists():
            try:
                _, content, _ = parse_entry(file_path)
            except ParseError as e:
                log.warning("Failed to parse %s for hydration: %s", file_path, e)

        # Create new result with content
        hydrated.append(
            SearchResult(
                path=result.path,
                title=result.title,
                snippet=result.snippet,
                score=result.score,
                tags=result.tags,
                section=result.section,
                created=result.created,
                updated=result.updated,
                token_count=result.token_count,
                content=content,
            )
        )

    return hydrated


# Paths to exclude from link suggestions (meta-entries that aren't useful to link to)
_EXCLUDED_FROM_SUGGESTIONS = {"_index.md", "_index"}


def compute_link_suggestions(
    title: str,
    content: str,
    tags: list[str],
    self_path: str,
    existing_links: set[str] | None = None,
    limit: int = 5,
    min_score: float = LINK_SUGGESTION_MIN_SCORE,
) -> list[dict]:
    """Compute link suggestions based on semantic similarity.

    This is the core suggestion logic, extracted for reuse by add/update tools.

    Args:
        title: Entry title.
        content: Entry content.
        tags: Entry tags.
        self_path: Path to the entry itself (to exclude from results).
        existing_links: Set of paths already linked to (to exclude).
        limit: Maximum suggestions to return.
        min_score: Minimum similarity score threshold (0.0-1.0).

    Returns:
        List of {path, title, score, reason} for suggested links.
    """
    if existing_links is None:
        existing_links = set()

    # Normalize self path for comparison
    self_key = self_path[:-3] if self_path.endswith(".md") else self_path
    exclude_set = existing_links | {self_key, self_path} | _EXCLUDED_FROM_SUGGESTIONS

    # Use semantic search to find similar entries
    searcher = get_searcher()

    # Search using the entry's title and first part of content as query
    query = f"{title} {content[:500]}"
    search_results = searcher.search(query, limit=limit + len(exclude_set) + 10, mode="semantic")

    suggestions = []
    tags_set = set(tags)

    for result in search_results:
        result_key = result.path[:-3] if result.path.endswith(".md") else result.path

        # Skip excluded paths (self, already linked, meta-entries)
        if result_key in exclude_set or result.path in exclude_set:
            continue

        # Skip results below score threshold
        if result.score < min_score:
            continue

        # Determine reason based on shared tags
        shared_tags = tags_set & set(result.tags)
        if shared_tags:
            reason = f"Shares tags: {', '.join(sorted(shared_tags))}"
        else:
            reason = "Semantically similar content"

        suggestions.append({
            "path": result.path,
            "title": result.title,
            "score": round(result.score, 3),
            "reason": reason,
        })

        if len(suggestions) >= limit:
            break

    return suggestions


@dataclass
class LinkingResult:
    """Result of creating bidirectional semantic links.

    Contains both the forward links for the new entry and
    the list of neighbors that may need evolution.
    """

    forward_links: list[SemanticLink]
    """Links to add to the new entry's semantic_links field."""

    neighbors_for_evolution: list[tuple[str, float]]
    """List of (neighbor_path, score) tuples for memory evolution."""


def create_bidirectional_semantic_links(
    entry_path: str,
    title: str,
    content: str,
    tags: list[str],
    k: int = SEMANTIC_LINK_K,
    min_score: float = SEMANTIC_LINK_MIN_SCORE,
) -> LinkingResult:
    """Find similar entries and create bidirectional semantic links.

    This function:
    1. Searches for similar entries by embedding similarity
    2. Creates SemanticLink objects for the new entry (forward links)
    3. Adds backlinks to neighbor entries
    4. Re-indexes affected entries
    5. Collects neighbors for potential memory evolution

    Args:
        entry_path: Relative path of the new/updated entry (e.g., "guides/python.md")
        title: Title of the entry for search query
        content: Content of the entry for search query
        tags: Tags of the entry for context
        k: Maximum number of neighbors to link to
        min_score: Minimum similarity score threshold (0.0-1.0)

    Returns:
        LinkingResult with forward links and neighbors for evolution.
    """
    if not SEMANTIC_LINK_ENABLED:
        return LinkingResult(forward_links=[], neighbors_for_evolution=[])

    kb_root = get_kb_root()
    searcher = get_searcher()

    # Normalize entry path for comparison
    entry_key = entry_path[:-3] if entry_path.endswith(".md") else entry_path

    # Search for similar entries using title and content
    query = f"{title} {content[:500]}"
    search_results = searcher.search(query, limit=k + 10, mode="semantic")

    forward_links: list[SemanticLink] = []
    neighbors_for_evolution: list[tuple[str, float]] = []

    for result in search_results:
        # Skip self
        result_key = result.path[:-3] if result.path.endswith(".md") else result.path
        if result_key == entry_key or result.path == entry_path:
            continue

        # Skip results below threshold
        if result.score < min_score:
            continue

        # Create forward link for the new entry
        forward_link = SemanticLink(
            path=result.path,
            score=round(result.score, 3),
            reason="embedding_similarity",
        )
        forward_links.append(forward_link)

        # Create backlink on the neighbor entry
        _add_backlink_to_neighbor(
            kb_root=kb_root,
            neighbor_path=result.path,
            new_entry_path=entry_path,
            score=result.score,
            searcher=searcher,
        )

        # Collect for potential evolution
        neighbors_for_evolution.append((result.path, result.score))

        if len(forward_links) >= k:
            break

    return LinkingResult(
        forward_links=forward_links,
        neighbors_for_evolution=neighbors_for_evolution,
    )


def _add_backlink_to_neighbor(
    kb_root: Path,
    neighbor_path: str,
    new_entry_path: str,
    score: float,
    searcher: HybridSearcher,
) -> None:
    """Add a semantic backlink to a neighbor entry.

    This is a helper function that:
    1. Parses the neighbor entry
    2. Adds/updates the backlink in semantic_links
    3. Saves the neighbor entry
    4. Re-indexes the neighbor

    Args:
        kb_root: KB root directory
        neighbor_path: Relative path to the neighbor entry
        new_entry_path: Relative path to the new entry (for backlink)
        score: Similarity score for the backlink
        searcher: HybridSearcher instance for re-indexing
    """
    neighbor_file = kb_root / neighbor_path

    if not neighbor_file.exists():
        log.warning("Neighbor entry not found for backlinking: %s", neighbor_path)
        return

    try:
        metadata, content, chunks = parse_entry(neighbor_file)
    except ParseError as e:
        log.warning("Failed to parse neighbor for backlinking %s: %s", neighbor_path, e)
        return

    # Check if backlink already exists
    existing_paths = {link.path for link in metadata.semantic_links}
    if new_entry_path in existing_paths:
        # Backlink already exists, skip
        return

    # Add the backlink
    backlink = SemanticLink(
        path=new_entry_path,
        score=round(score, 3),
        reason="bidirectional",
    )
    updated_links = list(metadata.semantic_links) + [backlink]

    # Update metadata with new semantic links
    updated_metadata = update_metadata_for_edit(
        metadata,
        semantic_links=updated_links,
    )

    # Write updated file
    frontmatter = build_frontmatter(updated_metadata)
    neighbor_file.write_text(frontmatter + content, encoding="utf-8")

    # Re-index the neighbor entry
    try:
        _, _, updated_chunks = parse_entry(neighbor_file)
        if updated_chunks:
            searcher.delete_document(neighbor_path)
            normalized_chunks = normalize_chunks(updated_chunks, neighbor_path)
            searcher.index_chunks(normalized_chunks)
    except ParseError as e:
        log.warning("Updated neighbor but failed to re-index %s: %s", neighbor_path, e)


def _queue_evolution(
    new_entry_path: str,
    neighbors_to_evolve: list[tuple[str, float]],
    kb_root: Path,
) -> int:
    """Queue evolution work for later processing.

    This replaces inline evolution - work is queued and processed by `mx evolve`.

    Args:
        new_entry_path: Path to the newly added entry.
        neighbors_to_evolve: List of (neighbor_path, score) tuples.
        kb_root: KB root directory.

    Returns:
        Number of items queued.
    """
    config = get_memory_evolution_config()
    if not config.enabled or not neighbors_to_evolve:
        return 0

    # Filter by min_score before queuing
    filtered = [(path, score) for path, score in neighbors_to_evolve if score >= config.min_score]
    if not filtered:
        return 0

    from .evolution_queue import queue_evolution
    return queue_evolution(new_entry_path, filtered, kb_root)


def _build_neighbor_info_for_strengthen(
    kb_root: Path,
    neighbors_to_evolve: list[tuple[str, float]],
) -> list:
    """Build NeighborInfo objects for strengthen analysis.

    Reads neighbor entry files and extracts the information needed
    for the strengthen LLM analysis.

    Args:
        kb_root: KB root directory.
        neighbors_to_evolve: List of (neighbor_path, score) tuples.

    Returns:
        List of NeighborInfo objects (from llm module).
    """
    from .llm import NeighborInfo

    neighbors_info = []
    for neighbor_path, score in neighbors_to_evolve:
        neighbor_file = kb_root / neighbor_path
        if not neighbor_file.exists():
            log.warning("Neighbor not found for strengthen: %s", neighbor_path)
            continue

        try:
            metadata, content, _ = parse_entry(neighbor_file)
            neighbors_info.append(
                NeighborInfo(
                    path=neighbor_path,
                    title=metadata.title,
                    content=content,
                    keywords=list(metadata.keywords),
                    score=score,
                )
            )
        except ParseError as e:
            log.warning("Failed to parse neighbor for strengthen %s: %s", neighbor_path, e)
            continue

    return neighbors_info


@dataclass
class EvolutionResult:
    """Result of processing evolution queue items."""

    processed: int
    """Number of items processed."""

    keywords_added: int
    """Total keywords added across all neighbors."""

    errors: int
    """Number of errors encountered."""


async def process_evolution_items(
    items: list,  # list[QueueItem] - avoid circular import
    kb_root: Path | None = None,
) -> EvolutionResult:
    """Process queued evolution items.

    Groups items by new_entry and processes each batch.
    Called by `mx evolve` command.

    Args:
        items: List of QueueItem objects to process.
        kb_root: KB root directory. If None, auto-discovers.

    Returns:
        EvolutionResult with processing statistics.
    """
    from .evolution_queue import QueueItem

    if kb_root is None:
        kb_root = get_kb_root()

    config = get_memory_evolution_config()
    if not config.enabled:
        log.warning("Memory evolution is disabled in config")
        return EvolutionResult(processed=0, keywords_added=0, errors=0)

    # Import LLM module
    try:
        from .llm import NeighborInfo, analyze_evolution
    except ImportError as e:
        log.warning("Memory evolution requires openai package: %s", e)
        return EvolutionResult(processed=0, keywords_added=0, errors=0)

    # Group items by new_entry for batched processing
    from collections import defaultdict
    groups: dict[str, list[QueueItem]] = defaultdict(list)
    for item in items:
        groups[item.new_entry].append(item)

    searcher = get_searcher()
    total_processed = 0
    total_keywords = 0
    total_errors = 0

    for new_entry_path, group_items in groups.items():
        # Read new entry info
        new_entry_file = kb_root / new_entry_path
        if not new_entry_file.exists():
            log.warning("New entry not found, skipping: %s", new_entry_path)
            total_errors += len(group_items)
            continue

        try:
            new_metadata, new_content, _ = parse_entry(new_entry_file)
        except ParseError as e:
            log.warning("Failed to parse new entry %s: %s", new_entry_path, e)
            total_errors += len(group_items)
            continue

        # Collect neighbor info for this batch
        neighbors_info: list[NeighborInfo] = []
        for item in group_items:
            neighbor_file = kb_root / item.neighbor
            if not neighbor_file.exists():
                log.warning("Neighbor not found: %s", item.neighbor)
                total_errors += 1
                continue

            try:
                metadata, content, _ = parse_entry(neighbor_file)
                neighbors_info.append(
                    NeighborInfo(
                        path=item.neighbor,
                        title=metadata.title,
                        content=content,
                        keywords=list(metadata.keywords),
                        score=item.score,
                    )
                )
            except ParseError as e:
                log.warning("Failed to parse neighbor %s: %s", item.neighbor, e)
                total_errors += 1
                continue

        if not neighbors_info:
            continue

        # Get evolution decision from LLM (A-Mem parity: explicit should_evolve)
        try:
            decision = await analyze_evolution(
                new_entry_title=new_metadata.title,
                new_entry_content=new_content,
                new_entry_keywords=list(new_metadata.keywords),
                neighbors=neighbors_info,
                model=config.model,
            )
        except Exception as e:
            log.warning("LLM evolution failed for %s: %s", new_entry_path, e)
            total_errors += len(neighbors_info)
            continue

        # A-Mem parity: respect LLM's should_evolve decision
        if not decision.should_evolve:
            log.info("LLM decided not to evolve neighbors for: %s", new_entry_path)
            continue

        # Apply updates to neighbors
        for update in decision.neighbor_updates:
            total_processed += 1

            neighbor_file = kb_root / update.path
            if not neighbor_file.exists():
                continue

            try:
                metadata, content, _ = parse_entry(neighbor_file)

                # Check what's changing
                existing_keywords = set(kw.lower() for kw in metadata.keywords)
                new_keywords_set = set(kw.lower() for kw in update.new_keywords)
                keywords_changed = existing_keywords != new_keywords_set

                # Check if description should be updated (only if LLM provided new context)
                new_description = update.new_context if update.new_context else None
                description_changed = new_description is not None

                if not keywords_changed and not description_changed:
                    continue  # No changes needed

                updated_keywords = update.new_keywords if keywords_changed else None
                if keywords_changed:
                    keywords_added = len(new_keywords_set - existing_keywords)
                    total_keywords += keywords_added

                log.info(
                    "Evolving %s:%s%s",
                    update.path,
                    f" keywords {list(metadata.keywords)} -> {update.new_keywords}" if keywords_changed else "",
                    f" description: {new_description}" if description_changed else "",
                )

                # Record evolution history for auditing/debugging
                evolution_record = EvolutionRecord(
                    timestamp=datetime.now(UTC),
                    trigger_entry=new_entry_path,
                    previous_keywords=list(metadata.keywords) if keywords_changed else [],
                    new_keywords=update.new_keywords if keywords_changed else [],
                    previous_description=metadata.description if description_changed else None,
                    new_description=new_description,
                )
                updated_history = list(metadata.evolution_history) + [evolution_record]

                # Update metadata with new keywords and/or description
                updated_metadata = update_metadata_for_edit(
                    metadata,
                    keywords=updated_keywords,
                    description=new_description,
                    evolution_history=updated_history,
                )

                # Write updated file
                frontmatter = build_frontmatter(updated_metadata)
                neighbor_file.write_text(frontmatter + content, encoding="utf-8")

                # Re-index the evolved entry
                try:
                    _, _, updated_chunks = parse_entry(neighbor_file)
                    if updated_chunks:
                        searcher.delete_document(update.path)
                        normalized_chunks = normalize_chunks(updated_chunks, update.path)
                        searcher.index_chunks(normalized_chunks)
                except ParseError as e:
                    log.warning("Evolved neighbor but failed to re-index %s: %s", update.path, e)

            except ParseError as e:
                log.warning("Failed to apply evolution to %s: %s", update.path, e)
                total_errors += 1
            except OSError as e:
                log.warning("Failed to write evolved neighbor %s: %s", update.path, e)
                total_errors += 1

    return EvolutionResult(
        processed=total_processed,
        keywords_added=total_keywords,
        errors=total_errors,
    )


def get_tag_taxonomy() -> dict[str, int]:
    """Get all tags currently used in the KB with their usage counts.

    Returns:
        Dict mapping tag name to usage count.
    """
    kb_root = get_kb_root()
    if not kb_root.exists():
        return {}

    tag_counts: dict[str, int] = {}

    for md_file in kb_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue
        try:
            metadata, _, _ = parse_entry(md_file)
            for tag in metadata.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        except ParseError:
            continue

    return tag_counts


def compute_tag_suggestions(
    title: str,
    content: str,
    existing_tags: list[str] | None = None,
    limit: int = 5,
    min_score: float = TAG_SUGGESTION_MIN_SCORE,
) -> list[dict]:
    """Compute tag suggestions based on content analysis and semantic similarity.

    Combines three signals:
    1. Semantic similarity - Tags from semantically similar KB entries
    2. Keyword matching - Direct matches between content and existing tags
    3. Taxonomy frequency - Slight preference for commonly used tags

    Args:
        title: Entry title.
        content: Entry content.
        existing_tags: Tags already assigned (to exclude from suggestions).
        limit: Maximum suggestions to return.
        min_score: Minimum relevance score threshold (0.0-1.0).

    Returns:
        List of {tag, score, reason} for suggested tags.
    """
    if existing_tags is None:
        existing_tags = []

    existing_set = set(existing_tags)

    # Get the existing tag taxonomy
    tag_taxonomy = get_tag_taxonomy()
    if not tag_taxonomy:
        return []

    # Scoring accumulators: tag -> (score, reasons)
    tag_scores: dict[str, float] = {}
    tag_reasons: dict[str, list[str]] = {}

    # ===== Signal 1: Semantic similarity =====
    # Find similar entries and collect their tags
    searcher = get_searcher()
    query = f"{title} {content[:500]}"
    search_results = searcher.search(query, limit=10, mode="semantic")

    # Weight tags by similarity score of the entry they come from
    for result in search_results:
        if result.score < TAG_SUGGESTION_MIN_SCORE:
            continue
        for tag in result.tags:
            if tag in existing_set:
                continue
            # Score contribution: similarity_score * 0.6 (primary signal)
            contribution = result.score * 0.6
            tag_scores[tag] = tag_scores.get(tag, 0) + contribution
            if tag not in tag_reasons:
                tag_reasons[tag] = []
            tag_reasons[tag].append(f"Similar entry: {result.title}")

    # ===== Signal 2: Keyword matching =====
    # Check if existing tags appear as words in title or content
    text_lower = f"{title} {content}".lower()
    for tag in tag_taxonomy:
        if tag in existing_set:
            continue
        tag_lower = tag.lower()
        # Check for tag as a word boundary match
        if re.search(rf"\b{re.escape(tag_lower)}\b", text_lower):
            # Direct keyword match is a strong signal
            tag_scores[tag] = tag_scores.get(tag, 0) + SIMILAR_ENTRY_TAG_WEIGHT
            if tag not in tag_reasons:
                tag_reasons[tag] = []
            tag_reasons[tag].append("Keyword match in content")

    # ===== Signal 3: Taxonomy frequency =====
    # Slight boost for commonly used tags (encourages consistency)
    max_count = max(tag_taxonomy.values()) if tag_taxonomy else 1
    for tag in tag_scores:
        if tag in tag_taxonomy:
            # Small boost proportional to usage (max 0.1 for most common)
            frequency_boost = (tag_taxonomy[tag] / max_count) * 0.1
            tag_scores[tag] += frequency_boost

    # Filter and sort by score
    suggestions = []
    for tag, score in sorted(tag_scores.items(), key=lambda x: -x[1]):
        if score < min_score:
            continue
        if tag in existing_set:
            continue

        # Normalize score to 0-1 range (cap at 1.0)
        normalized_score = min(1.0, score)

        # Summarize reasons
        reasons = tag_reasons.get(tag, [])
        if len(reasons) > 2:
            reason = f"{reasons[0]} (+{len(reasons)-1} more)"
        elif reasons:
            reason = reasons[0]
        else:
            reason = "Taxonomy preference"

        suggestions.append({
            "tag": tag,
            "score": round(normalized_score, 3),
            "reason": reason,
        })

        if len(suggestions) >= limit:
            break

    return suggestions


def detect_potential_duplicates(
    title: str,
    content: str,
    searcher: HybridSearcher,
    min_score: float = DUPLICATE_DETECTION_MIN_SCORE,
    limit: int = DUPLICATE_DETECTION_LIMIT,
) -> list[PotentialDuplicate]:
    """Detect potential duplicate entries based on semantic similarity.

    Uses a combination of title and content to search for highly similar
    existing entries that might be duplicates.

    Args:
        title: The title of the new entry.
        content: The content of the new entry.
        searcher: The hybrid searcher instance.
        min_score: Minimum similarity score to consider as potential duplicate.
        limit: Maximum number of potential duplicates to return.

    Returns:
        List of PotentialDuplicate objects for high-similarity matches.
    """
    # Combine title and first ~500 chars of content for semantic comparison
    # This captures the essence of the entry without being too long
    search_text = f"{title} {content[:500]}"

    # Use semantic search to find similar entries
    try:
        results = searcher.search(
            search_text,
            limit=limit * 2,
            mode="semantic",
        )
    except Exception as e:
        log.warning("Duplicate detection failed: %s", e)
        return []

    duplicates = []
    for result in results:
        if result.score >= min_score:
            duplicates.append(
                PotentialDuplicate(
                    path=result.path,
                    title=result.title,
                    score=round(result.score, 3),
                    tags=result.tags,
                )
            )
            if len(duplicates) >= limit:
                break

    return duplicates


def _generate_description_from_content(
    content: str, max_length: int = 120, title: str | None = None
) -> str:
    """Generate a one-line description from entry content.

    Extracts the first meaningful sentence or phrase from the content,
    stripping markdown formatting. If the content starts with the title
    (common in markdown with H1 headers), it skips past the title.

    Args:
        content: Raw markdown content of the entry.
        max_length: Maximum description length (default 120).
        title: Optional title to skip if content starts with it.

    Returns:
        A clean one-line description.
    """
    if not content:
        return ""

    # Import here to avoid circular imports
    from .indexer import strip_markdown_for_snippet

    # Get a longer snippet to work with
    clean_text = strip_markdown_for_snippet(content, max_length=500)

    if not clean_text:
        return ""

    # If content starts with the title (common when H1 matches title), skip it
    if title:
        # Check if clean_text starts with the title (possibly followed by whitespace)
        title_normalized = title.strip()
        if clean_text.startswith(title_normalized):
            clean_text = clean_text[len(title_normalized):].strip()
            if not clean_text:
                return ""

    # Try to find the first complete sentence
    # Look for sentence-ending punctuation followed by space or end
    sentence_match = re.match(r"^(.+?[.!?])(?:\s|$)", clean_text)
    if sentence_match:
        description = sentence_match.group(1).strip()
        if len(description) <= max_length:
            return description

    # Fallback: truncate at word boundary
    if len(clean_text) <= max_length:
        return clean_text

    # Find last space before max_length
    truncated = clean_text[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length * 0.5:  # At least half the content
        return truncated[:last_space] + "..."
    return truncated + "..."


# ─────────────────────────────────────────────────────────────────────────────
# Core business logic functions
# ─────────────────────────────────────────────────────────────────────────────


async def search(
    query: str,
    limit: int = 10,
    mode: Literal["hybrid", "keyword", "semantic"] = "hybrid",
    tags: list[str] | None = None,
    include_content: bool = False,
    kb_context: KBContext | None = None,
    strict: bool = False,
    scope: str | None = None,
) -> SearchResponse:
    """Search the knowledge base.

    Args:
        query: Search query string.
        limit: Maximum number of results (default 10).
        mode: Search mode - "hybrid" (default), "keyword", or "semantic".
        tags: Optional list of tags to filter results.
        include_content: If True, include full document content in results.
                         Default False (snippet only). Limited to MAX_CONTENT_RESULTS.
        kb_context: Optional project context for path-based boosting.
                    If not provided, auto-discovered from cwd.
        strict: If True, use a higher similarity threshold to filter out
                low-confidence semantic matches. Prevents misleading scores
                for gibberish or unrelated queries.
        scope: Filter to specific scope - "project", "user", or None for all.

    Returns:
        SearchResponse with results and optional warnings.
    """
    searcher = get_searcher()
    # Auto-detect project context for boosting entries from current project
    project_context = get_current_project()
    # Auto-discover KB context if not provided
    if kb_context is None:
        kb_context = get_kb_context()
    results = searcher.search(
        query, limit=limit, mode=mode, project_context=project_context, kb_context=kb_context, strict=strict
    )
    warnings: list[str] = []

    # Filter by scope if specified (matches @project/ or @user/ prefix)
    if scope:
        scope_prefix = f"@{scope}/"
        results = [r for r in results if r.path.startswith(scope_prefix)]

    # Filter by tags if specified
    if tags:
        tag_set = set(tags)
        results = [r for r in results if tag_set.intersection(r.tags)]

    # Hydrate with full content if requested
    if include_content:
        if len(results) > MAX_CONTENT_RESULTS:
            warnings.append(
                f"Results limited to {MAX_CONTENT_RESULTS} for content hydration "
                f"(requested {len(results)}). Reduce limit or use get tool for remaining."
            )
            results = results[:MAX_CONTENT_RESULTS]
        results = hydrate_content(results)

    return SearchResponse(results=results, warnings=warnings)


async def expand_search_with_neighbors(
    results: list[SearchResult],
    depth: int = 1,
    include_content: bool = False,
) -> list[dict]:
    """Expand search results to include semantically linked entries (neighbors).

    For each search result, reads its semantic_links from metadata and fetches
    those linked entries. Performs graph traversal up to the specified depth.

    Args:
        results: Initial search results to expand.
        depth: Number of hops to traverse (default 1). A depth of 1 means
               only direct neighbors are included. A depth of 2 includes
               neighbors of neighbors, etc.
        include_content: If True, include full document content in results.

    Returns:
        List of dicts with search result data plus is_neighbor and linked_from fields.
        Direct matches have is_neighbor=False, neighbors have is_neighbor=True
        and linked_from set to the path that linked to them.
    """
    from .config import resolve_scoped_path

    # Build output list - start with direct matches
    output: list[dict] = []
    seen_paths: set[str] = set()

    # Track which paths link to which for the linked_from field
    # Format: {neighbor_path: first_linker_path}
    linked_from_map: dict[str, str] = {}

    # Queue for BFS traversal: (path, current_depth, score, linked_from_path)
    queue: list[tuple[str, int, float, str | None]] = []

    # Add initial results
    for r in results:
        if r.path not in seen_paths:
            seen_paths.add(r.path)
            item: dict = {
                "path": r.path,
                "title": r.title,
                "score": r.score,
                "is_neighbor": False,
            }
            if include_content and r.content:
                item["content"] = r.content
            else:
                item["snippet"] = r.snippet
            output.append(item)
            # Enqueue for neighbor discovery
            queue.append((r.path, 0, r.score, None))

    # BFS traversal up to specified depth
    while queue:
        current_path, current_depth, parent_score, _ = queue.pop(0)

        if current_depth >= depth:
            continue

        # Read the entry to get its semantic_links
        try:
            file_path = resolve_scoped_path(current_path)
            if not file_path.exists():
                continue

            metadata, content, _ = parse_entry(file_path)
            semantic_links = metadata.semantic_links

            for link in semantic_links:
                neighbor_path = link.path
                if neighbor_path not in seen_paths:
                    seen_paths.add(neighbor_path)

                    # Track who linked to this neighbor (first one wins)
                    if neighbor_path not in linked_from_map:
                        linked_from_map[neighbor_path] = current_path

                    # Try to read the neighbor entry for title/snippet
                    try:
                        neighbor_file = resolve_scoped_path(neighbor_path)
                        if neighbor_file.exists():
                            neighbor_meta, neighbor_content, _ = parse_entry(neighbor_file)

                            # Generate snippet from content
                            snippet = neighbor_content[:200].strip()
                            if len(neighbor_content) > 200:
                                snippet += "..."

                            neighbor_item: dict = {
                                "path": neighbor_path,
                                "title": neighbor_meta.title,
                                "score": link.score,
                                "is_neighbor": True,
                                "linked_from": linked_from_map[neighbor_path],
                            }
                            if include_content:
                                neighbor_item["content"] = neighbor_content
                            else:
                                neighbor_item["snippet"] = snippet
                            output.append(neighbor_item)

                            # Add to queue for further traversal
                            queue.append((neighbor_path, current_depth + 1, link.score, current_path))
                    except (ValueError, ParseError) as e:
                        log.debug("Could not read neighbor entry %s: %s", neighbor_path, e)
                        continue

        except (ValueError, ParseError) as e:
            log.debug("Could not read entry for neighbor expansion %s: %s", current_path, e)
            continue

    return output


async def quality(limit: int = 5, cutoff: int = 3) -> QualityReport:
    """Evaluate MCP search accuracy."""
    searcher = get_searcher()
    return run_quality_checks(searcher, limit=limit, cutoff=cutoff)


async def add_entry(
    title: str,
    content: str,
    tags: list[str],
    category: str = "",
    directory: str | None = None,
    links: list[str] | None = None,
    kb_context: KBContext | None = None,
    scope: str | None = None,
    keywords: list[str] | None = None,
    semantic_links: list | None = None,
    relations: list | None = None,
) -> dict:
    """Create a new KB entry.

    Args:
        title: Entry title.
        content: Markdown content (without frontmatter).
        tags: List of tags for the entry.
        category: Top-level category directory (e.g., "development", "devops").
                  Deprecated - use 'directory' for nested paths.
        directory: Full directory path (e.g., "development/python/frameworks").
                   Takes precedence over 'category' if both are provided.
        links: Optional list of paths to link to using [[link]] syntax.
        kb_context: Optional project context. If not provided, auto-discovered from cwd.
                   Used for default directory (primary) and tag suggestions (default_tags).
        scope: Optional scope for KB selection ("project" or "user").
               If not provided, uses auto-discovery (project KB if in project, else user KB).
        keywords: Optional list of key concepts for semantic linking.
        semantic_links: Optional list of SemanticLink objects for manual linking.
        relations: Optional list of RelationLink objects for typed relations.

    Returns:
        Dict with 'path' of created file, 'suggested_links' to consider adding,
        and 'suggested_tags' based on content similarity and existing taxonomy.
    """
    # Determine KB root based on scope parameter or auto-discovery
    if scope:
        kb_root = get_kb_root_by_scope(scope)
    else:
        kb_root = get_kb_root()

    # Auto-discover KB context if not provided
    if kb_context is None:
        kb_context = get_kb_context()

    # Determine target directory: prefer 'directory' over 'category' over context.primary
    if directory:
        # Validate the directory is within KB, auto-create if needed
        abs_dir, normalized_dir = validate_nested_path(directory)
        if abs_dir.exists() and not abs_dir.is_dir():
            raise ValueError(f"Path exists but is not a directory: {directory}")
        # Auto-create directory if it doesn't exist
        abs_dir.mkdir(parents=True, exist_ok=True)
        target_dir = abs_dir
        rel_dir = normalized_dir
    elif category:
        # Auto-create category directory if it doesn't exist
        target_dir = kb_root / category
        target_dir.mkdir(parents=True, exist_ok=True)
        rel_dir = category
    elif kb_context and kb_context.primary:
        # Use context primary directory
        abs_dir, normalized_dir = validate_nested_path(kb_context.primary)
        if abs_dir.exists() and not abs_dir.is_dir():
            raise ValueError(f"Context primary path exists but is not a directory: {kb_context.primary}")
        # Auto-create directory if it doesn't exist
        abs_dir.mkdir(parents=True, exist_ok=True)
        target_dir = abs_dir
        rel_dir = normalized_dir
    else:
        valid_categories = get_valid_categories()
        context_hint = (
            " (no .kbcontext file found with 'primary' field)"
            if kb_context is None
            else ""
        )
        raise ValueError(
            f"Either 'category' or 'directory' must be provided{context_hint}. "
            f"Existing categories: {', '.join(valid_categories)}"
        )

    if not tags:
        raise ValueError("At least one tag is required")

    # Check A-Mem strict mode - require keywords if enabled
    if is_amem_strict_enabled() and not keywords:
        raise AMEMStrictError(AMEM_STRICT_ERROR_MESSAGE)

    # Generate slug and path
    slug = slugify(title)
    if not slug:
        raise ValueError("Title must contain at least one alphanumeric character")

    file_path = target_dir / f"{slug}.md"
    rel_path = f"{rel_dir}/{slug}.md"

    if file_path.exists():
        raise ValueError(f"Entry already exists at {rel_path}")

    # Add links to content if specified
    final_content = content
    if links:
        link_section = "\n\n## Related\n\n"
        for link in links:
            link_section += f"- [[{link}]]\n"
        final_content += link_section

    # Build metadata and frontmatter using utilities
    metadata = create_new_metadata(
        title=title,
        tags=tags,
        source_project=get_current_project(),
        contributor=get_current_contributor(),
        model=get_llm_model(),
        git_branch=get_git_branch(),
        actor=get_actor_identity(),
        keywords=keywords,
        semantic_links=semantic_links,
        relations=relations,
    )
    frontmatter = build_frontmatter(metadata)

    # Write the file
    file_path.write_text(frontmatter + final_content, encoding="utf-8")
    rebuild_backlink_cache(kb_root)

    # Reindex the new file
    try:
        _, _, chunks = parse_entry(file_path)
        searcher = get_searcher()
        if chunks:
            relative_path = relative_kb_path(kb_root, file_path)
            normalized_chunks = normalize_chunks(chunks, relative_path)
            searcher.index_chunks(normalized_chunks)
    except ParseError as e:
        log.warning("Created entry but failed to index %s: %s", rel_path, e)

    # Auto-create bidirectional semantic links if enabled and no manual links provided
    if semantic_links is None and SEMANTIC_LINK_ENABLED:
        linking_result = create_bidirectional_semantic_links(
            entry_path=rel_path,
            title=title,
            content=final_content,
            tags=tags,
        )
        if linking_result.forward_links:
            # Re-save entry with auto-generated semantic links
            try:
                updated_metadata = metadata.model_copy(update={"semantic_links": linking_result.forward_links})
                updated_frontmatter = build_frontmatter(updated_metadata)
                file_path.write_text(updated_frontmatter + final_content, encoding="utf-8")

                # Re-index with updated metadata
                _, _, updated_chunks = parse_entry(file_path)
                if updated_chunks:
                    searcher = get_searcher()
                    searcher.delete_document(rel_path)
                    normalized_chunks = normalize_chunks(updated_chunks, rel_path)
                    searcher.index_chunks(normalized_chunks)
            except (ParseError, OSError) as e:
                log.warning("Failed to update entry with auto semantic links %s: %s", rel_path, e)

        # Queue memory evolution for linked neighbors (non-blocking)
        if linking_result.neighbors_for_evolution:
            _queue_evolution(
                new_entry_path=rel_path,
                neighbors_to_evolve=linking_result.neighbors_for_evolution,
                kb_root=kb_root,
            )

            # Strengthen new entry based on neighbor context (A-Mem parity)
            evolution_config = get_memory_evolution_config()
            if evolution_config.strengthen_on_add:
                try:
                    # Build NeighborInfo objects for strengthen analysis
                    strengthen_neighbors = _build_neighbor_info_for_strengthen(
                        kb_root=kb_root,
                        neighbors_to_evolve=linking_result.neighbors_for_evolution,
                    )

                    if strengthen_neighbors:
                        from .llm import analyze_for_strengthen

                        # Re-read current metadata (may have been updated with semantic_links)
                        current_metadata, current_content, _ = parse_entry(file_path)

                        strengthen_result = await analyze_for_strengthen(
                            new_entry_content=current_content,
                            new_entry_keywords=list(current_metadata.keywords),
                            new_entry_title=current_metadata.title,
                            neighbors=strengthen_neighbors,
                            model=evolution_config.model,
                        )

                        if strengthen_result.should_strengthen:
                            # Apply strengthened keywords
                            updates: dict = {"keywords": strengthen_result.new_keywords}

                            # Add suggested links to semantic_links if not already present
                            if strengthen_result.suggested_links:
                                existing_link_paths = {
                                    sl.path for sl in (current_metadata.semantic_links or [])
                                }
                                new_links = [
                                    SemanticLink(
                                        path=p,
                                        score=0.8,  # High score for LLM-suggested links
                                        reason="strengthen_suggested",
                                    )
                                    for p in strengthen_result.suggested_links
                                    if p not in existing_link_paths
                                ]
                                if new_links:
                                    all_links = list(current_metadata.semantic_links or []) + new_links
                                    updates["semantic_links"] = all_links

                            # Re-save with strengthened metadata
                            strengthened_metadata = current_metadata.model_copy(update=updates)
                            strengthened_frontmatter = build_frontmatter(strengthened_metadata)
                            file_path.write_text(
                                strengthened_frontmatter + current_content, encoding="utf-8"
                            )

                            # Re-index with updated metadata
                            _, _, strengthened_chunks = parse_entry(file_path)
                            if strengthened_chunks:
                                searcher = get_searcher()
                                searcher.delete_document(rel_path)
                                normalized = normalize_chunks(strengthened_chunks, rel_path)
                                searcher.index_chunks(normalized)

                            log.info(
                                "Strengthened entry %s: keywords=%s",
                                rel_path,
                                strengthen_result.new_keywords,
                            )
                except Exception as e:
                    log.warning("Failed to strengthen entry %s: %s", rel_path, e)

    # Compute link suggestions for the new entry
    existing_links = set(links) if links else set()
    suggested_links = compute_link_suggestions(
        title=title,
        content=final_content,
        tags=tags,
        self_path=rel_path,
        existing_links=existing_links,
        limit=3,  # Suggest up to 3 high-confidence links
        min_score=0.5,
    )

    # Compute tag suggestions for additional tags to consider
    suggested_tags = compute_tag_suggestions(
        title=title,
        content=final_content,
        existing_tags=tags,
        limit=5,
        min_score=0.3,
    )

    # Collect default_tags from .kbconfig (KB-level) and .kbcontext (project-level)
    existing_tag_set = set(tags)
    suggested_tag_set = {s["tag"] for s in suggested_tags}
    config_suggestions = []

    # Check for .kbconfig in the target directory (KB-level default_tags)
    kbconfig = get_kbconfig(target_dir)
    if kbconfig and kbconfig.default_tags:
        for tag in kbconfig.default_tags:
            if tag not in existing_tag_set and tag not in suggested_tag_set:
                config_suggestions.append({
                    "tag": tag,
                    "score": 1.0,  # High priority for KB config tags
                    "reason": "From .kbconfig",
                })
                suggested_tag_set.add(tag)

    # Also check .kbcontext default_tags (project-level, for backwards compatibility)
    if kb_context and kb_context.default_tags:
        for tag in kb_context.default_tags:
            if tag not in existing_tag_set and tag not in suggested_tag_set:
                config_suggestions.append({
                    "tag": tag,
                    "score": 1.0,  # High priority for context tags
                    "reason": "From project .kbcontext",
                })
                suggested_tag_set.add(tag)

    # Prepend config suggestions to semantic suggestions
    suggested_tags = config_suggestions + suggested_tags

    return {"path": rel_path, "suggested_links": suggested_links, "suggested_tags": suggested_tags}


async def preview_add_entry(
    title: str,
    content: str,
    tags: list[str],
    category: str = "",
    directory: str | None = None,
    links: list[str] | None = None,
    kb_context: KBContext | None = None,
    check_duplicates: bool = True,
    force: bool = False,
) -> AddEntryPreview:
    """Preview a new KB entry without writing to disk.

    Returns path, generated frontmatter, final content, and any duplicate warnings.

    Args:
        title: Entry title.
        content: Markdown content (without frontmatter).
        tags: List of tags for the entry.
        category: Top-level category directory (deprecated - use 'directory').
        directory: Full directory path (e.g., "development/python/frameworks").
        links: Optional list of paths to link to using [[link]] syntax.
        kb_context: Optional project context.
        check_duplicates: If True, check for potential duplicates.
        force: If True, skip duplicate warnings.

    Returns:
        AddEntryPreview with path, frontmatter, content, and duplicate info.

    Raises:
        ValueError: If entry already exists or validation fails.
    """
    kb_root = get_kb_root()

    if kb_context is None:
        kb_context = get_kb_context()

    # Determine target directory
    if directory:
        abs_dir, normalized_dir = validate_nested_path(directory)
        target_dir = abs_dir
        rel_dir = normalized_dir
    elif category:
        target_dir = kb_root / category
        rel_dir = category
    elif kb_context and kb_context.primary:
        abs_dir, normalized_dir = validate_nested_path(kb_context.primary)
        target_dir = abs_dir
        rel_dir = normalized_dir
    else:
        valid_categories = get_valid_categories()
        raise ValueError(
            f"Either 'category' or 'directory' must be provided. "
            f"Existing categories: {', '.join(valid_categories)}"
        )

    if not tags:
        raise ValueError("At least one tag is required")

    slug = slugify(title)
    if not slug:
        raise ValueError("Title must contain at least one alphanumeric character")

    file_path = target_dir / f"{slug}.md"
    rel_path = f"{rel_dir}/{slug}.md"

    if file_path.exists():
        raise ValueError(f"Entry already exists at {rel_path}")

    potential_duplicates: list[PotentialDuplicate] = []
    warning = None
    if check_duplicates and not force:
        searcher = get_searcher()
        potential_duplicates = detect_potential_duplicates(title, content, searcher)
        if potential_duplicates:
            top_match = potential_duplicates[0]
            warning = (
                f"Potential duplicate detected: '{top_match.title}' ({top_match.path}) "
                f"with {top_match.score:.0%} similarity. "
                f"Use force=True to create anyway, or update the existing entry."
            )

    final_content = content
    if links:
        link_section = "\n\n## Related\n\n"
        for link in links:
            link_section += f"- [[{link}]]\n"
        final_content += link_section

    metadata = create_new_metadata(
        title=title,
        tags=tags,
        source_project=get_current_project(),
        contributor=get_current_contributor(),
        model=get_llm_model(),
        git_branch=get_git_branch(),
        actor=get_actor_identity(),
    )
    frontmatter = build_frontmatter(metadata)

    return AddEntryPreview(
        path=rel_path,
        absolute_path=str(file_path),
        frontmatter=frontmatter,
        content=final_content,
        potential_duplicates=potential_duplicates,
        warning=warning,
    )


async def ingest_file(
    file_path: str | Path,
    title: str | None = None,
    tags: list[str] | None = None,
    directory: str | None = None,
    scope: str | None = None,
    dry_run: bool = False,
) -> IngestResult:
    """Ingest a markdown file into the knowledge base.

    Takes an existing markdown file, prepends frontmatter if missing,
    and moves it to the KB directory if it's not already there.

    Args:
        file_path: Path to the markdown file to ingest.
        title: Optional title override. If not provided, extracted from first H1 or filename.
        tags: Optional tags. If not provided, defaults to empty list with suggestions.
        directory: Target directory within KB. If not provided, uses context.primary.
        scope: KB scope ("project" or "user"). If not provided, auto-detects.
        dry_run: If True, show what would be done without making changes.

    Returns:
        IngestResult with path, moved status, frontmatter_added status.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is not a markdown file or validation fails.
    """
    import frontmatter as fm

    source_path = Path(file_path).resolve()

    # Validate file exists and is markdown
    if not source_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not source_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    if source_path.suffix.lower() not in ('.md', '.markdown'):
        raise ValueError(f"File must be markdown (.md or .markdown): {file_path}")

    # Determine KB root
    if scope:
        kb_root = get_kb_root_by_scope(scope)
    else:
        kb_root = get_kb_root()

    kb_root = kb_root.resolve()

    # Load file and check for existing frontmatter
    raw_content = source_path.read_text(encoding='utf-8')
    post = fm.loads(raw_content)
    has_frontmatter = bool(post.metadata)

    # Extract or derive title
    if title:
        entry_title = title
    elif has_frontmatter and post.metadata.get('title'):
        entry_title = post.metadata['title']
    else:
        # Try to extract from first H1 heading
        h1_match = re.match(r'^#\s+(.+)$', post.content.strip(), re.MULTILINE)
        if h1_match:
            entry_title = h1_match.group(1).strip()
        else:
            # Fall back to filename without extension
            entry_title = source_path.stem.replace('-', ' ').replace('_', ' ').title()

    # Determine tags
    if tags:
        entry_tags = tags
    elif has_frontmatter and post.metadata.get('tags'):
        existing_tags = post.metadata['tags']
        entry_tags = existing_tags if isinstance(existing_tags, list) else [existing_tags]
    else:
        # Default to empty list - tags will be suggested
        entry_tags = []

    # Check if file is already within KB
    try:
        source_path.relative_to(kb_root)
        is_in_kb = True
    except ValueError:
        is_in_kb = False

    # Determine target location
    if is_in_kb:
        # File is already in KB, keep it in place
        target_path = source_path
        rel_dir = str(source_path.parent.relative_to(kb_root))
    else:
        # File needs to be moved into KB
        kb_context = get_kb_context()

        if directory:
            abs_dir, normalized_dir = validate_nested_path(directory)
            target_dir = abs_dir
            rel_dir = normalized_dir
        elif kb_context and kb_context.primary:
            abs_dir, normalized_dir = validate_nested_path(kb_context.primary)
            target_dir = abs_dir
            rel_dir = normalized_dir
        else:
            # Default to root of KB
            target_dir = kb_root
            rel_dir = ""

        # Generate slug for filename if moving
        slug = slugify(entry_title)
        if not slug:
            slug = source_path.stem  # Fall back to original filename

        target_path = target_dir / f"{slug}.md"

        # Check for collision
        if target_path.exists() and target_path != source_path:
            raise ValueError(f"Target file already exists: {target_path.relative_to(kb_root)}")

    # Compute relative path within KB
    rel_path = str(target_path.relative_to(kb_root))

    # Build frontmatter if needed
    frontmatter_added = False
    if not has_frontmatter or not entry_tags:
        # Need to add or update frontmatter
        metadata = create_new_metadata(
            title=entry_title,
            tags=entry_tags if entry_tags else ["untagged"],  # Require at least one tag
            source_project=get_current_project(),
            contributor=get_current_contributor(),
            model=get_llm_model(),
            git_branch=get_git_branch(),
            actor=get_actor_identity(),
        )

        # If there was existing frontmatter with extra fields, preserve them
        if has_frontmatter:
            # Preserve fields that create_new_metadata doesn't set
            if post.metadata.get('description'):
                metadata.description = post.metadata['description']
            if post.metadata.get('aliases'):
                metadata.aliases = post.metadata['aliases']
            if post.metadata.get('status'):
                metadata.status = post.metadata['status']
            # Preserve original created date if present
            if post.metadata.get('created'):
                try:
                    from datetime import datetime
                    created = post.metadata['created']
                    if isinstance(created, datetime):
                        metadata.created = created
                except Exception:
                    pass

        new_frontmatter = build_frontmatter(metadata)
        new_content = new_frontmatter + post.content
        frontmatter_added = not has_frontmatter
    else:
        new_content = raw_content

    # Compute tag suggestions
    suggested_tags = []
    if entry_tags == ["untagged"] or not entry_tags:
        suggested_tags = compute_tag_suggestions(
            title=entry_title,
            content=post.content,
            existing_tags=entry_tags,
            limit=5,
            min_score=0.3,
        )

    if dry_run:
        return IngestResult(
            path=rel_path,
            absolute_path=str(target_path),
            moved=not is_in_kb,
            frontmatter_added=frontmatter_added,
            original_path=str(source_path) if not is_in_kb else None,
            title=entry_title,
            tags=entry_tags if entry_tags else ["untagged"],
            suggested_tags=suggested_tags,
        )

    # Write the file
    if not is_in_kb:
        # Create target directory if needed
        target_path.parent.mkdir(parents=True, exist_ok=True)

    target_path.write_text(new_content, encoding='utf-8')

    # If moved, delete original
    if not is_in_kb and source_path.exists() and source_path != target_path:
        source_path.unlink()

    # Rebuild backlink cache and index
    rebuild_backlink_cache(kb_root)

    try:
        _, _, chunks = parse_entry(target_path)
        searcher = get_searcher()
        if chunks:
            normalized_chunks = normalize_chunks(chunks, rel_path)
            searcher.index_chunks(normalized_chunks)
    except ParseError as e:
        log.warning("Ingested file but failed to index %s: %s", rel_path, e)

    return IngestResult(
        path=rel_path,
        absolute_path=str(target_path),
        moved=not is_in_kb,
        frontmatter_added=frontmatter_added,
        original_path=str(source_path) if not is_in_kb else None,
        title=entry_title,
        tags=entry_tags if entry_tags else ["untagged"],
        suggested_tags=suggested_tags,
    )


async def append_entry(
    title: str,
    content: str,
    tags: list[str] | None = None,
    category: str = "",
    directory: str | None = None,
    no_create: bool = False,
    kb_context: KBContext | None = None,
) -> dict:
    """Append content to an existing entry by title, or create new if not found.

    This is the 'append' operation - a user-friendly way to add content to an
    existing entry without knowing its exact path.

    Args:
        title: Entry title to search for.
        content: Content to append (or use as initial content for new entries).
        tags: Tags for new entries (required if creating new entry).
        category: Category for new entries (e.g., "development").
        directory: Directory for new entries (takes precedence over category).
        no_create: If True, error if entry not found instead of creating.
        kb_context: Optional project context. If not provided, auto-discovered.

    Returns:
        Dict with 'path', 'action' ('appended' or 'created'), and suggestions.

    Raises:
        ValueError: If entry not found with no_create=True, or missing required
                    fields for creating new entry.
    """
    kb_root = get_kb_root()

    # Auto-discover KB context if not provided
    if kb_context is None:
        kb_context = get_kb_context()

    # Search for existing entry by title
    existing_entry: Path | None = None
    existing_path: str | None = None

    for md_file in kb_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue
        try:
            metadata, _, _ = parse_entry(md_file)
            if metadata.title.lower() == title.lower():
                existing_entry = md_file
                existing_path = str(md_file.relative_to(kb_root))
                break
        except ParseError:
            continue

    if existing_entry is None:
        # Entry not found
        if no_create:
            raise ValueError(f"Entry not found with title: {title}")

        # Create new entry - tags are required
        if not tags:
            raise ValueError("Tags are required when creating a new entry")

        result = await add_entry(
            title=title,
            content=content,
            tags=tags,
            category=category,
            directory=directory,
            kb_context=kb_context,
        )
        result["action"] = "created"
        return result

    # Entry exists - append content
    try:
        metadata, existing_content, _ = parse_entry(existing_entry)
    except ParseError as e:
        raise ValueError(f"Failed to parse existing entry: {e}") from e

    # Append content with separator
    new_content = existing_content.rstrip() + "\n\n" + content

    # Update metadata (preserve tags unless new ones provided)
    new_tags = tags if tags is not None else list(metadata.tags)
    if not new_tags:
        raise ValueError("At least one tag is required")

    updated_metadata = update_metadata_for_edit(
        metadata,
        new_tags=new_tags,
        new_contributor=get_current_contributor(),
        edit_source=get_current_project(),
        model=get_llm_model(),
        git_branch=get_git_branch(),
        actor=get_actor_identity(),
    )
    frontmatter = build_frontmatter(updated_metadata)

    # Write updated file
    existing_entry.write_text(frontmatter + new_content, encoding="utf-8")
    rebuild_backlink_cache(kb_root)

    # At this point existing_entry is not None, so existing_path is also set
    assert existing_path is not None

    # Reindex
    searcher = get_searcher()
    try:
        searcher.delete_document(existing_path)
        _, _, chunks = parse_entry(existing_entry)
        if chunks:
            normalized_chunks = normalize_chunks(chunks, existing_path)
            searcher.index_chunks(normalized_chunks)
    except ParseError as e:
        log.warning("Appended to entry but failed to re-index %s: %s", existing_path, e)
    except Exception as e:
        log.error("Unexpected error re-indexing %s: %s", existing_path, e)

    # Compute suggestions (existing_path was asserted above)
    existing_links = set(extract_links(new_content))
    suggested_links = compute_link_suggestions(
        title=metadata.title,
        content=new_content,
        tags=new_tags,
        self_path=existing_path,
        existing_links=existing_links,
        limit=3,
        min_score=0.5,
    )

    suggested_tags = compute_tag_suggestions(
        title=metadata.title,
        content=new_content,
        existing_tags=new_tags,
        limit=5,
        min_score=0.3,
    )

    return {
        "path": existing_path,
        "action": "appended",
        "suggested_links": suggested_links,
        "suggested_tags": suggested_tags,
    }


async def update_entry(
    path: str,
    content: str | None = None,
    tags: list[str] | None = None,
    section_updates: dict[str, str] | None = None,
    keywords: list[str] | None = None,
    semantic_links: list | None = None,
    relations: list | None = None,
) -> dict:
    """Update an existing KB entry.

    Args:
        path: Path to the entry relative to KB root (e.g., "development/python-tooling.md").
        content: New markdown content (without frontmatter).
        tags: Optional new list of tags. If provided, replaces existing tags.
        section_updates: Optional dict of section heading -> new content.
        keywords: Optional new list of keywords. If provided, replaces existing keywords.
        semantic_links: Optional list of SemanticLink objects. If provided, replaces existing.
        relations: Optional list of RelationLink objects. If provided, replaces existing.

    Returns:
        Dict with 'path' of updated file, 'suggested_links' to consider adding,
        and 'suggested_tags' based on content similarity and existing taxonomy.
    """
    kb_root = get_kb_root()
    file_path = kb_root / path

    if not file_path.exists():
        raise ValueError(f"Entry not found: {path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if (
        content is None
        and not section_updates
        and tags is None
        and keywords is None
        and semantic_links is None
        and relations is None
    ):
        raise ValueError("Provide new content, section_updates, tags, keywords, semantic_links, or relations")

    # Parse existing entry to get metadata
    try:
        metadata, existing_content, _ = parse_entry(file_path)
    except ParseError as e:
        raise ValueError(f"Failed to parse existing entry: {e}") from e

    # Update metadata using utilities
    new_tags = tags if tags is not None else list(metadata.tags)
    if not new_tags:
        raise ValueError("At least one tag is required")

    # Check A-Mem strict mode - require keywords when content changes
    # Only enforced if neither existing entry nor update provides keywords
    content_changing = content is not None or section_updates is not None
    has_existing_keywords = bool(metadata.keywords)
    has_new_keywords = bool(keywords)
    needs_keywords = content_changing and not has_existing_keywords and not has_new_keywords
    if is_amem_strict_enabled() and needs_keywords:
        raise AMEMStrictError(AMEM_STRICT_ERROR_MESSAGE)

    updated_metadata = update_metadata_for_edit(
        metadata,
        new_tags=new_tags,
        new_contributor=get_current_contributor(),
        edit_source=get_current_project(),
        model=get_llm_model(),
        git_branch=get_git_branch(),
        actor=get_actor_identity(),
        keywords=keywords,
        semantic_links=semantic_links,
        relations=relations,
    )
    frontmatter = build_frontmatter(updated_metadata)

    new_content = content if content is not None else existing_content
    if section_updates:
        new_content = apply_section_updates(new_content, section_updates)

    # Write updated file
    file_path.write_text(frontmatter + new_content, encoding="utf-8")
    rebuild_backlink_cache(kb_root)

    relative_path = relative_kb_path(kb_root, file_path)

    # Reindex
    searcher = get_searcher()
    try:
        # Remove old index entries
        searcher.delete_document(relative_path)
        # Parse and index new content
        _, _, chunks = parse_entry(file_path)
        if chunks:
            normalized_chunks = normalize_chunks(chunks, relative_path)
            searcher.index_chunks(normalized_chunks)
    except ParseError as e:
        log.warning("Updated entry but failed to re-index %s: %s", relative_path, e)
    except Exception as e:
        log.error("Unexpected error re-indexing %s: %s", relative_path, e)

    # Auto-create bidirectional semantic links if enabled and no manual links provided
    # Only run when content is updated (not just tags/keywords)
    content_changed = content is not None or section_updates
    if semantic_links is None and content_changed and SEMANTIC_LINK_ENABLED:
        linking_result = create_bidirectional_semantic_links(
            entry_path=relative_path,
            title=updated_metadata.title,
            content=new_content,
            tags=new_tags,
        )
        if linking_result.forward_links:
            # Re-save entry with auto-generated semantic links
            try:
                final_metadata = updated_metadata.model_copy(update={"semantic_links": linking_result.forward_links})
                final_frontmatter = build_frontmatter(final_metadata)
                file_path.write_text(final_frontmatter + new_content, encoding="utf-8")

                # Re-index with updated metadata
                _, _, final_chunks = parse_entry(file_path)
                if final_chunks:
                    searcher.delete_document(relative_path)
                    normalized_chunks = normalize_chunks(final_chunks, relative_path)
                    searcher.index_chunks(normalized_chunks)
            except (ParseError, OSError) as e:
                log.warning(
                    "Failed to update entry with auto semantic links %s: %s",
                    relative_path, e
                )

        # Queue memory evolution for linked neighbors (non-blocking)
        if linking_result.neighbors_for_evolution:
            _queue_evolution(
                new_entry_path=relative_path,
                neighbors_to_evolve=linking_result.neighbors_for_evolution,
                kb_root=kb_root,
            )

    # Compute link suggestions based on updated content
    existing_links = set(extract_links(new_content))
    suggested_links = compute_link_suggestions(
        title=metadata.title,
        content=new_content,
        tags=new_tags,
        self_path=relative_path,
        existing_links=existing_links,
        limit=3,  # Suggest up to 3 high-confidence links
        min_score=0.5,
    )

    # Compute tag suggestions for additional tags to consider
    suggested_tags = compute_tag_suggestions(
        title=metadata.title,
        content=new_content,
        existing_tags=new_tags,
        limit=5,
        min_score=0.3,
    )

    return {"path": relative_path, "suggested_links": suggested_links, "suggested_tags": suggested_tags}


async def get_entry(path: str) -> KBEntry:
    """Read a KB entry.

    Args:
        path: Path to the entry relative to KB root (e.g., "development/python-tooling.md").
              Supports scoped paths like "@project/path.md" or "@user/path.md".

    Returns:
        KBEntry with metadata, content, links, and backlinks.
    """
    from .config import resolve_scoped_path

    # Resolve scoped paths (handles @project/ and @user/ prefixes)
    file_path = resolve_scoped_path(path)

    if not file_path.exists():
        raise ValueError(f"Entry not found: {path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    try:
        metadata, content, _ = parse_entry(file_path)
    except ParseError as e:
        raise ValueError(f"Failed to parse entry: {e}") from e

    # Extract links from content
    links = extract_links(content)

    # Get backlinks
    all_backlinks = get_backlink_index()
    # Normalize path for lookup (remove .md extension)
    path_key = path[:-3] if path.endswith(".md") else path
    entry_backlinks = all_backlinks.get(path_key, [])

    # Record the view (non-blocking, fire-and-forget)
    try:
        from .views_tracker import record_view

        record_view(path)
    except Exception as e:
        log.debug("Failed to record view for %s: %s", path, e)

    return KBEntry(
        path=path,
        metadata=metadata,
        content=content,
        links=links,
        backlinks=entry_backlinks,
    )


async def list_entries(
    category: str | None = None,
    directory: str | None = None,
    tag: str | None = None,
    recursive: bool = True,
    limit: int = 20,
    scope: str | None = None,
) -> list[dict]:
    """List KB entries.

    Args:
        category: Optional top-level category to filter by (e.g., "development").
        directory: Optional directory path to filter by (e.g., "development/python").
                   Takes precedence over 'category' if both are provided.
        tag: Optional tag to filter by.
        recursive: If True (default), include entries in subdirectories.
                   If False, only list direct children of the directory.
        limit: Maximum number of entries to return (default 20).
        scope: Filter to specific scope - "project", "user", or None for all.

    Returns:
        List of {path, title, tags, created, updated} dictionaries.
    """
    from .config import get_kb_roots_for_indexing

    kb_roots = get_kb_roots_for_indexing(scope=scope)
    if not kb_roots:
        return []

    results = []

    for scope_label, kb_root in kb_roots:
        if not kb_root.exists():
            continue

        # Determine search path: prefer 'directory' over 'category'
        if directory:
            search_path = kb_root / directory
            if not search_path.exists() or not search_path.is_dir():
                continue  # Skip this KB if directory doesn't exist
        elif category:
            search_path = kb_root / category
            if not search_path.exists() or not search_path.is_dir():
                continue  # Skip this KB if category doesn't exist
        else:
            search_path = kb_root

        # Choose glob pattern based on recursive flag
        md_files = search_path.rglob("*.md") if recursive else search_path.glob("*.md")

        for md_file in md_files:
            # Skip index files
            if md_file.name.startswith("_"):
                continue

            rel_path = str(md_file.relative_to(kb_root))
            # Add scope prefix for multi-KB mode
            display_path = f"@{scope_label}/{rel_path}" if scope_label else rel_path

            try:
                metadata, _, _ = parse_entry(md_file)
            except ParseError:
                continue

            # Filter by tag if specified
            if tag and tag not in metadata.tags:
                continue

            results.append(
                {
                    "path": display_path,
                    "title": metadata.title,
                    "tags": metadata.tags,
                    "created": metadata.created.isoformat() if metadata.created else None,
                    "updated": metadata.updated.isoformat() if metadata.updated else None,
                }
            )

            if len(results) >= limit:
                return results

    return results


async def find_entries_by_title(
    title: str,
    exact: bool = True,
) -> list[dict]:
    """Find KB entries by title.

    Args:
        title: Title to search for.
        exact: If True (default), require exact case-insensitive match.
               If False, include partial matches.

    Returns:
        List of {path, title, tags} dictionaries for matching entries.
    """
    kb_root = get_kb_root()

    if not kb_root.exists():
        return []

    results = []
    title_lower = title.lower()

    for md_file in kb_root.rglob("*.md"):
        # Skip index files
        if md_file.name.startswith("_"):
            continue

        try:
            metadata, _, _ = parse_entry(md_file)
        except ParseError:
            continue

        entry_title_lower = metadata.title.lower()

        if exact:
            if entry_title_lower == title_lower:
                rel_path = str(md_file.relative_to(kb_root))
                results.append({
                    "path": rel_path,
                    "title": metadata.title,
                    "tags": metadata.tags,
                })
        else:
            if title_lower in entry_title_lower:
                rel_path = str(md_file.relative_to(kb_root))
                results.append({
                    "path": rel_path,
                    "title": metadata.title,
                    "tags": metadata.tags,
                })

    return results


async def get_similar_titles(title: str, limit: int = 5) -> list[str]:
    """Get titles similar to the given title for suggestions.

    Args:
        title: Title to find similar matches for.
        limit: Maximum number of suggestions.

    Returns:
        List of similar titles.
    """
    import difflib

    kb_root = get_kb_root()

    if not kb_root.exists():
        return []

    all_titles = []
    for md_file in kb_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue
        try:
            metadata, _, _ = parse_entry(md_file)
            all_titles.append(metadata.title)
        except ParseError:
            continue

    return difflib.get_close_matches(title, all_titles, n=limit, cutoff=0.4)


async def whats_new(
    days: int = 30,
    limit: int = 10,
    include_created: bool = True,
    include_updated: bool = True,
    category: str | None = None,
    tag: str | None = None,
    scope: str | None = None,
) -> list[dict]:
    """List recent KB entries.

    Args:
        days: Look back period in days (default 30).
        limit: Maximum entries to return (default 10).
        include_created: Include newly created entries (default True).
        include_updated: Include recently updated entries (default True).
        category: Optional category filter.
        tag: Optional tag filter.
        scope: Filter to specific scope - "project", "user", or None for all.

    Returns:
        List of {path, title, tags, created, updated, activity_type, activity_date, source_project}.
    """
    from .config import get_kb_roots_for_indexing

    kb_roots = get_kb_roots_for_indexing(scope=scope)
    if not kb_roots:
        return []

    cutoff_datetime = datetime.now(UTC) - timedelta(days=days)
    candidates: list[dict] = []

    for scope_label, kb_root in kb_roots:
        if not kb_root.exists():
            continue

        # Determine search path
        if category:
            search_path = kb_root / category
            if not search_path.exists() or not search_path.is_dir():
                continue  # Skip this KB if category doesn't exist
        else:
            search_path = kb_root

        for md_file in search_path.rglob("*.md"):
            if md_file.name.startswith("_"):
                continue

            rel_path = str(md_file.relative_to(kb_root))
            # Add scope prefix for multi-KB mode
            display_path = f"@{scope_label}/{rel_path}" if scope_label else rel_path

            try:
                metadata, _, _ = parse_entry(md_file)
            except ParseError:
                continue

            # Filter by tag if specified
            if tag and tag not in metadata.tags:
                continue

            # Determine activity type and datetime
            activity_type: str | None = None
            activity_date: datetime | None = None

            # Normalize dates to aware (assume UTC for naive dates from legacy entries)
            created_aware = _ensure_aware(metadata.created)
            updated_aware = _ensure_aware(metadata.updated)

            # Check updated first (takes precedence if both qualify)
            if include_updated and updated_aware and updated_aware >= cutoff_datetime:
                activity_type = "updated"
                activity_date = updated_aware
            elif include_created and created_aware and created_aware >= cutoff_datetime:
                activity_type = "created"
                activity_date = created_aware

            if activity_type is None or activity_date is None:
                continue

            candidates.append(
                {
                    "path": display_path,
                    "title": metadata.title,
                    "tags": metadata.tags,
                    "created": metadata.created.isoformat() if metadata.created else None,
                    "updated": metadata.updated.isoformat() if metadata.updated else None,
                    "activity_type": activity_type,
                    "activity_date": activity_date.isoformat(),
                    "source_project": metadata.source_project,
                }
            )

    # Sort by activity_date descending
    candidates.sort(key=lambda x: x["activity_date"], reverse=True)

    return candidates[:limit]


async def popular(
    limit: int = 10,
    days: int | None = None,
    category: str | None = None,
    tag: str | None = None,
) -> list[dict]:
    """List popular KB entries by view count.

    Args:
        limit: Maximum entries to return (default 10).
        days: Optional time window (None = all time).
        category: Optional category filter.
        tag: Optional tag filter.

    Returns:
        List of {path, title, tags, view_count, last_viewed}.
    """
    from .views_tracker import get_popular

    kb_root = get_kb_root()

    if not kb_root.exists():
        return []

    # Get popular entries (fetch extra to allow for filtering)
    popular_entries = get_popular(limit=limit * 3, days=days)

    results: list[dict] = []

    for entry_path, stats in popular_entries:
        file_path = kb_root / entry_path

        if not file_path.exists():
            continue

        # Apply category filter
        if category:
            parts = entry_path.split("/")
            if len(parts) < 2 or parts[0] != category:
                continue

        try:
            metadata, _, _ = parse_entry(file_path)
        except ParseError:
            continue

        # Filter by tag if specified
        if tag and tag not in metadata.tags:
            continue

        results.append(
            {
                "path": entry_path,
                "title": metadata.title,
                "tags": metadata.tags,
                "view_count": stats.total_views,
                "last_viewed": stats.last_viewed.isoformat()
                if stats.last_viewed
                else None,
            }
        )

        if len(results) >= limit:
            break

    return results


async def backlinks(path: str) -> list[str]:
    """Find entries that link to this path.

    Args:
        path: Path to the entry (e.g., "development/python-tooling.md"
            or "development/python-tooling").

    Returns:
        List of paths that link to this entry.
    """
    # Normalize path (remove .md extension for lookup)
    path_key = path[:-3] if path.endswith(".md") else path

    all_backlinks = get_backlink_index()
    return all_backlinks.get(path_key, [])


async def reindex(scope: str | None = None) -> IndexStatus:
    """Rebuild search indices.

    Args:
        scope: Filter to specific scope - "project", "user", or None for all.

    Returns:
        IndexStatus with document counts.
    """
    from .config import get_kb_roots_for_indexing

    searcher = get_searcher()
    kb_roots = get_kb_roots_for_indexing(scope=scope)

    # Index all KBs
    if kb_roots:
        searcher.reindex(kb_roots=kb_roots)

    # Rebuild backlink cache for each KB
    for scope, kb_root in kb_roots:
        rebuild_backlink_cache(kb_root)

    # Clean up views for deleted entries (collect from all KBs)
    try:
        from .views_tracker import cleanup_stale_entries

        valid_paths = set()
        for scope, kb_root in kb_roots:
            for p in kb_root.rglob("*.md"):
                if not p.name.startswith("_"):
                    relative = str(p.relative_to(kb_root))
                    # Add scope prefix if in multi-KB mode
                    if scope:
                        valid_paths.add(f"@{scope}/{relative}")
                    else:
                        valid_paths.add(relative)
        cleanup_stale_entries(valid_paths)
    except Exception as e:
        log.warning("Failed to cleanup stale view entries: %s", e)

    status = searcher.status()
    return status


async def tree(
    path: str = "",
    depth: int = 3,
    include_files: bool = True,
    scope: str | None = None,
) -> dict:
    """Show directory tree.

    Args:
        path: Starting path relative to KB root (empty for root).
        depth: Maximum depth to display (default 3).
        include_files: Whether to include .md files (default True).
        scope: Filter to specific scope - "project", "user", or None for all.

    Returns:
        Dict with tree structure and counts.
    """
    from .config import get_kb_roots_for_indexing

    kb_roots = get_kb_roots_for_indexing(scope=scope)
    if not kb_roots:
        return {"tree": {}, "directories": 0, "files": 0}

    def _build_tree(current: Path, current_depth: int, max_depth: int) -> dict:
        """Recursively build tree structure."""
        result = {}

        if current_depth >= max_depth:
            return result

        try:
            items = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return result

        for item in items:
            # Skip hidden and special files
            if item.name.startswith(".") or item.name.startswith("_"):
                continue

            if item.is_dir():
                children = _build_tree(item, current_depth + 1, max_depth)
                result[item.name] = {"_type": "directory", **children}
            elif item.is_file() and item.suffix == ".md" and include_files:
                # Try to get title from frontmatter
                title = None
                try:
                    metadata, _, _ = parse_entry(item)
                    title = metadata.title
                except (ParseError, Exception) as e:
                    log.debug("Could not parse title from %s: %s", item, e)
                result[item.name] = {"_type": "file", "title": title}

        return result

    # Count directories and files
    def _count(node: dict) -> tuple[int, int]:
        dirs, files = 0, 0
        for key, value in node.items():
            if key == "_type":
                continue
            if isinstance(value, dict):
                if value.get("_type") == "directory":
                    dirs += 1
                    sub_dirs, sub_files = _count(value)
                    dirs += sub_dirs
                    files += sub_files
                elif value.get("_type") == "file":
                    files += 1
        return dirs, files

    # Build tree for each KB root
    combined_tree = {}
    total_dirs, total_files = 0, 0

    for scope_label, kb_root in kb_roots:
        if not kb_root.exists():
            continue

        if path:
            start_path = kb_root / path
            if not start_path.is_dir():
                continue  # Skip this KB if path doesn't exist
        else:
            start_path = kb_root

        tree_data = _build_tree(start_path, 0, depth)
        dirs, files = _count(tree_data)
        total_dirs += dirs
        total_files += files

        # In multi-KB mode, nest under @scope/ prefix
        if scope_label:
            combined_tree[f"@{scope_label}"] = {"_type": "directory", **tree_data}
            total_dirs += 1  # Count the scope directory itself
        else:
            # Single KB mode - use tree directly
            combined_tree = tree_data

    return {
        "tree": combined_tree,
        "directories": total_dirs,
        "files": total_files,
    }


async def mkdir(path: str) -> str:
    """Create a directory.

    Args:
        path: Directory path relative to KB root (e.g., "development/python/frameworks").
              Can create new top-level categories or nested directories.

    Returns:
        Created directory path.

    Raises:
        ValueError: If path is invalid or directory already exists.
    """
    # Validate path format
    abs_path, normalized = validate_nested_path(path)

    # Check if already exists
    if abs_path.exists():
        raise ValueError(f"Directory already exists: {path}")

    # Create the directory (and any missing parents)
    abs_path.mkdir(parents=True, exist_ok=False)

    return normalized


async def move(
    source: str,
    destination: str,
    update_links: bool = True,
) -> dict:
    """Move entry or directory.

    Args:
        source: Current path (e.g., "development/old-entry.md" or "development/python/").
        destination: New path (e.g., "architecture/old-entry.md" or "patterns/python/").
        update_links: Whether to update [[links]] in other files (default True).

    Returns:
        Dict with moved paths and updated link count.

    Raises:
        ValueError: If source doesn't exist or destination conflicts.
    """
    kb_root = get_kb_root()

    # Validate source path
    source_abs, source_normalized = validate_nested_path(source)
    if not source_abs.exists():
        raise ValueError(f"Source not found: {source}")

    # Validate destination path
    dest_abs, dest_normalized = validate_nested_path(destination)

    # Check destination doesn't already exist
    if dest_abs.exists():
        raise ValueError(f"Destination already exists: {destination}")

    # Ensure destination parent directory exists
    dest_parent = dest_abs.parent
    if not dest_parent.exists():
        raise ValueError(
            f"Destination parent directory does not exist: {dest_parent.relative_to(kb_root)}. "
            "Use mkdir to create it first."
        )

    # Ensure destination is within a valid category
    dest_category = get_parent_category(dest_normalized)
    valid_categories = get_valid_categories()
    if dest_category not in valid_categories:
        raise ValueError(
            f"Destination must be within a valid category. Valid categories: {', '.join(valid_categories)}"
        )

    # Protect category root directories from being moved
    if source_normalized in valid_categories:
        raise ValueError(f"Cannot move category root directory: {source}")

    # Collect files to move and build path mapping for link updates
    path_mapping: dict[str, str] = {}  # old_path -> new_path (without .md)
    moved_files: list[str] = []

    if source_abs.is_file():
        # Moving a single file
        old_rel = source_normalized[:-3] if source_normalized.endswith(".md") else source_normalized
        new_rel = dest_normalized[:-3] if dest_normalized.endswith(".md") else dest_normalized
        path_mapping[old_rel] = new_rel
        moved_files.append(f"{source_normalized} -> {dest_normalized}")
    else:
        # Moving a directory - collect all .md files
        for md_file in source_abs.rglob("*.md"):
            old_rel_path = str(md_file.relative_to(kb_root))
            old_rel = old_rel_path[:-3] if old_rel_path.endswith(".md") else old_rel_path

            # Compute new path
            relative_to_source = md_file.relative_to(source_abs)
            new_rel_path = str(Path(dest_normalized) / relative_to_source)
            new_rel = new_rel_path[:-3] if new_rel_path.endswith(".md") else new_rel_path

            path_mapping[old_rel] = new_rel
            moved_files.append(f"{old_rel_path} -> {new_rel_path}")

    # Perform the move
    shutil.move(str(source_abs), str(dest_abs))

    # Update links in other files
    links_updated = 0
    if update_links and path_mapping:
        links_updated = update_links_batch(kb_root, path_mapping)

    # Reindex moved files
    searcher = get_searcher()
    entries_reindexed = 0

    for old_path, new_path in path_mapping.items():
        old_path_md = f"{old_path}.md"
        new_path_md = f"{new_path}.md"

        # Delete old index entries
        searcher.delete_document(old_path_md)

        # Index under new path
        new_file_path = kb_root / new_path_md
        if new_file_path.exists():
            try:
                _, _, chunks = parse_entry(new_file_path)
                if chunks:
                    normalized_chunks = normalize_chunks(chunks, new_path_md)
                    searcher.index_chunks(normalized_chunks)
                    entries_reindexed += 1
            except ParseError as e:
                log.warning("Could not reindex moved file %s: %s", new_path_md, e)

    # Rebuild backlink cache
    rebuild_backlink_cache(kb_root)

    return {
        "moved": moved_files,
        "links_updated": links_updated,
        "entries_reindexed": entries_reindexed,
    }


async def rmdir(path: str, force: bool = False) -> str:
    """Remove directory.

    Args:
        path: Directory path relative to KB root.
        force: If True, also remove empty subdirectories recursively.

    Returns:
        Removed directory path.

    Raises:
        ValueError: If directory contains entries, not empty (without force), or doesn't exist.
    """
    kb_root = get_kb_root()

    # Validate path
    abs_path, normalized = validate_nested_path(path)

    if not abs_path.exists():
        raise ValueError(f"Directory not found: {path}")

    if not abs_path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    # Check if directory is empty (or force is True)
    has_files = any(abs_path.rglob("*"))
    has_md_files = any(abs_path.rglob("*.md"))

    if has_md_files:
        raise ValueError(
            f"Directory contains entries: {path}. Move or delete entries first."
        )

    if has_files and not force:
        raise ValueError(
            f"Directory not empty: {path}. Use force=True to remove empty subdirectories."
        )

    # Remove the directory
    if force:
        shutil.rmtree(str(abs_path))
    else:
        abs_path.rmdir()

    return normalized


async def delete_entry(path: str, force: bool = False) -> dict:
    """Delete an entry.

    Args:
        path: Path to the entry (e.g., "development/old-entry.md").
        force: If True, delete even if other entries link to this one.

    Returns:
        Dict with deleted path and any backlinks that were pointing to it.

    Raises:
        ValueError: If entry doesn't exist or has backlinks (without force).
    """
    kb_root = get_kb_root()

    # Validate path
    abs_path, normalized = validate_nested_path(path)

    if not abs_path.exists():
        raise ValueError(f"Entry not found: {path}")

    if not abs_path.is_file():
        raise ValueError(f"Path is not a file: {path}. Use rmdir for directories.")

    if not normalized.endswith(".md"):
        raise ValueError(f"Path must be a markdown file: {path}")

    # Check for backlinks
    path_key = normalized[:-3] if normalized.endswith(".md") else normalized
    all_backlinks = get_backlink_index()
    entry_backlinks = all_backlinks.get(path_key, [])

    if entry_backlinks and not force:
        raise ValueError(
            f"Entry has {len(entry_backlinks)} backlink(s): {', '.join(entry_backlinks)}. "
            "Use force=True to delete anyway, or update linking entries first."
        )

    # Remove from search index
    searcher = get_searcher()
    searcher.delete_document(normalized)

    # Remove from view tracking
    try:
        from .views_tracker import delete_entry_views

        delete_entry_views(normalized)
    except Exception as e:
        log.debug("Failed to delete view tracking for %s: %s", normalized, e)

    # Delete the file
    abs_path.unlink()

    # Rebuild backlink cache
    rebuild_backlink_cache(kb_root)

    return {
        "deleted": normalized,
        "had_backlinks": entry_backlinks,
    }


async def tags(
    min_count: int = 1,
    include_entries: bool = False,
) -> list[dict]:
    """List all tags with usage counts.

    Args:
        min_count: Only include tags used at least this many times (default 1).
        include_entries: If True, include list of entry paths for each tag.

    Returns:
        List of {tag, count, entries?} sorted by count descending.
    """
    kb_root = get_kb_root()

    if not kb_root.exists():
        return []

    # Collect all tags
    tag_entries: dict[str, list[str]] = {}

    for md_file in kb_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue

        rel_path = str(md_file.relative_to(kb_root))

        try:
            metadata, _, _ = parse_entry(md_file)
        except ParseError:
            continue

        for tag in metadata.tags:
            if tag not in tag_entries:
                tag_entries[tag] = []
            tag_entries[tag].append(rel_path)

    # Build results
    results = []
    for tag, entries in sorted(tag_entries.items(), key=lambda x: -len(x[1])):
        if len(entries) >= min_count:
            result: dict = {"tag": tag, "count": len(entries)}
            if include_entries:
                result["entries"] = entries
            results.append(result)

    return results


async def health(
    stale_days: int = 90,
    check_orphans: bool = True,
    check_broken_links: bool = True,
    check_stale: bool = True,
    check_empty_dirs: bool = True,
) -> dict:
    """Audit KB health.

    Args:
        stale_days: Entries not updated in this many days are considered stale.
        check_orphans: Check for entries with no incoming backlinks.
        check_broken_links: Check for [[links]] pointing to non-existent entries.
        check_stale: Check for entries not updated recently.
        check_empty_dirs: Check for directories with no entries.

    Returns:
        Dict with lists of issues found in each category.
    """
    kb_root = get_kb_root()

    if not kb_root.exists():
        return {
            "orphans": [],
            "broken_links": [],
            "stale": [],
            "empty_dirs": [],
            "summary": {"total_issues": 0},
        }

    results: dict = {
        "orphans": [],
        "broken_links": [],
        "stale": [],
        "empty_dirs": [],
    }

    # Collect all entries and their metadata
    all_entries: dict[str, dict] = {}  # path -> {title, tags, created, updated, links}
    # Use timezone-aware datetime (UTC) for comparison
    cutoff_datetime = datetime.now(UTC) - timedelta(days=stale_days)

    def make_aware(dt: datetime | None) -> datetime | None:
        """Ensure datetime is timezone-aware for comparison."""
        if dt is None:
            return None
        if dt.tzinfo is None:
            # Assume naive datetimes are UTC
            return dt.replace(tzinfo=UTC)
        return dt

    for md_file in kb_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue

        rel_path = str(md_file.relative_to(kb_root))
        path_key = rel_path[:-3] if rel_path.endswith(".md") else rel_path

        try:
            metadata, content, _ = parse_entry(md_file)
            links = extract_links(content)
        except ParseError:
            continue

        all_entries[path_key] = {
            "path": rel_path,
            "title": metadata.title,
            "tags": metadata.tags,
            "created": metadata.created,
            "updated": metadata.updated,
            "links": links,
        }

    # Get backlink index
    all_backlinks = get_backlink_index()

    # Check for orphans (entries with no incoming backlinks)
    if check_orphans:
        for path_key, entry in all_entries.items():
            incoming = all_backlinks.get(path_key, [])
            if not incoming:
                results["orphans"].append({
                    "path": entry["path"],
                    "title": entry["title"],
                })

    # Check for broken links
    if check_broken_links:
        valid_targets = set(all_entries.keys())
        # Also accept paths with .md extension
        valid_targets.update(f"{k}.md" for k in all_entries.keys())
        # Also accept titles as link targets
        title_to_path = {e["title"]: k for k, e in all_entries.items()}

        for path_key, entry in all_entries.items():
            for link in entry["links"]:
                # Normalize link target
                link_normalized = link[:-3] if link.endswith(".md") else link

                # Check if link is valid (by path or by title)
                if (
                    link not in valid_targets
                    and link_normalized not in valid_targets
                    and link not in title_to_path
                ):
                    results["broken_links"].append({
                        "source": entry["path"],
                        "broken_link": link,
                    })

    # Check for stale entries
    if check_stale:
        for path_key, entry in all_entries.items():
            last_activity = make_aware(entry["updated"]) or make_aware(entry["created"])
            if last_activity and last_activity < cutoff_datetime:
                days_old = (datetime.now(UTC) - last_activity).days
                results["stale"].append({
                    "path": entry["path"],
                    "title": entry["title"],
                    "last_activity": last_activity.isoformat(),
                    "days_old": days_old,
                })

    # Check for empty directories
    if check_empty_dirs:
        for dir_path in kb_root.rglob("*"):
            if not dir_path.is_dir():
                continue
            if dir_path.name.startswith(".") or dir_path.name.startswith("_"):
                continue

            # Check if directory has any .md files
            has_entries = any(
                f.suffix == ".md" and not f.name.startswith("_")
                for f in dir_path.rglob("*")
                if f.is_file()
            )

            if not has_entries:
                rel_dir = str(dir_path.relative_to(kb_root))
                results["empty_dirs"].append(rel_dir)

    # Add summary with health score
    total_issues = sum(len(v) for v in results.values() if isinstance(v, list))
    total_entries = len(all_entries)

    # Calculate health score (0-100)
    # Deduct points for issues, weighted by severity
    score = 100.0
    if total_entries > 0:
        # Broken links are serious (5 points each, max 30 deduction)
        broken_penalty = min(30, len(results["broken_links"]) * 5)
        # Orphans are moderate (2 points each, max 25 deduction)
        orphan_penalty = min(25, len(results["orphans"]) * 2)
        # Stale content is minor (1 point each, max 20 deduction)
        stale_penalty = min(20, len(results["stale"]) * 1)
        # Empty dirs are trivial (1 point each, max 10 deduction)
        empty_penalty = min(10, len(results["empty_dirs"]) * 1)

        score = max(0, 100 - broken_penalty - orphan_penalty - stale_penalty - empty_penalty)

    results["summary"] = {
        "health_score": round(score),
        "total_issues": total_issues,
        "orphans_count": len(results["orphans"]),
        "broken_links_count": len(results["broken_links"]),
        "stale_count": len(results["stale"]),
        "empty_dirs_count": len(results["empty_dirs"]),
        "total_entries": total_entries,
    }

    return results


async def hubs(limit: int = 10) -> list[dict]:
    """Find hub entries with most connections.

    Args:
        limit: Maximum entries to return (default 10).

    Returns:
        List of {path, title, incoming, outgoing, total} sorted by total connections.
    """
    kb_root = get_kb_root()

    if not kb_root.exists():
        return []

    # Build connection counts
    all_backlinks = get_backlink_index()
    connection_counts: dict[str, dict] = {}

    # Count incoming links from backlinks index
    for path_key, sources in all_backlinks.items():
        if path_key not in connection_counts:
            connection_counts[path_key] = {"incoming": 0, "outgoing": 0}
        connection_counts[path_key]["incoming"] = len(sources)

    # Count outgoing links from each entry
    for md_file in kb_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue

        rel_path = str(md_file.relative_to(kb_root))
        path_key = rel_path[:-3] if rel_path.endswith(".md") else rel_path

        try:
            _, content, _ = parse_entry(md_file)
            links = extract_links(content)
        except ParseError:
            continue

        if path_key not in connection_counts:
            connection_counts[path_key] = {"incoming": 0, "outgoing": 0}
        connection_counts[path_key]["outgoing"] = len(links)

    # Calculate totals and sort
    results = []
    for path_key, counts in connection_counts.items():
        total = counts["incoming"] + counts["outgoing"]
        if total == 0:
            continue

        # Get title
        file_path = kb_root / f"{path_key}.md"
        title = path_key
        if file_path.exists():
            try:
                metadata, _, _ = parse_entry(file_path)
                title = metadata.title
            except ParseError as e:
                log.debug("Could not parse title from %s: %s", file_path, e)

        results.append({
            "path": f"{path_key}.md",
            "title": title,
            "incoming": counts["incoming"],
            "outgoing": counts["outgoing"],
            "total": total,
        })

    # Sort by total connections descending
    results.sort(key=lambda x: x["total"], reverse=True)

    return results[:limit]


async def dead_ends(limit: int = 10) -> list[dict]:
    """Find entries with incoming links but no outgoing links.

    Args:
        limit: Maximum entries to return (default 10).

    Returns:
        List of {path, title, incoming_count} sorted by incoming links descending.
    """
    kb_root = get_kb_root()

    if not kb_root.exists():
        return []

    all_backlinks = get_backlink_index()
    results = []

    for md_file in kb_root.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue

        rel_path = str(md_file.relative_to(kb_root))
        path_key = rel_path[:-3] if rel_path.endswith(".md") else rel_path

        try:
            metadata, content, _ = parse_entry(md_file)
            outgoing_links = extract_links(content)
        except ParseError:
            continue

        incoming_count = len(all_backlinks.get(path_key, []))

        # Dead end: has incoming links but no outgoing
        if incoming_count > 0 and len(outgoing_links) == 0:
            results.append({
                "path": rel_path,
                "title": metadata.title,
                "incoming_count": incoming_count,
            })

    # Sort by incoming count descending (most linked-to dead ends first)
    results.sort(key=lambda x: x["incoming_count"], reverse=True)

    return results[:limit]


async def suggest_links(
    path: str,
    limit: int = 5,
    min_score: float = LINK_SUGGESTION_MIN_SCORE,
) -> list[dict]:
    """Suggest links to add to an entry based on content similarity.

    Args:
        path: Path to the entry to suggest links for.
        limit: Maximum suggestions to return (default 5).
        min_score: Minimum similarity score threshold.

    Returns:
        List of {path, title, score, reason} for suggested links.
    """
    kb_root = get_kb_root()
    file_path = kb_root / path

    if not file_path.exists():
        raise ValueError(f"Entry not found: {path}")

    try:
        metadata, content, _ = parse_entry(file_path)
    except ParseError as e:
        raise ValueError(f"Failed to parse entry: {e}") from e

    # Get existing links to exclude
    existing_links = set(extract_links(content))

    return compute_link_suggestions(
        title=metadata.title,
        content=content,
        tags=list(metadata.tags),
        self_path=path,
        existing_links=existing_links,
        limit=limit,
        min_score=min_score,
    )


async def patch_entry(
    path: str,
    find_string: str,
    replace_string: str,
    replace_all: bool = False,
    dry_run: bool = False,
    backup: bool = False,
) -> dict:
    """Apply surgical find-replace patch to a KB entry.

    Operates on body content only; frontmatter is preserved unchanged.
    After successful patch, triggers re-indexing for semantic search.

    Args:
        path: Path to entry relative to KB root (e.g., "tooling/notes.md").
        find_string: Exact text to find and replace.
        replace_string: Replacement text.
        replace_all: If True, replace all occurrences.
        dry_run: If True, preview changes without writing.
        backup: If True, create .bak backup before patching.

    Returns:
        Dict with:
            - success: bool
            - exit_code: int (0=success, 1=not found, 2=ambiguous, 3=file error)
            - message: str
            - replacements: int (number of replacements made)
            - diff: str | None (unified diff if dry_run=True)
            - match_contexts: list | None (for ambiguous errors)
            - path: str (relative path, on success)
    """
    from .patch import (
        PatchExitCode,
        apply_patch,
        generate_diff,
        read_file_safely,
        write_file_atomically,
    )

    kb_root = get_kb_root()
    file_path = kb_root / path

    if not file_path.exists():
        return {
            "success": False,
            "exit_code": int(PatchExitCode.FILE_ERROR),
            "message": f"Entry not found: {path}",
        }

    if not file_path.is_file():
        return {
            "success": False,
            "exit_code": int(PatchExitCode.FILE_ERROR),
            "message": f"Path is not a file: {path}",
        }

    # Read file, separating frontmatter from body
    frontmatter, body, read_error = read_file_safely(file_path)
    if read_error:
        return read_error.to_dict()

    # Apply patch to body only
    result = apply_patch(
        content=body,
        find_string=find_string,
        replace_string=replace_string,
        replace_all=replace_all,
    )

    if not result.success:
        return result.to_dict()

    # When success is True, new_content is guaranteed to be set
    assert result.new_content is not None

    # Handle dry-run: return diff without writing
    if dry_run:
        diff = generate_diff(body, result.new_content, filename=path)
        return {
            "success": True,
            "exit_code": 0,
            "message": "Dry run - no changes made",
            "replacements": result.replacements_made,
            "diff": diff,
            "path": path,
        }

    # Parse existing metadata for update
    try:
        metadata, _, _ = parse_entry(file_path)
    except ParseError as e:
        return {
            "success": False,
            "exit_code": int(PatchExitCode.FILE_ERROR),
            "message": f"Failed to parse entry metadata: {e}",
        }

    # Update metadata (updated date, contributor info)
    updated_metadata = update_metadata_for_edit(
        metadata,
        new_tags=list(metadata.tags),  # Preserve tags
        new_contributor=get_current_contributor(),
        edit_source=get_current_project(),
        model=get_llm_model(),
        git_branch=get_git_branch(),
        actor=get_actor_identity(),
    )
    new_frontmatter = build_frontmatter(updated_metadata)

    # Write atomically
    write_error = write_file_atomically(
        path=file_path,
        frontmatter=new_frontmatter,
        content=result.new_content,
        backup=backup,
    )
    if write_error:
        return write_error.to_dict()

    # Rebuild caches
    rebuild_backlink_cache(kb_root)

    # Reindex for semantic search
    relative_path = relative_kb_path(kb_root, file_path)
    searcher = get_searcher()
    try:
        # Remove old index entries
        searcher.delete_document(relative_path)
        # Parse and index new content
        _, _, chunks = parse_entry(file_path)
        if chunks:
            normalized_chunks = normalize_chunks(chunks, relative_path)
            searcher.index_chunks(normalized_chunks)
    except ParseError as e:
        log.warning("Patched entry but failed to re-index %s: %s", relative_path, e)
    except Exception as e:
        log.error("Unexpected error re-indexing %s: %s", relative_path, e)

    return {
        "success": True,
        "exit_code": 0,
        "message": f"Patched {path} ({result.replacements_made} replacement(s))",
        "replacements": result.replacements_made,
        "path": relative_path,
    }


async def publish(
    output_dir: Path | str | None = None,
    base_url: str = "",
    site_title: str = "Memex",
    index_entry: str | None = None,
    include_drafts: bool = False,
    include_archived: bool = False,
    clean: bool = True,
    kb_root: Path | str | None = None,
) -> dict:
    """Generate static HTML site from knowledge base.

    Produces a complete static site suitable for hosting on GitHub Pages,
    with resolved wikilinks, client-side search, and a minimal theme.

    Args:
        output_dir: Output directory (default: _site)
        base_url: Base URL prefix for links (e.g., "/my-kb" for subdirectory hosting)
        site_title: Site title for header and page titles (default: "Memex")
        index_entry: Path to entry to use as landing page (e.g., "guides/welcome")
        include_drafts: Include entries with status="draft"
        include_archived: Include entries with status="archived"
        clean: Remove output directory before build (default True)
        kb_root: KB source directory (overrides auto-detected KB if provided)

    Returns:
        Dict with:
        - entries_published: Number of entries written
        - broken_links: List of {source, target} for unresolved wikilinks
        - output_dir: Path to output directory
        - search_index_path: Path to generated search index
    """
    from .publisher import PublishConfig, SiteGenerator

    resolved_kb_root = Path(kb_root) if kb_root else get_kb_root()

    config = PublishConfig(
        output_dir=Path(output_dir) if output_dir else Path("_site"),
        base_url=base_url,
        site_title=site_title,
        index_entry=index_entry,
        include_drafts=include_drafts,
        include_archived=include_archived,
        clean=clean,
    )

    generator = SiteGenerator(config, resolved_kb_root)
    result = await generator.generate()

    return {
        "entries_published": result.entries_published,
        "broken_links": result.broken_links,
        "output_dir": result.output_dir,
        "search_index_path": result.search_index_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Description Generation
# ─────────────────────────────────────────────────────────────────────────────


async def generate_descriptions(
    dry_run: bool = False,
    limit: int | None = None,
) -> list[dict]:
    """Generate descriptions for entries that are missing them.

    Reads entry content and generates a one-line summary based on the
    first meaningful sentence or paragraph.

    Args:
        dry_run: If True, only preview changes without writing.
        limit: Maximum number of entries to process.

    Returns:
        List of {path, title, description, status} for each processed entry.
        status is 'updated', 'skipped' (already has description), or 'preview' (dry_run).
    """
    from .health_cache import get_entry_metadata

    kb_root = get_kb_root()

    if not kb_root.exists():
        return []

    results: list[dict] = []
    processed = 0

    # Get entries missing descriptions from health cache
    all_entries = get_entry_metadata(kb_root)

    for path_key, entry in all_entries.items():
        if entry.get("description"):
            # Already has description
            continue

        if limit and processed >= limit:
            break

        rel_path = entry["path"]
        file_path = kb_root / rel_path

        if not file_path.exists():
            continue

        try:
            metadata, content, _ = parse_entry(file_path)

            # Generate description from content, passing title to skip if it appears at start
            description = _generate_description_from_content(content, title=metadata.title)

            if not description:
                results.append({
                    "path": rel_path,
                    "title": entry["title"],
                    "description": None,
                    "status": "skipped",
                    "reason": "Could not generate description from content",
                })
                continue

            if dry_run:
                results.append({
                    "path": rel_path,
                    "title": entry["title"],
                    "description": description,
                    "status": "preview",
                })
            else:
                # Update the file with new description
                metadata.description = description

                # Rebuild frontmatter with new description
                new_frontmatter = build_frontmatter(metadata)

                # Write updated content
                # Strip leading whitespace from content to prevent blank line accumulation
                new_content = f"{new_frontmatter}{content.lstrip()}"
                file_path.write_text(new_content)

                results.append({
                    "path": rel_path,
                    "title": entry["title"],
                    "description": description,
                    "status": "updated",
                })

            processed += 1

        except Exception as e:
            results.append({
                "path": rel_path,
                "title": entry["title"],
                "description": None,
                "status": "error",
                "reason": str(e),
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Title Resolution and Upsert Operations
# ─────────────────────────────────────────────────────────────────────────────


class AmbiguousMatchError(ValueError):
    """Raised when title matches multiple entries with similar confidence."""

    def __init__(self, matches: list[UpsertMatch]):
        self.matches = matches
        titles = ", ".join(f"'{m.title}' ({m.score:.0%})" for m in matches[:3])
        super().__init__(f"Ambiguous title match. Candidates: {titles}")


def resolve_entry_by_title(
    title: str,
    kb_root: Path | None = None,
    min_score: float = 0.6,
) -> UpsertMatch | None:
    """Resolve a title to an existing entry path.

    Search order:
    1. Exact title match (case-insensitive)
    2. Alias match
    3. Fuzzy title search (semantic similarity)

    Args:
        title: Title to search for.
        kb_root: KB root path (auto-detected if None).
        min_score: Minimum fuzzy match score (0-1).

    Returns:
        UpsertMatch if found, None if no match above threshold.

    Raises:
        AmbiguousMatchError: If multiple matches with similar scores.
    """
    from .parser.title_index import TitleIndex, build_title_index

    if kb_root is None:
        kb_root = get_kb_root()

    # Build title index
    index = build_title_index(kb_root, include_filename_index=True)
    if isinstance(index, TitleIndex):
        title_to_path = index.title_to_path
    else:
        title_to_path = index

    title_lower = title.lower().strip()

    # 1. Check exact title match
    if title_lower in title_to_path:
        path = title_to_path[title_lower]
        return UpsertMatch(
            path=f"{path}.md",
            title=title,
            score=1.0,
            match_type="exact_title",
        )

    # 2. Check if any title starts with or contains our search
    # This catches aliases since they're also in title_to_path
    for indexed_title, path in title_to_path.items():
        if indexed_title == title_lower:
            return UpsertMatch(
                path=f"{path}.md",
                title=indexed_title,
                score=1.0,
                match_type="alias",
            )

    # 3. Fuzzy search using semantic similarity
    searcher = get_searcher()
    results = searcher.search(title, limit=5, mode="semantic")

    if not results:
        return None

    # Convert to UpsertMatch objects
    matches: list[UpsertMatch] = []
    for r in results:
        if r.score >= min_score:
            matches.append(UpsertMatch(
                path=r.path,
                title=r.title,
                score=r.score,
                match_type="fuzzy",
            ))

    if not matches:
        return None

    # Check for ambiguous matches (multiple results with similar scores)
    if len(matches) > 1:
        score_gap = matches[0].score - matches[1].score
        if score_gap < 0.2:  # Less than 20% difference - ambiguous
            raise AmbiguousMatchError(matches)

    return matches[0]


# ─────────────────────────────────────────────────────────────────────────────
# A-Mem Init Functions
# ─────────────────────────────────────────────────────────────────────────────


async def amem_init_inventory(
    scope: Literal["project", "user"] | None = None,
    missing_keywords: Literal["error", "skip", "llm"] | None = None,
    limit: int | None = None,
) -> InitInventoryResult:
    """Build inventory of KB entries for A-Mem initialization.

    Lists all entries in scope, validates keyword presence, and categorizes
    entries by their keyword status. Entries are sorted by created timestamp
    (oldest first) for chronological processing in later phases.

    Args:
        scope: Filter to specific KB scope - "project", "user", or None for all.
        missing_keywords: How to handle entries without keywords:
            - "error": Return error list, all missing entries marked as errors
            - "skip": Mark entries as skipped, continue with others
            - "llm": Mark entries as needs_llm for Phase 2 keyword extraction
            - None: Default based on amem_strict config (error if true, skip if false)
        limit: Maximum number of entries to process (for testing).

    Returns:
        InitInventoryResult with entries sorted by created date and statistics.

    Example:
        >>> result = await amem_init_inventory(scope="project", missing_keywords="skip")
        >>> print(f"Found {result.total_count} entries, {result.with_keywords} ready")
    """
    from .config import get_kb_roots_for_indexing

    # Determine default mode based on amem_strict config
    if missing_keywords is None:
        missing_keywords = "error" if is_amem_strict_enabled() else "skip"

    kb_roots = get_kb_roots_for_indexing(scope=scope)
    if not kb_roots:
        return InitInventoryResult(
            entries=[],
            total_count=0,
            with_keywords=0,
            missing_keywords=0,
            skipped_count=0,
            needs_llm_count=0,
            missing_keyword_mode=missing_keywords,
        )

    entries: list[InitInventoryEntry] = []
    errors: list[str] = []

    for scope_label, kb_root in kb_roots:
        if not kb_root.exists():
            continue

        for md_file in kb_root.rglob("*.md"):
            # Skip index files
            if md_file.name.startswith("_"):
                continue

            rel_path = str(md_file.relative_to(kb_root))
            # Add scope prefix for multi-KB mode
            display_path = f"@{scope_label}/{rel_path}" if scope_label else rel_path

            try:
                metadata, _, _ = parse_entry(md_file)
            except ParseError as e:
                errors.append(f"{display_path}: {e}")
                continue

            # Check for keywords
            has_keywords = bool(metadata.keywords)

            # Determine keyword status based on mode
            if has_keywords:
                keyword_status: Literal["ready", "skipped", "needs_llm"] = "ready"
            elif missing_keywords == "skip":
                keyword_status = "skipped"
            elif missing_keywords == "llm":
                keyword_status = "needs_llm"
            else:  # error mode
                keyword_status = "skipped"  # Will be reported as error

            # Ensure created timestamp is timezone-aware
            created = _ensure_aware(metadata.created) or datetime.now(UTC)

            entries.append(InitInventoryEntry(
                path=display_path,
                title=metadata.title,
                created=created,
                has_keywords=has_keywords,
                keyword_status=keyword_status,
                keywords=list(metadata.keywords) if metadata.keywords else [],
                absolute_path=str(md_file),
            ))

    # Sort by created timestamp (oldest first)
    entries.sort(key=lambda e: e.created)

    # Apply limit if specified
    if limit is not None and limit > 0:
        entries = entries[:limit]

    # Calculate statistics
    with_keywords = sum(1 for e in entries if e.has_keywords)
    missing_count = len(entries) - with_keywords
    skipped_count = sum(1 for e in entries if e.keyword_status == "skipped" and not e.has_keywords)
    needs_llm_count = sum(1 for e in entries if e.keyword_status == "needs_llm")

    # In error mode, add missing keyword entries to errors list
    if missing_keywords == "error":
        for e in entries:
            if not e.has_keywords:
                errors.append(f"{e.path}: missing keywords")

    return InitInventoryResult(
        entries=entries,
        total_count=len(entries),
        with_keywords=with_keywords,
        missing_keywords=missing_count,
        skipped_count=skipped_count,
        needs_llm_count=needs_llm_count,
        missing_keyword_mode=missing_keywords,
        errors=errors,
    )


async def amem_init_extract_keywords(
    inventory: InitInventoryResult,
    model: str | None = None,
) -> KeywordExtractionPhaseResult:
    """Phase 2: Extract keywords for entries marked as needs_llm.

    Processes entries from Phase 1 inventory that are missing keywords,
    using LLM to extract domain-specific keywords and updating frontmatter.

    Args:
        inventory: Result from amem_init_inventory() with entries marked needs_llm.
        model: OpenRouter model ID. If None, uses config from memory_evolution.model.

    Returns:
        KeywordExtractionPhaseResult with per-entry results and statistics.

    Example:
        >>> inventory = await amem_init_inventory(missing_keywords="llm")
        >>> result = await amem_init_extract_keywords(inventory)
        >>> print(f"Updated {result.entries_updated} entries with keywords")
    """
    from .llm import LLMConfigurationError, extract_keywords_llm

    # Get entries that need LLM keyword extraction
    needs_llm = [e for e in inventory.entries if e.keyword_status == "needs_llm"]

    if not needs_llm:
        return KeywordExtractionPhaseResult(
            entries_processed=0,
            entries_updated=0,
            entries_failed=0,
        )

    # Get model from config if not specified
    if model is None:
        config = get_memory_evolution_config()
        model = config.model

    results: list[KeywordExtractionEntry] = []
    errors: list[str] = []
    entries_updated = 0
    entries_failed = 0

    for entry in needs_llm:
        if not entry.absolute_path:
            errors.append(f"{entry.path}: no absolute path available")
            entries_failed += 1
            results.append(KeywordExtractionEntry(
                path=entry.path,
                title=entry.title,
                keywords=[],
                success=False,
                error="No absolute path available",
            ))
            continue

        # Read entry content
        try:
            file_path = Path(entry.absolute_path)
            _, content, _ = parse_entry(file_path)
        except Exception as e:
            error_msg = f"Failed to read entry: {e}"
            errors.append(f"{entry.path}: {error_msg}")
            entries_failed += 1
            results.append(KeywordExtractionEntry(
                path=entry.path,
                title=entry.title,
                keywords=[],
                success=False,
                error=error_msg,
            ))
            continue

        # Extract keywords using LLM
        try:
            extraction_result = await extract_keywords_llm(
                content=content,
                title=entry.title,
                model=model,
            )
        except LLMConfigurationError:
            # Configuration error - fail fast
            raise
        except Exception as e:
            error_msg = f"LLM error: {e}"
            errors.append(f"{entry.path}: {error_msg}")
            entries_failed += 1
            results.append(KeywordExtractionEntry(
                path=entry.path,
                title=entry.title,
                keywords=[],
                success=False,
                error=error_msg,
            ))
            continue

        if not extraction_result.success:
            errors.append(f"{entry.path}: {extraction_result.error}")
            entries_failed += 1
            results.append(KeywordExtractionEntry(
                path=entry.path,
                title=entry.title,
                keywords=[],
                success=False,
                error=extraction_result.error,
            ))
            continue

        # Update entry with extracted keywords
        # Get the relative path without scope prefix for update_entry
        rel_path = entry.path
        if rel_path.startswith("@"):
            # Remove @scope/ prefix
            rel_path = rel_path.split("/", 1)[1] if "/" in rel_path else rel_path

        try:
            await update_entry(
                path=rel_path,
                keywords=extraction_result.keywords,
            )
            entries_updated += 1
            results.append(KeywordExtractionEntry(
                path=entry.path,
                title=entry.title,
                keywords=extraction_result.keywords,
                success=True,
            ))
        except Exception as e:
            error_msg = f"Failed to update entry: {e}"
            errors.append(f"{entry.path}: {error_msg}")
            entries_failed += 1
            results.append(KeywordExtractionEntry(
                path=entry.path,
                title=entry.title,
                keywords=extraction_result.keywords,
                success=False,
                error=error_msg,
            ))

    return KeywordExtractionPhaseResult(
        entries_processed=len(needs_llm),
        entries_updated=entries_updated,
        entries_failed=entries_failed,
        results=results,
        errors=errors,
    )


async def amem_init_link_entries(
    inventory: InitInventoryResult,
    k: int = SEMANTIC_LINK_K,
    min_score: float = SEMANTIC_LINK_MIN_SCORE,
    queue_evolution_items: bool = True,
) -> LinkingPhaseResult:
    """Phase 3: Create bidirectional semantic links chronologically.

    Processes entries in chronological order (oldest first), creating semantic
    links only to entries that were "created before" the current entry. This
    simulates incremental A-Mem as if entries were added over time.

    For each entry:
    1. Search for similar entries using semantic search
    2. Filter results to only include entries with created < current.created
    3. Create bidirectional links (forward on current, backlinks on neighbors)
    4. Queue neighbor pairs for evolution (Phase 4 integration)

    Args:
        inventory: Result from amem_init_inventory() with entries sorted by created.
        k: Maximum number of neighbors to link per entry.
        min_score: Minimum similarity score threshold (0.0-1.0).
        queue_evolution_items: Whether to queue items for mx evolve.

    Returns:
        LinkingPhaseResult with per-entry results and statistics.

    Example:
        >>> inventory = await amem_init_inventory(missing_keywords="skip")
        >>> result = await amem_init_link_entries(inventory)
        >>> print(f"Created {result.total_links_created} links")
    """
    from .evolution_queue import queue_evolution

    if not SEMANTIC_LINK_ENABLED:
        return LinkingPhaseResult(
            entries_processed=0,
            entries_linked=0,
            entries_skipped=len(inventory.entries),
            total_links_created=0,
            total_evolution_items=0,
            errors=["Semantic linking is disabled (SEMANTIC_LINK_ENABLED=False)"],
        )

    # Get entries that can be processed (ready status, have keywords)
    processable = [
        e for e in inventory.entries
        if e.keyword_status == "ready" and e.has_keywords
    ]

    if not processable:
        return LinkingPhaseResult(
            entries_processed=0,
            entries_linked=0,
            entries_skipped=len(inventory.entries),
            total_links_created=0,
            total_evolution_items=0,
        )

    kb_root = get_kb_root()
    searcher = get_searcher()

    # Build a map of path -> created timestamp for filtering
    # Use all entries from inventory, not just processable
    path_to_created: dict[str, datetime] = {}
    for entry in inventory.entries:
        # Store with both display path and normalized path for lookup
        path_to_created[entry.path] = entry.created
        # Also store without @scope/ prefix
        if entry.path.startswith("@"):
            parts = entry.path.split("/", 1)
            if len(parts) > 1:
                path_to_created[parts[1]] = entry.created

    results: list[LinkingPhaseEntry] = []
    errors: list[str] = []
    total_links = 0
    total_evolution = 0
    entries_linked = 0
    skipped_count = len(inventory.entries) - len(processable)

    for entry in processable:
        if not entry.absolute_path:
            errors.append(f"{entry.path}: no absolute path available")
            skipped_count += 1
            continue

        # Read entry content
        try:
            file_path = Path(entry.absolute_path)
            metadata, content, _ = parse_entry(file_path)
        except Exception as e:
            errors.append(f"{entry.path}: failed to read - {e}")
            skipped_count += 1
            continue

        # Get relative path for linking
        rel_path = entry.path
        if rel_path.startswith("@"):
            parts = rel_path.split("/", 1)
            rel_path = parts[1] if len(parts) > 1 else rel_path

        # Normalize entry path for comparison
        entry_key = rel_path[:-3] if rel_path.endswith(".md") else rel_path

        # Search for similar entries - get extra to allow for date filtering
        query = f"{metadata.title} {content[:500]}"
        search_results = searcher.search(query, limit=k * 3, mode="semantic")

        # Filter results:
        # 1. Not self
        # 2. Score >= threshold
        # 3. Created BEFORE current entry (the chronological constraint)
        forward_links: list[SemanticLink] = []
        neighbors_for_evolution: list[tuple[str, float]] = []

        for result in search_results:
            # Skip self
            result_key = result.path[:-3] if result.path.endswith(".md") else result.path
            if result_key == entry_key or result.path == rel_path:
                continue

            # Skip results below threshold
            if result.score < min_score:
                continue

            # CRITICAL: Only link to entries created BEFORE this entry
            result_created = path_to_created.get(result.path)
            if result_created is None:
                # Try to get created from search result
                result_created = result.created
            if result_created is None:
                # Can't determine date, skip to be safe
                continue

            # Ensure both are timezone-aware for comparison
            entry_created = _ensure_aware(entry.created)
            result_created = _ensure_aware(result_created)

            if entry_created and result_created:
                if result_created >= entry_created:
                    # Neighbor is newer or same age, skip
                    continue

            # Create forward link for current entry
            forward_link = SemanticLink(
                path=result.path,
                score=round(result.score, 3),
                reason="embedding_similarity",
            )
            forward_links.append(forward_link)

            # Create backlink on the neighbor entry
            _add_backlink_to_neighbor(
                kb_root=kb_root,
                neighbor_path=result.path,
                new_entry_path=rel_path,
                score=result.score,
                searcher=searcher,
            )

            # Collect for potential evolution
            neighbors_for_evolution.append((result.path, result.score))

            if len(forward_links) >= k:
                break

        # Update current entry with forward links if any new ones
        if forward_links:
            try:
                # Merge with existing links
                existing_paths = {link.path for link in metadata.semantic_links}
                new_links = [lnk for lnk in forward_links if lnk.path not in existing_paths]

                if new_links:
                    all_links = list(metadata.semantic_links) + new_links
                    await update_entry(
                        path=rel_path,
                        semantic_links=[lnk.model_dump() for lnk in all_links],
                    )

                    # Queue evolution items
                    evolution_count = 0
                    if queue_evolution_items and neighbors_for_evolution:
                        evolution_count = queue_evolution(
                            new_entry_path=rel_path,
                            neighbors=neighbors_for_evolution,
                            kb_root=kb_root,
                        )
                        total_evolution += evolution_count

                    total_links += len(new_links)
                    entries_linked += 1

                    results.append(LinkingPhaseEntry(
                        path=entry.path,
                        title=entry.title,
                        links_created=len(new_links),
                        evolution_items_queued=evolution_count,
                        neighbors=[lnk.path for lnk in new_links],
                    ))
                else:
                    # All links already existed
                    results.append(LinkingPhaseEntry(
                        path=entry.path,
                        title=entry.title,
                        links_created=0,
                        evolution_items_queued=0,
                        neighbors=[],
                    ))
            except Exception as e:
                errors.append(f"{entry.path}: failed to update links - {e}")
        else:
            # No matching neighbors found
            results.append(LinkingPhaseEntry(
                path=entry.path,
                title=entry.title,
                links_created=0,
                evolution_items_queued=0,
                neighbors=[],
            ))

    return LinkingPhaseResult(
        entries_processed=len(processable),
        entries_linked=entries_linked,
        entries_skipped=skipped_count,
        total_links_created=total_links,
        total_evolution_items=total_evolution,
        results=results,
        errors=errors,
    )


