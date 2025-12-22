"""Core business logic for voidlabs-kb.

This module contains the pure business logic, separated from MCP protocol concerns.
Both the MCP server (server.py) and CLI (cli.py) import from here.

Design principles:
- All functions are async for consistency with MCP server
- No MCP decorators or protocol-specific code
- Lazy initialization of expensive resources (searcher, embeddings)
"""

import os
import re
import shutil
import subprocess
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from .backlinks_cache import ensure_backlink_cache, rebuild_backlink_cache
from .config import MAX_CONTENT_RESULTS, get_kb_root
from .evaluation import run_quality_checks
from .indexer import HybridSearcher
from .models import DocumentChunk, IndexStatus, KBEntry, QualityReport, SearchResponse, SearchResult
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
        Project name (e.g., "voidlabs-ansible") or None if unavailable.
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
    except (subprocess.TimeoutExpired, OSError):
        pass

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
    except (subprocess.TimeoutExpired, OSError):
        pass

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
    except (subprocess.TimeoutExpired, OSError):
        pass
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
    """Convert title to URL-friendly slug."""
    # Lowercase
    slug = title.lower()
    # Replace spaces and underscores with hyphens
    slug = re.sub(r"[\s_]+", "-", slug)
    # Remove non-alphanumeric characters (except hyphens)
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    # Remove consecutive hyphens
    slug = re.sub(r"-+", "-", slug)
    # Strip leading/trailing hyphens
    slug = slug.strip("-")
    return slug


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
            except ParseError:
                # Fall back to None if parsing fails
                pass

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
    min_score: float = 0.5,
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
    min_score: float = 0.3,
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
        if result.score < 0.3:  # Skip low-similarity entries
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
            tag_scores[tag] = tag_scores.get(tag, 0) + 0.5
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


# ─────────────────────────────────────────────────────────────────────────────
# Core business logic functions
# ─────────────────────────────────────────────────────────────────────────────


async def search(
    query: str,
    limit: int = 10,
    mode: Literal["hybrid", "keyword", "semantic"] = "hybrid",
    tags: list[str] | None = None,
    include_content: bool = False,
) -> SearchResponse:
    """Search the knowledge base.

    Args:
        query: Search query string.
        limit: Maximum number of results (default 10).
        mode: Search mode - "hybrid" (default), "keyword", or "semantic".
        tags: Optional list of tags to filter results.
        include_content: If True, include full document content in results.
                         Default False (snippet only). Limited to MAX_CONTENT_RESULTS.

    Returns:
        SearchResponse with results and optional warnings.
    """
    searcher = get_searcher()
    # Auto-detect project context for boosting entries from current project
    project_context = get_current_project()
    results = searcher.search(query, limit=limit, mode=mode, project_context=project_context)
    warnings: list[str] = []

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

    Returns:
        Dict with 'path' of created file, 'suggested_links' to consider adding,
        and 'suggested_tags' based on content similarity and existing taxonomy.
    """
    kb_root = get_kb_root()

    # Determine target directory: prefer 'directory' over 'category'
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
    else:
        valid_categories = get_valid_categories()
        raise ValueError(
            "Either 'category' or 'directory' must be provided. "
            f"Existing categories: {', '.join(valid_categories)}"
        )

    if not tags:
        raise ValueError("At least one tag is required")

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

    # Build frontmatter
    today = date.today().isoformat()
    tags_yaml = "\n".join(f"  - {tag}" for tag in tags)

    # Get contributor identity
    contributor = get_current_contributor()
    contributors_yaml = f"\ncontributors:\n  - {contributor}" if contributor else ""

    # Get source project context
    source_project = get_current_project()
    source_project_yaml = f"\nsource_project: {source_project}" if source_project else ""

    # Get breadcrumb metadata (agent/LLM provenance)
    model = get_llm_model()
    model_yaml = f"\nmodel: {model}" if model else ""

    git_branch = get_git_branch()
    git_branch_yaml = f"\ngit_branch: {git_branch}" if git_branch else ""

    actor = get_actor_identity()
    last_edited_by_yaml = f"\nlast_edited_by: {actor}" if actor else ""

    frontmatter = f"""---
title: {title}
tags:
{tags_yaml}
created: {today}{contributors_yaml}{source_project_yaml}{model_yaml}{git_branch_yaml}{last_edited_by_yaml}
---

"""
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
    except ParseError:
        # File was written but indexing failed - still return success
        pass

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

    return {"path": rel_path, "suggested_links": suggested_links, "suggested_tags": suggested_tags}


async def update_entry(
    path: str,
    content: str | None = None,
    tags: list[str] | None = None,
    section_updates: dict[str, str] | None = None,
) -> dict:
    """Update an existing KB entry.

    Args:
        path: Path to the entry relative to KB root (e.g., "development/python-tooling.md").
        content: New markdown content (without frontmatter).
        tags: Optional new list of tags. If provided, replaces existing tags.
        section_updates: Optional dict of section heading -> new content.

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

    if content is None and not section_updates:
        raise ValueError("Provide new content or section_updates")

    # Parse existing entry to get metadata
    try:
        metadata, existing_content, _ = parse_entry(file_path)
    except ParseError as e:
        raise ValueError(f"Failed to parse existing entry: {e}") from e

    # Update metadata
    today = date.today()
    new_tags = tags if tags is not None else list(metadata.tags)

    if not new_tags:
        raise ValueError("At least one tag is required")

    # Add current contributor if not already present
    contributors = list(metadata.contributors)
    current_contributor = get_current_contributor()
    if current_contributor and current_contributor not in contributors:
        contributors.append(current_contributor)

    # Track edit sources (projects that have edited this entry)
    edit_sources = list(metadata.edit_sources)
    current_project = get_current_project()
    if current_project and current_project not in edit_sources:
        # Don't add if it's the same as source_project (original creator)
        if current_project != metadata.source_project:
            edit_sources.append(current_project)

    # Build updated frontmatter
    tags_yaml = "\n".join(f"  - {tag}" for tag in new_tags)
    contributors_yaml = "\n".join(f"  - {c}" for c in contributors)
    aliases_yaml = "\n".join(f"  - {a}" for a in metadata.aliases)
    edit_sources_yaml = "\n".join(f"  - {s}" for s in edit_sources)

    frontmatter_parts = [
        "---",
        f"title: {metadata.title}",
        "tags:",
        tags_yaml,
        f"created: {metadata.created.isoformat()}",
        f"updated: {today.isoformat()}",
    ]

    if contributors:
        frontmatter_parts.append("contributors:")
        frontmatter_parts.append(contributors_yaml)

    if metadata.aliases:
        frontmatter_parts.append("aliases:")
        frontmatter_parts.append(aliases_yaml)

    if metadata.status != "published":
        frontmatter_parts.append(f"status: {metadata.status}")

    # Preserve source_project from original creation
    if metadata.source_project:
        frontmatter_parts.append(f"source_project: {metadata.source_project}")

    # Add edit_sources if any projects have edited this
    if edit_sources:
        frontmatter_parts.append("edit_sources:")
        frontmatter_parts.append(edit_sources_yaml)

    # Preserve beads integration fields
    if metadata.beads_issues:
        beads_issues_yaml = "\n".join(f"  - {i}" for i in metadata.beads_issues)
        frontmatter_parts.append("beads_issues:")
        frontmatter_parts.append(beads_issues_yaml)

    if metadata.beads_project:
        frontmatter_parts.append(f"beads_project: {metadata.beads_project}")

    # Update breadcrumb metadata (agent/LLM provenance)
    model = get_llm_model()
    if model:
        frontmatter_parts.append(f"model: {model}")

    git_branch = get_git_branch()
    if git_branch:
        frontmatter_parts.append(f"git_branch: {git_branch}")

    actor = get_actor_identity()
    if actor:
        frontmatter_parts.append(f"last_edited_by: {actor}")

    frontmatter_parts.append("---\n\n")
    frontmatter = "\n".join(frontmatter_parts)

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
    except (ParseError, Exception):
        # Indexing failed but file was updated
        pass

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


async def link_beads(
    path: str,
    issues: list[str] | None = None,
    project: str | None = None,
) -> dict:
    """Link a KB entry to beads issues.

    Args:
        path: Path to the entry relative to KB root (e.g., "projects/foo.md").
        issues: List of beads issue IDs to link (e.g., ["proj-123", "proj-456"]).
        project: Beads project name to link all its issues.

    Returns:
        Dict with 'path', 'linked_issues', and 'project' if set.
    """
    if not issues and not project:
        raise ValueError("Provide at least one of: issues, project")

    kb_root = get_kb_root()
    file_path = kb_root / path

    if not file_path.exists():
        raise ValueError(f"Entry not found: {path}")

    # Parse existing entry
    try:
        metadata, content, _ = parse_entry(file_path)
    except ParseError as e:
        raise ValueError(f"Failed to parse entry: {e}") from e

    # Merge with existing beads_issues (avoid duplicates)
    existing_issues = set(metadata.beads_issues)
    new_issues = list(existing_issues)
    for issue_id in (issues or []):
        if issue_id not in existing_issues:
            new_issues.append(issue_id)

    # Update beads_project if provided
    new_project = project if project else metadata.beads_project

    # Rebuild frontmatter with beads fields
    today = date.today()
    tags_yaml = "\n".join(f"  - {tag}" for tag in metadata.tags)
    contributors_yaml = "\n".join(f"  - {c}" for c in metadata.contributors)
    aliases_yaml = "\n".join(f"  - {a}" for a in metadata.aliases)
    edit_sources_yaml = "\n".join(f"  - {s}" for s in metadata.edit_sources)

    frontmatter_parts = [
        "---",
        f"title: {metadata.title}",
        "tags:",
        tags_yaml,
        f"created: {metadata.created.isoformat()}",
        f"updated: {today.isoformat()}",
    ]

    if metadata.contributors:
        frontmatter_parts.append("contributors:")
        frontmatter_parts.append(contributors_yaml)

    if metadata.aliases:
        frontmatter_parts.append("aliases:")
        frontmatter_parts.append(aliases_yaml)

    if metadata.status != "published":
        frontmatter_parts.append(f"status: {metadata.status}")

    if metadata.source_project:
        frontmatter_parts.append(f"source_project: {metadata.source_project}")

    if metadata.edit_sources:
        frontmatter_parts.append("edit_sources:")
        frontmatter_parts.append(edit_sources_yaml)

    # Add beads fields
    if new_issues:
        beads_issues_yaml = "\n".join(f"  - {i}" for i in new_issues)
        frontmatter_parts.append("beads_issues:")
        frontmatter_parts.append(beads_issues_yaml)

    if new_project:
        frontmatter_parts.append(f"beads_project: {new_project}")

    frontmatter_parts.append("---\n\n")
    frontmatter = "\n".join(frontmatter_parts)

    # Write updated file
    file_path.write_text(frontmatter + content, encoding="utf-8")

    return {
        "path": path,
        "linked_issues": new_issues,
        "project": new_project,
    }


async def register_beads_project(
    path: str,
    prefix: str | None = None,
) -> dict:
    """Register a beads project in the KB registry.

    Args:
        path: Absolute path to the project directory containing .beads/.
        prefix: Issue prefix (e.g., "dv", "epstein"). Auto-detected if not provided.

    Returns:
        Dict with 'prefix', 'path', and 'status' (added/exists/error).
    """
    from .beads_client import REGISTRY_FILE, BeadsClient, load_beads_registry

    project_path = Path(path).resolve()

    # Validate .beads/ exists
    if not (project_path / ".beads").exists():
        raise ValueError(f"No .beads/ directory found at: {project_path}")

    # Auto-detect prefix if not provided
    if not prefix:
        client = BeadsClient(beads_root=project_path)
        prefix = client.get_project_name()
        if not prefix:
            raise ValueError(
                "Could not auto-detect project prefix. "
                "Please provide it explicitly."
            )

    # Load existing registry
    kb_root = get_kb_root()
    registry_path = kb_root.parent / REGISTRY_FILE
    if not registry_path.exists():
        registry_path = kb_root / REGISTRY_FILE

    existing = load_beads_registry()

    # Check if already registered
    if prefix in existing:
        existing_path = existing[prefix]
        if existing_path == project_path:
            return {
                "prefix": prefix,
                "path": str(project_path),
                "status": "exists",
                "message": f"Project '{prefix}' already registered",
            }
        else:
            # Different path - update it
            pass

    # Add to registry file
    # Read existing content or create new
    if registry_path.exists():
        content = registry_path.read_text()
    else:
        content = "# Beads Registry\n# Maps issue ID prefixes to project directories\n\n"

    # Append new entry
    content += f"{prefix}: {project_path}\n"
    registry_path.write_text(content)

    return {
        "prefix": prefix,
        "path": str(project_path),
        "status": "added",
        "message": f"Registered project '{prefix}' at {project_path}",
    }


async def get_entry(path: str) -> KBEntry:
    """Read a KB entry.

    Args:
        path: Path to the entry relative to KB root (e.g., "development/python-tooling.md").

    Returns:
        KBEntry with metadata, content, links, and backlinks.
    """
    kb_root = get_kb_root()
    file_path = kb_root / path

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
    except Exception:
        pass  # Don't let view tracking failures affect get

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

    Returns:
        List of {path, title, tags, created, updated} dictionaries.
    """
    kb_root = get_kb_root()

    if not kb_root.exists():
        return []

    # Determine search path: prefer 'directory' over 'category'
    if directory:
        abs_path, _ = validate_nested_path(directory)
        if not abs_path.exists() or not abs_path.is_dir():
            raise ValueError(f"Directory not found: {directory}")
        search_path = abs_path
    elif category:
        search_path = kb_root / category
        if not search_path.exists() or not search_path.is_dir():
            raise ValueError(f"Category not found: {category}")
    else:
        search_path = kb_root

    results = []

    # Choose glob pattern based on recursive flag
    md_files = search_path.rglob("*.md") if recursive else search_path.glob("*.md")

    for md_file in md_files:
        # Skip index files
        if md_file.name.startswith("_"):
            continue

        rel_path = str(md_file.relative_to(kb_root))

        try:
            metadata, _, _ = parse_entry(md_file)
        except ParseError:
            continue

        # Filter by tag if specified
        if tag and tag not in metadata.tags:
            continue

        results.append(
            {
                "path": rel_path,
                "title": metadata.title,
                "tags": metadata.tags,
                "created": metadata.created.isoformat() if metadata.created else None,
                "updated": metadata.updated.isoformat() if metadata.updated else None,
            }
        )

        if len(results) >= limit:
            break

    return results


async def whats_new(
    days: int = 30,
    limit: int = 10,
    include_created: bool = True,
    include_updated: bool = True,
    category: str | None = None,
    tag: str | None = None,
) -> list[dict]:
    """List recent KB entries.

    Args:
        days: Look back period in days (default 30).
        limit: Maximum entries to return (default 10).
        include_created: Include newly created entries (default True).
        include_updated: Include recently updated entries (default True).
        category: Optional category filter.
        tag: Optional tag filter.

    Returns:
        List of {path, title, tags, created, updated, activity_type, activity_date}.
    """
    kb_root = get_kb_root()

    if not kb_root.exists():
        return []

    # Determine search path
    if category:
        search_path = kb_root / category
        if not search_path.exists() or not search_path.is_dir():
            raise ValueError(f"Category not found: {category}")
    else:
        search_path = kb_root

    cutoff_date = date.today() - timedelta(days=days)
    candidates: list[dict] = []

    for md_file in search_path.rglob("*.md"):
        if md_file.name.startswith("_"):
            continue

        rel_path = str(md_file.relative_to(kb_root))

        try:
            metadata, _, _ = parse_entry(md_file)
        except ParseError:
            continue

        # Filter by tag if specified
        if tag and tag not in metadata.tags:
            continue

        # Determine activity type and date
        activity_type: str | None = None
        activity_date: date | None = None

        # Check updated first (takes precedence if both qualify)
        if include_updated and metadata.updated and metadata.updated >= cutoff_date:
            activity_type = "updated"
            activity_date = metadata.updated
        elif include_created and metadata.created >= cutoff_date:
            activity_type = "created"
            activity_date = metadata.created

        if activity_type is None:
            continue

        candidates.append(
            {
                "path": rel_path,
                "title": metadata.title,
                "tags": metadata.tags,
                "created": metadata.created.isoformat() if metadata.created else None,
                "updated": metadata.updated.isoformat() if metadata.updated else None,
                "activity_type": activity_type,
                "activity_date": activity_date.isoformat(),
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


async def reindex() -> IndexStatus:
    """Rebuild search indices.

    Returns:
        IndexStatus with document counts.
    """
    kb_root = get_kb_root()
    searcher = get_searcher()

    searcher.reindex(kb_root)
    rebuild_backlink_cache(kb_root)

    # Clean up views for deleted entries
    try:
        from .views_tracker import cleanup_stale_entries

        valid_paths = {
            str(p.relative_to(kb_root))
            for p in kb_root.rglob("*.md")
            if not p.name.startswith("_")
        }
        cleanup_stale_entries(valid_paths)
    except Exception:
        pass  # Don't fail reindex if cleanup fails

    status = searcher.status()
    return status


async def tree(
    path: str = "",
    depth: int = 3,
    include_files: bool = True,
) -> dict:
    """Show directory tree.

    Args:
        path: Starting path relative to KB root (empty for root).
        depth: Maximum depth to display (default 3).
        include_files: Whether to include .md files (default True).

    Returns:
        Dict with tree structure and counts.
    """
    kb_root = get_kb_root()

    if path:
        abs_path, _ = validate_nested_path(path)
        if not abs_path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        start_path = abs_path
    else:
        start_path = kb_root

    def _build_tree(current: Path, current_depth: int) -> dict:
        """Recursively build tree structure."""
        result = {}

        if current_depth >= depth:
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
                children = _build_tree(item, current_depth + 1)
                result[item.name] = {"_type": "directory", **children}
            elif item.is_file() and item.suffix == ".md" and include_files:
                # Try to get title from frontmatter
                title = None
                try:
                    metadata, _, _ = parse_entry(item)
                    title = metadata.title
                except (ParseError, Exception):
                    pass
                result[item.name] = {"_type": "file", "title": title}

        return result

    tree_data = _build_tree(start_path, 0)

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

    total_dirs, total_files = _count(tree_data)

    return {
        "tree": tree_data,
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
            except ParseError:
                pass

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
    except Exception:
        pass  # Don't fail delete if view cleanup fails

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
    cutoff_date = date.today() - timedelta(days=stale_days)

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
            last_activity = entry["updated"] or entry["created"]
            if last_activity and last_activity < cutoff_date:
                days_old = (date.today() - last_activity).days
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
            except ParseError:
                pass

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
    min_score: float = 0.5,
) -> list[dict]:
    """Suggest links to add to an entry based on content similarity.

    Args:
        path: Path to the entry to suggest links for.
        limit: Maximum suggestions to return (default 5).
        min_score: Minimum similarity score threshold (default 0.5).

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
