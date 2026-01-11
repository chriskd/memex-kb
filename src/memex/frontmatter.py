"""Frontmatter building utilities for KB entries.

This module provides functions to serialize EntryMetadata to YAML frontmatter.
Extracted from core.py to reduce duplication in add_entry/update_entry.
"""

from datetime import datetime, timezone

from .models import EntryMetadata


def build_frontmatter(metadata: EntryMetadata) -> str:
    """Build YAML frontmatter string from metadata.

    Produces consistent, clean frontmatter by:
    - Always including required fields (title, tags, created)
    - Only including optional fields when they have non-default values
    - Using standard YAML list format for multi-value fields

    Args:
        metadata: The entry metadata to serialize.

    Returns:
        Complete frontmatter string including --- delimiters and trailing newlines.
    """
    parts = ["---"]

    # Required fields
    parts.append(f"title: {metadata.title}")
    parts.append("tags:")
    parts.append(_format_yaml_list(metadata.tags))
    parts.append(f"created: {metadata.created.isoformat()}")

    # Updated date (present on updates, not on creation)
    if metadata.updated:
        parts.append(f"updated: {metadata.updated.isoformat()}")

    # Contributors
    if metadata.contributors:
        parts.append("contributors:")
        parts.append(_format_yaml_list(metadata.contributors))

    # Aliases
    if metadata.aliases:
        parts.append("aliases:")
        parts.append(_format_yaml_list(metadata.aliases))

    # Status (only if not default)
    if metadata.status != "published":
        parts.append(f"status: {metadata.status}")

    # Source project (where entry was created)
    if metadata.source_project:
        parts.append(f"source_project: {metadata.source_project}")

    # Edit sources (projects that have edited this entry)
    if metadata.edit_sources:
        parts.append("edit_sources:")
        parts.append(_format_yaml_list(metadata.edit_sources))

    # Breadcrumb metadata (agent/LLM provenance)
    if metadata.model:
        parts.append(f"model: {metadata.model}")
    if metadata.git_branch:
        parts.append(f"git_branch: {metadata.git_branch}")
    if metadata.last_edited_by:
        parts.append(f"last_edited_by: {metadata.last_edited_by}")

    # Beads integration fields (preserved for backwards compatibility)
    if metadata.beads_issues:
        parts.append("beads_issues:")
        parts.append(_format_yaml_list(metadata.beads_issues))
    if metadata.beads_project:
        parts.append(f"beads_project: {metadata.beads_project}")

    parts.append("---\n\n")

    return "\n".join(parts)


def _format_yaml_list(items: list[str]) -> str:
    """Format a list as YAML list items with indentation.

    Args:
        items: List of strings to format.

    Returns:
        Multi-line string with "  - item" format, no trailing newline.
    """
    return "\n".join(f"  - {item}" for item in items)


def create_new_metadata(
    title: str,
    tags: list[str],
    *,
    source_project: str | None = None,
    contributor: str | None = None,
    model: str | None = None,
    git_branch: str | None = None,
    actor: str | None = None,
) -> EntryMetadata:
    """Create metadata for a new KB entry.

    This is a convenience function for creating new entries with
    all the standard breadcrumb metadata populated.

    Args:
        title: Entry title.
        tags: Entry tags (at least one required).
        source_project: Project context where entry is being created.
        contributor: Contributor identity (name or "Name <email>").
        model: LLM model identifier if created by an agent.
        git_branch: Current git branch.
        actor: Actor identity (agent name or human username).

    Returns:
        EntryMetadata populated with creation metadata.
    """
    return EntryMetadata(
        title=title,
        tags=tags,
        created=datetime.now(timezone.utc),
        updated=None,
        contributors=[contributor] if contributor else [],
        source_project=source_project,
        model=model,
        git_branch=git_branch,
        last_edited_by=actor,
    )


def update_metadata_for_edit(
    metadata: EntryMetadata,
    *,
    new_tags: list[str] | None = None,
    new_contributor: str | None = None,
    edit_source: str | None = None,
    model: str | None = None,
    git_branch: str | None = None,
    actor: str | None = None,
) -> EntryMetadata:
    """Create updated metadata for an existing entry.

    Preserves immutable fields (title, created, source_project) while
    updating mutable fields and adding edit provenance.

    Args:
        metadata: Existing entry metadata.
        new_tags: Updated tags (or None to preserve existing).
        new_contributor: New contributor to add to contributors list.
        edit_source: Project making the edit (added to edit_sources if different from source_project).
        model: LLM model identifier for the edit.
        git_branch: Current git branch.
        actor: Actor making the edit.

    Returns:
        New EntryMetadata with updated fields.
    """
    # Build updated contributors list
    contributors = list(metadata.contributors)
    if new_contributor and new_contributor not in contributors:
        contributors.append(new_contributor)

    # Build updated edit_sources list
    edit_sources = list(metadata.edit_sources)
    if edit_source and edit_source != metadata.source_project and edit_source not in edit_sources:
        edit_sources.append(edit_source)

    return EntryMetadata(
        title=metadata.title,
        tags=new_tags if new_tags is not None else list(metadata.tags),
        created=metadata.created,
        updated=datetime.now(timezone.utc),
        contributors=contributors,
        aliases=list(metadata.aliases),
        status=metadata.status,
        source_project=metadata.source_project,
        edit_sources=edit_sources,
        model=model,
        git_branch=git_branch,
        last_edited_by=actor,
        # Preserve beads fields for backwards compatibility
        beads_issues=list(metadata.beads_issues),
        beads_project=metadata.beads_project,
    )
