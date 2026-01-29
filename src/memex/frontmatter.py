"""Frontmatter building utilities for KB entries.

This module provides functions to serialize EntryMetadata to YAML frontmatter.
Extracted from core.py to reduce duplication in add_entry/update_entry.
"""

from datetime import UTC, datetime

import yaml

from .models import EntryMetadata, EvolutionRecord


def _yaml_quote_if_needed(value: str) -> str:
    """Quote a string value if it contains YAML special characters.

    Uses PyYAML to determine if quoting is needed by testing if the value
    roundtrips correctly through YAML parsing. Characters like `: `, `#`,
    and leading `*`, `&`, `%`, `@`, etc. require quoting.

    Args:
        value: The string value to potentially quote.

    Returns:
        The value, quoted if necessary for valid YAML.
    """
    test_yaml = f"key: {value}"
    try:
        parsed = yaml.safe_load(test_yaml)
        if isinstance(parsed, dict) and parsed.get("key") == value:
            return value  # Roundtrips safely, no quoting needed
    except yaml.YAMLError:
        pass
    # Need quoting - let PyYAML figure out proper escaping
    dumped = yaml.safe_dump({"key": value}, default_flow_style=False).strip()
    # Returns 'key: VALUE' or "key: 'VALUE'" - extract the value part
    return dumped[5:]


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
    parts.append(f"title: {_yaml_quote_if_needed(metadata.title)}")

    # Description (one-line summary for search results)
    if metadata.description:
        parts.append(f"description: {_yaml_quote_if_needed(metadata.description)}")

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
        parts.append(f"status: {_yaml_quote_if_needed(metadata.status)}")

    # Source project (where entry was created)
    if metadata.source_project:
        parts.append(f"source_project: {_yaml_quote_if_needed(metadata.source_project)}")

    # Edit sources (projects that have edited this entry)
    if metadata.edit_sources:
        parts.append("edit_sources:")
        parts.append(_format_yaml_list(metadata.edit_sources))

    # Breadcrumb metadata (agent/LLM provenance)
    if metadata.model:
        parts.append(f"model: {_yaml_quote_if_needed(metadata.model)}")
    if metadata.git_branch:
        parts.append(f"git_branch: {_yaml_quote_if_needed(metadata.git_branch)}")
    if metadata.last_edited_by:
        parts.append(f"last_edited_by: {_yaml_quote_if_needed(metadata.last_edited_by)}")

    # A-Mem semantic linking fields
    if metadata.keywords:
        parts.append("keywords:")
        parts.append(_format_yaml_list(metadata.keywords))
    if metadata.semantic_links:
        parts.append("semantic_links:")
        for link in metadata.semantic_links:
            parts.append(f"  - path: {_yaml_quote_if_needed(link.path)}")
            parts.append(f"    score: {link.score}")
            parts.append(f"    reason: {link.reason}")

    # Typed relations (manual, non-A-Mem)
    if metadata.relations:
        parts.append("relations:")
        for relation in metadata.relations:
            parts.append(f"  - path: {_yaml_quote_if_needed(relation.path)}")
            parts.append(f"    type: {_yaml_quote_if_needed(relation.type)}")

    # Evolution history (how this entry has been updated over time)
    if metadata.evolution_history:
        parts.append("evolution_history:")
        for record in metadata.evolution_history:
            parts.append(f"  - timestamp: {record.timestamp.isoformat()}")
            parts.append(f"    trigger_entry: {_yaml_quote_if_needed(record.trigger_entry)}")
            if record.previous_keywords:
                parts.append("    previous_keywords:")
                for kw in record.previous_keywords:
                    parts.append(f"      - {_yaml_quote_if_needed(kw)}")
            if record.new_keywords:
                parts.append("    new_keywords:")
                for kw in record.new_keywords:
                    parts.append(f"      - {_yaml_quote_if_needed(kw)}")
            if record.previous_description is not None:
                prev_desc = _yaml_quote_if_needed(record.previous_description)
                parts.append(f"    previous_description: {prev_desc}")
            if record.new_description is not None:
                new_desc = _yaml_quote_if_needed(record.new_description)
                parts.append(f"    new_description: {new_desc}")

    parts.append("---\n\n")

    return "\n".join(parts)


def _format_yaml_list(items: list[str]) -> str:
    """Format a list as YAML list items with indentation.

    Args:
        items: List of strings to format.

    Returns:
        Multi-line string with "  - item" format, no trailing newline.
    """
    return "\n".join(f"  - {_yaml_quote_if_needed(item)}" for item in items)


def create_new_metadata(
    title: str,
    tags: list[str],
    *,
    source_project: str | None = None,
    contributor: str | None = None,
    model: str | None = None,
    git_branch: str | None = None,
    actor: str | None = None,
    keywords: list[str] | None = None,
    semantic_links: list | None = None,
    relations: list | None = None,
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
        keywords: LLM-extracted key concepts for semantic linking.
        semantic_links: List of SemanticLink objects for manual linking.

    Returns:
        EntryMetadata populated with creation metadata.
    """
    return EntryMetadata(
        title=title,
        tags=tags,
        created=datetime.now(UTC),
        updated=None,
        contributors=[contributor] if contributor else [],
        source_project=source_project,
        model=model,
        git_branch=git_branch,
        last_edited_by=actor,
        keywords=keywords or [],
        semantic_links=semantic_links or [],
        relations=relations or [],
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
    keywords: list[str] | None = None,
    semantic_links: list | None = None,
    relations: list | None = None,
    description: str | None = None,
    evolution_history: list[EvolutionRecord] | None = None,
) -> EntryMetadata:
    """Create updated metadata for an existing entry.

    Preserves immutable fields (title, created, source_project) while
    updating mutable fields and adding edit provenance.

    Args:
        metadata: Existing entry metadata.
        new_tags: Updated tags (or None to preserve existing).
        new_contributor: New contributor to add to contributors list.
        edit_source: Project making the edit (added to edit_sources if different).
        model: LLM model identifier for the edit.
        git_branch: Current git branch.
        actor: Actor making the edit.
        keywords: Updated keywords (or None to preserve existing).
        semantic_links: Updated semantic links (or None to preserve existing).
        description: Updated description (or None to preserve existing).
        evolution_history: Updated evolution history (or None to preserve existing).

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
        description=description if description is not None else metadata.description,
        tags=new_tags if new_tags is not None else list(metadata.tags),
        created=metadata.created,
        updated=datetime.now(UTC),
        contributors=contributors,
        aliases=list(metadata.aliases),
        status=metadata.status,
        source_project=metadata.source_project,
        edit_sources=edit_sources,
        model=model,
        git_branch=git_branch,
        last_edited_by=actor,
        # A-Mem semantic linking fields
        keywords=keywords if keywords is not None else list(metadata.keywords),
        semantic_links=(
            semantic_links if semantic_links is not None else list(metadata.semantic_links)
        ),
        relations=relations if relations is not None else list(metadata.relations),
        evolution_history=(
            evolution_history if evolution_history is not None else list(metadata.evolution_history)
        ),
    )
