"""LLM integration for memory evolution.

This module provides async functions for LLM-driven memory evolution,
where neighbors of new entries are analyzed and their keywords/context
updated based on the relationship.

Supports both Anthropic (direct API) and OpenRouter (OpenAI-compatible gateway).
Provider is configured via .kbconfig or auto-detected from environment.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from .llm_providers import (
    LLMProviderError,
    get_async_client,
    resolve_model,
)

log = logging.getLogger(__name__)


@dataclass
class EvolutionSuggestion:
    """Suggested updates for a neighbor entry based on a new connection."""

    neighbor_path: str
    """Path to the neighbor entry being evolved."""

    new_keywords: list[str]
    """Complete new keyword list for the neighbor (replaces existing)."""

    relationship: str
    """One-sentence description of the relationship, or empty string."""

    new_context: str = ""
    """Updated context/description for the neighbor (one sentence describing semantic role)."""

    should_evolve: bool = True
    """Whether the LLM recommends evolving this neighbor (default True for backwards compat)."""


# Backwards compatibility alias
LLMConfigurationError = LLMProviderError


def _get_client() -> tuple[Any, str]:
    """Get an async LLM client with provider info.

    Returns:
        Tuple of (client, provider) where:
        - client is AsyncAnthropic or AsyncOpenAI
        - provider is 'anthropic' or 'openrouter'

    Raises:
        LLMProviderError: If provider cannot be determined.
    """
    return get_async_client()


async def evolve_single_neighbor(
    new_entry_title: str,
    new_entry_content: str,
    new_entry_keywords: list[str],
    neighbor_path: str,
    neighbor_title: str,
    neighbor_content: str,
    neighbor_keywords: list[str],
    link_score: float,
    model: str,
) -> EvolutionSuggestion:
    """Analyze relationship between new entry and neighbor, suggest updates.

    Args:
        new_entry_title: Title of the newly added entry.
        new_entry_content: Content of the new entry (may be truncated).
        new_entry_keywords: Keywords of the new entry.
        neighbor_path: Path to the neighbor entry.
        neighbor_title: Title of the neighbor entry.
        neighbor_content: Content of the neighbor (may be truncated).
        neighbor_keywords: Current keywords of the neighbor.
        link_score: Similarity score between entries (0.0-1.0).
        model: Model ID to use (canonical or provider-specific).

    Returns:
        EvolutionSuggestion with new keywords (replaces existing) and relationship.

    Raises:
        LLMProviderError: If provider not configured.
        Exception: On API errors.
    """
    client, provider = _get_client()
    resolved_model = resolve_model(model, provider)

    score_str = f"{link_score:.2f}"
    new_kw_str = ", ".join(new_entry_keywords) if new_entry_keywords else "none"
    neighbor_kw_str = ", ".join(neighbor_keywords) if neighbor_keywords else "none"

    prompt = f"""A new KB entry was linked to an existing entry (similarity: {score_str}).

NEW ENTRY:
Title: {new_entry_title}
Keywords: {new_kw_str}
Content (first 500 chars): {new_entry_content[:500]}

EXISTING ENTRY (to evolve):
Title: {neighbor_title}
Current keywords: {neighbor_kw_str}
Content (first 500 chars): {neighbor_content[:500]}

First, determine if the EXISTING entry should be evolved based on this connection.
Consider: Is this connection meaningful enough to warrant updating the existing entry?
High similarity scores don't always mean the entries are conceptually related in a way that benefits from evolution.

If should_evolve is true, suggest the COMPLETE new keyword list for the EXISTING entry.
The new list should:
- Keep relevant existing keywords
- Add new keywords based on the connection (if genuinely relevant)
- Remove keywords that are no longer appropriate
- Typically contain 3-7 keywords total

Also provide:
1. One sentence describing this relationship (or empty string if weak)
2. Updated context: a single sentence describing the EXISTING entry's semantic role in the KB

Respond with JSON only:
{{"should_evolve": true, "new_keywords": ["kw1", "kw2", "kw3"], "relationship": "sentence or empty",
"new_context": "one sentence describing what this entry is about"}}"""

    # Make provider-appropriate API call
    if provider == "anthropic":
        response = await client.messages.create(
            model=resolved_model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = response.content[0].text
    else:
        response = await client.chat.completions.create(
            model=resolved_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        response_text = response.choices[0].message.content or ""

    try:
        result = json.loads(response_text)
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        log.warning("Failed to parse LLM response for evolution: %s", e)
        return EvolutionSuggestion(
            neighbor_path=neighbor_path,
            new_keywords=neighbor_keywords,  # Preserve existing on error
            relationship="",
            new_context="",
        )

    # Validate and sanitize keywords
    new_keywords = result.get("new_keywords", [])
    if not isinstance(new_keywords, list):
        new_keywords = neighbor_keywords  # Preserve existing on invalid response
    else:
        new_keywords = [str(kw).strip().lower() for kw in new_keywords if kw]
        if not new_keywords:
            new_keywords = neighbor_keywords  # Preserve existing if empty

    relationship = result.get("relationship", "")
    if not isinstance(relationship, str):
        relationship = ""
    relationship = relationship.strip()

    new_context = result.get("new_context", "")
    if not isinstance(new_context, str):
        new_context = ""
    new_context = new_context.strip()

    # Parse should_evolve (default True for backwards compatibility)
    should_evolve = result.get("should_evolve", True)
    if not isinstance(should_evolve, bool):
        should_evolve = True  # Default to evolving if invalid value

    return EvolutionSuggestion(
        neighbor_path=neighbor_path,
        new_keywords=new_keywords,
        relationship=relationship,
        new_context=new_context,
        should_evolve=should_evolve,
    )


@dataclass
class NeighborInfo:
    """Information about a neighbor entry for batched evolution."""

    path: str
    title: str
    content: str
    keywords: list[str]
    score: float


async def evolve_neighbors_batched(
    new_entry_title: str,
    new_entry_content: str,
    new_entry_keywords: list[str],
    neighbors: list[NeighborInfo],
    model: str,
) -> list[EvolutionSuggestion]:
    """Analyze relationships with multiple neighbors in a single LLM call.

    More efficient than individual calls when linking to multiple neighbors.

    Args:
        new_entry_title: Title of the newly added entry.
        new_entry_content: Content of the new entry.
        new_entry_keywords: Keywords of the new entry.
        neighbors: List of neighbor entries to analyze.
        model: Model ID to use (canonical or provider-specific).

    Returns:
        List of EvolutionSuggestions, one per neighbor.
    """
    if not neighbors:
        return []

    # For single neighbor, use the simpler prompt
    if len(neighbors) == 1:
        n = neighbors[0]
        return [
            await evolve_single_neighbor(
                new_entry_title=new_entry_title,
                new_entry_content=new_entry_content,
                new_entry_keywords=new_entry_keywords,
                neighbor_path=n.path,
                neighbor_title=n.title,
                neighbor_content=n.content,
                neighbor_keywords=n.keywords,
                link_score=n.score,
                model=model,
            )
        ]

    client, provider = _get_client()
    resolved_model = resolve_model(model, provider)

    # Build neighbor descriptions
    neighbor_sections = []
    for i, n in enumerate(neighbors):
        section = f"""NEIGHBOR {i + 1} (path: {n.path}, similarity: {n.score:.2f}):
Title: {n.title}
Current keywords: {', '.join(n.keywords) if n.keywords else 'none'}
Content (first 300 chars): {n.content[:300]}"""
        neighbor_sections.append(section)

    new_kw_str = ", ".join(new_entry_keywords) if new_entry_keywords else "none"

    prompt = f"""A new KB entry was linked to multiple existing entries.

NEW ENTRY:
Title: {new_entry_title}
Keywords: {new_kw_str}
Content (first 400 chars): {new_entry_content[:400]}

{chr(10).join(neighbor_sections)}

For EACH neighbor, first determine if it should be evolved based on this connection.
Consider: Is this connection meaningful enough to warrant updating the existing entry?
High similarity scores don't always mean entries are conceptually related in a way that benefits from evolution.

If should_evolve is true, suggest the COMPLETE new keyword list.
Each new list should:
- Keep relevant existing keywords
- Add new keywords based on the connection (if genuinely relevant)
- Remove keywords that are no longer appropriate
- Typically contain 3-7 keywords total

Also provide:
1. One sentence describing the relationship (or empty string if weak)
2. Updated context: a single sentence describing that neighbor's semantic role in the KB

Respond with JSON array, one object per neighbor in order:
[{{"path": "neighbor_path", "should_evolve": true, "new_keywords": ["kw1", "kw2"], "relationship": "sentence",
"new_context": "one sentence describing what this entry is about"}}]"""

    try:
        # Make provider-appropriate API call
        if provider == "anthropic":
            response = await client.messages.create(
                model=resolved_model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
        else:
            response = await client.chat.completions.create(
                model=resolved_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            raw = response.choices[0].message.content or ""

        # Parse response - might be wrapped in an object
        parsed = json.loads(raw)

        # Handle both array and object with "neighbors" key
        if isinstance(parsed, list):
            results = parsed
        elif isinstance(parsed, dict):
            results = parsed.get("neighbors", parsed.get("results", []))
        else:
            results = []

    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        log.warning("Failed to parse batched LLM response: %s", e)
        results = []
    except Exception as e:
        log.warning("LLM API error during batched evolution: %s", e)
        results = []

    # Build suggestions, filling in defaults for missing/invalid entries
    suggestions = []
    for i, n in enumerate(neighbors):
        if i < len(results) and isinstance(results[i], dict):
            r = results[i]
            new_keywords = r.get("new_keywords", [])
            if not isinstance(new_keywords, list):
                new_keywords = n.keywords  # Preserve existing on invalid
            else:
                new_keywords = [str(kw).strip().lower() for kw in new_keywords if kw]
                if not new_keywords:
                    new_keywords = n.keywords  # Preserve existing if empty

            relationship = r.get("relationship", "")
            if not isinstance(relationship, str):
                relationship = ""

            new_context = r.get("new_context", "")
            if not isinstance(new_context, str):
                new_context = ""

            # Parse should_evolve (default True for backwards compatibility)
            should_evolve = r.get("should_evolve", True)
            if not isinstance(should_evolve, bool):
                should_evolve = True
        else:
            new_keywords = n.keywords  # Preserve existing on missing
            relationship = ""
            new_context = ""
            should_evolve = True  # Default to evolving on missing response

        suggestions.append(
            EvolutionSuggestion(
                neighbor_path=n.path,
                new_keywords=new_keywords,
                relationship=relationship.strip(),
                new_context=new_context.strip(),
                should_evolve=should_evolve,
            )
        )

    return suggestions


async def analyze_evolution(
    new_entry_title: str,
    new_entry_content: str,
    new_entry_keywords: list[str],
    neighbors: list[NeighborInfo],
    model: str,
) -> "EvolutionDecision":
    """Analyze whether a new entry should trigger evolution of its neighbors.

    Mirrors A-Mem's evolution decision: LLM explicitly decides whether
    evolution should happen, not just based on similarity score.

    Args:
        new_entry_title: Title of the newly added entry.
        new_entry_content: Content of the new entry.
        new_entry_keywords: Keywords of the new entry.
        neighbors: List of neighbor entries to potentially evolve.
        model: OpenRouter model ID to use.

    Returns:
        EvolutionDecision with should_evolve flag and per-neighbor updates.

    Raises:
        LLMConfigurationError: If API key not configured.
    """
    from .models import EvolutionDecision, NeighborUpdate

    if not neighbors:
        return EvolutionDecision(should_evolve=False)

    client, provider = _get_client()
    resolved_model = resolve_model(model, provider)

    # Build neighbor descriptions for the prompt
    neighbor_sections = []
    for i, n in enumerate(neighbors):
        kw_str = ", ".join(n.keywords) if n.keywords else "none"
        section = f"""NEIGHBOR {i + 1} (path: {n.path}, similarity: {n.score:.2f}):
Title: {n.title}
Current keywords: {kw_str}
Content preview: {n.content[:400]}"""
        neighbor_sections.append(section)

    new_kw_str = ", ".join(new_entry_keywords) if new_entry_keywords else "none"

    # A-Mem style prompt with explicit should_evolve decision
    prompt = f"""You are a memory evolution agent analyzing relationships between KB entries.

NEW ENTRY:
Title: {new_entry_title}
Keywords: {new_kw_str}
Content: {new_entry_content[:600]}

NEIGHBORS (semantically similar entries):
{chr(10).join(neighbor_sections)}

Analyze this new entry and its neighbors. Determine:

1. Should any evolution happen? Consider:
   - Is there a meaningful semantic relationship worth capturing?
   - Would updating neighbors add genuine value (not just noise)?
   - Are the connections strong enough to warrant keyword/context updates?

2. If evolving, for EACH neighbor provide:
   - Complete new keyword list (keep relevant existing, add new if meaningful)
   - Updated context (one sentence describing the entry's role)
   - Brief relationship description

Respond with JSON:
{{
    "should_evolve": true/false,
    "actions": ["update_keywords", "update_context"],
    "neighbor_updates": [
        {{
            "path": "neighbor_path",
            "new_keywords": ["kw1", "kw2", ...],
            "new_context": "Updated context sentence",
            "relationship": "How this relates to the new entry"
        }}
    ],
    "suggested_connections": []
}}

If should_evolve is false, neighbor_updates should be empty.
Total neighbors: {len(neighbors)}"""

    try:
        # Make provider-appropriate API call
        if provider == "anthropic":
            response = await client.messages.create(
                model=resolved_model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
        else:
            response = await client.chat.completions.create(
                model=resolved_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            raw = response.choices[0].message.content or "{}"

        parsed = _extract_first_json_object(raw)

        should_evolve = parsed.get("should_evolve", False)
        if not isinstance(should_evolve, bool):
            should_evolve = bool(should_evolve)

        actions = parsed.get("actions", [])
        if not isinstance(actions, list):
            actions = []

        suggested_connections = parsed.get("suggested_connections", [])
        if not isinstance(suggested_connections, list):
            suggested_connections = []

        # Parse neighbor updates
        neighbor_updates = []
        raw_updates = parsed.get("neighbor_updates", [])
        if isinstance(raw_updates, list):
            for i, n in enumerate(neighbors):
                if i < len(raw_updates) and isinstance(raw_updates[i], dict):
                    r = raw_updates[i]
                    new_keywords = r.get("new_keywords", [])
                    if not isinstance(new_keywords, list):
                        new_keywords = list(n.keywords)
                    else:
                        new_keywords = [
                            str(kw).strip().lower() for kw in new_keywords if kw
                        ]
                        if not new_keywords:
                            new_keywords = list(n.keywords)

                    new_context = r.get("new_context", "")
                    if not isinstance(new_context, str):
                        new_context = ""

                    relationship = r.get("relationship", "")
                    if not isinstance(relationship, str):
                        relationship = ""

                    neighbor_updates.append(
                        NeighborUpdate(
                            path=n.path,
                            new_keywords=new_keywords,
                            new_context=new_context.strip(),
                            relationship=relationship.strip(),
                        )
                    )
                elif should_evolve:
                    # If should_evolve but missing update, preserve existing
                    neighbor_updates.append(
                        NeighborUpdate(
                            path=n.path,
                            new_keywords=list(n.keywords),
                        )
                    )

        return EvolutionDecision(
            should_evolve=should_evolve,
            actions=actions,
            neighbor_updates=neighbor_updates,
            suggested_connections=suggested_connections,
        )

    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        log.warning("Failed to parse evolution decision: %s", e)
        return EvolutionDecision(should_evolve=False)
    except Exception as e:
        log.warning("LLM API error during evolution analysis: %s", e)
        return EvolutionDecision(should_evolve=False)


def _extract_first_json_object(text: str) -> dict:
    """Extract the first valid JSON object from text.

    Some models return extra content after the JSON object even with
    response_format=json_object. This function finds and parses just
    the first complete JSON object.

    Args:
        text: Raw text that may contain JSON plus extra content.

    Returns:
        Parsed dict from the first JSON object.

    Raises:
        json.JSONDecodeError: If no valid JSON object found.
    """
    # Try parsing the whole thing first (common case)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first { and try to match the closing }
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", text, 0)

    # Count braces to find matching close
    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                # Found complete object
                return json.loads(text[start : i + 1])

    # No complete object found
    raise json.JSONDecodeError("Incomplete JSON object", text, len(text))


@dataclass
class KeywordExtractionResult:
    """Result of LLM keyword extraction for an entry."""

    keywords: list[str]
    """Extracted keywords (3-6 typically)."""

    success: bool
    """Whether extraction succeeded."""

    error: str | None = None
    """Error message if extraction failed."""


async def extract_keywords_llm(
    content: str,
    title: str,
    model: str,
    min_keywords: int = 3,
    max_keywords: int = 6,
) -> KeywordExtractionResult:
    """Extract keywords from content using LLM.

    Args:
        content: The entry content to analyze.
        title: Entry title for context.
        model: Model ID to use (canonical or provider-specific).
        min_keywords: Minimum keywords to extract.
        max_keywords: Maximum keywords to extract.

    Returns:
        KeywordExtractionResult with extracted keywords or error info.

    Raises:
        LLMProviderError: If provider not configured.
    """
    client, provider = _get_client()
    resolved_model = resolve_model(model, provider)

    # Truncate content to avoid excessive tokens
    content_preview = content[:2000] if len(content) > 2000 else content

    prompt = f"""Extract {min_keywords}-{max_keywords} keywords from this content.
Keywords should be:
- Domain-specific concepts (not generic words)
- Key entities, technologies, or patterns mentioned
- Related concepts that aid discoverability

Title: {title}
Content:
{content_preview}

Return JSON: {{"keywords": ["keyword1", "keyword2", ...]}}"""

    try:
        # Make provider-appropriate API call
        if provider == "anthropic":
            response = await client.messages.create(
                model=resolved_model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            response_content = response.content[0].text
        else:
            response = await client.chat.completions.create(
                model=resolved_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=256,
            )
            response_content = response.choices[0].message.content or ""

        # Some models return extra data after JSON - extract first valid object
        result = _extract_first_json_object(response_content)
        keywords = result.get("keywords", [])

        # Validate and sanitize
        if not isinstance(keywords, list):
            keywords = []
        keywords = [str(kw).strip().lower() for kw in keywords if kw]
        keywords = [kw for kw in keywords if kw]  # Remove empty strings
        keywords = keywords[:max_keywords]

        if not keywords:
            return KeywordExtractionResult(
                keywords=[],
                success=False,
                error="LLM returned no keywords",
            )

        return KeywordExtractionResult(
            keywords=keywords,
            success=True,
        )

    except json.JSONDecodeError as e:
        log.warning("Failed to parse LLM keyword response: %s", e)
        return KeywordExtractionResult(
            keywords=[],
            success=False,
            error=f"JSON parse error: {e}",
        )
    except Exception as e:
        log.warning("LLM API error during keyword extraction: %s", e)
        return KeywordExtractionResult(
            keywords=[],
            success=False,
            error=str(e),
        )


@dataclass
class StrengthenResult:
    """Result of analyzing whether to strengthen a new entry based on neighbors.

    The "strengthen" action in A-Mem updates the NEW entry's keywords and links
    based on its relationship to existing neighbors. This complements the
    "update_neighbor" action which updates existing entries.
    """

    should_strengthen: bool
    """Whether the new entry should be updated."""

    new_keywords: list[str]
    """Updated keywords for the new entry (may include refinements based on neighbors)."""

    suggested_links: list[str]
    """Neighbor paths that should be explicitly linked to."""


async def analyze_for_strengthen(
    new_entry_content: str,
    new_entry_keywords: list[str],
    new_entry_title: str,
    neighbors: list[NeighborInfo],
    model: str,
) -> StrengthenResult:
    """Analyze if a new entry should be strengthened based on its neighbors.

    This implements the A-Mem "strengthen" action: when a new entry is linked
    to existing neighbors, the LLM analyzes whether the new entry's keywords
    should be refined based on the discovered relationships.

    Args:
        new_entry_content: Content of the newly added entry.
        new_entry_keywords: Current keywords of the new entry.
        new_entry_title: Title of the new entry.
        neighbors: List of neighbor entries that were linked to.
        model: OpenRouter model ID to use.

    Returns:
        StrengthenResult with updated keywords and suggested links.

    Raises:
        LLMConfigurationError: If API key not configured.
    """
    if not neighbors:
        return StrengthenResult(
            should_strengthen=False,
            new_keywords=new_entry_keywords,
            suggested_links=[],
        )

    client, provider = _get_client()
    resolved_model = resolve_model(model, provider)

    # Build neighbor descriptions
    neighbor_sections = []
    for i, n in enumerate(neighbors):
        kw_str = ", ".join(n.keywords) if n.keywords else "none"
        section = f"""NEIGHBOR {i + 1} (path: {n.path}, similarity: {n.score:.2f}):
Title: {n.title}
Keywords: {kw_str}
Content (first 200 chars): {n.content[:200]}"""
        neighbor_sections.append(section)

    new_kw_str = ", ".join(new_entry_keywords) if new_entry_keywords else "none"

    prompt = f"""A new KB entry was just added and linked to existing entries.
Analyze whether the NEW entry's keywords should be refined based on these connections.

NEW ENTRY (to potentially strengthen):
Title: {new_entry_title}
Current keywords: {new_kw_str}
Content (first 500 chars): {new_entry_content[:500]}

LINKED NEIGHBORS:
{chr(10).join(neighbor_sections)}

Decide whether to STRENGTHEN the new entry by:
1. Refining its keywords based on the discovered relationships
2. Suggesting which neighbors should be explicitly linked

Only strengthen if the neighbors reveal important context that should be captured.
Do NOT strengthen if the current keywords are already adequate.

The refined keyword list should:
- Keep relevant existing keywords
- Add keywords that capture important relationships to neighbors
- Typically contain 3-7 keywords total

Respond with JSON:
{{"should_strengthen": true/false, "new_keywords": ["kw1", "kw2", ...],
"suggested_links": ["path1", "path2"]}}"""

    try:
        # Make provider-appropriate API call
        if provider == "anthropic":
            response = await client.messages.create(
                model=resolved_model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text
        else:
            response = await client.chat.completions.create(
                model=resolved_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            content = response.choices[0].message.content or ""

        result = _extract_first_json_object(content)

        should_strengthen = result.get("should_strengthen", False)
        if not isinstance(should_strengthen, bool):
            should_strengthen = False

        new_keywords = result.get("new_keywords", [])
        if not isinstance(new_keywords, list):
            new_keywords = new_entry_keywords
        else:
            new_keywords = [str(kw).strip().lower() for kw in new_keywords if kw]
            if not new_keywords:
                new_keywords = new_entry_keywords

        suggested_links = result.get("suggested_links", [])
        if not isinstance(suggested_links, list):
            suggested_links = []
        else:
            suggested_links = [str(p).strip() for p in suggested_links if p]
            # Filter to only paths that are in our neighbors
            valid_paths = {n.path for n in neighbors}
            suggested_links = [p for p in suggested_links if p in valid_paths]

        return StrengthenResult(
            should_strengthen=should_strengthen,
            new_keywords=new_keywords,
            suggested_links=suggested_links,
        )

    except json.JSONDecodeError as e:
        log.warning("Failed to parse LLM strengthen response: %s", e)
        return StrengthenResult(
            should_strengthen=False,
            new_keywords=new_entry_keywords,
            suggested_links=[],
        )
    except Exception as e:
        log.warning("LLM API error during strengthen analysis: %s", e)
        return StrengthenResult(
            should_strengthen=False,
            new_keywords=new_entry_keywords,
            suggested_links=[],
        )
