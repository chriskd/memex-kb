"""Entry templates for structured knowledge base entries.

Templates provide scaffolding for common entry types, helping agents and users
create well-structured documentation with consistent sections.

Template sources (in priority order):
1. Project-specific templates from .kbcontext
2. User-defined templates from ~/.config/memex/templates/
3. Built-in defaults

Example usage:
    mx add --title="Fix login bug" --template=troubleshooting
    mx templates list
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .context import get_kb_context

log = logging.getLogger(__name__)

# User templates directory
USER_TEMPLATES_DIR = Path.home() / ".config" / "memex" / "templates"


@dataclass
class Template:
    """A knowledge base entry template."""

    name: str
    """Template identifier (e.g., 'troubleshooting')."""

    description: str
    """Short description of when to use this template."""

    content: str
    """Markdown content with section headers."""

    suggested_tags: list[str] = field(default_factory=list)
    """Tags commonly associated with this template type."""

    source: str = "builtin"
    """Where this template came from: 'builtin', 'user', or 'project'."""


# =============================================================================
# Built-in Templates
# =============================================================================

BUILTIN_TEMPLATES: dict[str, Template] = {
    "troubleshooting": Template(
        name="troubleshooting",
        description="Problem/solution format for debugging issues",
        content="""## Problem

[Describe the error, symptom, or unexpected behavior]

## Cause

[Root cause analysis - why did this happen?]

## Solution

[Step-by-step fix or workaround]

## Related

[Links to related entries, docs, or issues]
""",
        suggested_tags=["troubleshooting", "fix"],
    ),
    "project": Template(
        name="project",
        description="Project overview with setup and deployment",
        content="""## Overview

[Brief description of the project's purpose]

## Setup

```bash
# Installation steps
```

## Configuration

[Key configuration options and environment variables]

## Development

[How to run locally, test, and contribute]

## Deployment

[Production deployment process]
""",
        suggested_tags=["project", "setup"],
    ),
    "pattern": Template(
        name="pattern",
        description="Reusable pattern with use cases and examples",
        content="""## When to Use

[Scenarios where this pattern applies]

## Pattern

[Core concept or implementation approach]

## Example

```
[Code or usage example]
```

## Alternatives

[Other approaches and trade-offs]
""",
        suggested_tags=["pattern", "best-practice"],
    ),
    "decision": Template(
        name="decision",
        description="Architecture decision record (ADR) format",
        content="""## Context

[Background and constraints that led to this decision]

## Decision

[The choice that was made]

## Rationale

[Why this option was chosen over alternatives]

## Consequences

[Expected outcomes, both positive and negative]

## Status

[Proposed / Accepted / Deprecated / Superseded]
""",
        suggested_tags=["adr", "architecture", "decision"],
    ),
    "runbook": Template(
        name="runbook",
        description="Operational procedure with step-by-step instructions",
        content="""## Prerequisites

[Required access, tools, or knowledge]

## Procedure

1. [Step 1]
2. [Step 2]
3. [Step 3]

## Verification

[How to confirm the procedure succeeded]

## Rollback

[How to undo if something goes wrong]

## Alerts

[Related monitoring alerts or escalation paths]
""",
        suggested_tags=["runbook", "operations", "procedure"],
    ),
    "api": Template(
        name="api",
        description="API endpoint documentation",
        content="""## Endpoint

`METHOD /path/to/endpoint`

## Description

[What this endpoint does]

## Request

### Headers

| Header | Required | Description |
|--------|----------|-------------|
| Authorization | Yes | Bearer token |

### Body

```json
{
  "field": "value"
}
```

## Response

### Success (200)

```json
{
  "result": "value"
}
```

### Errors

| Code | Description |
|------|-------------|
| 400 | Bad request |
| 401 | Unauthorized |

## Example

```bash
curl -X POST /api/endpoint -H "Authorization: Bearer $TOKEN" -d '{"field": "value"}'
```
""",
        suggested_tags=["api", "endpoint", "reference"],
    ),
    "meeting": Template(
        name="meeting",
        description="Meeting notes with decisions and action items",
        content="""## Attendees

- [Name 1]
- [Name 2]

## Agenda

1. [Topic 1]
2. [Topic 2]

## Discussion

### [Topic 1]

[Key points discussed]

### [Topic 2]

[Key points discussed]

## Decisions

- [ ] [Decision 1]
- [ ] [Decision 2]

## Action Items

- [ ] [Owner]: [Task] by [Date]
- [ ] [Owner]: [Task] by [Date]

## Next Steps

[When to follow up, next meeting date]
""",
        suggested_tags=["meeting", "notes"],
    ),
    "blank": Template(
        name="blank",
        description="Empty template (just the title)",
        content="",
        suggested_tags=[],
    ),
}


def _load_user_template(template_path: Path) -> Template | None:
    """Load a user-defined template from a YAML or Markdown file.

    YAML format:
        name: my-template
        description: My custom template
        content: |
          ## Section 1
          ...
        suggested_tags:
          - tag1
          - tag2

    Markdown format:
        The file content is used directly as the template content.
        Name is derived from filename, description is the first line.

    Args:
        template_path: Path to the template file.

    Returns:
        Template if valid, None if loading fails.
    """
    try:
        content = template_path.read_text(encoding="utf-8")
        name = template_path.stem  # Filename without extension

        if template_path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(content)
            if not isinstance(data, dict):
                log.warning("Invalid template %s: expected YAML dict", template_path)
                return None

            return Template(
                name=data.get("name", name),
                description=data.get("description", "User-defined template"),
                content=data.get("content", ""),
                suggested_tags=data.get("suggested_tags", []),
                source="user",
            )

        # Markdown file - use content directly
        # Extract first line as description if it's a comment
        lines = content.split("\n", 1)
        description = "User-defined template"
        if lines[0].startswith("<!-- ") and lines[0].endswith(" -->"):
            description = lines[0][5:-4].strip()
            content = lines[1] if len(lines) > 1 else ""

        return Template(
            name=name,
            description=description,
            content=content,
            suggested_tags=[],
            source="user",
        )

    except OSError as e:
        log.warning("Could not read template %s: %s", template_path, e)
        return None
    except yaml.YAMLError as e:
        log.warning("Invalid YAML in template %s: %s", template_path, e)
        return None


def _load_user_templates() -> dict[str, Template]:
    """Load all user-defined templates from ~/.config/memex/templates/.

    Returns:
        Dict mapping template name to Template.
    """
    templates: dict[str, Template] = {}

    if not USER_TEMPLATES_DIR.exists():
        return templates

    for path in USER_TEMPLATES_DIR.iterdir():
        if path.suffix in (".yaml", ".yml", ".md"):
            template = _load_user_template(path)
            if template:
                templates[template.name] = template

    return templates


def _load_project_templates() -> dict[str, Template]:
    """Load project-specific templates from .kbcontext.

    .kbcontext format:
        templates:
          quick-fix:
            description: Quick bug fix
            content: |
              ## Problem
              ...

    Returns:
        Dict mapping template name to Template.
    """
    templates: dict[str, Template] = {}

    context = get_kb_context()
    if not context or not context.source_file:
        return templates

    try:
        content = context.source_file.read_text(encoding="utf-8")
        data = yaml.safe_load(content)

        if not isinstance(data, dict):
            return templates

        template_data = data.get("templates", {})
        if not isinstance(template_data, dict):
            return templates

        for name, tpl in template_data.items():
            if isinstance(tpl, dict):
                templates[name] = Template(
                    name=name,
                    description=tpl.get("description", "Project template"),
                    content=tpl.get("content", ""),
                    suggested_tags=tpl.get("suggested_tags", []),
                    source="project",
                )

    except (OSError, yaml.YAMLError) as e:
        log.warning("Could not load project templates: %s", e)

    return templates


def get_all_templates() -> dict[str, Template]:
    """Get all available templates (project > user > builtin priority).

    Returns:
        Dict mapping template name to Template.
    """
    # Start with builtins
    templates = dict(BUILTIN_TEMPLATES)

    # Override with user templates
    templates.update(_load_user_templates())

    # Override with project templates (highest priority)
    templates.update(_load_project_templates())

    return templates


def get_template(name: str) -> Template | None:
    """Get a template by name.

    Args:
        name: Template name (e.g., 'troubleshooting').

    Returns:
        Template if found, None otherwise.
    """
    templates = get_all_templates()
    return templates.get(name)


def list_templates() -> list[Template]:
    """List all available templates.

    Returns:
        List of Template objects sorted by name.
    """
    templates = get_all_templates()
    return sorted(templates.values(), key=lambda t: (t.source != "project", t.source != "user", t.name))


def apply_template(template: Template, title: str) -> str:
    """Apply a template to create entry content.

    Args:
        template: The template to apply.
        title: Entry title (added as H1).

    Returns:
        Markdown content with title and template sections.
    """
    # Start with the title as H1
    content = f"# {title}\n\n"

    # Add template content if not blank
    if template.content.strip():
        content += template.content.strip() + "\n"

    return content
