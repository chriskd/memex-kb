---
title: Templates Reference
tags: [reference, templates]
created: 2026-01-07
description: Built-in templates for creating structured knowledge base entries
---

# Templates Reference

Templates provide scaffolding for common entry types, helping create well-structured documentation with consistent sections.

## Using Templates

```bash
# List available templates
mx templates

# Create entry with template
mx add --title="Fix login bug" --tags="auth,fix" --template=troubleshooting

# Preview without creating
mx add --title="..." --tags="..." --template=runbook --dry-run
```

## Template Priority

Templates are loaded in this order (later sources override earlier):

1. **Built-in** - Default templates shipped with memex
2. **User** - Custom templates from `~/.config/memex/templates/`
3. **Project** - Templates defined in `.kbcontext`

## Built-in Templates

### troubleshooting

**Purpose**: Problem/solution format for debugging issues

**When to use**: Documenting bugs, errors, and their fixes

**Suggested tags**: `troubleshooting`, `fix`

**Structure**:
```markdown
## Problem
[Describe the error, symptom, or unexpected behavior]

## Cause
[Root cause analysis - why did this happen?]

## Solution
[Step-by-step fix or workaround]

## Related
[Links to related entries, docs, or issues]
```

---

### project

**Purpose**: Project overview with setup and deployment

**When to use**: Creating documentation for a new project or service

**Suggested tags**: `project`, `setup`

**Structure**:
```markdown
## Overview
[Brief description of the project's purpose]

## Setup
[Installation steps]

## Configuration
[Key configuration options and environment variables]

## Development
[How to run locally, test, and contribute]

## Deployment
[Production deployment process]
```

---

### pattern

**Purpose**: Reusable pattern with use cases and examples

**When to use**: Documenting best practices, design patterns, or reusable solutions

**Suggested tags**: `pattern`, `best-practice`

**Structure**:
```markdown
## When to Use
[Scenarios where this pattern applies]

## Pattern
[Core concept or implementation approach]

## Example
[Code or usage example]

## Alternatives
[Other approaches and trade-offs]
```

---

### decision

**Purpose**: Architecture decision record (ADR) format

**When to use**: Recording significant technical or architectural decisions

**Suggested tags**: `adr`, `architecture`, `decision`

**Structure**:
```markdown
## Context
[Background and constraints that led to this decision]

## Decision
[The choice that was made]

## Rationale
[Why this option was chosen over alternatives]

## Consequences
[Expected outcomes, both positive and negative]

## Status
[Proposed / Accepted / Deprecated / Superseded]
```

---

### runbook

**Purpose**: Operational procedure with step-by-step instructions

**When to use**: Documenting deployment procedures, incident response, or maintenance tasks

**Suggested tags**: `runbook`, `operations`, `procedure`

**Structure**:
```markdown
## Prerequisites
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
```

---

### api

**Purpose**: API endpoint documentation

**When to use**: Documenting REST API endpoints, webhooks, or service interfaces

**Suggested tags**: `api`, `endpoint`, `reference`

**Structure**:
```markdown
## Endpoint
`METHOD /path/to/endpoint`

## Description
[What this endpoint does]

## Request
### Headers
| Header | Required | Description |
|--------|----------|-------------|
| Authorization | Yes | Bearer token |

### Body
[Request body schema]

## Response
### Success (200)
[Response schema]

### Errors
| Code | Description |
|------|-------------|
| 400 | Bad request |
| 401 | Unauthorized |

## Example
[curl or code example]
```

---

### meeting

**Purpose**: Meeting notes with decisions and action items

**When to use**: Recording meeting outcomes, decisions made, and follow-up tasks

**Suggested tags**: `meeting`, `notes`

**Structure**:
```markdown
## Attendees
- [Name 1]
- [Name 2]

## Agenda
1. [Topic 1]
2. [Topic 2]

## Discussion
### [Topic 1]
[Key points discussed]

## Decisions
- [ ] [Decision 1]
- [ ] [Decision 2]

## Action Items
- [ ] [Owner]: [Task] by [Date]

## Next Steps
[When to follow up, next meeting date]
```

---

### blank

**Purpose**: Empty template (just the title)

**When to use**: When you want full control over the structure

**Suggested tags**: None

Creates an entry with only the title heading.

## Custom Templates

### User Templates

Create custom templates in `~/.config/memex/templates/`:

**YAML format** (`~/.config/memex/templates/my-template.yaml`):
```yaml
name: my-template
description: My custom template for X
content: |
  ## Section 1
  [Content]

  ## Section 2
  [More content]
suggested_tags:
  - custom
  - tag
```

**Markdown format** (`~/.config/memex/templates/my-template.md`):
```markdown
<!-- Description of this template -->

## Section 1
[Content]

## Section 2
[More content]
```

### Project Templates

Define templates in `.kbcontext` for project-specific needs:

```yaml
# .kbcontext
project: my-project
primary_category: docs

templates:
  quick-fix:
    description: Quick bug fix documentation
    content: |
      ## Issue
      [What went wrong]

      ## Fix
      [What was changed]
    suggested_tags:
      - fix
      - quick

  sprint-notes:
    description: Sprint retrospective notes
    content: |
      ## Completed
      - [Item 1]

      ## Blocked
      - [Item 1]

      ## Next Sprint
      - [Item 1]
    suggested_tags:
      - sprint
      - notes
```

## See Also

- [[reference/cli|CLI Reference]] - Full `mx add` command options
- [[reference/entry-format|Entry Format]] - Frontmatter and content guidelines
