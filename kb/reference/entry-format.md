---
title: Entry Format Reference
description: Markdown entry format with YAML frontmatter and bidirectional links
tags:
  - reference
  - frontmatter
  - markdown
  - links
created: 2026-01-06T00:00:00
updated: 2026-01-25T23:30:00+00:00
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
edit_sources:
  - memex
git_branch: main
last_edited_by: chris
keywords:
  - memex
  - markdown
  - yaml frontmatter
  - bidirectional links
  - knowledge management
---

# Entry Format Reference

Memex entries are Markdown files with YAML frontmatter.

## Basic Structure

```markdown
---
title: Entry Title
tags: [tag1, tag2, tag3]
created: 2025-01-15
description: One-line summary for search results
---

# Entry Title

Your content here with [[bidirectional links]] to other entries.
```

## Frontmatter Fields

### Required

| Field | Type | Description |
|-------|------|-------------|
| `title` | string | Entry title (used for display and title matching) |
| `tags` | list | Tags for filtering and discovery |

### Optional

| Field | Type | Description |
|-------|------|-------------|
| `created` | date | Creation date (YYYY-MM-DD) |
| `updated` | date | Last update date |
| `description` | string | One-line summary for search results |
| `aliases` | list | Alternative titles for title matching |
| `draft` | boolean | Exclude from `mx publish` unless `--include-drafts` |
| `archived` | boolean | Exclude from search and publish |
| `source_project` | string | Project that created this entry |
| `semantic_links` | list | Semantic links (auto-managed unless set via CLI) |
| `relations` | list | Typed relations to other entries |

### Semantic Links (optional)

Semantic links are usually computed by Memex when semantic linking is enabled.
They can also be set explicitly via CLI (`mx add --semantic-links` or `mx replace --semantic-links`).

```yaml
semantic_links:
  - path: reference/cli.md
    score: 0.82
    reason: embedding_similarity
  - path: guides/quick-start.md
    score: 0.74
    reason: shared_tags
```

### Typed Relations (optional)

Typed relations are manual, directed links that carry a relation type.
Add them by editing frontmatter directly (or via `mx patch` / `mx replace`).

```yaml
relations:
  - path: reference/cli.md
    type: documents
  - path: guides/installation.md
    type: depends_on
```

#### Canonical relation types

Use **snake_case** and prefer these canonical types. If none fit, use `related` as the fallback.

| Type | Meaning |
|------|---------|
| `depends_on` | A requires B to function or make progress |
| `implements` | A implements the spec, API, or plan described by B |
| `extends` | A builds on or extends B with additional behavior |
| `documents` | A documents or explains B (guides, readmes, specs) |
| `references` | A cites B for supporting detail or context |
| `blocks` | A blocks B from proceeding (directional) |
| `related` | A is generally related to B when no stronger type fits |

Use `mx relations-lint` to audit unknown or inconsistent relation types.

## Bidirectional Links

Link to other entries using wiki-style syntax:

```markdown
See [[guides/installation]] for setup instructions.

Or use display text: [[guides/installation|the installation guide]].
```

### Link Resolution

Links resolve in this order:
1. Exact path match (with or without `.md`)
2. Title match (case-insensitive)
3. Filename match (for short links)

### Examples

```markdown
[[guides/installation]]           # Path link
[[Installation Guide]]            # Title link
[[installation]]                  # Filename link
[[guides/installation|Setup]]     # With display text
```

## Tags

Tags enable filtering and discovery:

```yaml
tags: [infrastructure, deployment, docker]
```

### Best Practices

- Use lowercase, hyphenated tags
- Check existing tags with `mx tags`
- Be consistent across entries
- Use 2-5 tags per entry

### Common Tag Patterns

| Pattern | Example |
|---------|---------|
| Technology | `docker`, `kubernetes`, `python` |
| Category | `troubleshooting`, `patterns`, `reference` |
| Project | `myapp`, `api-gateway` |
| Status | `draft`, `needs-review` |

## Content Guidelines

### Headings

Use heading hierarchy for structure:

```markdown
# Main Title (matches frontmatter title)

## Section

### Subsection
```

### Code Blocks

Use fenced code blocks with language hints:

```markdown
\`\`\`bash
mx search "query"
\`\`\`

\`\`\`python
def example():
    pass
\`\`\`
```

### Lists

Use consistent list formatting:

```markdown
- Item one
- Item two
  - Nested item

1. First step
2. Second step
```

## File Organization

### Categories

Organize entries into directories:

```
kb/
├── guides/           # How-to guides
├── reference/        # Reference documentation
├── patterns/         # Reusable patterns
├── troubleshooting/  # Problem-solution pairs
├── projects/         # Project-specific docs
└── infrastructure/   # Infrastructure docs
```

### Naming

- Use lowercase with hyphens: `my-entry-title.md`
- Keep names descriptive but concise
- Avoid special characters

## Templates

Use templates for consistent structure:

```bash
# List available templates
mx templates

# Show a template and copy it into a new entry
mx templates show troubleshooting
```

### Built-in Templates

| Template | Use case |
|----------|----------|
| `troubleshooting` | Problem/solution documentation |
| `pattern` | Reusable code/design patterns |
| `runbook` | Operational procedures |
| `decision` | Architecture decision records |
| `api` | API endpoint documentation |

## See Also

- [[reference/cli|CLI Reference]]
- [[guides/quick-start|Quick Start Guide]]
