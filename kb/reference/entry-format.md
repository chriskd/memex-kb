---
title: Entry Format Reference
tags: [reference, frontmatter, markdown, links]
created: 2026-01-06
description: Markdown entry format with YAML frontmatter and bidirectional links
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

# Create entry from template
mx add --title="..." --tags="..." --category=... --template=troubleshooting
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
