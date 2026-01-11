---
name: kb-add
description: Add a new entry to the knowledge base
allowed-tools:
  - mcp__memex__add
  - mcp__memex__search
  - mcp__memex__list
  - AskUserQuestion
argument-hint: "[title]"
---

Interactive workflow to add a new knowledge base entry.

## Workflow

### 1. Check for Duplicates
If a title is provided, search for similar entries first:
```
mcp__memex__search(query=<title>)
```
If duplicates found, ask user if they want to continue or update existing.

### 2. Gather Entry Details

Use `AskUserQuestion` to collect:

**Title** (if not provided as argument):
- Should be descriptive and searchable
- Example: "Kubernetes Pod Networking Troubleshooting"

**Category** (required):
- `infrastructure` - servers, networking, storage, cloud resources
- `devops` - CI/CD, monitoring, deployment tools
- `development` - coding practices, languages, frameworks
- `troubleshooting` - problem solutions, debugging guides
- `architecture` - system design, patterns, decisions
- `patterns` - reusable solutions, templates, conventions

**Content** (required):
- Markdown formatted
- Include code examples where relevant
- Document the "why" not just the "how"

**Tags** (optional):
- Comma-separated list
- Helps with search and discovery

### 3. Create the Entry
```
mcp__memex__add(
  title=<title>,
  category=<category>,
  content=<content>,
  tags=<tags>
)
```

### 4. Suggest Related Links
After creation, search for related entries and suggest linking them.

## Example Session

```
User: /kb-add Nginx reverse proxy setup

Claude: Let me check for similar entries...
[searches]
No duplicates found.

Which category?
- infrastructure, devops, development, troubleshooting, architecture, patterns

User: infrastructure

Claude: Please provide the content (markdown):

User: [provides content]

Claude: Any tags? (comma-separated, or skip)

User: nginx, proxy, web

Claude: Created: infrastructure/nginx-reverse-proxy-setup.md
```
