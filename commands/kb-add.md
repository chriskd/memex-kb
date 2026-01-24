---
name: kb-add
description: Add a new entry to the knowledge base
allowed-tools:
  - Bash
  - AskUserQuestion
argument-hint: "[title]"
---

Interactive workflow to add a new knowledge base entry.

## Workflow

### 1. Check for Duplicates
If a title is provided, search for similar entries first:
```bash
mx search "<title>"
```
If duplicates found, ask user if they want to continue or update existing.

### 2. Determine Scope

Ask user which KB to use (if both project and user KBs exist):

| Scope | Use For |
|-------|---------|
| `project` | Team knowledge, infra docs, shared patterns, API docs |
| `user` | Personal notes, experiments, drafts, individual workflow tips |

If only one KB exists, use it automatically.

### 3. Gather Entry Details

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

### 4. Create the Entry

```bash
mx add \
  --title="<title>" \
  --tags="<tags>" \
  --category="<category>" \
  --scope=<project|user> \
  --content="<content>"
```

Or with file input:
```bash
mx add \
  --title="<title>" \
  --tags="<tags>" \
  --category="<category>" \
  --scope=<project|user> \
  --file=<path>
```

### 5. Suggest Related Links
After creation, search for related entries and suggest linking them:
```bash
mx suggest-links <created-path>
```

## Example Session

```
User: /kb-add Nginx reverse proxy setup

Claude: Let me check for similar entries...
$ mx search "Nginx reverse proxy"
No duplicates found.

Which scope should I use?
- project (shared with team)
- user (personal notes)

User: project

Which category?
- infrastructure, devops, development, troubleshooting, architecture, patterns

User: infrastructure

Claude: Please provide the content (markdown):

User: [provides content]

Claude: Any tags? (comma-separated, or skip)

User: nginx, proxy, web

Claude:
$ mx add --title="Nginx reverse proxy setup" --tags="nginx,proxy,web" \
    --category="infrastructure" --scope=project --content="..."

Created: @project/infrastructure/nginx-reverse-proxy-setup.md
```

## Scope Selection Heuristics

When the user doesn't specify, suggest based on content type:

**Suggest `project` scope for:**
- Infrastructure documentation
- API references
- Team workflows
- Debugging guides for shared systems
- Architecture decisions

**Suggest `user` scope for:**
- Personal notes or drafts
- Experimental findings
- Individual productivity tips
- Work-in-progress content
- Content mentioning "my", "personal", "draft"
