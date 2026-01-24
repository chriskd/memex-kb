---
title: Quick Start Guide
tags: [getting-started, tutorial, basics]
created: 2026-01-06
description: Get productive with memex in 5 minutes
---

# Quick Start Guide

Get productive with memex in 5 minutes.

## 1. Set Up Your Knowledge Base

```bash
# Create a directory for your KB
mkdir -p ~/kb

# Set environment variables (add to shell profile for persistence)
export MEMEX_KB_ROOT=~/kb
export MEMEX_INDEX_ROOT=~/.memex-indices

# Initialize the KB
mx init
```

## 2. Create Your First Entry

```bash
mx add \
  --title="Git Stash Workflow" \
  --tags="git,workflow,cli" \
  --category=tooling \
  --content="# Git Stash Workflow

Quick save work in progress:
\`\`\`bash
git stash push -m 'WIP: feature X'
git stash list
git stash pop
\`\`\`

Use \`git stash apply\` to keep the stash after applying."
```

This creates `tooling/git-stash-workflow.md` in your KB.

## 3. Search for It

```bash
# Keyword search
mx search "stash"

# With semantic search (if installed)
mx search "save work in progress"

# Filter by tag
mx search "git" --tags=workflow
```

## 4. Read It Back

```bash
# Full entry with content
mx get tooling/git-stash-workflow.md

# Metadata only
mx get tooling/git-stash-workflow.md --metadata
```

## 5. Check KB Health

```bash
mx health
```

Audits your KB for:
- Missing frontmatter
- Broken links
- Orphaned entries
- Index sync issues

## Essential Commands

| Command | Description |
|---------|-------------|
| `mx search "query"` | Search the KB |
| `mx get path/entry.md` | Read an entry |
| `mx add --title="..." --tags="..." --category=...` | Create entry |
| `mx tree` | Browse structure |
| `mx tags` | List all tags |
| `mx whats-new` | Recent changes |
| `mx health` | Audit KB |

## Next Steps

- [[reference/cli|CLI Reference]] - Full command documentation
- [[reference/entry-format|Entry Format]] - Frontmatter and linking
- [[guides/ai-integration|AI Agent Integration]] - Use with Claude Code
