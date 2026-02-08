---
title: Quick Start Guide
tags: [getting-started, tutorial, basics]
created: 2026-01-06
description: Get productive with memex in 5 minutes
---

# Quick Start Guide

Get productive with memex in 5 minutes.

## 5-minute flow (agents)

Run these in order; each step confirms the previous one worked.

```bash
mx init                                 # Creates KB + .kbconfig
mx add --title="First Entry" --tags=docs --category=guides --content="Hello KB"  # Confirms write path
mx list --limit=5                       # Confirms entry exists (path printed)
mx get guides/first-entry.md            # Confirms read path
mx health                               # Confirms basic KB health (warnings are OK early)
```

Optional check: `mx search "Hello KB"` to verify indexing.
If `mx search` fails, run `mx doctor` for an install hint. If you see `No module named 'whoosh'`, install
keyword search deps with `pip install whoosh-reloaded` (or reinstall `memex-kb`).

Optional: set a default category (so you can omit `--category` in `mx add`):

```yaml
# .kbconfig (project root)
primary: guides
```

## 1. Set Up Your Knowledge Base

Choose a KB scope:

```bash
# Project KB (shared with your repo)
mx init

# User KB (personal, available everywhere)
mx init --user
```

Optional overrides:

```bash
# Point to an existing user KB in a custom location
export MEMEX_USER_KB_ROOT=~/kb

# Store indices outside the KB directory
export MEMEX_INDEX_ROOT=~/.memex-indices
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

**Note:** If `--category` is omitted and no `.kbconfig` `primary` exists, `mx add` defaults to the KB root (`.`) and prints a warning.

## 3. Search for It

```bash
# Keyword search
mx search "stash"

# With semantic search (if installed)
mx search "save work in progress"

# Filter by tag
mx search "git" --tags=workflow

# Limit to a specific KB
mx search "stash" --scope=project
mx search "stash" --scope=user
```

When both KBs exist, results are prefixed with `@project/...` and `@user/...`.

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

Orphans are entries with no incoming links yet. This is normal for new KBs; add links or use `mx suggest-links` when you have more entries.
Quick fixes:
- Create a simple index entry (e.g., `guides/index.md`) and link to key entries with `[[wikilinks]]`.
- Add at least one link from an existing entry to each orphan (typed relations count too).

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
