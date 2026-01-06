---
title: CLI Reference
tags: [cli, reference, commands]
created: 2026-01-06
description: Complete reference for the mx command-line interface
---

# CLI Reference

The `mx` CLI provides token-efficient access to your knowledge base.

## Search Commands

### mx search

Search the knowledge base with hybrid keyword + semantic search.

```bash
mx search "query"                    # Hybrid search
mx search "docker" --tags=infra      # Filter by tag
mx search "api" --mode=semantic      # Semantic only
mx search "api" --mode=keyword       # Keyword only
mx search "query" --limit=20         # More results
mx search "query" --content          # Include full content
mx search "query" --strict           # No semantic fallback
mx search "query" --terse            # Paths only
mx search "query" --json             # JSON output
```

**Options:**
- `--tags, -t`: Filter by tags (comma-separated)
- `--mode`: Search mode (hybrid, keyword, semantic)
- `--limit, -n`: Maximum results (default: 10)
- `--content, -c`: Include full content in results
- `--strict`: Disable semantic fallback
- `--terse`: Output paths only
- `--json`: JSON output

### mx history

View and re-run past searches.

```bash
mx history                  # Show last 10 searches
mx history -n 20            # Show last 20
mx history --rerun 1        # Re-run most recent
mx history --clear          # Clear history
```

## Read Commands

### mx get

Read a knowledge base entry.

```bash
mx get tooling/my-entry.md            # Full entry
mx get tooling/my-entry.md --metadata # Metadata only
mx get tooling/my-entry.md --json     # JSON output
```

### mx list

List entries with optional filters.

```bash
mx list                        # All entries
mx list --tag=infrastructure   # Filter by tag
mx list --category=tooling     # Filter by category
mx list --limit=50             # More results
```

### mx tree

Display directory structure.

```bash
mx tree                    # Full tree
mx tree tooling            # Specific path
mx tree --depth=2          # Limit depth
```

## Write Commands

### mx add

Create a new entry.

```bash
mx add --title="My Entry" --tags="foo,bar" --category=tooling --content="..."
mx add --title="..." --tags="..." --category=... --file=content.md
cat content.md | mx add --title="..." --tags="..." --category=... --stdin
mx add --title="..." --tags="..." --template=troubleshooting
mx add --title="..." --tags="..." --dry-run  # Preview only
```

**Required:**
- `--title, -t`: Entry title
- `--tags`: Tags (comma-separated)
- `--category, -c`: Target directory (unless .kbcontext sets primary)

### mx update

Update an existing entry.

```bash
mx update path/entry.md --tags="new,tags"
mx update path/entry.md --content="New content"
mx update path/entry.md --content="Append this" --append
mx update path/entry.md --content="..." --append --timestamp
mx update path/entry.md --file=new-content.md
```

### mx patch

Surgical find-replace edits.

```bash
mx patch path/entry.md --old="old text" --new="new text"
mx patch path/entry.md --old="TODO" --new="DONE" --replace-all
mx patch path/entry.md --old="..." --new="..." --dry-run
```

### mx upsert

Create or append to entry by title.

```bash
mx upsert "Daily Log" --content="Session summary"
mx upsert "API Docs" --file=api.md --tags="api,docs"
mx upsert "Debug Log" --content="..." --no-create  # Error if not found
```

### mx delete

Delete an entry.

```bash
mx delete path/entry.md
mx delete path/entry.md --force  # Delete even with backlinks
```

## Analysis Commands

### mx health

Audit KB for problems.

```bash
mx health
mx health --json
```

Checks for:
- Orphaned entries (no backlinks)
- Broken links
- Stale content (>90 days)
- Missing frontmatter

### mx hubs

Show most connected entries.

```bash
mx hubs
mx hubs --limit=5
```

### mx suggest-links

Find semantically related entries.

```bash
mx suggest-links path/entry.md
mx suggest-links path/entry.md --limit=10
```

### mx tags

List all tags with counts.

```bash
mx tags
mx tags --min-count=3
```

## Browse Commands

### mx info

Show KB configuration and stats.

```bash
mx info
mx info --json
```

### mx whats-new

Show recently modified entries.

```bash
mx whats-new                      # Last 30 days
mx whats-new --days=7             # Last week
mx whats-new --project=myapp      # Filter by project
mx whats-new --limit=20           # More results
```

## Context Commands

### mx context

Manage project-specific KB context.

```bash
mx context                  # Show current context
mx context init             # Create .kbcontext file
mx context validate         # Validate context paths
```

## Publishing

### mx publish

Generate static HTML site for GitHub Pages or other static hosting.

**KB Source Resolution:**

The publish command resolves which KB to publish in this order:
1. `--kb-root ./path` - explicit CLI override
2. `project_kb` in `.kbcontext` - project-local KB
3. Requires `--global` flag to use `MEMEX_KB_ROOT`

This prevents accidentally publishing your organizational KB when you meant to publish project docs.

```bash
# Using .kbcontext (recommended)
mx publish -o docs                   # Uses project_kb from .kbcontext

# Explicit KB source
mx publish --kb-root ./kb -o docs    # Specify KB directory
mx publish --global -o docs          # Use global MEMEX_KB_ROOT

# Base URL for subdirectory hosting
mx publish -o docs --base-url /repo-name   # For username.github.io/repo-name
```

**When to use --base-url:**

If your site is hosted at a subdirectory (e.g., `username.github.io/my-repo`), you need `--base-url /my-repo` so all links work correctly. Without it, links will point to the root domain and 404.

**Recommended: Configure in .kbcontext:**

```yaml
# .kbcontext
project_kb: ./kb              # Project's documentation folder
publish_base_url: /my-repo    # Auto-applied to mx publish
```

Then just run `mx publish -o docs` - both settings are applied automatically.

**Options:**
- `--kb-root, -k`: KB source directory
- `--global`: Use global MEMEX_KB_ROOT
- `--output, -o`: Output directory (default: _site)
- `--base-url, -b`: URL prefix for links
- `--title, -t`: Site title
- `--index, -i`: Entry to use as landing page
- `--include-drafts`: Include draft entries
- `--include-archived`: Include archived entries

## Maintenance

### mx reindex

Rebuild search indices.

```bash
mx reindex
```

### mx prime

Output agent workflow context (for Claude Code hooks).

```bash
mx prime                    # Auto-detect mode
mx prime --full             # Force full output
mx prime --compact          # Force minimal output (for PreCompact hooks)
mx prime --project=myapp    # Include recent entries for project
```

## Global Options

All commands support:
- `--json`: JSON output
- `--help`: Show help

## See Also

- [[guides/installation|Installation Guide]]
- [[guides/ai-integration|AI Agent Integration]]
- [[reference/entry-format|Entry Format]]
