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

Generate static HTML site.

```bash
mx publish                           # Build to _site/
mx publish -o docs                   # Build to docs/
mx publish --base-url /my-kb         # For subdirectory hosting
mx publish --include-drafts          # Include draft entries
```

## Maintenance

### mx reindex

Rebuild search indices.

```bash
mx reindex
```

### mx prime

Output agent workflow context (for hooks).

```bash
mx prime                    # Auto-detect mode
mx prime --full             # Force full output
mx prime --mcp              # Force minimal output
```

## Global Options

All commands support:
- `--json`: JSON output
- `--help`: Show help

## See Also

- [[guides/installation|Installation Guide]]
- [[guides/ai-integration|AI Agent Integration]]
- [[reference/entry-format|Entry Format]]
