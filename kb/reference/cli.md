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
mx search "query" --min-score=0.5    # Only confident results
mx search "query" --content          # Include full content
mx search "query" --strict           # No semantic fallback
mx search "query" --terse            # Paths only
mx search "query" --full-titles      # Show untruncated titles
mx search "query" --json             # JSON output
```

**Options:**
- `--tags, -t`: Filter by tags (comma-separated)
- `--mode`: Search mode (hybrid, keyword, semantic)
- `--limit, -n`: Maximum results (default: 10)
- `--min-score`: Minimum score threshold (0.0-1.0)
- `--content, -c`: Include full document content in results (replaces snippet)
- `--strict`: Disable semantic fallback
- `--terse`: Output paths only
- `--full-titles`: Show full titles without truncation
- `--json`: JSON output

**Content Flag Behavior:**

When `--content` is used:
- **JSON output**: Returns `content` field with full document text instead of `snippet`
- **Table output**: Shows the standard results table followed by a content section displaying full text for each result

Without `--content`, only a brief snippet is shown (default behavior).

**Notes:**
- Query cannot be empty. An error is returned for empty or whitespace-only queries.

#### Understanding Search Scores

All search scores are normalized to **0.0-1.0** (higher = better match).

| Score Range | Confidence | Interpretation |
|-------------|------------|----------------|
| >= 0.7 | High | Strong keyword or semantic match. Trust this result. |
| 0.4 - 0.7 | Moderate | Partial match. Worth reviewing but may be tangential. |
| < 0.4 | Weak | Tangential relevance only. May be noise. |

**How scores are calculated:**

- **Hybrid mode** (default): Combines keyword and semantic search using Reciprocal Rank Fusion (RRF). Results appearing in both rankings score higher.
- **Keyword mode**: BM25 text matching. Exact term matches matter most.
- **Semantic mode**: Cosine similarity of embeddings. Conceptual meaning matters more than exact words.

**Score adjustments:**

After initial ranking, scores receive context-aware boosts:
- **Tag match**: +0.05 per matching tag (e.g., searching "python" boosts entries tagged "python")
- **Project context**: +0.15 if entry's source_project matches your current project
- **KB path context**: +0.12 if entry matches patterns in your `.kbcontext` file

Scores are re-normalized to 0-1 after boosts, so the top result is always 1.0.

**Recommended thresholds:**

```bash
# High-confidence results only (for automated workflows)
mx search "deployment" --min-score=0.7

# Moderate confidence (good default for exploration)
mx search "deployment" --min-score=0.4

# All results (when you want broad coverage)
mx search "deployment"
```

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
mx get tooling/my-entry.md            # Full entry by path
mx get tooling/my-entry.md --metadata # Metadata only
mx get tooling/my-entry.md --json     # JSON output
mx get --title="Docker Guide"         # Get by title
mx get -t "Python Tooling"            # Short form
```

**Options:**
- `--title, -t`: Get entry by title instead of path (case-insensitive)
- `--metadata, -m`: Show only metadata
- `--json`: JSON output

**Title Lookup Behavior:**
- If one match found: returns that entry
- If multiple matches: shows error with candidate paths
- If no match: shows error with similar title suggestions

### mx list

List entries with optional filters.

```bash
mx list                        # All entries
mx list --tag=infrastructure   # Filter by tag
mx list --category=tooling     # Filter by category
mx list --limit=50             # More results
mx list --full-titles          # Show untruncated titles
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

### mx replace

Replace content or metadata in an existing entry.

```bash
mx replace path/entry.md --tags="new,tags"
mx replace path/entry.md --content="New content"
mx replace path/entry.md --file=new-content.md
```

Note: For appending content, use `mx append`. For surgical edits, use `mx patch`.

### mx patch

Surgical find-replace edits.

```bash
mx patch path/entry.md --find="old text" --replace="new text"
mx patch path/entry.md --find="TODO" --replace="DONE" --replace-all
mx patch path/entry.md --find="..." --replace="..." --dry-run
```

**Intent Detection:** If you use flags that suggest a different command (e.g., `--content` without `--find`), the CLI will suggest the correct command:

```bash
$ mx patch entry.md --content='new stuff'
Error: --find is required for find-replace operations.

Did you mean:
  - To append content:  mx append 'Title' --content='...'
  - To replace text:    mx patch entry.md --find 'x' --replace 'y'
  - To overwrite entry: mx replace entry.md --content='...'
```

### mx append

Append content to existing entry by title, or create new if not found.

```bash
mx append "Daily Log" --content="Session summary"
mx append "API Docs" --file=api.md --tags="api,docs"
mx append "Debug Log" --content="..." --no-create  # Error if not found
cat notes.md | mx append "Meeting Notes" --stdin --tags="meetings"
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
mx whats-new --scope=project      # Project KB only
mx whats-new --limit=20           # More results
```

## Project Setup Commands

### mx init

Initialize a local project KB. Creates a `kb/` directory for project-specific knowledge that stays with the project and is GitHub Pages compatible.

```bash
mx init                    # Create kb/ in current directory
mx init --path docs/kb     # Custom location
mx init --force            # Reinitialize existing
mx init --json             # JSON output
```

**Options:**
- `--path, -p`: Custom location for local KB (default: kb/)
- `--force, -f`: Reinitialize existing local KB
- `--json`: JSON output

**What gets created:**
- `kb/README.md` - Documentation for the local KB
- `kb/.kbconfig` - Configuration file marking this as a memex KB

**When to use:**
- Starting a new project that needs project-specific documentation
- Creating a knowledge base that should be versioned with the code
- Setting up for GitHub Pages publishing

### mx context

Manage project-specific KB context (for routing to global KB).

```bash
mx context                  # Show current context
mx context init             # Create .kbcontext file
mx context init --json      # JSON output
mx context validate         # Validate context paths
mx context validate --json  # JSON output
```

**Note:** `mx init` creates a local `kb/` directory. `mx context init` creates a `.kbcontext` file that routes entries to the global KB.

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
mx reindex --json
```

### mx prime

Output agent workflow context (for hooks).

```bash
mx prime                    # Auto-detect mode
mx prime --full             # Force full output
mx prime --mcp              # Force minimal output
```

### mx schema

Output CLI schema with agent-friendly metadata for introspection.

```bash
mx schema                    # Full schema (JSON)
mx schema --command=patch    # Schema for specific command only
mx schema -c search          # Short form
mx schema --compact          # Minimal output (commands and options only)
```

**Schema includes:**
- All commands with their arguments and options
- Related commands (cross-references)
- Common mistakes and how to avoid them
- Example invocations
- Recommended workflows

**Use cases:**
- Agent introspection to understand available commands
- Proactive error avoidance via common_mistakes field
- Discovering related commands for a task
- Understanding option types and defaults

**Example output structure:**
```json
{
  "version": "0.1.0",
  "commands": {
    "patch": {
      "description": "Apply surgical find-replace edits",
      "options": [...],
      "related": ["update", "append"],
      "common_mistakes": {
        "--find without --replace": "Both are required"
      },
      "examples": ["mx patch path.md --find \"old\" --replace \"new\""]
    }
  },
  "workflows": {...}
}
```

## Global Options

All commands support:
- `--json`: JSON output
- `--help`: Show help

## See Also

- [[guides/installation|Installation Guide]]
- [[guides/ai-integration|AI Agent Integration]]
- [[reference/entry-format|Entry Format]]
