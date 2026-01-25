---
title: CLI Reference
description: Complete reference for the mx command-line interface
tags:
  - cli
  - reference
  - commands
created: 2026-01-06T00:00:00
updated: 2026-01-25T23:32:00+00:00
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
edit_sources:
  - memex
git_branch: main
last_edited_by: chris
keywords:
  - hybrid search
  - semantic search
  - cli knowledge base
  - search scoring
  - query filtering
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
mx search "query" --scope=project    # Project KB only
mx search "query" --include-neighbors
mx search "query" --include-neighbors --neighbor-depth=2
mx search "query" --terse            # Paths only
mx search "query" --full-titles      # Show untruncated titles
mx search "query" --json             # JSON output
```

**Options:**
- `--tag, --tags`: Filter by tags (comma-separated)
- `--mode`: Search mode (hybrid, keyword, semantic)
- `--limit, -n`: Maximum results (default: 10)
- `--min-score`: Minimum score threshold (0.0-1.0)
- `--content`: Include full document content in results (replaces snippet)
- `--strict`: Disable semantic fallback
- `--terse`: Output paths only
- `--full-titles`: Show full titles without truncation
- `--scope`: Limit to KB scope (project or user)
- `--include-neighbors`: Include semantic links + typed relations
- `--neighbor-depth`: Neighbor traversal depth (default: 1)
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
- **Boost paths**: +0.12 if entry matches patterns in `boost_paths` in your `.kbconfig`

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
mx get --title "Python Tooling"       # Get by title (alt form)
```

**Options:**
- `--title`: Get entry by title instead of path (case-insensitive)
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
mx list --tags=infrastructure  # Filter by tag
mx list --category=tooling     # Filter by category
mx list --limit=50             # More results
mx list --full-titles          # Show untruncated titles
mx list --scope=project        # Project KB only
```

### mx tree

Display directory structure.

```bash
mx tree                    # Full tree
mx tree tooling            # Specific path
mx tree --depth=2          # Limit depth
mx tree --scope=project    # Project KB only
```

## Write Commands

### mx add

Create a new entry.

```bash
mx add --title="My Entry" --tags="foo,bar" --category=tooling --content="..."
mx add --title="..." --tags="..." --category=... --file=content.md
cat content.md | mx add --title="..." --tags="..." --category=... --stdin
```

**Required:**
- `--title`: Entry title
- `--tag, --tags`: Tags (comma-separated)
- `--category`: Target directory (required unless `primary` is set in `.kbconfig`)

**Common options:**
- `--scope`: Target KB scope (project or user)
- `--keywords`: Key concepts for semantic linking (comma-separated)
- `--semantic-links`: Set semantic links as JSON array

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

### mx quick-add

Quickly add content with auto-generated metadata.

```bash
mx quick-add --stdin              # Paste content, auto-generate all
mx quick-add -f notes.md          # From file with auto metadata
mx quick-add --content "..." --confirm  # Auto-confirm creation
echo "..." | mx quick-add --stdin --json
```

### mx ingest

Ingest a markdown file into the KB, adding frontmatter if missing.

```bash
mx ingest notes.md                          # Auto-detect title/tags
mx ingest draft.md --title="My Entry"       # Override title
mx ingest doc.md --tags="api,docs"          # Set tags
mx ingest doc.md --directory="guides"       # Place in guides/
mx ingest doc.md --scope=project            # Project KB only
mx ingest doc.md --dry-run                  # Preview changes
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

### mx relations-lint

Warn on unknown or inconsistent typed relation types (non-blocking by default).

```bash
mx relations-lint
mx relations-lint --json
mx relations-lint --strict   # Exit non-zero if issues found
```

Use `mx relations-lint` to align relation types with the canonical taxonomy.

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

Alias: `mx config`

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

Initialize a knowledge base (project or user scope).

```bash
mx init                    # Project KB in ./kb/
mx init --user             # User KB in ~/.memex/kb/
mx init --path docs/kb     # Custom location
mx init --force            # Reinitialize existing
mx init --json             # JSON output
```

**Options:**
- `--path, -p`: Custom location for local KB (default: kb/)
- `--user, -u`: Create user KB at ~/.memex/kb/
- `--force, -f`: Reinitialize existing KB
- `--json`: JSON output

**What gets created:**
- Project: `kb/README.md` and `.kbconfig` at the project root
- User: `~/.memex/kb/README.md` and `~/.memex/kb/.kbconfig`

**When to use:**
- Starting a new project that needs project-specific documentation
- Creating a knowledge base that should be versioned with the code
- Setting up for GitHub Pages publishing

### mx context

Show or validate project KB configuration.

```bash
mx context                  # Show current config
mx context show             # Same as above
mx context validate         # Validate config paths
mx context validate --json  # JSON output
```

**Note:** `.kbconfig` is created by `mx init`.

## Automation Commands

### mx batch

Execute multiple KB operations in a single invocation.

```bash
mx batch << 'EOF'
add --title='Note 1' --tags='tag1' --category=tooling --content='Content'
search 'api'
EOF
```

Reads commands from stdin (or `--file`) and returns JSON results.

## Agent Memory Commands

### mx memory

Agent memory for AI coding assistants.

```bash
mx memory              # Show memory status
mx memory init         # Enable memory for this project
mx memory add "note"   # Add a manual memory note
mx memory inject       # Preview injected context
mx memory capture      # Manually trigger capture
```

### mx evolve

Process queued memory evolution.

```bash
mx evolve              # Process all queued items
mx evolve --status     # Show queue statistics
mx evolve --dry-run    # Preview what would be evolved
mx evolve --limit 10   # Process up to 10 items
mx evolve --clear      # Clear all queued items
```

## Templates

List or show available entry templates.

```bash
mx templates
mx templates show troubleshooting
mx templates --json
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

### mx summarize

Generate descriptions for entries missing them.

```bash
mx summarize --dry-run         # Preview what would be generated
mx summarize                   # Generate and write descriptions
mx summarize --limit 5         # Process only 5 entries
mx summarize --json            # Output as JSON
```

### mx reindex

Rebuild search indices.

```bash
mx reindex
mx reindex --json
```

### mx a-mem-init

Initialize A-Mem structures (semantic links + evolution queue) for existing KB entries.

```bash
mx a-mem-init                        # Run all phases
mx a-mem-init --dry-run              # Preview without executing
mx a-mem-init --missing-keywords=llm # Use LLM to extract missing keywords
mx a-mem-init --missing-keywords=skip # Skip entries without keywords
mx a-mem-init --scope=project        # Only process project KB
mx a-mem-init --limit=10             # Process first 10 entries only
mx a-mem-init --json                 # JSON output
```

**Phases:**
1. **Inventory & Validation** - Lists entries, validates keywords
2. **Keyword Extraction** - Uses LLM to extract keywords (when mode=llm)
3. **Semantic Linking** - Creates bidirectional links chronologically
4. **Evolution Queue** - Queues items for `mx evolve`

**Missing Keywords Modes:**
- `error` - Stop and list missing keywords (default if `amem_strict: true`)
- `skip` - Skip entries without keywords (default otherwise)
- `llm` - Extract keywords using LLM (requires `OPENROUTER_API_KEY`)

**Notes:**
- Processes entries chronologically (oldest first) to simulate incremental A-Mem
- Only links to entries created BEFORE the current entry
- Idempotent: safe to re-run without creating duplicates

See [[a-mem-parity/a-mem-init-command-specification.md]] for full specification.

### mx prime

Output agent workflow context (for hooks).

```bash
mx prime                    # Auto-detect mode
mx prime --full             # Force full output
mx prime --mcp              # Force minimal output
mx prime --json             # JSON output
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
  "version": "0.2.0",
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

Global flags:
- `--version`: Show the version and exit
- `--json-errors`: Output errors as JSON (for programmatic use)
- `-q, --quiet`: Suppress warnings, show only errors and essential output
- `--help`: Show help

Per-command (when available):
- `--json`: JSON output

## See Also

- [[guides/installation|Installation Guide]]
- [[guides/ai-integration|AI Agent Integration]]
- [[reference/entry-format|Entry Format]]
- [[memex/state-diagrams|State diagrams for MX flows]]
