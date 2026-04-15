---
title: CLI Reference
description: Complete reference for the mx command-line interface
tags:
  - cli
  - reference
  - commands
created: 2026-01-06T00:00:00
updated: 2026-04-15T00:00:00+00:00
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
edit_sources:
  - memex
git_branch: main
last_edited_by: chris
---

# CLI Reference

The `mx` CLI provides token-efficient access to your knowledge base.

## Search Commands

### mx help

Show top-level help or help for a specific command.

```bash
mx help
mx help search
mx help publish
```

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
- `--include-neighbors`: Include semantic links + typed relations + wikilinks
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

### mx doctor

Inspect installation status and optional dependency availability, or audit/fix
frontmatter timestamps from filesystem metadata.

```bash
mx doctor                                   # Deps + KB detection
mx doctor --json                            # Same as JSON
mx doctor --timestamps                      # Report missing/invalid/stale timestamps
mx doctor --timestamps --fix                # Apply timestamp fixes
mx doctor --timestamps --fix --dry-run      # Preview fixes without writing
mx doctor --timestamps --force              # Recompute even valid timestamps
mx doctor --timestamps --scope=project      # Audit project KB only
mx doctor --timestamps --limit=25 --json    # Bound scan and emit per-file JSON
```

**Timestamp options:**
- `--timestamps`: Audit `created` / `updated` frontmatter fields against filesystem metadata
- `--fix`: Write proposed timestamp updates in-place
- `--dry-run`: Preview `--fix` results without writing
- `--force`: Overwrite valid timestamps from filesystem metadata
- `--scope`: Limit timestamp audit to KB scope (project or user)
- `--limit`: Maximum entries to check during timestamp audit
- `--json`: Include per-file before/after values and timestamp source

## Read Commands

### mx relations

Query the unified relations graph across wikilinks, semantic links, and typed relations.

```bash
mx relations path/entry.md
mx relations path/entry.md --depth=2
mx relations path/entry.md --direction=outgoing
mx relations path/entry.md --origin=relations --type=documents
mx relations path/entry.md --graph --json
```

### mx get

Read a knowledge base entry.

```bash
mx get tooling/my-entry.md            # Full entry by path
mx get tooling/my-entry.md --metadata # Metadata only
mx get tooling/my-entry.md --json     # JSON output
mx get @project/tooling/my-entry.md   # Explicit project scope
mx get @user/tooling/my-entry.md      # Explicit user scope
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

### mx categories

List available top-level categories (directories) in the KB.

```bash
mx categories
mx categories --scope=project
mx categories --json
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

### mx relations-add

Add typed relations without replacing the full frontmatter block.

```bash
mx relations-add path/entry.md --relation "reference/cli.md=documents"
mx relations-add path/entry.md --relations '[{"path":"guides/quick-start.md","type":"depends_on"}]'
```

### mx relations-remove

Remove typed relations without replacing the full frontmatter block.

```bash
mx relations-remove path/entry.md --relation "reference/cli.md=documents"
mx relations-remove path/entry.md --relations '[{"path":"guides/quick-start.md","type":"depends_on"}]'
```

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

**Category behavior:**
- `--category`: Optional target directory.
- If omitted and `.kbconfig` sets `primary`, `primary` is used.
- If omitted and no `primary` is set, `mx add` writes to KB root (`.`) and warns by default.
  Tip: Use `mx categories` to discover available categories.
- Silence the warning with:
  - `.kbconfig`: `warn_on_implicit_category: false`

**Common options:**
- `--scope`: Target KB scope (project or user)
- `--semantic-links`: Set semantic links as JSON array

### mx replace

Replace content or metadata in an existing entry.

```bash
mx replace path/entry.md --tags="new,tags"
mx replace path/entry.md --content="New content"
mx replace path/entry.md --file=new-content.md
mx replace @user/path/entry.md --content="New content"  # Explicit scope
```

Note: For appending content, use `mx append`. For surgical edits, use `mx patch`.

### mx patch

Surgical find-replace edits.

```bash
mx patch path/entry.md --find="old text" --replace="new text"
mx patch path/entry.md --find="TODO" --replace="DONE" --replace-all
mx patch path/entry.md --find="..." --replace="..." --dry-run
mx patch @project/path/entry.md --find="old" --replace="new"  # Explicit scope
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
mx delete @user/path/entry.md    # Explicit scope
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

Orphans are entries with no incoming links (no `[[wikilinks]]` or typed relations pointing at them).
To reduce false alarms in brand-new KBs, orphan findings are suppressed until the KB has at least 5 entries.

Quick fixes:
- Add an index/hub entry that links to everything important.
- Add at least one link from an existing entry to each orphan.
- Use `mx suggest-links path/to/entry.md` to find candidates to connect.

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

Recency uses frontmatter when available, but falls back to filesystem timestamps when
timestamp fields are missing. If `updated` is present but older than the file `mtime`,
recent views use the newer filesystem time and `mx doctor --timestamps --fix` can sync
frontmatter explicitly.

```bash
mx whats-new                      # Last 30 days
mx whats-new --days=7             # Last week
mx whats-new --scope=project      # Project KB only
mx whats-new --limit=20           # More results
```

## Project Setup Commands

### mx onboard

Guided setup check for first-time users and agents.

```bash
mx onboard
mx onboard --init --yes
mx onboard --init --yes --cwd-only
mx onboard --init --user --yes
mx onboard --init --yes --json
```

**Options:**
- `--init`: Initialize a KB if none is configured
- `--user`: Create a user KB when initializing
- `--sample / --no-sample`: Control whether a starter entry is created
- `--cwd-only`: Ignore parent-directory `.kbconfig` discovery
- `--yes, -y`: Assume yes for initialization prompts
- `--json`: JSON output

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

Common next step: set a default write directory so you can omit `--category` in `mx add`:

```yaml
# .kbconfig (project root)
primary: guides
```

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
mx publish --kb-root ./kb -o docs    # Publish an explicit KB path
mx publish --scope=project -o docs   # Publish one scope
mx publish --base-url /my-kb         # For subdirectory hosting
mx publish --index guides/index      # Choose landing page
mx publish --include-drafts          # Include draft entries
mx publish --include-archived        # Include archived entries
mx publish --no-clean                # Keep existing output dir contents
mx publish --setup-github-actions
mx publish --setup-github-actions --dry-run
```

**Resolution order:**

- KB source: `--kb-root`, then `--scope`, then project KB from `.kbconfig`
- Base URL: `--base-url`, then `publish_base_url` in `.kbconfig`
- Landing page: `--index`, then `publish_index_entry` in `.kbconfig`

**Options:**
- `--kb-root, -k`: Publish an explicit KB path
- `--scope, -s`: Publish project or user scope
- `--yes, -y`: Skip confirmation when publishing user KB content
- `--output, -o`: Output directory
- `--base-url, -b`: Site base path
- `--title`: Site title
- `--index, -i`: Landing page entry path
- `--include-drafts`: Include draft entries
- `--include-archived`: Include archived entries
- `--no-clean`: Keep the existing output directory contents
- `--json`: JSON output
- `--setup-github-actions`: Create a GitHub Pages workflow
- `--dry-run`: Preview the GitHub Actions workflow

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

### mx eval

Evaluate search accuracy against a query dataset.

```bash
mx eval --dataset /path/to/queries.json         # Run against an explicit dataset
mx eval --dataset /path/to/queries.json --mode=keyword
mx eval --dataset /path/to/queries.json --scope=project
mx eval --dataset /path/to/queries.json --json  # JSON to stdout (includes meta)
mx eval --dataset /path/to/queries.json --save  # Also write .memex-eval/results/<timestamp>.json
mx eval --dataset /path/to/queries.json --out /tmp/mx-eval.json
```

### mx prime

Output agent workflow context (for hooks).

```bash
mx prime                    # Output onboarding + quick reference
mx prime --json             # JSON output
```

### mx session-context

Output dynamic project-relevant context (for hooks).

```bash
mx session-context                   # Print session context
mx session-context --max-entries 4   # Limit relevant entries
mx session-context --install         # Update the Claude settings file hook
mx session-context --install --install-path .claude/settings.local.json
mx session-context --json            # JSON output
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

## Machine-Readable JSON Output (Stable v1)

Many commands support `--json` for agent/tool integration.

### Common fields

All documented JSON outputs include:
- `schema_version`: Stable output schema version (currently `1`)
- `version`: `mx` tool version (string, e.g. `0.3.2`)

### Scoped paths (@project/@user)

When both a project KB and a user KB are active, paths in machine output (and many human outputs) are **scope-qualified**:
- `@project/<relpath>.md`
- `@user/<relpath>.md`

Unscoped paths (e.g. `guides/x.md`) are still accepted and resolve to the **primary** KB (project if present, else user). If names collide across scopes, prefer explicit `@project/...` / `@user/...`.

### mx info --json

Top-level object (excerpt):

```json
{
  "schema_version": 1,
  "version": "0.3.2",
  "kb_configured": true,
  "primary_scope": "project",
  "primary_kb": "/abs/path/to/kb",
  "kbs": [
    {"scope": "project", "path": "/abs/path/to/project/kb", "entries": 12},
    {"scope": "user", "path": "/home/me/.memex/kb", "entries": 3}
  ]
}
```

### mx search --json

JSON list of result objects:

```json
[
  {
    "schema_version": 1,
    "version": "0.3.2",
    "path": "@project/guides/first-entry.md",
    "scope": "project",
    "title": "First Entry",
    "score": 1.0,
    "confidence": "high",
    "snippet": "..."
  }
]
```

### mx add --json

Top-level object:

```json
{
  "schema_version": 1,
  "version": "0.3.2",
  "path": "@project/guides/first-entry.md",
  "scope": "project",
  "suggested_links": [],
  "suggested_tags": [],
  "warnings": []
}
```

### --json-errors (errors as JSON)

Use the global `--json-errors` flag to get structured errors for **all** error sources (including Click usage/validation errors).

Error payload shape:

```json
{
  "schema_version": 1,
  "version": "0.3.2",
  "error": "USAGE_ERROR",
  "code": 1304,
  "message": "Query cannot be empty."
}
```

## See Also

- [[guides/installation|Installation Guide]]
- [[guides/ai-integration|AI Agent Integration]]
- [[reference/entry-format|Entry Format]]
- [[design/state-diagrams|State diagrams for MX flows]]
