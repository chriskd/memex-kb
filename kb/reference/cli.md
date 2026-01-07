---
title: CLI Reference
tags: [cli, reference, commands]
created: 2026-01-06
description: Complete reference for the mx command-line interface
---

# CLI Reference

The `mx` CLI provides token-efficient access to your knowledge base.

## Global Options

All commands support these top-level options:

```bash
mx --version                     # Show version
mx --json-errors <command>       # Output errors as JSON (for agents)
mx --quiet <command>             # Suppress warnings
mx -q <command>                  # Shorthand for --quiet
```

**Environment Variables:**
- `MEMEX_KB_ROOT`: Knowledge base root directory
- `MEMEX_INDEX_ROOT`: Search index directory
- `MEMEX_QUIET=1`: Suppress warnings (equivalent to --quiet)

---

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
mx search "query" --compact          # Minimal JSON (short keys)
mx search "query" --no-session       # Ignore session context
mx search "query" --no-history       # Don't record in history
```

**Options:**
- `--tags, -t`: Filter by tags (comma-separated)
- `--mode`: Search mode (hybrid, keyword, semantic)
- `--limit, -n`: Maximum results (default: 10)
- `--content, -c`: Include full content in results
- `--strict`: Disable semantic fallback
- `--terse`: Output paths only (one per line)
- `--json`: JSON output
- `--compact`: Minimal JSON (short keys: p, t, s)
- `--no-session`: Ignore session context for this search
- `--no-history`: Don't record this search in history

**JSON Output Example:**
```json
[
  {
    "path": "tooling/beads.md",
    "title": "Beads Issue Tracker",
    "score": 0.95,
    "snippet": "Lightweight issue tracking...",
    "match_type": "keyword"
  }
]
```

**Compact JSON Output:**
```json
[{"p": "tooling/beads.md", "t": "Beads Issue Tracker", "s": 0.95}]
```

### mx history

View and re-run past searches.

```bash
mx history                  # Show last 10 searches
mx history -n 20            # Show last 20
mx history --rerun 1        # Re-run most recent
mx history -r 3             # Re-run 3rd most recent
mx history --clear          # Clear history
mx history --json           # JSON output
```

**Options:**
- `-n, --limit`: Max entries to show (default: 10)
- `-r, --rerun`: Re-execute search at position N (1=most recent)
- `--clear`: Clear all search history
- `--json`: Output as JSON

---

## Read Commands

### mx get

Read a knowledge base entry.

```bash
mx get tooling/my-entry.md            # Full entry
mx get tooling/my-entry.md --metadata # Metadata only
mx get tooling/my-entry.md --json     # JSON output
```

**Options:**
- `-m, --metadata`: Show only metadata (frontmatter)
- `--json`: Output as JSON with metadata

### mx list

List entries with optional filters.

```bash
mx list                        # All entries
mx list --tag=infrastructure   # Filter by tag
mx list --category=tooling     # Filter by category
mx list --limit=50             # More results
mx list --json                 # JSON output
```

**Options:**
- `-t, --tag`: Filter by tag
- `-c, --category`: Filter by category
- `-n, --limit`: Max results (default: 20)
- `--json`: Output as JSON

### mx tree

Display directory structure.

```bash
mx tree                    # Full tree
mx tree tooling            # Specific path
mx tree --depth=2          # Limit depth
mx tree --json             # JSON output
```

**Options:**
- `-d, --depth`: Max depth to display
- `--json`: Output as JSON

---

## Write Commands

### mx add

Create a new entry.

```bash
mx add --title="My Entry" --tags="foo,bar" --category=tooling --content="..."
mx add --title="..." --tags="..." --category=... --file=content.md
cat content.md | mx add --title="..." --tags="..." --category=... --stdin
mx add --title="..." --tags="..." --template=troubleshooting
mx add --title="..." --tags="..." --dry-run  # Preview only
mx add --title="..." --tags="..." --force    # Override duplicate detection
mx add --title="..." --tags="..." --json     # JSON output
```

**Required:**
- `--title, -t`: Entry title
- `--tags`: Tags (comma-separated)
- `--category, -c`: Target directory (required unless .kbcontext sets a primary path)

**Options:**
- `--content`: Content (or use --file/--stdin)
- `-f, --file`: Read content from file
- `--stdin`: Read content from stdin
- `-T, --template`: Use a template (see `mx templates`)
- `--force`: Create even if duplicates detected
- `--dry-run`: Preview path/frontmatter/content without creating
- `--json`: Output as JSON

### mx update

Update an existing entry.

```bash
mx update path/entry.md --tags="new,tags"
mx update path/entry.md --content="New content"
mx update path/entry.md --content="Append this" --append
mx update path/entry.md --content="..." --append --timestamp
mx update path/entry.md --file=new-content.md
mx update path/entry.md --stdin --append
mx update path/entry.md --json
```

**Options:**
- `--tags`: Replace tags (comma-separated)
- `--content`: New content (replaces existing unless --append)
- `-f, --file`: Read content from file
- `--stdin`: Read content from stdin
- `--append`: Append to end instead of replacing
- `--timestamp`: Add `## YYYY-MM-DD HH:MM UTC` header
- `--json`: Output as JSON

### mx patch

Surgical find-replace edits.

```bash
mx patch path/entry.md --old="old text" --new="new text"
mx patch path/entry.md --old="TODO" --new="DONE" --replace-all
mx patch path/entry.md --old-file=old.txt --new-file=new.txt
mx patch path/entry.md --old="..." --new="..." --dry-run
mx patch path/entry.md --old="..." --new="..." --backup
mx patch path/entry.md --old="..." --new="..." --json
```

**Options:**
- `--old`: Exact text to find and replace
- `--new`: Replacement text
- `--old-file`: Read --old text from file (for multi-line)
- `--new-file`: Read --new text from file (for multi-line)
- `--replace-all`: Replace all occurrences
- `--dry-run`: Preview changes without modifying
- `--backup`: Create .bak backup before patching
- `--json`: Output as JSON

**Exit Codes:**
- 0: Success
- 1: Text not found
- 2: Multiple matches (ambiguous, use --replace-all)
- 3: Input error (file not found, permission, encoding)

### mx upsert

Create or append to entry by title.

```bash
mx upsert "Daily Log" --content="Session summary"
mx upsert "API Docs" --file=api.md --tags="api,docs"
mx upsert "Debug Log" --content="..." --no-create  # Error if not found
mx upsert "Notes" --content="..." --replace        # Replace instead of append
mx upsert "Notes" --content="..." --no-timestamp   # Skip timestamp header
mx upsert "Notes" --stdin --directory=projects/myapp
mx upsert "Notes" --content="..." --json
```

**Options:**
- `-c, --content`: Content to add
- `-f, --file`: Read content from file
- `--stdin`: Read content from stdin
- `--tags`: Tags for new entry (comma-separated)
- `-d, --directory`: Target directory for new entry
- `--no-timestamp`: Don't add timestamp header
- `--replace`: Replace content instead of appending
- `--create / --no-create`: Create entry if not found (default: create)
- `--json`: Output as JSON

### mx quick-add

Quickly add content with auto-generated metadata.

```bash
mx quick-add --stdin              # Paste content, auto-generate all
mx quick-add -f notes.md          # From file with auto metadata
mx quick-add -c "..." -y          # Auto-confirm creation
echo "..." | mx quick-add --stdin --json  # Machine-readable
mx quick-add --stdin --title="Override Title"  # Override auto-detection
```

**Options:**
- `-f, --file`: Read content from file
- `--stdin`: Read content from stdin
- `-c, --content`: Raw content to add
- `-t, --title`: Override auto-detected title
- `--tags`: Override auto-suggested tags (comma-separated)
- `--category`: Override auto-suggested category
- `-y, --confirm`: Auto-confirm without prompting
- `--json`: Output as JSON

### mx delete

Delete an entry.

```bash
mx delete path/entry.md
mx delete path/entry.md --force  # Delete even with backlinks
mx delete path/entry.md --json
```

**Options:**
- `-f, --force`: Delete even if has backlinks
- `--json`: Output as JSON

---

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
- Empty directories
- Entries missing descriptions

### mx hubs

Show most connected entries.

```bash
mx hubs
mx hubs --limit=5
mx hubs --json
```

**Options:**
- `-n, --limit`: Max results (default: 10)
- `--json`: Output as JSON

### mx suggest-links

Find semantically related entries.

```bash
mx suggest-links path/entry.md
mx suggest-links path/entry.md --limit=10
mx suggest-links path/entry.md --json
```

**Options:**
- `-n, --limit`: Max suggestions (default: 5)
- `--json`: Output as JSON

### mx tags

List all tags with counts.

```bash
mx tags
mx tags --min-count=3
mx tags --json
```

**Options:**
- `--min-count`: Minimum usage count
- `--json`: Output as JSON

### mx summarize

Generate descriptions for entries missing them.

```bash
mx summarize --dry-run         # Preview what would be generated
mx summarize                   # Generate and write descriptions
mx summarize --limit 5         # Process only 5 entries
mx summarize --json            # Output as JSON
```

**Options:**
- `--dry-run`: Preview changes without writing
- `--limit`: Maximum entries to process
- `--json`: Output as JSON

---

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
mx whats-new --project=myapp      # Filter by project
mx whats-new --limit=20           # More results
mx whats-new --json               # JSON output
```

**Options:**
- `-d, --days`: Look back N days
- `-n, --limit`: Max results
- `-p, --project`: Filter by project name
- `--json`: Output as JSON

---

## Session Commands

### mx session

Manage session search context. Session context persists until explicitly cleared.

```bash
mx session                    # Show current session (alias for 'show')
mx session show               # Show current session
mx session show --json        # JSON output
mx session show --suggest     # Show bootstrap suggestions
```

### mx session start

Start a new session with given context.

```bash
mx session start --tags=infra,docker
mx session start --project=api-service
mx session start --tags=python --project=memex
mx session start --json
```

**Options:**
- `-t, --tags`: Tags to filter by (comma-separated)
- `-p, --project`: Project to boost in results
- `--json`: Output as JSON

### mx session set

Update the current session context.

```bash
mx session set --tags=docker          # Replace tags
mx session set --add-tags=kubernetes  # Add tag
mx session set --project=new-project  # Change project
mx session set --json
```

**Options:**
- `-t, --tags`: Tags to filter by (replaces existing)
- `--add-tags`: Tags to add (comma-separated)
- `-p, --project`: Project to boost
- `--json`: Output as JSON

### mx session clear

Clear the session context.

```bash
mx session clear
mx session clear --json
```

### mx session-log

Log a session summary to the project's session entry.

```bash
mx session-log --message="Fixed auth bug, added tests"
mx session-log --stdin < session_notes.md
mx session-log -m "Deployed v2.1" --tags="deployment,release"
mx session-log -m "..." --entry=projects/myapp/devlog.md
mx session-log -m "..." --links="tooling/beads,projects/api"
mx session-log -m "..." --no-timestamp
mx session-log -f session.md --json
```

**Entry Resolution Order:**
1. `--entry` flag (explicit)
2. `.kbcontext` session_entry field
3. `{.kbcontext primary}/sessions.md`
4. Error with guidance if no context

**Options:**
- `-m, --message`: Session summary message
- `-f, --file`: Read message from file
- `--stdin`: Read message from stdin
- `-e, --entry`: Explicit entry path (overrides context)
- `--tags`: Additional tags (comma-separated)
- `--links`: Wiki-style links to include (comma-separated)
- `--no-timestamp`: Don't add timestamp header
- `--json`: Output as JSON

---

## Context Commands

### mx context

Manage project-specific KB context.

```bash
mx context                  # Show current context (alias for 'show')
mx context show             # Show current context
mx context show --suggest   # Show bootstrap suggestions if no context
mx context show --json      # JSON output
```

### mx context init

Create a new .kbcontext file in the current directory.

```bash
mx context init
mx context init --project myapp
mx context init --project myapp --directory projects/myapp/docs
mx context init --force     # Overwrite existing
```

**Options:**
- `-p, --project`: Project name (auto-detected from directory if not provided)
- `-d, --directory`: KB directory (defaults to projects/<project>)
- `-f, --force`: Overwrite existing .kbcontext file

### mx context validate

Validate the current .kbcontext file against the knowledge base.

```bash
mx context validate
mx context validate --json
```

Checks that:
- Primary directory exists (or can be created)
- Paths reference valid locations (warning only)

---

## Beads Commands

Browse beads issue tracking across registered projects.

### mx beads

```bash
mx beads list                    # List issues
mx beads show epstein-42         # Show issue details
mx beads kanban                  # Kanban board view
mx beads status                  # Project statistics
mx beads projects                # List registered projects
```

### mx beads list

List issues from a beads project.

```bash
mx beads list                         # All issues from detected project
mx beads list -p epstein              # Issues from epstein project
mx beads list --status=open           # Only open issues
mx beads list --type=bug --limit=10   # 10 bugs
mx beads list --json                  # JSON output
```

**Options:**
- `-p, --project`: Beads project prefix from registry
- `-s, --status`: Filter by status (open, in_progress, closed, all)
- `-t, --type`: Filter by type (task, bug, feature, epic)
- `-n, --limit`: Max results (default: 50)
- `--json`: Output as JSON

### mx beads show

Show detailed information for a specific issue.

```bash
mx beads show epstein-42              # Full issue details with comments
mx beads show 42 -p epstein           # Equivalent with explicit project
mx beads show epstein-42 --no-comments # Without comments
mx beads show epstein-42 --json       # JSON output
```

**Options:**
- `-p, --project`: Beads project prefix (auto-detected from issue ID)
- `--no-comments`: Exclude comments
- `--json`: Output as JSON

### mx beads kanban

Display issues grouped by status (kanban board view).

```bash
mx beads kanban                       # Kanban for detected project
mx beads kanban -p voidlabs-ansible   # Specific project
mx beads kanban --compact             # Titles only
mx beads kanban --json                # JSON output
```

**Options:**
- `-p, --project`: Beads project prefix from registry
- `--compact`: Compact view (titles only)
- `--json`: Output as JSON

### mx beads status

Show project statistics and health summary.

```bash
mx beads status                       # Stats for detected project
mx beads status -p memex              # Stats for memex project
mx beads status --json                # JSON output
```

**Options:**
- `-p, --project`: Beads project prefix from registry
- `--json`: Output as JSON

### mx beads projects

List all registered beads projects.

```bash
mx beads projects                     # List all projects
mx beads projects --json              # JSON output
```

---

## Batch Operations

### mx batch

Execute multiple KB operations in a single invocation.

```bash
mx batch << 'EOF'
add --title='Note 1' --tags='tag1' --category=tooling --content='Content'
search 'api'
get tooling/beads.md
update tooling/notes.md --content='New content' --append
delete tooling/old-entry.md --force
EOF

mx batch -f commands.txt              # Read from file
mx batch --stop-on-error              # Stop on first error
```

**Supported Commands:**
- `add --title='...' --tags='...' [--category=...] [--content='...'] [--force]`
- `update <path> [--tags='...'] [--content='...'] [--append]`
- `upsert <title> [--content='...'] [--tags='...'] [--directory=...]`
- `search <query> [--tags='...'] [--mode=...] [--limit=N]`
- `get <path> [--metadata]`
- `delete <path> [--force]`

**Options:**
- `-f, --file`: Read commands from file instead of stdin
- `--continue-on-error / --stop-on-error`: Continue after errors (default: continue)

**Output Format:**
```json
{
  "total": 2,
  "succeeded": 2,
  "failed": 0,
  "results": [
    {"index": 0, "command": "add ...", "success": true, "result": {...}},
    {"index": 1, "command": "search ...", "success": true, "result": {...}}
  ]
}
```

---

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

# Other options
mx publish -o docs --title "My Docs"       # Custom site title
mx publish -o docs --index guides/welcome  # Custom landing page
mx publish -o docs --include-drafts        # Include draft entries
mx publish -o docs --include-archived      # Include archived entries
mx publish -o docs --no-clean              # Don't clean output dir first
mx publish -o docs --json                  # JSON output
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
- `--title, -t`: Site title (default: Memex)
- `--index, -i`: Entry to use as landing page
- `--include-drafts`: Include draft entries in output
- `--include-archived`: Include archived entries in output
- `--no-clean`: Don't remove output directory before build
- `--json`: Output as JSON

---

## Maintenance Commands

### mx init

Initialize a new memex knowledge base.

```bash
mx init                           # Use defaults
mx init --kb-root ~/my-kb         # Custom location
mx init --index-root ~/my-indices # Custom index location
mx init --no-context              # Skip .kbcontext setup
mx init --force                   # Recreate dirs (safe, keeps entries)
```

**Default Structure:**
```
<kb-root>/
  ├── projects/       # Project-specific documentation
  ├── tooling/        # Tools and utilities
  └── infrastructure/ # Infrastructure and DevOps
```

**Options:**
- `--kb-root`: KB root path (defaults to MEMEX_KB_ROOT or ~/.memex/kb)
- `--index-root`: Index root path
- `--no-context`: Skip .kbcontext creation prompt
- `-f, --force`: Recreate directory structure (preserves existing entries)

### mx reindex

Rebuild search indices.

```bash
mx reindex
```

Use this after bulk imports or if search results seem stale.

### mx prime

Output agent workflow context (for Claude Code hooks).

```bash
mx prime                    # Auto-detect mode
mx prime --full             # Force full output
mx prime --compact          # Force minimal output (for PreCompact hooks)
mx prime --project=myapp    # Include recent entries for project
mx prime -p myapp -d 14     # Last 14 days of myapp changes
mx prime --json             # JSON output
```

**Options:**
- `--full`: Force full CLI output (ignore MCP detection)
- `--compact`: Force compact output (minimal)
- `-p, --project`: Include recent entries for project
- `-d, --days`: Days to look back for project entries (default: 7)
- `--json`: Output as JSON

### mx templates

List or show entry templates.

```bash
mx templates              # List all available templates
mx templates list         # Same as above
mx templates show pattern # Show the 'pattern' template content
mx templates --json       # JSON output
```

**Template Sources (priority order):**
1. Project: `.kbcontext` templates section
2. User: `~/.config/memex/templates/*.yaml`
3. Built-in: troubleshooting, project, pattern, decision, runbook, api, meeting

### mx schema

Output machine-readable schema of all mx commands.

```bash
mx schema                    # Full schema as JSON
mx schema -c add             # Schema for 'add' command only
mx schema --compact          # Minified JSON output
```

Useful for LLM agents and tools that need to programmatically understand available commands.

**Options:**
- `-c, --command`: Get schema for specific command only
- `--compact`: Compact JSON output (no indentation)

---

## Programmatic Error Handling

### --json-errors Flag

Use `--json-errors` for structured error output that agents can parse:

```bash
mx --json-errors get nonexistent/file.md
```

**Output:**
```json
{"error": "ENTRY_NOT_FOUND", "code": 1002, "message": "Entry not found: nonexistent/file.md"}
```

**Error Format:**
```json
{
  "error": "ERROR_NAME",
  "code": 1001,
  "message": "Human-readable message",
  "details": {
    "similar_entries": ["path/to/similar.md"],
    "suggestion": "Use --force to override"
  }
}
```

### Error Codes Reference

**Entry Errors (1001-1099):**
| Code | Name | Description |
|------|------|-------------|
| 1001 | DUPLICATE_DETECTED | Entry with similar title exists |
| 1002 | ENTRY_NOT_FOUND | Entry does not exist |
| 1003 | INVALID_PATH | Path is malformed or invalid |
| 1004 | ENTRY_EXISTS | Entry already exists (for create operations) |
| 1005 | PARSE_ERROR | Failed to parse entry content |
| 1006 | AMBIGUOUS_MATCH | Multiple entries match the query |

**Index/Search Errors (1101-1199):**
| Code | Name | Description |
|------|------|-------------|
| 1101 | INDEX_UNAVAILABLE | Search index is not available |
| 1102 | SEMANTIC_SEARCH_UNAVAILABLE | Semantic search not installed |
| 1103 | SEARCH_FAILED | Search operation failed |

**Configuration Errors (1201-1299):**
| Code | Name | Description |
|------|------|-------------|
| 1201 | KB_NOT_CONFIGURED | MEMEX_KB_ROOT not set |
| 1202 | INVALID_CATEGORY | Category/directory does not exist |
| 1203 | CONTEXT_NOT_FOUND | No .kbcontext file found |

**Validation Errors (1301-1399):**
| Code | Name | Description |
|------|------|-------------|
| 1301 | MISSING_REQUIRED_FIELD | Required option not provided |
| 1302 | INVALID_CONTENT | Content failed validation |
| 1303 | INVALID_TAGS | Tags format is invalid |
| 1304 | VALIDATION_ERROR | General input validation failure |

**File Operation Errors (1401-1499):**
| Code | Name | Description |
|------|------|-------------|
| 1401 | FILE_READ_ERROR | Failed to read file |
| 1402 | FILE_WRITE_ERROR | Failed to write file |
| 1403 | PERMISSION_DENIED | Insufficient permissions |

**Batch Operation Errors (1501-1599):**
| Code | Name | Description |
|------|------|-------------|
| 1501 | BATCH_PARSE_ERROR | Failed to parse batch command |
| 1502 | BATCH_UNKNOWN_COMMAND | Unknown command in batch |
| 1503 | BATCH_MISSING_ARGUMENT | Missing required argument |

### Agent Error Handling Example

```bash
# Agent workflow: try to get entry, handle errors programmatically
result=$(mx --json-errors get tooling/api-docs.md 2>&1)
exit_code=$?

if [ $exit_code -ne 0 ]; then
  error_code=$(echo "$result" | jq -r '.code')

  case $error_code in
    1002)
      # Entry not found - create it
      mx add --title="API Docs" --tags="api" --category=tooling --content="..."
      ;;
    1201)
      # KB not configured - initialize
      mx init
      ;;
    *)
      echo "Unexpected error: $result"
      ;;
  esac
fi
```

---

## See Also

- [[guides/installation|Installation Guide]]
- [[guides/ai-integration|AI Agent Integration]]
- [[reference/entry-format|Entry Format]]
