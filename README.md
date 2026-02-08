![Memex](https://raw.githubusercontent.com/chriskd/memex-kb/main/social-banner.png)

# Memex

CLI-first knowledge base with hybrid search, typed relations, and static site publishing. Stores Markdown with YAML frontmatter across project and user scopes. Built for humans and AI coding agents.

## Features

- **Hybrid search** — BM25 keyword + semantic embeddings, merged with Reciprocal Rank Fusion
- **Typed relations** — explicit directed links in frontmatter with CLI helpers and linting
- **Graph view** — wikilinks, relations, and semantic links queryable via CLI and published sites
- **Dual scopes** — project KB (`./kb/`) and user KB (`~/.memex/kb/`) with shared search
- **Publishing** — static HTML with embedded search, tag pages, and graph visualization
- **Agent-friendly** — `mx prime`, `mx schema`, `mx batch` for low-token AI workflows

## Installation

```bash
# With uv (recommended)
uv tool install memex-kb

# Enable semantic search (optional heavier deps)
uv tool install 'memex-kb[search]'

# With pip
pip install memex-kb

# Enable semantic search (optional heavier deps)
pip install 'memex-kb[search]'

# Verify
mx --version
```

Keyword search ships by default. Semantic search is optional (to avoid heavyweight ML installs by default).
If semantic search is missing, run `mx doctor` for an install hint.
If `mx search` fails with `ModuleNotFoundError: No module named 'whoosh'`, install keyword search deps with
`pip install whoosh-reloaded` (or reinstall `memex-kb`), or install the full extra with `memex-kb[search]`.
Requires Python 3.11+.

## Quick Start

```bash
mx init                                    # Create kb/ and .kbconfig
mx add --title="Setup" --tags="docs" \
       --category=guides --content="..."   # Add an entry
mx list --limit=5                          # Confirm entry path
mx get guides/setup.md                     # Read an entry
mx health                                  # Audit KB (orphans are normal early)
mx search "setup"                          # Optional: search (keyword; semantic if installed)
```

Note: If `--category` is omitted and no `.kbconfig` `primary` exists, `mx add` defaults to the KB root (`.`) and prints a warning.
Tip: Set `.kbconfig` `primary: guides` to make `--category` optional. Check scope/config with `mx info` and `mx context show`.

## Entries

Markdown files with YAML frontmatter:

```markdown
---
title: API Guide
tags: [api, docs]
description: Endpoints and auth
---

# API Guide

See [[reference/auth]] for auth details.
```

Typed relations go in frontmatter:

```yaml
relations:
  - path: reference/cli.md
    type: documents
  - path: guides/installation.md
    type: depends_on
```

Types: `depends_on`, `implements`, `extends`, `documents`, `references`, `blocks`, `related`.

## CLI Overview

```bash
# Search and browse
mx search "query"                          # Hybrid search (default)
mx search "query" --mode=semantic          # Semantic only
mx search "query" --include-neighbors      # Include linked entries
mx get path/entry.md                       # Read entry
mx list --tags=docs                        # List by tag
mx tree                                    # Directory structure
mx whats-new --days=7                      # Recent changes

# Create and edit
mx add --title="..." --tags="..." --content="..."
mx append "Title" --content="..."          # Append or create
mx patch path.md --find="old" --replace="new"
mx ingest notes.md --directory=guides      # Import file

# Relations
mx relations path/entry.md --depth=2       # Query graph
mx relations-add path.md --relation "other.md=documents"
mx relations-lint --strict                 # Check consistency

# Maintenance
mx health                                  # Audit KB
mx hubs                                    # Find high-connectivity entries
mx suggest-links path.md                   # Semantic link suggestions
mx reindex                                 # Rebuild indices
mx eval                                    # Search quality metrics

# Agent tools
mx prime                                   # Session context for agents
mx schema --compact                        # CLI schema (for LLMs)
mx batch < commands.txt                    # Batch operations
```

Full reference at `kb/reference/cli.md`.

## Publishing

```bash
mx publish -o _site                        # Generate static site
mx publish --base-url /my-kb               # Custom base path
mx publish --setup-github-actions          # Add GH Pages workflow
```

## AI Integration

### Claude Code

Add to `.claude/settings.json`:

```json
{
  "permissions": { "allow": ["Bash(mx:*)"] },
  "hooks": {
    "SessionStart": [{ "hooks": [{ "type": "command", "command": "mx session-context" }] }]
  }
}
```

Or install automatically:

```bash
mx session-context --install
```

### Plugin

```text
/plugin marketplace add ./.claude-plugin/marketplace.json
/plugin install memex@memex
```

### Agent Commands

```bash
mx prime              # Context for session start
mx schema --compact   # CLI schema (minimal tokens)
mx batch              # Multiple commands, one invocation
```

## Configuration

`.kbconfig` at project root:

```yaml
kb_path: ./kb
primary: guides              # Default category for mx add
boost_paths:
  - guides/*                 # Search ranking boost
default_tags:
  - myproject
publish_base_url: /my-kb
```

Environment variables:

| Variable | Description |
|----------|-------------|
| `MEMEX_USER_KB_ROOT` | Override user KB location |
| `MEMEX_INDEX_ROOT` | Override index directory |
| `MEMEX_QUIET` | Suppress warnings |

CLI flags:

- `--json-errors` — errors as JSON with error codes
- `-q, --quiet` — suppress warnings

## Index Storage

Indices live at `{kb}/.indices/`:

- `whoosh/` — BM25 keyword index
- `chroma/` — semantic vectors
- `embedding_cache.sqlite` — cached embeddings

## Development

```bash
git clone https://github.com/chriskd/memex-kb.git
cd memex
uv sync --dev
uv run pytest
```

See `CONTRIBUTING.md` for guidelines.

## License

MIT
