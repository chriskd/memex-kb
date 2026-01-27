# Memex

Memex is a CLI-first, dual-scope knowledge base for teams and LLM workflows.
It stores Markdown with YAML frontmatter, supports hybrid full-text + semantic search,
typed relations, bidirectional links, and can publish a static site with a graph view.
Designed for humans and AI coding agents across project and user knowledge bases.

## Features

- **Hybrid search**: BM25 keyword + semantic embeddings (Whoosh + ChromaDB + sentence-transformers)
  merged with Reciprocal Rank Fusion and relevance boosts.
- **Typed relations**: Canonical relation types in frontmatter, CLI add/remove helpers,
  and linting (`mx relations-lint`) for consistency.
- **Relations graph**: Unified graph of wikilinks + typed relations, queryable via `mx relations`,
  and surfaced in published sites.
- **Bidirectional links**: Obsidian-style `[[links]]`, backlinks, and semantic links for discovery.
- **Project + user scopes**: Separate KBs with `@project/` and `@user/` paths, shared search,
  and scope-aware indexing/publishing.
- **Entry tooling**: Quick-add, ingest, templates, and surgical edits (append/patch/replace).
- **Publishing**: Static HTML site with search index, tag pages, and graph data; optional
  GitHub Pages workflow generation.
- **Agent workflows**: `mx session-context`, `mx prime`, `mx schema`, `mx batch`, and
  optional memory/evolution tooling for AI assistants.
- **Quality tooling**: `mx health`, `mx suggest-links`, `mx hubs`, `mx summarize`, and
  `mx reindex` for maintenance.

## Who It Is For

- Developers and teams who want searchable, versioned project knowledge.
- AI-assisted workflows that need low-token, scriptable KB access.
- Anyone building a "second brain" with Markdown + graph connections.

## Installation

```bash
# With uv (recommended)
uv tool install memex-kb

# With pip
pip install memex-kb

# Verify
mx --version
```

Semantic search is included by default. First run downloads the embedding model.
Python 3.11+ is required.

## Quick Start

```bash
# Initialize a project KB
mx init

# Add an entry
mx add --title="Setup Guide" --tags="docs" --category=guides \
  --content="# Setup\n\nInstructions here."

# Search
mx search "setup"

# Read an entry
mx get guides/setup-guide.md
```

This creates a `kb/` directory and a `.kbconfig` at your project root.
Note: `mx add` requires `--category` unless `primary` is set in `.kbconfig`.

## Core Concepts

### Entries (Markdown + Frontmatter)

Entries are Markdown files with YAML frontmatter:

```markdown
---
title: API Guide
tags: [api, docs]
description: Endpoints and auth rules
---

# API Guide

See [[reference/auth]] for auth details.
```

### Typed Relations

Typed relations are explicit, directed links in frontmatter:

```yaml
relations:
  - path: reference/cli.md
    type: documents
  - path: guides/installation.md
    type: depends_on
```

Common relation types: `depends_on`, `implements`, `extends`, `documents`,
`references`, `blocks`, `related`.

### Scopes

Memex supports two KB scopes:

- **Project** (`./kb/`): shared with your repo
- **User** (`~/.memex/kb/`): personal across projects

Commands accept `--scope=project|user`, and paths are prefixed when both exist:
`@project/guides/setup.md`, `@user/notes/ideas.md`.

## CLI At a Glance

```bash
# Search and browse
mx search "query" --mode=hybrid|keyword|semantic --tags=docs --min-score=0.4
mx search "query" --include-neighbors --neighbor-depth=2
mx get path/to/entry.md --metadata
mx list --tags=docs --scope=project
mx tree --depth=2
mx whats-new --days=7
mx history
mx info
mx context validate

# Write and edit
mx add --title="..." --tags="..." --category=guides --content="..."
mx append "Title" --content="..."
mx patch path.md --find="old" --replace="new"
mx replace path.md --content="new content"
mx delete path.md
mx quick-add --stdin
mx ingest notes.md --directory=guides
mx templates

# Relations and graph
mx relations path/to/entry.md --depth=2 --origin relations
mx relations --graph --json
mx relations-add path/to/entry.md --relation "reference/cli.md=documents"
mx relations-remove path/to/entry.md --relation "reference/cli.md=documents"
mx relations-lint --strict

# Maintenance and discovery
mx health
mx hubs
mx suggest-links path/to/entry.md
mx summarize --dry-run
mx reindex

# Automation and schema
mx batch << 'EOF'
search "deployment"
get guides/installation.md
EOF
mx schema --compact
```

See the full CLI reference at `kb/reference/cli.md`.

## Publishing

Generate a static site with search, tags, and graph data:

```bash
mx publish -o _site
mx publish --base-url /my-kb
mx publish --include-drafts
```

Optional GitHub Pages workflow:

```bash
mx publish --setup-github-actions
```

Published pages surface typed relations (direction + type) and a graph view.

## AI / Agent Integration

Memex is optimized for AI coding assistants.

### Claude Code Hooks (Recommended)

```json
{
  "permissions": { "allow": ["Bash(mx:*)"] },
  "hooks": {
    "SessionStart": [{ "hooks": [{ "type": "command", "command": "mx session-context" }] }]
  }
}
```

Install the hook automatically:

```bash
mx session-context --install
```

### Claude Code Plugin

This repo ships a Claude Code plugin manifest in `.claude-plugin/`. From the repo
root, add the local marketplace and install the plugin from Claude Code:

```text
/plugin marketplace add ./.claude-plugin/marketplace.json
/plugin install memex@memex
```

Restart Claude Code after installing or updating the plugin.

### Codex Skills

This repo includes a Memex skill at `skills/kb-usage/`.
Codex CLI discovers skills from well-known directories (for example
`.codex/skills` in the repo or `~/.codex/skills` for user-level installs).
To install this repo's Memex skill, copy or symlink `skills/kb-usage/` into a
Codex skills directory, then restart Codex.

Examples:

```bash
mkdir -p .codex/skills
cp -r skills/kb-usage .codex/skills/

mkdir -p ~/.codex/skills
cp -r skills/kb-usage ~/.codex/skills/
```

### Agent-Friendly Commands

Other agent-friendly commands:

```bash
mx prime
mx schema --compact
mx batch << 'EOF'
search "deployment"
get guides/installation.md
EOF
```

Agent memory tools:

```bash
mx memory init
mx memory add "Decision: use Redis for caching"
mx a-mem-init --scope=project
mx evolve --status
```

## Configuration

Project config lives in `.kbconfig`:

```yaml
kb_path: ./kb
primary: guides
boost_paths:
  - guides/*
default_tags:
  - memex
publish_base_url: /my-kb
```

`primary` sets the default directory for new entries, `boost_paths` affects search
ranking, and `publish_base_url` is used when publishing to a subpath.

Environment variables:

| Variable | Description |
|----------|-------------|
| `MEMEX_USER_KB_ROOT` | Override user KB root (default: `~/.memex/kb/`) |
| `MEMEX_INDEX_ROOT` | Index directory (default: `{kb}/.indices`) |

## Documentation

- `kb/guides/installation.md`
- `kb/guides/quick-start.md`
- `kb/guides/ai-integration.md`
- `kb/reference/cli.md`
- `kb/reference/entry-format.md`
- `kb/memex/state-diagrams.md`

## Development

```bash
git clone https://github.com/chriskd/memex.git
cd memex
uv sync --dev
uv run pytest
```

See `CONTRIBUTING.md` for guidelines.

## License

MIT - see `LICENSE`
