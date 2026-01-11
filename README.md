# Memex

Personal knowledge base with hybrid search (keyword + semantic).

## Features

- **Hybrid search** - Combines keyword (Whoosh) and semantic (ChromaDB + sentence-transformers) search
- **CLI tool** - `mx` for token-efficient access from any environment
- **MCP server** - For Claude Desktop and MCP-compatible tools
- **Bidirectional links** - Obsidian-style `[[links]]` with backlink tracking
- **Web explorer** - Visual knowledge browser with graph view

## Installation

```bash
# Install with uv
uv tool install memex

# Or install from source
git clone https://github.com/chriskd/memex.git
cd memex
uv tool install -e .

# Verify installation
mx --version
```

## Quick Start

```bash
# Initialize a knowledge base
mkdir -p kb
export MEMEX_KB_ROOT=$(pwd)/kb

# Create your first entry
mx add --title="My First Note" --tags="example" --content="Hello, world!"

# Search for entries
mx search "hello"

# View an entry
mx get example/my-first-note.md
```

## CLI Reference

```bash
# Search (hybrid keyword + semantic)
mx search "deployment"              # Find entries
mx search "docker" --tags=infra     # Filter by tag
mx search "api" --mode=semantic     # Semantic only

# Read entries
mx get tooling/notes.md             # Full entry with content
mx get tooling/notes.md --metadata  # Just metadata

# Browse
mx tree                             # Directory structure
mx list --tag=infrastructure        # Filter by tag
mx whats-new --days=7               # Recent changes
mx tags                             # List all tags with counts

# Create/update entries
mx add --title="My Entry" --tags="foo,bar" --content="# Content..."
mx update path/entry.md --tags="new,tags"

# Analysis
mx hubs                             # Most connected entries
mx popular                          # Most viewed entries
mx dead-ends                        # Entries with no outgoing links

# Maintenance
mx health                           # Audit for problems
mx reindex                          # Rebuild search indices
```

## Claude Code Integration

### Hooks (Recommended)

Add to `.claude/settings.local.json`:

```json
{
  "env": { "MEMEX_KB_ROOT": "/path/to/kb", "MEMEX_INDEX_ROOT": "/path/to/indices" },
  "permissions": { "allow": ["Bash(mx:*)"] },
  "hooks": {
    "SessionStart": [{ "hooks": [{ "type": "command", "command": "mx prime" }] }],
    "PreCompact": [{ "hooks": [{ "type": "command", "command": "mx prime --compact" }] }]
  }
}
```

This automatically injects KB context at session start and before context compaction.

### MCP Server

Add to your Claude Code settings or MCP configuration:

```json
{
  "mcpServers": {
    "memex": {
      "type": "stdio",
      "command": "memex"
    }
  }
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `search` | Hybrid keyword + semantic search |
| `get` | Retrieve entry with content and links |
| `add` | Create new KB entry |
| `update` | Modify existing entry |
| `delete` | Remove an entry |
| `list` | List entries by category/tag |
| `whats_new` | Recently modified entries |
| `tree` | Directory structure |
| `tags` | List all tags |
| `backlinks` | Find entries linking to a path |
| `suggest_links` | Find semantically related entries |
| `health` | KB audit |

## Web Explorer

```bash
# Start the web interface
memex-web

# Open http://localhost:8080 in your browser
```

Features:
- Interactive search with live results
- Graph visualization of entry connections
- Directory tree navigation
- Markdown rendering with syntax highlighting

## Entry Format

Entries are Markdown files with YAML frontmatter:

```markdown
---
title: My Knowledge Entry
tags: [topic, category]
created: 2025-01-01
---

# My Knowledge Entry

Content with [[bidirectional links]] to other entries.

Use `[[path/to/entry|Display Text]]` for custom link text.
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMEX_KB_ROOT` | Knowledge base directory | `./kb` |
| `MEMEX_INDEX_ROOT` | Search index directory | `./.indices` |
| `MEMEX_PRELOAD` | Preload embedding model | `false` |
| `MEMEX_LOG_LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR) | `INFO` |

## Development

```bash
# Clone and install
git clone https://github.com/chriskd/memex.git
cd memex
uv sync --dev

# Run tests
uv run pytest

# Lint
uv run ruff check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT - see [LICENSE](LICENSE)
