# Memex

Personal knowledge base with hybrid search (keyword + semantic).

## Features

- **Hybrid search** - Combines keyword (Whoosh) and semantic (ChromaDB + sentence-transformers) search
- **CLI tool** - `mx` for token-efficient access from any environment
- **MCP server** - For Claude Desktop and MCP-compatible tools
- **Bidirectional links** - Obsidian-style `[[links]]` with backlink tracking
- **Web explorer** - Visual knowledge browser with graph view

## Installation

### Minimal Install (Keyword Search)

Fast, lightweight installation with keyword search only:

```bash
# With uv (recommended)
uv tool install memex

# With pip
pip install memex

# Verify
mx --version
```

This gives you full CLI and MCP functionality with BM25 keyword search. No heavy ML dependencies.

### Full Install (Semantic Search)

Add semantic search for meaning-based queries:

```bash
# With uv
uv tool install "memex[semantic]"

# With pip
pip install "memex[semantic]"
```

Adds ~500MB of dependencies (ChromaDB, sentence-transformers). First search will download the embedding model (~100MB).

### From Source

```bash
git clone https://github.com/chriskd/memex.git
cd memex
uv sync --dev              # Development install
uv pip install -e ".[semantic]"  # Add semantic search
```

**Note:** On ARM64 (Apple Silicon, etc.), ChromaDB is capped at <1.0.0 for onnxruntime compatibility.

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

**New to Memex?** See the [First 5 Minutes Tutorial](TUTORIAL.md) for a complete walkthrough.

## CLI Reference

The `mx` CLI is designed for token-efficient access from AI agents and humans alike.

```bash
# Search (hybrid keyword + semantic)
mx search "deployment"              # Find entries
mx search "docker" --tags=infra     # Filter by tag
mx search "api" --mode=semantic     # Semantic only (requires [semantic])
mx search "api" --mode=keyword      # Fast keyword-only search

# Read entries
mx get tooling/notes.md             # Full entry with content
mx get tooling/notes.md --metadata  # Just metadata

# Browse
mx info                             # Show KB configuration
mx tree                             # Directory structure
mx list --tag=infrastructure        # Filter by tag
mx whats-new --days=7               # Recent changes
mx whats-new --project=myapp        # Recent changes for a project
mx tags                             # List all tags with counts
mx history                          # View search history

# Create/update entries
mx add --title="My Entry" --tags="foo,bar" --content="# Content..."
mx add --title="..." --tags="..." --category=tooling --file=notes.md
mx update path/entry.md --tags="new,tags"
mx update path/entry.md --content="Append this" --append --timestamp
mx upsert --title="Daily Log" --content="New entry" --append
mx quick-add --stdin                # Auto-generate metadata from content
mx patch path/entry.md --find="old" --replace="new" --dry-run

# Analysis
mx hubs                             # Most connected entries
mx suggest-links path/entry.md      # Find semantically related entries
mx health                           # Audit for problems

# Project context
mx context init                     # Create .kbcontext for current project
mx context show                     # Show active project context

# Static site publishing
mx publish -o docs                  # Generate HTML site
mx publish -o docs --base-url /repo # For GitHub Pages subdirectory

# Maintenance
mx reindex                          # Rebuild search indices
```

## MCP Server

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
| `mkdir` | Create a directory |
| `move` | Move/rename an entry |
| `rmdir` | Remove an empty directory |
| `quality` | Detailed quality report |

## AI Agent Integration

Memex is designed for AI coding assistants. **CLI is the recommended interface** - it uses ~0 tokens vs MCP's schema overhead.

### Claude Code

Add to your `.claude/settings.local.json`:

```json
{
  "permissions": {
    "allow": ["Bash(mx:*)"]
  }
}
```

Or use hooks for automatic context injection:

```json
{
  "hooks": {
    "SessionStart": [{ "command": "mx prime" }]
  }
}
```

The `mx prime` command injects KB workflow guidance at session start and auto-detects whether MCP is configured.

### Codex CLI

Codex can use memex via shell commands:

```bash
# In your AGENTS.md or system prompt
mx search "query"     # Search KB
mx get path/entry.md  # Read entry
mx add --title="..." --tags="..." --content="..."  # Add entry
```

### Other AI Agents

Any agent with shell access can use the `mx` CLI. Key patterns:

```bash
# Check for relevant knowledge before implementing
mx search "authentication patterns"

# Add discoveries for future sessions
mx add --title="API Rate Limiting" --tags="api,patterns" --content="..."

# View recent project updates
mx whats-new --project=myapp --days=7
```

### CLI vs MCP

| Aspect | CLI (`mx`) | MCP Server |
|--------|------------|------------|
| Token cost | ~0 (shell command) | ~500+ (tool schema) |
| Setup | Just install | Configure mcpServers |
| Output | Human-readable | Structured JSON |
| Best for | Most agents | Claude Desktop app |

**Recommendation:** Use CLI unless you specifically need MCP's structured responses.

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
