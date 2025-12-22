# voidlabs-kb

Organization-wide knowledge base with hybrid search (keyword + semantic).

## Features

- **Hybrid search** - Combines keyword (Whoosh) and semantic (ChromaDB + sentence-transformers) search
- **CLI tool** - `vl-kb` for token-efficient access from any environment
- **MCP server** - For Claude Desktop and MCP-only environments
- **Bidirectional links** - Obsidian-style `[[links]]` with backlink tracking
- **Agent-optimized** - `vl-kb prime` for session context injection

## CLI vs MCP

**RECOMMENDED**: Use the `vl-kb` CLI for the best experience. This approach:

- **Minimizes context usage** - CLI calls use ~0 tokens vs MCP tool schemas
- **Lower latency** - Direct CLI calls are faster than MCP protocol overhead
- **Universal** - Works with any AI assistant, not just MCP-compatible ones

| Environment | Recommendation |
|-------------|----------------|
| ✅ Claude Code, Cursor, any shell access | Use `vl-kb` CLI |
| ✅ MCP-only environments (Claude Desktop) | Use MCP server |

## Installation

### CLI (Recommended)

```bash
# Install via uv (auto-installed in voidlabs devcontainers)
uv tool install -e /path/to/voidlabs-kb

# Verify installation
vl-kb --version
```

### MCP Server

Add to your Claude Code settings or `.mcp.json`:

```json
{
  "mcpServers": {
    "voidlabs-kb": {
      "type": "stdio",
      "command": "uv",
      "args": ["--directory", "/path/to/voidlabs-kb", "run", "python", "-m", "voidlabs_kb"]
    }
  }
}
```

## CLI Quick Reference

```bash
# Search (hybrid keyword + semantic)
vl-kb search "deployment"              # Find entries
vl-kb search "docker" --tags=infra     # Filter by tag
vl-kb search "api" --mode=semantic     # Semantic only

# Read entries
vl-kb get tooling/beads.md             # Full entry with content
vl-kb get tooling/beads.md --metadata  # Just metadata
vl-kb get tooling/beads.md --json      # JSON output

# Browse
vl-kb tree                             # Directory structure
vl-kb list --tag=infrastructure        # Filter by tag
vl-kb whats-new --days=7               # Recent changes
vl-kb whats-new --project=myapp        # Recent changes for a project
vl-kb tags                             # List all tags with counts

# Create entries
vl-kb add --title="My Entry" --tags="foo,bar" --content="# Content..."
vl-kb add --title="..." --tags="..." --file=content.md
cat notes.md | vl-kb add --title="..." --tags="..." --stdin

# Update entries
vl-kb update path/entry.md --tags="new,tags"
vl-kb update path/entry.md --file=updated-content.md

# Maintenance
vl-kb health                           # Audit for problems
vl-kb suggest-links path/entry.md      # Find related entries
vl-kb hubs                             # Most connected entries
vl-kb reindex                          # Rebuild search indices
```

### Agent Context with `vl-kb prime`

Get AI-optimized workflow context at session start:

```bash
vl-kb prime              # Full CLI reference (~1-2k tokens)
vl-kb prime --mcp        # Minimal output for MCP environments
vl-kb prime --json       # JSON format for programmatic use
```

Designed for Claude Code hooks (SessionStart, PreCompact) to prevent agents from forgetting KB workflow after context compaction.

## Knowledge Base Structure

Entries are stored in `kb/` with this category structure:

```
kb/
├── infrastructure/   # servers, networking, cloud
├── devops/          # CI/CD, monitoring, deployment
├── development/     # coding practices, languages
├── troubleshooting/ # problem solutions
├── architecture/    # system design, patterns
├── tooling/         # tools and utilities
└── projects/        # project-specific knowledge
```

Each entry is a Markdown file with YAML frontmatter:

```markdown
---
title: Kubernetes Pod Networking
tags: [kubernetes, networking, infrastructure]
created: 2024-01-15
---

# Kubernetes Pod Networking

Content with [[bidirectional links]] to other entries.
```

Use `[[path/to/entry.md|Display Text]]` for wiki-style links.

## MCP Tools Reference

For MCP-only environments, these tools are available:

| Tool | Description |
|------|-------------|
| `search` | Hybrid keyword + semantic search |
| `add` | Create new KB entry |
| `update` | Modify existing entry |
| `get` | Retrieve entry by path |
| `list` | List entries (optionally by category) |
| `whats_new` | Recently modified entries |
| `backlinks` | Find entries linking to a given entry |
| `suggest_links` | Find semantically related entries |
| `tags` | List all tags with counts |
| `health` | Audit KB for problems |
| `hubs` | Most connected entries |
| `reindex` | Rebuild search indices |

## Configuration

Environment variables (set automatically by plugin/devcontainer):

- `KB_ROOT` - Path to knowledge base directory (default: `${CLAUDE_PLUGIN_ROOT}/kb`)
- `INDEX_ROOT` - Path to search indices (default: `${CLAUDE_PLUGIN_ROOT}/.indices`)
- `KB_PRELOAD` - Set to `1` to preload embedding model at startup

## Prerequisites

- Python 3.11+
- `uv` package manager

Dependencies are installed automatically via `uv`.
