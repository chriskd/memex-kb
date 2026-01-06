---
title: MCP Server Setup
tags: [mcp, claude-desktop, setup, configuration]
created: 2026-01-06
description: Configure memex as an MCP server for Claude Desktop
---

# MCP Server Setup

Memex provides a Model Context Protocol (MCP) server for integration with Claude Desktop and other MCP-compatible tools.

## Basic Configuration

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

## Configuration Locations

**Claude Desktop (macOS):**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Claude Desktop (Windows):**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**Claude Code:**
```
.claude/settings.local.json
```

## Environment Variables

Configure the KB location via environment:

```json
{
  "mcpServers": {
    "memex": {
      "type": "stdio",
      "command": "memex",
      "env": {
        "MEMEX_KB_ROOT": "/path/to/kb",
        "MEMEX_INDEX_ROOT": "/path/to/indices"
      }
    }
  }
}
```

## Available MCP Tools

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
| `rmdir` | Remove empty directory |
| `quality` | Detailed quality report |

## Preloading Embedding Model

For faster first search, preload the embedding model:

```json
{
  "mcpServers": {
    "memex": {
      "type": "stdio",
      "command": "memex",
      "env": {
        "MEMEX_PRELOAD": "true"
      }
    }
  }
}
```

## CLI vs MCP

| Aspect | CLI (`mx`) | MCP Server |
|--------|------------|------------|
| Token cost | ~0 (shell command) | ~500+ (tool schema) |
| Setup | Just install | Configure mcpServers |
| Output | Human-readable | Structured JSON |
| Best for | Most agents | Claude Desktop app |

**Recommendation:** Use CLI for AI agents unless you specifically need MCP's structured responses.

## Troubleshooting

### Server not starting

1. Verify memex is installed: `memex --version`
2. Check KB path exists: `ls $MEMEX_KB_ROOT`
3. Test manually: `echo '{}' | memex`

### Search not returning results

1. Verify entries exist: `mx list`
2. Rebuild indices: `mx reindex`
3. Check search mode: semantic requires `[semantic]` extras

## See Also

- [[guides/installation|Installation Guide]]
- [[guides/ai-integration|AI Agent Integration]]
- [[reference/cli|CLI Reference]]
