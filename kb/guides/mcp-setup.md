---
title: MCP Server Setup
tags: [mcp, claude-desktop, setup, configuration]
created: 2026-01-06
description: Configure memex as an MCP server for Claude Desktop
---

# MCP Server Setup

Memex provides a Model Context Protocol (MCP) server for integration with Claude Desktop and other MCP-compatible tools.

## MCP Philosophy

Memex MCP is a **minimal core** focused on search and CRUD operations. This is intentional:

- **For AI Agents**: Use the CLI (`mx`). It has ~0 token overhead vs MCP's ~500+ token schema cost.
- **For Claude Desktop**: Use MCP. It provides structured JSON responses without shell access.

The CLI has additional features (sessions, templates, batch operations, publishing) that are not exposed via MCP because they are better suited to shell workflows. See [[reference/mcp-audit-2026-01|MCP Audit]] for the full feature matrix.

## When to Use MCP

| Use Case | Recommended | Why |
|----------|-------------|-----|
| Claude Desktop app | MCP | No shell access available |
| Claude Code / AI agents | CLI | ~0 tokens vs ~500+ for MCP schemas |
| IDE extensions | MCP | Structured JSON responses |
| Automation scripts | CLI | Shell pipelines, batch operations |
| Need sessions/templates | CLI | Not available in MCP |

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

### search

Hybrid keyword + semantic search.

**Parameters:**
- `query` (string, required): Search query
- `limit` (int, default: 10): Maximum results
- `mode` (string, default: "hybrid"): `"hybrid"` | `"keyword"` | `"semantic"`
- `tags` (array): Filter by tags
- `include_content` (bool, default: false): Include full content
- `strict` (bool, default: false): No semantic fallback

**Example:**
```json
{"query": "authentication", "limit": 5, "tags": ["patterns"]}
```

---

### add

Create a new KB entry with duplicate detection.

**Parameters:**
- `title` (string, required): Entry title
- `content` (string, required): Markdown content
- `tags` (array, required): At least one tag
- `category` (string): Top-level category
- `directory` (string): Full directory path (overrides category)
- `links` (array): Paths to link to
- `force` (bool, default: false): Bypass duplicate detection

**Example:**
```json
{
  "title": "OAuth2 Setup",
  "content": "# OAuth2\n\nConfiguration steps...",
  "tags": ["auth", "patterns"],
  "category": "development"
}
```

---

### update

Update an existing entry.

**Parameters:**
- `path` (string, required): Entry path (e.g., "development/oauth2.md")
- `content` (string): New markdown content
- `tags` (array): New tag list
- `section_updates` (object): Section heading → new content

**Example (section update):**
```json
{
  "path": "development/oauth2.md",
  "section_updates": {"Configuration": "Updated config steps..."}
}
```

---

### get

Retrieve an entry with content, links, and backlinks.

**Parameters:**
- `path` (string, required): Entry path

**Response includes:** path, metadata (title, tags, created, updated), content, links, backlinks

---

### delete

Delete an entry.

**Parameters:**
- `path` (string, required): Entry path
- `force` (bool, default: false): Delete even if other entries link to it

---

### list

List entries with optional filters.

**Parameters:**
- `category` (string): Filter by category
- `directory` (string): Filter by directory path
- `tag` (string): Filter by tag
- `limit` (int, default: 20): Maximum entries

---

### whats_new

Recently created or modified entries.

**Parameters:**
- `days` (int, default: 30): Look-back period
- `limit` (int, default: 10): Maximum entries
- `project` (string): Filter by source project

---

### tree

Directory structure.

**Parameters:**
- `path` (string, default: ""): Starting path
- `depth` (int, default: 3): Maximum depth

---

### tags

List all tags with usage counts.

**Parameters:** None

---

### backlinks

Find entries that link to a path.

**Parameters:**
- `path` (string, required): Entry path

---

### suggest_links

Find semantically related entries.

**Parameters:**
- `path` (string, required): Entry path
- `limit` (int, default: 5): Maximum suggestions
- `min_score` (float, default: 0.5): Minimum similarity (0-1)

---

### health

Audit KB for problems.

**Parameters:**
- `stale_days` (int, default: 90): Threshold for stale entries

**Returns:** orphans, broken_links, stale, empty_dirs, parse_errors, summary

---

### quality

Evaluate search accuracy.

**Parameters:**
- `limit` (int, default: 5): Number of test queries
- `cutoff` (int, default: 3): Ranking threshold

---

### mkdir

Create a directory.

**Parameters:**
- `path` (string, required): Directory path

---

### move

Move or rename an entry/directory.

**Parameters:**
- `source` (string, required): Current path
- `destination` (string, required): New path

**Returns:** moved files, links_updated count

---

### rmdir

Remove an empty directory.

**Parameters:**
- `path` (string, required): Directory path
- `force` (bool, default: false): Remove empty subdirectories recursively

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

## Error Handling

MCP tools return structured errors:

**ValidationError** (invalid input):
```json
{"error": "ValidationError", "message": "title cannot be empty or whitespace-only"}
```

**ValueError** (business logic):
```json
{"error": "ValueError", "message": "Entry not found: path/to/missing.md"}
```

### Common Errors

| Error Message | Cause | Resolution |
|---------------|-------|------------|
| `title cannot be empty` | Empty/whitespace title | Provide meaningful title |
| `tags cannot be empty` | Empty tags array | Provide at least one tag |
| `Entry not found` | Invalid path | Check path with `list` or `tree` |
| `path traversal not allowed` | Path contains `..` | Use relative path within KB |
| `absolute paths not allowed` | Path starts with `/` | Use relative path |
| `Entry has N backlink(s)` | Delete blocked | Use `force: true` or update linking entries |

## CLI vs MCP

| Aspect | CLI (`mx`) | MCP Server |
|--------|------------|------------|
| Token cost | ~0 (shell command) | ~500+ (tool schema) |
| Setup | Just install | Configure mcpServers |
| Output | Human-readable | Structured JSON |
| Sessions | Yes | No |
| Templates | Yes | No |
| Batch operations | Yes | No |
| Best for | AI agents, scripts | Claude Desktop |

## Troubleshooting

### Server Issues

| Problem | Diagnosis | Solution |
|---------|-----------|----------|
| Server won't start | `memex --version` fails | Reinstall: `uv tool install memex` |
| "KB root not found" | `$MEMEX_KB_ROOT` invalid | Check path: `ls $MEMEX_KB_ROOT` |
| Slow first search | Embedding model loading | Use `MEMEX_PRELOAD=true` |

### Tool Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Entry not found" | Path doesn't exist | Use `tree` or `list` to find entries |
| "Category not found" | Invalid category | Check `tree` for valid categories |
| "tags cannot be empty" | Empty tags array | Provide at least one tag |
| "title cannot be empty" | Whitespace-only title | Provide meaningful title |

### Search Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| No results | KB empty or unindexed | Run `mx reindex` via CLI |
| Unexpected results | Semantic fallback | Use `strict: true` for keyword-only |
| Stale results | Index out of date | Run `mx reindex` via CLI |

## See Also

- [[guides/installation|Installation Guide]]
- [[guides/ai-integration|AI Agent Integration]]
- [[reference/cli|CLI Reference]]
- [[reference/mcp-audit-2026-01|MCP Audit Report]]
