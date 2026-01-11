---
title: Memex Knowledge Base
tags:
  - infrastructure
  - plugins
  - documentation
  - claude-code
  - mcp
  - memex
created: 2025-12-20
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
---

# Memex Knowledge Base

Memex is a knowledge management system integrated with Claude Code. It provides persistent, searchable documentation for patterns, decisions, guides, and institutional knowledge.

## Purpose

- **Capture institutional knowledge** - Document decisions, patterns, and lessons learned so they persist across sessions and team members
- **Enable semantic search** - Find relevant information using natural language queries, not just keyword matching
- **Build organizational memory** - Create a living reference that grows with the organization

## Core Features

### Hybrid Search
Combines keyword matching with semantic understanding. Search for concepts, not just exact phrases.

### Bidirectional Links
Entries can reference each other using wiki-style `[[link]]` syntax, creating a connected knowledge graph.

### Tagging System
Consistent taxonomy for categorizing and filtering entries.

### MCP Integration
Exposed as an MCP server, making it accessible to Claude Code sessions automatically.

## Available Operations

| Tool | Purpose |
|------|---------|
| `search` | Find entries using hybrid keyword + semantic search |
| `add` | Create new knowledge base entries |
| `update` | Modify existing entries |
| `get` | Retrieve full entry content |
| `list` | Browse entries by category or tag |
| `backlinks` | Find entries linking to a specific entry |
| `reindex` | Rebuild search indices |

## Best Practices

1. **Search before creating** - Avoid duplicates by checking existing entries
2. **Use consistent tags** - Follow the existing taxonomy
3. **Add bidirectional links** - Connect related concepts
4. **Keep entries focused** - One concept per entry works better than sprawling docs

## Technical Details

The plugin uses a markdown-based storage format with YAML frontmatter for metadata. Search indices support both traditional keyword matching and vector-based semantic similarity.