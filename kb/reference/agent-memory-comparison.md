---
title: Agent Memory Comparison
tags:
  - memory
  - comparison
  - basic-memory
  - claude-mem
created: 2026-01-13T01:25:36.081714+00:00
updated: 2026-01-15T00:56:26.823073+00:00
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
semantic_links:
  - path: a-mem-parity/a-mem-parity-analysis.md
    score: 0.701
    reason: bidirectional
  - path: a-mem-parity/memory-evolution-queue-architecture.md
    score: 0.606
    reason: bidirectional
  - path: reference/memory-evolution-queue-architecture.md
    score: 0.629
    reason: bidirectional
---

# Agent Memory Tool Comparison

Comparison of memex memory with basic-memory and claude-mem (January 2026).

## Feature Matrix

| Aspect | **Memex Memory** | **Basic-Memory** | **Claude-Mem** |
|--------|-----------------|------------------|----------------|
| **Storage** | Markdown files in `kb/sessions/` | Markdown + SQLite | SQLite + Chroma vector DB |
| **Format** | Per-day `.md` files | Semantic markdown with relations | Structured database tables |
| **Dependencies** | Just \`anthropic\` package | Python 3.12+, MCP server | Node.js, Bun, SQLite, Chroma |
| **Setup** | \`mx memory init\` | MCP server config in Claude settings | Plugin install + background service |
| **Capture** | Stop/PreCompact hooks â Haiku | Manual via MCP tools | All hooks (every tool use captured) |
| **Injection** | SessionStart â ~1000 tokens | \`build_context\` with \`memory://\` URLs | SessionStart â configurable limit |
| **Search** | Sequential read (no index) | SQLite + pattern matching | Hybrid semantic + FTS5 |
| **Knowledge Graph** | Wikilinks (existing KB) | Built-in relations & traversal | Observation hierarchy |
| **UI** | CLI only | Works with Obsidian | Web viewer on port 37777 |
| **Privacy** | Local markdown, git-trackable | Local-first, optional cloud | Local with \`<private>\` tags |
| **Token Cost** | ~$0.001/capture (Haiku) | None (no LLM for storage) | Higher (processes all tool use) |

## Key Differentiators

### Memex Memory (lightest)
- Zero infrastructure - just markdown files
- Git-trackable sessions
- Leverages existing KB system
- Minimal capture (session end only)
- ~700 lines of Python

### Basic-Memory (knowledge graph)
- Rich semantic linking between documents
- \`memory://\` URL scheme for graph traversal
- MCP tools for structured CRUD operations
- Works alongside Obsidian
- Typed relations: \`implements\`, \`depends_on\`, \`extends\`

### Claude-Mem (most comprehensive)
- Captures everything (every tool use)
- Vector search for semantic retrieval
- Web UI for visualization
- Background service architecture
- ~4k GitHub stars, active development
- Configurable observation types and concepts

## When to Use What

| Use Case | Recommendation |
|----------|----------------|
| Simple, git-tracked memory | **Memex** |
| Building connected knowledge graph | **Basic-Memory** |
| Using Obsidian for notes | **Basic-Memory** |
| Want comprehensive capture + semantic search | **Claude-Mem** |
| Minimal dependencies | **Memex** |
| Don't mind complexity for features | **Claude-Mem** |

## Architecture Comparison

### Memex Memory
```
SessionStart Hook â mx memory inject â read kb/sessions/*.md â output context
Stop/PreCompact Hook â mx memory capture â read conversation â Haiku â write session file
```

### Basic-Memory
```
User Request â MCP Tool (write_note, search_notes, build_context) â SQLite + Markdown files
memory:// URLs enable graph traversal across linked documents
```

### Claude-Mem
```
All Hooks â Worker Service (port 37777) â SQLite + Chroma
SessionStart â inject from session_summaries table
PostToolUse â capture to observations table
Stop â generate summary via Claude SDK
```

## Observation Categories

| Memex | Basic-Memory | Claude-Mem |
|-------|--------------|------------|
| \`[learned]\` | \`[tech]\` | \`discovery\` |
| \`[decision]\` | \`[decision]\` | \`decision\` |
| \`[pattern]\` | \`[design]\` | \`pattern\` |
| \`[issue]\` | \`[feature]\` | \`bugfix\` |
| \`[todo]\` | - | \`feature\`, \`refactor\` |

## See Also

- [[guides/agent-memory]] - Memex memory setup guide
- [Basic-Memory GitHub](https://github.com/basicmachines-co/basic-memory)
- [Claude-Mem GitHub](https://github.com/thedotmack/claude-mem)