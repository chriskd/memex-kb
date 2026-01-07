---
title: MCP Support Audit - January 2026
tags: [mcp, audit, reference]
created: 2026-01-07
description: Audit results for memex MCP server implementation
---

# MCP Support Audit - January 2026

## Summary

| Metric | Result |
|--------|--------|
| **Status** | PASS |
| **Tools Tested** | 16/16 |
| **Tests Passed** | 88/88 |
| **Critical Bugs** | 0 |
| **Documentation Improvements** | 5 items |

## Identity Decision

MCP is positioned as a **minimal core** focused on search and CRUD operations:

- **For AI Agents**: Use CLI (`mx`) - ~0 token overhead
- **For Claude Desktop**: Use MCP - structured JSON, no shell required

CLI intentionally has more features (sessions, templates, batch operations, publishing) because these are better suited to shell workflows.

## Tool Verification Results

All 16 MCP tools pass comprehensive testing:

| Tool | Status | Test Coverage |
|------|--------|---------------|
| search | PASS | modes, tags, content, limits, empty results |
| add | PASS | directories, links, duplicates, force, validation |
| update | PASS | content, tags, section_updates, timestamps |
| get | PASS | content, links, backlinks, errors |
| delete | PASS | basic, backlinks warning, force |
| list | PASS | category, directory, tag, limit, empty |
| whats_new | PASS | days, limit, project filter |
| backlinks | PASS | incoming links, no links |
| tree | PASS | path, depth |
| mkdir | PASS | nested directories |
| move | PASS | files, link updates |
| rmdir | PASS | empty, force |
| tags | PASS | counts, empty KB |
| health | PASS | orphans, broken links, stale |
| quality | PASS | accuracy metrics, cutoff |
| suggest_links | PASS | limit, min_score, excludes self |

## Input Validation

Comprehensive validation at MCP layer:

| Validation | Tools | Behavior |
|------------|-------|----------|
| Empty/whitespace title | add | ValidationError |
| Empty/whitespace content | add, update | ValidationError |
| Empty tags array | add, update | ValidationError |
| Path traversal (`..`) | all path-based | ValidationError |
| Absolute paths | all path-based | ValidationError |
| Empty section keys | update | ValidationError |

## CLI vs MCP Feature Matrix

### MCP Tools (16)

Core CRUD and search operations available via MCP.

### CLI-Only Features

These stay CLI-only by design:

| Feature | CLI Command | Rationale |
|---------|-------------|-----------|
| Session management | `mx session` | Shell workflow |
| Templates | `mx add --template` | Interactive creation |
| Batch operations | `mx batch` | File-based input |
| Quick-add | `mx quick-add` | Auto-detect metadata |
| Upsert by title | `mx upsert` | Search + append |
| Text patching | `mx patch` | Regex find/replace |
| Static publishing | `mx publish` | Build pipeline |
| Beads integration | `mx beads` | Issue tracking UI |
| Search history | `mx history` | Replay searches |
| Hubs analysis | `mx hubs` | Central concepts |
| Summarize | `mx summarize` | Extract summaries |
| Reindex | `mx reindex` | Rebuild indices |
| Prime context | `mx prime` | Agent injection |

### MCP-Only Features

| Feature | Tool | Description |
|---------|------|-------------|
| Section updates | `update` | Update named sections directly |
| Quality metrics | `quality` | Search accuracy testing |

## Documentation Changes Made

1. **MCP Philosophy section** - Explains minimal core positioning
2. **Tool parameter examples** - Full parameters with JSON examples
3. **Error handling documentation** - ValidationError and ValueError formats
4. **When to use MCP guidance** - Clear decision criteria
5. **Expanded troubleshooting** - Server, tool, and search issues

## Recommendations

1. **Maintain minimal MCP** - Do not add CLI-only features to MCP
2. **Update docs on changes** - When `core.py` changes, update `mcp-setup.md`
3. **Run tests before releases** - `uv run pytest tests/test_mcp*.py`

## Test Command

```bash
uv run pytest tests/test_mcp_tools.py tests/test_mcp_validation.py -v
```

## See Also

- [[guides/mcp-setup|MCP Setup Guide]]
- [[guides/ai-integration|AI Agent Integration]]
- [[reference/cli|CLI Reference]]
