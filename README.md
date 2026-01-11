# Memex

Knowledge base for AI agents with hybrid search and context injection.

## Why Memex?

AI agents lose context. Memex solves this:

- **Hook integration** - Auto-inject KB awareness at session start/compact recovery
- **Zero-token CLI** - `mx` uses ~0 tokens vs MCP's ~500+ token schema overhead
- **Agent-optimized output** - `--json`, `--json-errors`, `--terse` modes for programmatic use
- **Edit tool integration** - `mx patch`, `mx update --append` for surgical updates

### Before/After: Agent Workflow

```
Without memex:
  Agent starts → Searches codebase → Rediscovers same patterns → Forgets on compaction

With memex:
  Agent starts → SessionStart hook runs `mx prime` → Agent knows to check KB first
  Agent discovers pattern → Adds to KB → Future agents find it immediately
  Context compacts → PreCompact hook preserves KB awareness
```

## Quick Start

```bash
# Install
uv tool install memex-kb     # Keyword search only (~100MB)
uv tool install "memex-kb[semantic]"  # + Semantic search (~600MB)

# Initialize
mkdir -p kb && export MEMEX_KB_ROOT=$(pwd)/kb
mx init

# Use
mx add --title="Auth Pattern" --tags="auth,patterns" --content="OAuth2 flow..."
mx search "authentication"
mx get patterns/auth-pattern.md
```

See [TUTORIAL.md](TUTORIAL.md) for complete walkthrough.

## Agent Integration

### Claude Code Hooks (Recommended)

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

**`mx prime` output** (injected at session start):

```
# Memex Knowledge Base

> Search organizational knowledge before reinventing. Add discoveries for future agents.

**Use `mx` CLI instead of MCP tools** - CLI uses ~0 tokens vs MCP schema overhead.

## Session Protocol

**Before searching codebase**: Check if memex has project patterns first
  `mx search "deployment"` or `mx whats-new --project=<project>`

**After discovering patterns**: Consider adding to KB for future agents
  `mx add --title="..." --tags="..." --content="..."`

## CLI Quick Reference
mx search "query"     # Hybrid keyword + semantic search
mx get path/entry.md  # Read entry
mx add --title="..." --tags="..." --content="..."
mx whats-new --days=7 # Recent changes
```

The command auto-detects MCP mode and adapts output accordingly.

### Other Agents (Codex, etc.)

Any agent with shell access can use `mx`:

```bash
# Check KB before implementing
mx search "rate limiting"

# Add discoveries
mx add --title="Redis Caching" --tags="redis,patterns" --content="..."

# Project-aware recent changes
mx whats-new --project=myapp --days=7
```

## CLI Reference

```bash
# Search
mx search "deployment"              # Hybrid search
mx search "docker" --tags=infra     # Filter by tag
mx search "api" --mode=semantic     # Semantic only
mx search "api" --terse             # Paths only (agent-friendly)

# Read
mx get tooling/notes.md             # Full entry
mx get tooling/notes.md --metadata  # Just frontmatter
mx get tooling/notes.md --json      # Structured output

# Create/Update
mx add --title="Entry" --tags="a,b" --content="..."
mx update path.md --content="New section" --append --timestamp
mx upsert --title="Log" --content="Append this" --append
mx patch path.md --old="draft" --new="published"  # Surgical edit
mx quick-add --stdin                # Auto-generate metadata

# Browse
mx tree                             # Directory structure
mx list --tag=infrastructure        # Filter by tag
mx whats-new --days=7               # Recent changes
mx tags                             # All tags with counts
mx info                             # KB configuration
mx health                           # Audit for problems

# Project context
mx context init                     # Create .kbcontext
mx prime                            # Session context injection
mx prime --project=myapp -d 14      # Include recent project entries
```

### Agent-Optimized Flags

| Flag | Purpose |
|------|---------|
| `--json` | Structured output for parsing |
| `--json-errors` | Machine-readable errors with codes |
| `--terse` | Minimal output (paths only) |
| `--dry-run` | Preview changes safely |

## MCP Server

Memex MCP is a **minimal core** for Claude Desktop and tools without shell access:

```json
{
  "mcpServers": {
    "memex": { "type": "stdio", "command": "memex" }
  }
}
```

**For AI agents (Claude Code, etc.)**: Use the CLI instead. It has ~0 token overhead vs MCP's ~500+ token schema cost.

See [MCP Setup Guide](kb/guides/mcp-setup.md) for configuration details and tool documentation.

## Installation

### Minimal (Keyword Search)

```bash
uv tool install memex-kb    # or: pip install memex-kb
mx --version
```

### Full (Semantic Search)

```bash
uv tool install "memex-kb[semantic]"
```

Adds ~500MB (ChromaDB, sentence-transformers). First search downloads embedding model (~100MB).

### From Source

```bash
git clone https://github.com/chriskd/memex.git
cd memex
uv sync                    # Core only (~100MB)
uv sync --all-extras       # With semantic (~600MB)
```

### GPU Support (Optional)

Requires source install. Edit `pyproject.toml` to change `pytorch-cpu` to `pytorch-gpu` in `[tool.uv.sources]` and update the index URL to `https://download.pytorch.org/whl/cu124`, then:

```bash
uv sync --all-extras
```

## Entry Format

```markdown
---
title: My Knowledge Entry
tags: [topic, category]
created: 2025-01-01
---

# My Knowledge Entry

Content with [[bidirectional links]] to other entries.
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMEX_KB_ROOT` | KB directory | `./kb` |
| `MEMEX_INDEX_ROOT` | Index directory | `./.indices` |
| `MX_JSON_ERRORS` | Always use JSON errors | `false` |

## Development

```bash
git clone https://github.com/chriskd/memex.git
cd memex && uv sync --dev
uv run pytest
uv run ruff check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT - see [LICENSE](LICENSE)
