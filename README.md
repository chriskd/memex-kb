# Memex

Personal knowledge base with hybrid search (keyword + semantic).

## Features

- **Hybrid search** - Keyword (Whoosh) + semantic (ChromaDB) search
- **CLI tool** - `mx` command for terminal and agent workflows
- **Bidirectional links** - Obsidian-style `[[links]]` with backlink tracking

## Installation

```bash
# With uv (recommended)
uv tool install memex-kb

# With pip
pip install memex-kb

# Verify
mx --version
```

For semantic search, install with extras: `pip install memex-kb[semantic]`

## Quick Start

```bash
# Initialize a KB in your project
mx init

# Add an entry
mx add --title="Setup Guide" --tags="docs" --content="# Setup\n\nInstructions here."

# Search
mx search "setup"

# Read an entry
mx get guides/setup-guide.md
```

This creates a `kb/` directory for entries and `.kbconfig` at your project root.

## Project and User KBs

Memex supports two knowledge base locations:

| Location | Created with | Use for |
|----------|--------------|---------|
| **Project** (`./kb/`) | `mx init` | Team docs, project-specific knowledge. Commit to git. |
| **User** (`~/.memex/kb/`) | `mx init --user` | Personal notes, available across all projects. |

By default, searches include both KBs. Results show prefixes when both exist:
- `@project/guides/setup.md`
- `@user/notes/ideas.md`

### Additive Scope (Default)

By default, operations span **both** project and user KBs:

```bash
# Search finds entries from project KB AND user KB
mx search "deployment"

# Restrict to project KB only
mx search "deployment" --scope=project
mx list --scope=project
```

Results from different KBs use scope prefixes when both exist:
- `@project/guides/setup.md` - Entry from project KB
- `@user/personal/notes.md` - Entry from user KB

### Explicit Scope for Writes

When adding entries, use `--scope` to explicitly choose which KB:

```bash
# Add to project KB (shared with team)
mx add --title="API Guide" --tags="api" --scope=project --content="..."

# Add to user KB (personal notes)
mx add --title="My Notes" --tags="personal" --scope=user --content="..."

# Auto-detect (default): project KB if in project, else user KB
mx add --title="Note" --tags="test" --content="..."
```

**When to use each scope:**

| Scope | Use For |
|-------|---------|
| `project` | Team knowledge, infra docs, shared patterns, API docs |
| `user` | Personal notes, experiments, drafts, individual workflow tips |

## CLI Reference

```bash
# Search
mx search "query"                  # Hybrid search
mx search "api" --tags=docs        # Filter by tag
mx search "api" --mode=semantic    # Semantic only

# Read
mx get path/to/entry.md            # Full entry
mx get path/to/entry.md --metadata # Metadata only

# Browse
mx tree                            # Directory structure
mx list --tag=docs                 # Filter by tag
mx whats-new --days=7              # Recent changes
mx tags                            # All tags

# Write
mx add --title="Title" --tags="a,b" --content="..."
mx replace path/entry.md --content="new content"
mx patch path/entry.md --find="old" --replace="new"

# Maintenance
mx health                          # Audit KB
mx reindex                         # Rebuild indices
```

## Claude Code Integration

Add to `.claude/settings.local.json`:

```json
{
  "permissions": { "allow": ["Bash(mx:*)"] },
  "hooks": {
    "SessionStart": [{ "hooks": [{ "type": "command", "command": "mx prime" }] }]
  }
}
```

## Configuration

Project config lives in `.kbconfig` at your project root:

```yaml
kb_path: ./kb              # Required: path to KB directory
default_tags: [myproject]  # Tags suggested when adding entries
boost_paths: [guides/*]    # Prioritize in search results
primary: guides            # Default directory for new entries
```

Environment variables:

| Variable | Description |
|----------|-------------|
| `MEMEX_KB_ROOT` | Override KB discovery |
| `MEMEX_INDEX_ROOT` | Index directory (default: `{kb}/.indices`) |

## Development

```bash
git clone https://github.com/chriskd/memex.git
cd memex && uv sync --dev
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT - see [LICENSE](LICENSE)
