---
title: AI Agent Integration
description: How to use memex with AI coding assistants
tags:
  - ai
  - claude-code
  - codex
  - agents
  - integration
created: 2026-01-06T00:00:00
updated: 2026-01-27T00:00:00+00:00
semantic_links: []
---

# AI Agent Integration

Memex is designed for AI coding assistants. The CLI is the recommended interface.

## Claude Code

### Permission Setup

Add to `.claude/settings.local.json`:

```json
{
  "permissions": {
    "allow": ["Bash(mx:*)"]
  }
}
```

This grants Claude Code permission to run any `mx` command without prompting.

### Session Hooks

For automatic context injection, use hooks:

```json
{
  "hooks": {
    "SessionStart": [{ "hooks": [{ "type": "command", "command": "mx session-context" }] }]
  }
}
```

Write the hook into `.claude/settings.json`:

```bash
mx session-context --install
```

The `mx session-context` command:
- Injects project-relevant KB context at session start
- Provides a concise workflow reminder plus related entries

### Claude Code Plugin (Optional)

This repo ships a Claude Code plugin manifest in `.claude-plugin/`. From the repo
root, add the local marketplace and install the plugin:

```text
/plugin marketplace add ./.claude-plugin/marketplace.json
/plugin install memex@memex
```

Restart Claude Code after installing or updating the plugin.

### Workflow Pattern

```bash
# Before implementing: search KB for existing patterns
mx search "authentication patterns"

# During work: add discoveries for future sessions
mx add --title="OAuth2 Setup" --tags="auth,patterns" --category=patterns \
  --content="..."
```

## Codex CLI

Codex can use memex via shell commands in AGENTS.md:

```markdown
## Knowledge Base

Search organizational knowledge before implementing:
- `mx search "query"` - Find existing patterns
- `mx get path/entry.md` - Read specific entry
- `mx add --title="..." --tags="..." --category=... --content="..."` - Add discoveries
```

### Codex Skills (Optional)

This repo includes a Memex skill at `skills/kb-usage/`. Codex CLI discovers skills
from well-known directories (for example `.codex/skills` in the repo or
`~/.codex/skills` for user installs). Team Config can also load skills from
`/etc/codex/skills` when centrally managed. Copy or symlink the skill into a
Codex skills directory and restart Codex.

Examples:

```bash
mkdir -p .codex/skills
cp -r skills/kb-usage .codex/skills/

mkdir -p ~/.codex/skills
cp -r skills/kb-usage ~/.codex/skills/

sudo mkdir -p /etc/codex/skills
sudo cp -r skills/kb-usage /etc/codex/skills/
```

## Other AI Agents

Any agent with shell access can use the `mx` CLI.

### Common Patterns

```bash
# Check for relevant knowledge before implementing
mx search "deployment strategies"

# Add discoveries for future sessions
mx add --title="API Rate Limiting" \
  --tags="api,patterns" \
  --category=patterns \
  --content="..."

# View recent project KB updates
mx whats-new --scope=project --days=7

# Quick status check
mx info
```

### Search Strategy

1. **Before implementing**: Search for existing patterns
2. **When stuck**: Search for troubleshooting guides
3. **After solving**: Add solution to KB

### When to Search KB

- Looking for organizational patterns or guides
- Before implementing something that might exist
- Understanding infrastructure or deployment
- Troubleshooting known issues

### When to Contribute

- Discovered reusable pattern or solution
- Troubleshooting steps worth preserving
- Infrastructure or deployment knowledge
- Project-specific conventions

## Project Configuration

Configure project-specific KB settings in `.kbconfig`:

```bash
# In your project directory
cat <<'EOF' > .kbconfig
kb_path: ./kb
primary: projects/memex
boost_paths:
  - projects/memex/*
default_tags:
  - memex
EOF
```

This `.kbconfig` file:
- Points `mx` at the project KB (`kb_path`)
- Routes new entries to `projects/<name>` by default (`primary`)
- Boosts project entries in search results (`boost_paths`)
- Suggests project-specific tags (`default_tags`)

## Best Practices

1. **Search before creating** - Avoid duplicate entries
2. **Tag consistently** - Use `mx tags` to see existing tags
3. **Link related entries** - Use `[[path/to/entry]]` syntax
4. **Keep entries focused** - One topic per entry
5. **Update, don't duplicate** - Append to existing entries

## See Also

- [[reference/cli|CLI Reference]]
- [[reference/entry-format|Entry Format]]
