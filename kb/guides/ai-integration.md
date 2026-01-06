---
title: AI Agent Integration
tags: [ai, claude-code, codex, agents, integration]
created: 2026-01-06
description: How to use memex with AI coding assistants
---

# AI Agent Integration

Memex is designed for AI coding assistants. The CLI is the recommended interface - it uses ~0 tokens vs MCP's schema overhead.

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
    "SessionStart": [
      { "command": "mx prime" }
    ]
  }
}
```

The `mx prime` command:
- Injects KB workflow guidance at session start
- Auto-detects MCP vs CLI mode
- Adapts output format accordingly

### Workflow Pattern

```bash
# Before implementing: search KB for existing patterns
mx search "authentication patterns"

# During work: add discoveries for future sessions
mx add --title="OAuth2 Setup" --tags="auth,patterns" --category=patterns \
  --content="..."

# Track progress: update project session log
mx session-log --message="Implemented OAuth2 flow"
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

# View recent project updates
mx whats-new --project=myapp --days=7

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

## Project Context

Set up project-specific KB context with `.kbcontext`:

```bash
# In your project directory
mx context init
```

This creates a `.kbcontext` file that:
- Routes new entries to `projects/<name>` by default
- Boosts project entries in search results
- Suggests project-specific tags

## Session Management

Track work across sessions:

```bash
# Start a session with context
mx session start --tags=infrastructure --project=myapp

# Log session activity
mx session-log --message="Fixed auth bug, added tests"

# Clear session context
mx session clear
```

## Best Practices

1. **Search before creating** - Avoid duplicate entries
2. **Tag consistently** - Use `mx tags` to see existing tags
3. **Link related entries** - Use `[[path/to/entry]]` syntax
4. **Keep entries focused** - One topic per entry
5. **Update, don't duplicate** - Append to existing entries

## See Also

- [[reference/cli|CLI Reference]]
- [[guides/mcp-setup|MCP Server Setup]]
- [[reference/entry-format|Entry Format]]
