---
title: AI Agent Integration
tags: [ai, claude-code, codex, agents, integration]
created: 2026-01-06
updated: 2026-01-07
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

For automatic context injection, configure both SessionStart and PreCompact hooks:

```json
{
  "hooks": {
    "SessionStart": [
      { "command": "mx prime" }
    ],
    "PreCompact": [
      { "command": "mx prime --compact" }
    ]
  }
}
```

**SessionStart** runs when a new session begins:
- Injects full KB workflow guidance (~1-2k tokens)
- Auto-detects MCP vs CLI mode and adapts output
- Includes recent project entries if `.kbcontext` is present

**PreCompact** runs before Claude's context is summarized:
- Re-injects minimal KB reminders into the compacted context
- Prevents agents from "forgetting" KB workflow after compaction
- Uses `--compact` flag for minimal token usage (~50 tokens)

### Context Preservation Across Compaction

When Claude's context window fills up, it compacts (summarizes) older conversation. Without PreCompact hooks, agents often forget about memex entirely after compaction.

The PreCompact hook solves this by injecting a brief reminder:

```
# Memex KB Active

**Session Start**: Search KB before implementing: `mx search "query"`
**Session End**: Consider adding discoveries: `mx add --title="..." --tags="..."`

Quick: search | get | add | tree | whats-new | health
```

This ensures agents continue using the KB throughout long sessions.

### The `mx prime` Command

The prime command adapts its output based on context:

```bash
mx prime                    # Auto-detect mode (full if CLI, minimal if MCP)
mx prime --full             # Force full output (~1-2k tokens)
mx prime --compact          # Force minimal output (~50 tokens)
mx prime --project=myapp    # Include recent entries for project
mx prime -p myapp -d 14     # Last 14 days of project changes
```

**Full output** (SessionStart) includes:
- Complete CLI quick reference
- Search and contribution guidelines
- Entry format documentation

**Compact output** (PreCompact) includes:
- Brief workflow reminder
- Essential command list

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

## Typical Agent Session Workflow

A complete agent session with memex follows this pattern:

### 1. Session Start (Automatic)

The SessionStart hook runs `mx prime`, which:
- Outputs KB workflow guidance
- Shows recent project entries (if `.kbcontext` exists)
- Reminds agent to search before implementing

### 2. Before Implementation

```bash
# Check for existing patterns before writing code
mx search "authentication"
mx search "api rate limiting" --tags=patterns

# Review project-specific knowledge
mx whats-new --project=myapp --days=7
```

### 3. During Work

```bash
# Quick reference while implementing
mx get patterns/oauth2.md

# Log progress to session entry
mx session-log -m "Implemented OAuth2 flow"
```

### 4. After Discovery

```bash
# Add reusable patterns for future agents
mx add --title="OAuth2 Setup" \
  --tags="auth,patterns" \
  --category=patterns \
  --content="..."

# Or append to existing entry
mx upsert "Redis Troubleshooting" \
  --content="## New Finding\n\nDetails..."
```

### 5. Context Recovery (After Compaction)

When context compacts, the PreCompact hook re-injects KB reminders. The agent should:

```bash
# Recover project context
mx whats-new --project=myapp --days=7

# Check session log for previous progress
mx get projects/myapp/sessions.md
```

### 6. Session End

```bash
# Log final status
mx session-log -m "Completed: auth module. TODO: tests"

# Ensure knowledge is captured
mx health  # Check for issues
```

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

## Agent Workflow Examples

### Session Logging Pattern

Maintain a continuous log of work for context recovery and progress tracking:

```bash
# At session start - create/append to session log
mx session-log -m 'Started work on feature X' --entry=projects/myapp/sessions.md

# During work - log progress with links to related entries
mx session-log -m 'Implemented auth module' --tags=auth,progress --links='patterns/oauth2.md'

# At session end - summarize completed and remaining work
mx session-log -m 'Completed: auth module. TODO: tests' --links='tooling/testing.md'
```

The session log creates a persistent record that:
- Helps agents recover context after interruptions
- Documents decisions and progress for future sessions
- Links to related KB entries for context

### Incremental Knowledge Capture

Build knowledge entries incrementally as you discover information:

```bash
# First encounter - create entry with initial knowledge
mx upsert 'Redis Troubleshooting' \
  --content='## Connection Timeouts

When Redis connections timeout, check:
1. Connection pool exhaustion
2. Network latency
3. Server load' \
  --tags=redis,ops,troubleshooting

# Later - add more to same entry (appends by default)
mx upsert 'Redis Troubleshooting' \
  --content='## Memory Issues

Redis memory issues typically stem from:
1. Large key accumulation
2. Missing TTLs on temporary data
3. Fragmentation'

# Replace content instead of appending
mx upsert 'Redis Troubleshooting' \
  --content='Complete updated guide...' \
  --replace
```

Benefits for agents:
- No need to read-modify-write for simple additions
- Automatic deduplication prevents duplicate entries
- `--replace` mode for complete rewrites

### Surgical Content Updates

Make precise edits to existing entries without full rewrites:

```bash
# Preview before patching (safe mode)
mx patch guides/deployment.md \
  --old 'status: draft' \
  --new 'status: published' \
  --dry-run

# Apply the change
mx patch guides/deployment.md \
  --old 'status: draft' \
  --new 'status: published'

# Replace all occurrences (e.g., rename a term)
mx patch reference/api.md \
  --old 'getUserData' \
  --new 'fetchUserProfile' \
  --replace-all

# Create backup before modifying
mx patch critical-doc.md \
  --old 'old value' \
  --new 'new value' \
  --backup
```

Use patch when:
- Making targeted changes to large entries
- Updating status fields or metadata
- Renaming terms across a document
- Fixing typos or updating outdated info

### Context-Aware Operations

Leverage `.kbcontext` for project-scoped operations:

```bash
# Initialize project context (run once per project)
mx context init  # Creates .kbcontext in current directory

# With .kbcontext present, entries auto-route to project category
mx add --title="API Endpoints" --tags=api --content="..."
# → Creates at projects/myapp/api-endpoints.md

# Search with project boosting
mx search "authentication"
# → Project entries rank higher in results

# View project-specific recent changes
mx whats-new --project=myapp --days=7
```

### Batch Operations

Process multiple operations efficiently:

```bash
# Add multiple entries from a directory
for f in discovered-patterns/*.md; do
  mx add --title="$(basename $f .md)" --tags=patterns --file="$f"
done

# Batch tag updates
mx search "kubernetes" --json | \
  jq -r '.[].path' | \
  xargs -I {} mx update {} --tags=k8s,infrastructure

# Export entries matching a query
mx search "deployment" --full-text | mx export --format=json
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
