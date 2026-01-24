---
title: Agent Memory
tags:
  - agent-memory
  - hooks
  - claude-code
  - sessions
created: 2026-01-13T00:02:55.930011+00:00
updated: 2026-01-13
---

# Agent Memory

Automatic session memory capture and injection for Claude Code. Remembers what you worked on across sessions without manual effort.

## Quick Start

```bash
# 1. Enable memory for this project
mx memory init

# 2. That's it! Memory is now active.
#    - Sessions auto-captured when you exit
#    - Context auto-injected when you start
```

## Commands

| Command | Description |
|---------|-------------|
| `mx memory` | Show memory status |
| `mx memory init` | Enable memory for this project |
| `mx memory init --user` | Enable memory user-wide |
| `mx memory add "note"` | Add a manual memory note |
| `mx memory inject` | Preview/output session context |
| `mx memory capture` | Summarize current session |
| `mx memory disable` | Remove memory hooks |

### mx memory add

Appends a timestamped note to today's session file. Creates the file with frontmatter if it doesn't exist.

```bash
mx memory add "Fixed auth bug using refresh tokens"
mx memory add "Deployed v2.0" --tags=deployment,release
mx memory add --file=notes.md
mx memory add --stdin < notes.txt
mx memory add "Quick note" --no-timestamp
```

**What it writes:**
```markdown
## 2026-01-12 15:30 UTC

Fixed auth bug using refresh tokens

Tags: deployment, release
```

**Options:**
- `--tags` - Comma-separated tags appended to the entry
- `--file` - Read content from a file instead of argument
- `--stdin` - Read content from stdin (for piping)
- `--no-timestamp` - Skip the timestamp header
- `--json` - Output result as JSON

Use for quick notes during a session. For automatic summaries with observations, see `mx memory capture`.

### mx memory inject

Reads recent session files and outputs formatted context. Called automatically by the SessionStart hook to inject memory into new sessions.

```bash
mx memory inject          # Preview what agents see
mx memory inject --json   # Machine-readable output
```

Output is capped at ~1000 tokens and includes entries from the last 7 days.

### mx memory capture

Reads the current conversation from `~/.claude/projects/`, calls an LLM to extract a summary and observations, then writes to today's session file.

```bash
mx memory capture               # Manual capture
mx memory capture --event=stop  # Simulate stop hook
```

Called automatically by Stop and PreCompact hooks. Requires either `ANTHROPIC_API_KEY` or `OPENROUTER_API_KEY`.

## How It Works

### Automatic Capture (Stop/PreCompact)

When you end a session or context compacts:

1. Hook reads your conversation from `~/.claude/projects/`
2. Calls Claude haiku to extract structured observations
3. Writes to today's session file: `kb/sessions/2026-01-12.md`

**Observation categories extracted:**
- `[learned]` - New knowledge or insights
- `[decision]` - Choices made and why
- `[pattern]` - Recurring approaches or conventions
- `[issue]` - Problems encountered
- `[todo]` - Follow-up work identified

### Automatic Injection (SessionStart)

When you start a Claude Code session:

1. Hook calls `mx memory inject`
2. Reads recent session files from `kb/sessions/`
3. Formats ~1000 tokens of context
4. Outputs as system reminder

**Example injection:**
```
## Recent Memory (myproject)

**2026-01-12 15:30 UTC**
Fixed authentication bug in login flow

### Observations
- [learned] OAuth tokens expire after 1 hour, need refresh logic
- [decision] Use httpx instead of requests for async support

**2026-01-11 10:00 UTC**
Refactored database connection pooling

### Observations
- [pattern] Connection pools should be initialized once at startup
```

## Configuration

### .kbconfig

```yaml
# Required
kb_path: kb

# Memory settings (set by mx memory init)
session_dir: sessions              # Where session files go
session_retention_days: 30         # Auto-cleanup after N days
```

### LLM Provider

Memory capture uses an LLM to summarize sessions. Either provider works:

| Provider | API Key | Description |
|----------|---------|-------------|
| Anthropic | `ANTHROPIC_API_KEY` | Direct Anthropic API access |
| OpenRouter | `OPENROUTER_API_KEY` | Multi-model gateway (supports Anthropic, OpenAI, etc.) |

If both keys are set, specify which provider to use in `.kbconfig`:

```yaml
llm:
  provider: anthropic  # or "openrouter"
  model: claude-3.5-haiku  # optional model override
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | One of these | API key for Anthropic |
| `OPENROUTER_API_KEY` | required | API key for OpenRouter |
| `CLAUDE_PROJECT_DIR` | Auto | Set by Claude Code hooks |
| `CLAUDE_SESSION_ID` | Auto | Set by Claude Code hooks |

## Session Files

Sessions are stored as per-day markdown files:

```
kb/sessions/
  2026-01-12.md
  2026-01-11.md
  2026-01-10.md
```

Each file contains multiple session entries:

```markdown
---
title: Session Log 2026-01-12
tags: [sessions, memory]
created: 2026-01-12T10:00:00
---

# Session Log - 2026-01-12

## 2026-01-12 10:30 UTC

Implemented user authentication with OAuth2.

### Observations
- [learned] OAuth tokens need refresh logic
- [decision] Using httpx for async HTTP
- [pattern] Store tokens in httponly cookies

### Files
- src/auth.py
- tests/test_auth.py

## 2026-01-12 15:00 UTC

Fixed rate limiting bug in API.
...
```

## Troubleshooting

### Check Status

```bash
mx memory status
```

Shows:
- Whether hooks are installed
- Session directory configuration
- API key status

### "Hooks not installed"

Run `mx memory init` to install hooks.

### "API key not set"

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Preview Injection

```bash
mx memory inject
```

Shows what context would be injected at session start.

## Architecture

```
SessionStart Hook
       │
       ▼
┌─────────────────┐
│ mx memory inject│ ─── reads ──▶ kb/sessions/*.md
└────────┬────────┘
         │
         ▼
  System Reminder
  (injected ctx)


Stop/PreCompact Hook
       │
       ▼
┌──────────────────┐      ┌─────────────┐
│ mx memory capture│ ───▶ │Claude Haiku │
└────────┬─────────┘      │(summarize)  │
         │                └─────────────┘
         ▼
  kb/sessions/2026-01-12.md
```

## See Also

- [[guides/ai-integration]] - General AI agent setup
- [[reference/cli]] - CLI command reference
