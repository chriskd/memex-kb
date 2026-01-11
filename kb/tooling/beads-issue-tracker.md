---
title: Beads Issue Tracker
tags:
  - tooling
  - issue-tracking
  - ai-agents
  - git
created: 2025-12-20
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
---

# Beads Issue Tracker

A git-backed distributed issue tracker designed for AI-supervised coding workflows. Command: `bd`

## Repository

- **Source:** `github.com/steveyegge/beads`
- **Location in voidlabs-ansible:** `/srv/fast/code/voidlabs-ansible/beads`

## Core Concept

Beads enables a distributed, git-backed issue tracker that feels like a centralized database through a three-layer architecture:

```
SQLite DB (.beads/beads.db, gitignored)
    ↕ auto-sync (5s debounce)
JSONL (.beads/issues.jsonl, git-tracked)
    ↕ git push/pull
Remote JSONL (shared across machines)
```

## Installation

```bash
# Homebrew (recommended)
brew tap steveyegge/beads
brew install bd

# Or via go install
go install github.com/steveyegge/beads/cmd/bd@latest

# Initialize in project
cd your-project
bd init
```

## Quick Reference

```bash
# Issue management
bd create "Task title" -p 1 -t task
bd list
bd show <id>
bd update <id> --status in_progress
bd close <id> --reason "Done"

# Workflow
bd ready           # Show unblocked issues
bd blocked         # Show blocked issues
bd stats           # View statistics

# Dependencies
bd dep add <id> <depends-on-id>
bd dep tree <id>

# Sync
bd sync            # Force sync now
```

## Issue ID Format

Hash-based IDs prevent collisions when multiple agents work concurrently:
- Format: `bd-a1b2`, `bd-f14c` (4-6 char hash)
- Hierarchical: `bd-a3f8e9.1`, `bd-a3f8e9.2` for subtasks

## Daemon Architecture

Each workspace runs its own background daemon:
- Auto-starts on first command
- Batches operations before export
- Socket at `.beads/bd.sock`
- Disable with `BEADS_NO_DAEMON=1` or `--no-daemon`

```bash
bd daemons list    # Show running daemons
bd --no-daemon ready  # Skip daemon (for git worktrees)
```

## Multi-Agent Workflows (Agent Mail)

For 2+ AI agents working concurrently, Agent Mail provides real-time coordination:

```bash
# Start server
git clone https://github.com/Dicklesworthstone/mcp_agent_mail.git
cd mcp_agent_mail && python -m mcp_agent_mail.cli serve-http

# Configure each agent
export BEADS_AGENT_MAIL_URL=http://127.0.0.1:8765
export BEADS_AGENT_NAME=assistant-alpha
export BEADS_PROJECT_ID=my-project
```

Benefits:
- 20-50x latency reduction (<100ms vs 2-5s git sync)
- File reservations prevent collision
- Agents can't claim same issue

## Data Types

| Type | Description |
|------|-------------|
| Issue | Work item with status, priority, type |
| Dependency | Relationship (blocks, related, parent-child) |
| Label | Tags with color |
| Comment | Threaded discussions |
| Event | Audit trail |

## Dependency Types

| Type | Affects `bd ready`? |
|------|---------------------|
| `blocks` | Yes - X must close before Y starts |
| `parent-child` | Yes - children blocked if parent blocked |
| `related` | No - soft link for reference |
| `discovered-from` | No - found during work on parent |

## IDE Integration

### Claude Code
```bash
bd setup claude    # Installs SessionStart/PreCompact hooks
```

### Cursor
```bash
bd setup cursor    # Creates .cursor/rules/beads.mdc
```

Context injection uses `bd prime` which provides ~1-2k tokens of workflow context.

## Key Files

| Path | Purpose |
|------|---------|
| `.beads/beads.db` | SQLite database (gitignored) |
| `.beads/issues.jsonl` | JSONL source of truth (git-tracked) |
| `.beads/bd.sock` | Daemon socket (gitignored) |
| `.beads/config.yaml` | Project config |

## Related Entries

- [[Voidlabs Devtools]]
