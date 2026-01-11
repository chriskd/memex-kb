---
title: vl-mail Agent Messaging
tags:
  - tooling
  - ai-agents
  - messaging
  - beads
  - cross-project
  - voidlabs-devtools
created: 2025-12-22
updated: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
---

# vl-mail Agent Messaging

Lightweight CLI for agent-to-agent messaging with cross-project routing and provenance tracking.

## Overview

vl-mail provides asynchronous message passing between AI agents working on the same or different projects. Messages are stored in JSONL format and synced via git, enabling persistent communication across sessions.

**Key Features:**
- Cross-project addressing (`user@project` routing)
- Breadcrumb metadata for provenance tracking
- bd mail integration (via BEADS_MAIL_DELEGATE)
- JSONL storage (`.vl-mail/messages.jsonl`)

## Installation

Auto-installed in voidlabs devcontainers. For manual installation:

```bash
cd /srv/fast/code/voidlabs-devtools
make install-vl-mail
```

## Quick Reference

| Command | Purpose |
|---------|---------|
| `vl-mail inbox` | Show unread messages |
| `vl-mail inbox --all` | Include read messages |
| `vl-mail send <to> -s "Subj" -m "Body"` | Send message |
| `vl-mail read <id>` | Display message |
| `vl-mail ack <id>` | Mark as read |
| `vl-mail reply <id> -m "Body"` | Reply to message |
| `vl-mail sync` | Git sync mail changes |
| `vl-mail help <cmd>` | Per-command help |

## Cross-Project Addressing

```bash
# Local delivery (same project)
vl-mail send worker-2 -s "Subject" -m "Body"

# Cross-project (routes to /srv/fast/code/epstein/.vl-mail/)
vl-mail send worker-2@epstein -s "Schema ready" -m "Migrations done"

# Global mailbox (routes to ~/.vl-mail/)
vl-mail send worker-2@global -s "Announcement" -m "System update"
```

Replies auto-route to the sender's project based on the `From` field.

## Breadcrumb Metadata

Every message includes provenance metadata captured automatically:

| Field | Source |
|-------|--------|
| `actor` | BD_ACTOR or USER env var |
| `project` | Git repository name |
| `git_branch` | Current branch |
| `git_commit` | HEAD commit SHA |
| `session_id` | CLAUDE_SESSION_ID |
| `beads_issue` | BD_ISSUE |
| `timestamp` | Message creation time |

View with `vl-mail read msg-xxx --json`.

## Typical Workflow

```bash
# 1. Check for messages at session start
vl-mail inbox

# 2. Read and acknowledge
vl-mail read msg-abc123
vl-mail ack msg-abc123

# 3. Do the work...

# 4. Reply with status
vl-mail reply msg-abc123 -m "Completed. Tests passing."

# 5. Sync at session end
vl-mail sync
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `BD_ACTOR` / `USER` | Your identity |
| `VL_MAIL_BASE` | Project base path (default: `/srv/fast/code`) |
| `BEADS_MAIL_DELEGATE` | Set to "vl-mail" for bd mail passthrough |

## bd mail Integration

When `BEADS_MAIL_DELEGATE=vl-mail` is set (auto-configured in voidlabs devcontainers):

```bash
bd mail inbox              # Passes through to vl-mail
bd mail send worker-2 -s "Subject" -m "Body"
```

## Storage

- `.vl-mail/` in git repo root (preferred, syncs with repo)
- `~/.vl-mail/` global fallback (for @global or non-git contexts)
- Messages stored in `messages.jsonl` as newline-delimited JSON

## Source Code

Located at `/srv/fast/code/voidlabs-devtools/cmd/vl-mail/main.go`

Uses shared breadcrumb package at `pkg/breadcrumb/breadcrumb.go`