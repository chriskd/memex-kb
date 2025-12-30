---
title: vl-mail: Lightweight Agent Mail CLI
tags:
  - tooling
  - agent-mail
  - beads
  - go
  - cli
  - multi-agent
created: 2025-12-22
updated: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
beads_issues:
  - voidlabs-devtools-6m5
---

# vl-mail: Lightweight Agent Mail CLI

A small, single-binary Go CLI for agent-to-agent messaging. Works as `bd mail` delegate or standalone.

## Status

**In Development** - Tracked in beads epic `voidlabs-devtools-6m5`

## Why vl-mail?

Beads v0.32.0 removed the built-in `bd mail` commands, delegating mail to external providers like Gas Town (`gt`). However, Gas Town isn't publicly released yet. vl-mail fills this gap with a lightweight alternative.

## Features

- **6 commands**: `inbox`, `send`, `read`, `ack`, `reply`, `sync`
- **Auto-detect storage**: `.vl-mail/` in repo (git-synced) or `~/.vl-mail/` global
- **Zero deps**: Go stdlib only, ~400-500 lines
- **bd integration**: Works as `bd mail` delegate

## Interface

| Command | Args | Description |
|---------|------|-------------|
| `inbox` | `[--json]` | List unread messages for current identity |
| `send` | `<to> -s "Subj" -m "Body" [--urgent]` | Send message |
| `read` | `<id>` | Show message details |
| `ack` | `<id>` | Mark as read |
| `reply` | `<id> -m "Body"` | Reply to message |
| `sync` | | Git pull/push `.vl-mail/` |

## Data Model

Messages stored as JSONL:

```jsonl
{"id":"msg-a1b2","to":"worker-2","from":"claude-main","subject":"Handoff","body":"Your turn","status":"unread","priority":2,"reply_to":"","created":"2025-12-22T10:00:00Z"}
```

## Setup

### As bd mail delegate
```bash
bd config set mail.delegate "/srv/fast/code/voidlabs-devtools/bin/vl-mail"
# Then use: bd mail inbox, bd mail send, etc.
```

### Standalone
```bash
vl-mail inbox
vl-mail send worker-2 -s "Handoff" -m "Your turn on bd-xyz"
```

## Location

Source: `voidlabs-devtools/cmd/vl-mail/`
Binary: `voidlabs-devtools/bin/vl-mail`

## Related

- [[beads-workflow]] - Beads issue tracking integration
- [[voidlabs-devtools]] - Parent tooling repository