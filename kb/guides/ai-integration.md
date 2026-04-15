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
updated: 2026-04-15T00:00:00+00:00
semantic_links: []
---

# AI Agent Integration

Memex is designed to work through the `mx` CLI. Treat the CLI as the stable interface and keep
assistant-specific config thin.

## Claude Code

### Permissions

Add this to `.claude/settings.local.json`:

```json
{
  "permissions": {
    "allow": ["Bash(mx:*)"]
  }
}
```

Use `.claude/settings.local.json` for machine-local permissions. Keep shared team settings in
`.claude/settings.json` if you need them tracked in the repo.

### Session Hooks

Install the session hook into local settings:

```bash
mx session-context --install --install-path .claude/settings.local.json
```

`mx session-context` injects project-relevant KB context at session start. It is the recommended
way to give an agent a compact snapshot of the active KB, relevant entries, and recent workflow
context.

Useful companion commands:

```bash
mx prime
mx session-context
mx schema --compact
mx --json-errors search "query"
mx batch
```

### Claude Code Workflow

```bash
# Before implementing: search for existing patterns
mx search "authentication patterns"

# During work: capture discoveries for future sessions
mx add --title="OAuth2 Setup" --tags="auth,patterns" --category=guides --content="..."
```

`mx prime` is the shortest onboarding/context payload. `mx schema --compact` is the best option
when an agent needs structured command metadata. `mx --json-errors` makes failures stable for
automation. `mx batch` is useful when you want to combine several KB operations in one run.

## Codex

Codex can use Memex directly from shell commands.

```bash
# Search before implementing
mx search "deployment strategy"

# Read the source note
mx get reference/cli.md

# Capture a new discovery
mx quick-add --stdin
```

Helpful commands for Codex-style workflows:

- `mx info` - show KB roots, counts, and active config
- `mx context show` - show the resolved project context
- `mx context validate` - validate `kb_path` / `project_kb` and related paths
- `mx prime` - emit a concise agent startup payload
- `mx session-context` - emit live project context for the current session
- `mx schema --compact` - emit command metadata for introspection
- `mx --json-errors` - keep errors machine-parseable

## Other Agents

Any agent with shell access can use the same patterns:

```bash
mx search "deployment strategies"
mx whats-new --scope=project --days=7
mx info
mx context show
```

Search before implementing, add discoveries after solving, and use `mx quick-add --stdin` when you
already have Markdown that should become an entry.

## Project Configuration

Configure project-specific KB settings in `.kbconfig`:

```yaml
kb_path: ./kb
primary: guides
boost_paths:
  - guides/*
  - reference/*
default_tags:
  - memex
publish_base_url: /my-kb
publish_index_entry: guides/index
```

Use `kb_path` as the canonical project KB setting. You will also see `project_kb` in resolved
context and generated output, which is the same setting after it has been loaded.

This `.kbconfig` file:
- Points `mx` at the project KB
- Routes new entries to `guides/` by default
- Boosts project entries in search results
- Suggests project-specific tags
- Sets the published site base URL and landing page

Related commands:

- `mx context show`
- `mx context validate`
- `mx publish`
- `mx info`

## Best Practices

1. Search before creating to avoid duplicate entries.
2. Keep assistant-specific permissions in `.claude/settings.local.json` unless they need to be shared.
3. Link related entries with `[[wikilinks]]`.
4. Keep entries focused on one topic.
5. Update existing entries instead of duplicating them.

## See Also

- [[guides/index|Guides Index]]
- [[reference/cli|CLI Reference]]
- [[reference/entry-format|Entry Format Reference]]
