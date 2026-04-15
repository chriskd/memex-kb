---
title: AI Agent Integration
description: How to use Memex with Claude Code, Codex, and other skill-aware agent harnesses
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

Memex supports agent workflows in two layers:

1. `mx` is the portable interface for any harness with shell access.
2. `skills/memex-kb/` is the bundled reusable skill for harnesses that support `SKILL.md`-style
   installs.

Treat `mx` as the source of truth for KB state. Treat the skill as a reusable workflow layer on top
of the CLI.

## Choose an Integration Style

Use direct `mx` commands when:

- the harness can run shell commands but does not have a skill/plugin system
- you want the thinnest possible integration
- you need a portable fallback across multiple harnesses

Use the bundled skill when:

- the harness supports installable `SKILL.md`-style skills
- you want a reusable workflow instead of repeating the same prompt instructions
- you want the bundled references and metadata to travel with the skill

## Bundled Skill Layout

This repo ships a first-class skill in `skills/memex-kb/`:

- `skills/memex-kb/SKILL.md` - core workflow and when to use it
- `skills/memex-kb/references/` - lightweight supporting notes
- `skills/memex-kb/agents/openai.yaml` - UI metadata for Codex/OpenAI-style skill pickers

If a harness supports skills, copy or symlink the entire directory and keep the structure intact:

```bash
mkdir -p /path/to/harness/skills
ln -s "$PWD/skills/memex-kb" /path/to/harness/skills/memex-kb

# Or copy it instead of symlinking
cp -R skills/memex-kb /path/to/harness/skills/memex-kb
```

Preserve `SKILL.md` at the skill root. Keep `references/` next to it. If the harness ignores
`agents/openai.yaml`, that is fine.

## Claude Code

Claude Code integration is CLI-first today: grant `mx` access, keep shared prompt/config thin, and
use the session hook for startup context.

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
`.claude/settings.json` only when they should be committed for everyone.

### Session Hook

Install the session hook into local settings:

```bash
mx session-context --install --install-path .claude/settings.local.json
```

`mx session-context` injects project-relevant KB context at session start. This install flow is
Claude-specific; it is not a generic skill installer.

### Claude Code Workflow

Useful companion commands:

```bash
mx prime
mx session-context
mx schema --compact
mx --json-errors search "query"
mx batch
```

Typical flow:

```bash
# Before implementing: search for existing patterns
mx search "authentication patterns"

# Inspect current KB state
mx info
mx context show

# During work: capture discoveries for future sessions
mx add --title="OAuth2 Setup" --tags="auth,patterns" --category=guides --content="..."
```

`mx prime` is the shortest onboarding/context payload. `mx schema --compact` is the best option
when an agent needs structured command metadata. `mx --json-errors` makes failures stable for
automation. `mx batch` is useful when you want to combine several KB operations in one run.

## Codex

Codex can use Memex in both supported ways: direct CLI commands or the bundled skill.

### Install the Bundled Skill

Codex installs user skills into `$CODEX_HOME/skills` (default `~/.codex/skills`). From the repo
root:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
ln -s "$PWD/skills/memex-kb" "${CODEX_HOME:-$HOME/.codex}/skills/memex-kb"

# Or copy the directory instead of symlinking
cp -R skills/memex-kb "${CODEX_HOME:-$HOME/.codex}/skills/memex-kb"
```

Restart Codex after installing the skill so it is picked up on the next session.

The `agents/openai.yaml` file is optional UI metadata for Codex/OpenAI-style skill pickers. Keep it
next to `SKILL.md` when you install the skill.

### Use the CLI Directly

Codex can also use Memex directly from shell commands:

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

Use the skill when you want reusable Memex-specific workflow guidance. Use direct `mx` commands when
you want a thin shell-only integration.

## Other Harnesses

For other harnesses, use one of these patterns:

1. Direct CLI integration: run `mx` from shell commands and keep the harness prompt thin.
2. Skill installation: copy or symlink `skills/memex-kb/` into the harness's skill directory if it
   supports `SKILL.md`-style bundles.

Portable command set:

```bash
mx prime
mx info
mx context show
mx search "deployment strategies"
mx whats-new --scope=project --days=7
mx quick-add --stdin
```

If the harness has no native skill system, point it at the CLI and optionally keep
`skills/memex-kb/SKILL.md` nearby as reusable operator guidance.

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
2. Treat `mx` as the canonical interface, even when a harness-specific skill is installed.
3. Keep assistant-specific permissions in `.claude/settings.local.json` unless they need to be shared.
4. Link related entries with `[[wikilinks]]`.
5. Keep entries focused on one topic.
6. Update existing entries instead of duplicating them.

## See Also

- [[guides/index|Guides Index]]
- [[guides/quick-start|Quick Start Guide]]
- [[reference/cli|CLI Reference]]
- [[reference/entry-format|Entry Format Reference]]
