---
title: Quick Start Guide
tags: [getting-started, tutorial, basics]
created: 2026-01-06
updated: 2026-04-15
description: Get productive with memex in 5 minutes
---

# Quick Start Guide

If you're starting from scratch, read [[guides/installation]] first. If you already have `mx`,
`mx onboard --init --yes` is the fastest way to create a project KB in the current repo.

## First Run

```bash
mx onboard --init --yes                 # Create a project KB and .kbconfig
mx onboard --cwd-only --init --yes      # Skip parent discovery when you want a clean repo-local KB
mx init --user --sample                 # Create a personal KB with starter content
mx init --path docs/kb --no-sample      # Create a KB at a custom path without sample content
mx init --force --sample                # Rebuild an existing KB
```

Notes:
- `mx onboard` and `mx init` both create the KB, config, and starter directories.
- `--sample` creates a starter entry; `--no-sample` leaves the KB empty.
- `MEMEX_CONTEXT_NO_PARENT=1` disables parent `.kbconfig` discovery for the current session.
- `MEMEX_USER_KB_ROOT` and `MEMEX_INDEX_ROOT` override the user KB and index locations.

## Create Content

```bash
mx add --title="Git Stash Workflow" --tags="git,workflow,cli" --category=guides --content="..."
mx quick-add --stdin
cat notes.md | mx quick-add --stdin
mx ingest notes.md --scope=project
```

Use `mx add` when you know the title, tags, and destination category. Use `mx quick-add` when you
already have raw Markdown and want Memex to infer the metadata.

## Search and Read

```bash
mx search "stash"
mx search "stash" --scope=project
mx search "save work in progress" --strict
mx search "stash" --include-neighbors --neighbor-depth=2
mx get guides/git-stash-workflow.md
mx get guides/git-stash-workflow.md --metadata
```

When both KBs exist, results are prefixed with `@project/...` and `@user/...`.

## Keep It Healthy

```bash
mx health
mx doctor --timestamps
mx doctor --timestamps --fix
mx relations-lint --strict
mx whats-new --scope=project
```

`mx doctor` also reports and repairs missing, invalid, or stale `created` / `updated` timestamps.
If search fails with a missing keyword dependency, install `whoosh-reloaded` or run `mx doctor` to
check the environment first.

## Publish

```bash
mx publish
mx publish --index guides/index
mx publish --include-drafts
```

The published site starts at [[guides/index]].

## Next Steps

- [[guides/index|Guides Index]]
- [[reference/cli|CLI Reference]]
- [[reference/entry-format|Entry Format Reference]]
- [[guides/ai-integration|AI Integration]]
