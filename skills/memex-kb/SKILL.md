---
name: memex-kb
description: Search, read, create, edit, and maintain knowledge in a Memex KB via the `mx` CLI. Use when the user asks to search the KB, find documentation, check project or organizational docs, add or update knowledge-base entries, audit KB quality, or when reusable knowledge should be captured for future agents.
---

# Memex KB Workflow

Use `mx` as the source of truth for KB state. If `mx` is not installed globally and you are working from a repo checkout, prefix examples with `uv run`.

## Start With Context

Run these first when you need to write, interpret scoped paths, or understand the current KB shape:

```bash
mx prime
mx info
mx context show
```

Use them to answer:
- Which KBs are active (`project`, `user`, or both)
- Which categories/directories exist
- Whether `.kbconfig` sets a `primary` write directory
- Whether scoped paths like `@project/...` or `@user/...` are necessary

Run `mx categories` before creating entries. Categories are KB-specific and may differ across repos.

## Search Before Writing

Search for existing knowledge before answering from memory or adding new content.

Start broad, then tighten:

```bash
mx search "query"
mx search "query" --scope=project
mx search "query" --mode=keyword
mx search "query" --mode=semantic
mx search "query" --include-neighbors
mx search "query" --content
mx search "query" --json
```

Use:
- `--mode=keyword` for exact terms, command names, filenames, error strings
- `--mode=semantic` for conceptual matches
- `--include-neighbors` when relations or wikilinks matter
- `--content` or `mx get <path>` once you have a candidate entry
- `--scope=project|user` to avoid mixing team docs with personal notes

If results are ambiguous, inspect candidates directly:

```bash
mx get path/to/entry.md
mx get @project/path/to/entry.md
mx get --title="Entry Title"
```

When both project and user KBs are active, prefer explicit scoped paths in edits and citations to avoid collisions.

## Choose The Right Write Path

Prefer this order:
1. Update an existing entry if the knowledge belongs there.
2. Append to an existing log or ongoing note if the new material is incremental.
3. Create a new entry only when the topic deserves its own durable page.

Decide scope with the content's audience:
- `project`: team conventions, architecture, repo workflows, shared troubleshooting, operational docs
- `user`: personal notes, drafts, experiments, cross-project personal workflows

Decide category from the live KB, not from a hard-coded taxonomy:

```bash
mx categories
mx tree --depth=2
```

If `.kbconfig` has no `primary`, `mx add` without `--category` writes to the KB root and warns. Prefer setting `--category` explicitly unless the existing KB conventions say otherwise.

## Create Entries Deliberately

Use the command that matches how much structure you already have:

```bash
mx add --title="..." --tags="..." --category=... --content="..."
mx quick-add --content="..."
mx ingest notes.md --directory=guides
```

Prefer:
- `mx add` when you already know the title, tags, category, and scope
- `mx quick-add` when you have raw notes and want Memex to suggest metadata
- `mx ingest` when importing an existing Markdown file into the KB

After creating an entry:

```bash
mx list --limit=5
mx get path/to/new-entry.md
mx suggest-links path/to/new-entry.md
```

Use `mx tags` before inventing new tags. Reuse existing tags unless a new one is clearly warranted.

For frontmatter, relations, and templates, read `references/entry-format.md`.

## Update Entries Conservatively

Prefer the least destructive edit command:

```bash
mx patch path.md --find="old" --replace="new"
mx append "Entry Title" --content="new section"
mx replace path.md --content="full replacement"
```

Choose commands by intent:
- `mx patch`: surgical edits, typo fixes, targeted updates
- `mx append`: add notes or sections while preserving the existing document
- `mx replace`: full rewrites or metadata/content replacement

Use `--dry-run` with `mx patch` when the replacement is risky or repetitive.

For relation maintenance, prefer dedicated relation commands when possible:

```bash
mx relations path.md --depth=2
mx relations-add path.md --relation "other.md=documents"
mx relations-remove path.md --relation "other.md=documents"
mx relations-lint
```

## Audit And Maintain Quality

Run these after meaningful writes or when the user asks about KB health:

```bash
mx health
mx relations-lint
mx whats-new --days=7
mx hubs
mx doctor --timestamps
```

Use them to catch:
- broken links
- orphaned entries
- stale content
- relation type drift
- timestamp problems

`mx health` suppresses orphan warnings in very small KBs, so interpret orphan output in context.

## Keep Canonical Docs Canonical

Do not maintain a second copy of the CLI or schema inside this skill.

Read these repo docs when you need detail:
- `kb/reference/cli.md` for current command options and JSON output shapes
- `kb/reference/entry-format.md` for frontmatter, links, semantic links, and typed relations
- `kb/guides/quick-start.md` for onboarding flow and category conventions in the current repo
- `kb/design/relations-graph/relations-graph-overview.md` when relation traversal or publish behavior matters

The reference files bundled with this skill are navigation notes, not an independent source of truth.

## Anti-Patterns

Avoid:
- adding a new entry before searching for an existing one
- hard-coding category names from an old repo into a new KB
- using unscoped paths when both project and user KBs are active and collisions are possible
- using `mx replace` when `mx patch` or `mx append` would preserve context
- creating personal scratch notes in the project KB
- inventing new tags without checking `mx tags`
