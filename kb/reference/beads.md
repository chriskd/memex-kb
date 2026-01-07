---
title: Beads Integration
tags: [reference, beads, issue-tracking]
created: 2026-01-07
description: How memex integrates with beads for AI-native issue tracking
---

# Beads Integration

Memex provides native integration with [beads](https://github.com/steveyegge/beads) - an AI-native issue tracker that stores issues in your git repository.

## Why Beads for AI Agents?

Beads is designed specifically for AI-assisted development:

- **CLI-first**: The `bd` command works seamlessly with AI coding agents
- **Git-backed**: Issues stored in `.beads/issues.jsonl` sync with your code
- **No context switching**: No web UI needed - everything happens in terminal
- **Hash-based IDs**: Prevents conflicts when multiple agents work concurrently

## Memex Integration Features

### 1. Frontmatter Fields

Link KB entries to beads issues using frontmatter:

```yaml
---
title: My Entry
tags: [documentation]
created: 2026-01-07
beads_issues: [project-abc1, project-def2]  # Link to specific issues
beads_project: /path/to/project             # Link to all issues in a project
---
```

| Field | Type | Description |
|-------|------|-------------|
| `beads_issues` | list | Specific issue IDs to link |
| `beads_project` | string | Path to beads project (shows all open issues) |

### 2. Beads Registry

The `.beads-registry.yaml` file at KB root maps project prefixes to paths:

```yaml
# .beads-registry.yaml
myapp: /path/to/myapp
infra: /path/to/infrastructure
memex: .                       # "." means KB root itself
voidlabs-kb: .                 # Alias for same project
```

This enables cross-project issue browsing from a single command.

### 3. CLI Commands

Memex adds `mx beads` commands for read-only issue browsing:

```bash
# List issues
mx beads list                           # From auto-detected project
mx beads list -p myapp                  # From specific project
mx beads list --status=open             # Filter by status
mx beads list --type=bug --limit=10     # Filter by type

# Show issue details
mx beads show myapp-42                  # Full details with comments
mx beads show 42 -p myapp               # Equivalent with explicit project
mx beads show myapp-42 --no-comments    # Without comments

# Kanban board view
mx beads kanban                         # Current project
mx beads kanban -p infra --compact      # Titles only

# Project status
mx beads status                         # Counts by status/priority/type
mx beads status -p myapp

# List registered projects
mx beads projects                       # Show all projects in registry
```

All commands support `--json` for machine-readable output.

### 4. Webapp API

The memex webapp exposes beads data via REST API:

```bash
# Check if beads is available
GET /api/beads/config

# Kanban board
GET /api/beads/kanban
GET /api/beads/kanban?project_path=/path/to/project

# Issue details
GET /api/beads/issues/{issue_id}
GET /api/beads/issues/{issue_id}?project_path=/path/to/project

# Entry-linked issues
GET /api/entries/{path}/beads
```

The webapp renders linked issues when viewing KB entries that have `beads_issues` or `beads_project` frontmatter.

## Common Beads Commands

While `mx beads` provides read-only browsing, use `bd` directly for mutations:

```bash
# Create issues
bd create "Add user authentication"
bd create "Fix login bug" -t bug -p 1    # Type=bug, Priority=high

# Update status
bd update issue-abc1 --status in_progress
bd close issue-abc1 --reason "Completed"

# Dependencies
bd dep add issue-abc1 issue-def2         # abc1 depends on def2
bd ready                                 # Show unblocked issues
bd blocked                               # Show blocked issues

# Sync with git
bd sync                                  # Push/pull issues
```

## Issue ID Format

Beads uses hash-based IDs to prevent collision:
- Format: `prefix-hash` (e.g., `myapp-a1b2`, `infra-f14c`)
- 4-6 character hash ensures uniqueness across concurrent agents
- Hierarchical IDs for subtasks: `myapp-a3f8.1`, `myapp-a3f8.2`

## Workflow: KB Entry Per Issue

A common pattern is creating a KB entry for significant issues:

```bash
# Create issue
bd create "Implement caching layer"
# Output: Created issue myapp-7g3h

# Create corresponding KB entry
mx add --title="Caching Layer Design" \
       --tags="architecture,myapp" \
       --category=projects

# Link them in the entry frontmatter
# beads_issues: [myapp-7g3h]
```

## Best Practices

1. **Use registry for cross-project access**: Add all your beads projects to `.beads-registry.yaml`

2. **Link major issues to KB**: Create KB entries for epics, design decisions, or complex bugs

3. **Keep issues atomic**: One issue = one deliverable. Use dependencies for sequencing.

4. **Let agents create issues**: AI agents can use `bd create` to track discovered work

5. **Use `mx beads` for context**: Quick issue lookup without leaving the KB workflow

## Configuration

### Beads Project Setup

```bash
# Initialize beads in a project
cd /path/to/project
bd init

# This creates:
# .beads/
#   beads.db         # SQLite (gitignored)
#   issues.jsonl     # Source of truth (git-tracked)
#   config.yaml      # Project config
#   bd.sock          # Daemon socket (gitignored)
```

### Register with Memex

Add to your KB's `.beads-registry.yaml`:

```yaml
myproject: /path/to/myproject
```

Now `mx beads list -p myproject` works.

## See Also

- [[tooling/beads-issue-tracker|Beads Issue Tracker]] - Full beads reference
- [[reference/cli|CLI Reference]] - All mx commands
- [[reference/entry-format|Entry Format]] - Frontmatter fields
