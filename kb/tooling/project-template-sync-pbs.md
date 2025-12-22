---
title: Project Template Sync (pbs)
tags:
  - tooling
  - devcontainer
  - workflow
created: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: voidlabs-devtools
---

# Project Template Sync (pbs)

**pbs** (Project Bootstrap and Sync) keeps devcontainer configurations in sync across all voidlabs projects.

## Concept

The template at `voidlabs-devtools/devcontainers/template/.devcontainer/` is the **source of truth**. Projects should mirror this template, with pbs managing the sync.

## Commands

| Command | Description |
|---------|-------------|
| `pbs status` | Show drift from template (current project) |
| `pbs status --all` | Show drift across all projects |
| `pbs diff` | Show actual file diffs |
| `pbs diff <file>` | Show diff for specific file |
| `pbs sync` | Apply template to current project |
| `pbs sync --all` | Apply template to all projects |
| `pbs edit <file>` | Edit template file, then apply to all |
| `pbs files` | List which files are synced vs excluded |
| `pbs new <path>` | Bootstrap a new devcontainer project |

## Synced Files

These files are managed by pbs (edit in template, not in projects):
- `devcontainer.json`
- `docker-compose.yml`
- `scripts/post-start-common.sh`
- `configs/worktrunk-user-config.toml`

## Project-Specific Files

These files are NOT synced (edit in each project):
- `Dockerfile` - Project-specific build
- `post-start-project.sh` - Project-specific setup
- `voidlabs.conf` - Project feature flags
- `docker-compose.override.yml` - Project overrides

## Workflow for Agents

1. **Improving shared setup**: Edit template in `devcontainers/template/`, then `pbs sync --all`
2. **Project-specific changes**: Edit directly in project's `.devcontainer/`
3. **Check sync status**: Run `pbs status` to see if project has drifted

## Environment Variables

- `PBS_TEMPLATE_DIR` - Path to template .devcontainer/ directory
- `PBS_PROJECTS_ROOT` - Path to scan for projects (for --all)