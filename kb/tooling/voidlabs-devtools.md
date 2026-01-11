---
title: Voidlabs Devtools
tags:
  - tooling
  - devcontainer
  - developer-experience
  - project
created: 2025-12-20
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
---

# Voidlabs Devtools

Shared devcontainer tooling, Claude Code configuration, and project scaffolding for voidlabs projects.

## Repository

- **Location:** `/srv/fast/code/voidlabs-devtools`
- **GitHub:** `git@github.com:chriskd/voidlabs-devtools.git`

## Purpose

Central tooling repository that provides:
- Devcontainer templates and scaffolding scripts
- AI coding agent configuration (Claude Code, Cursor, Factory Droid)
- Shared shell integration and aliases
- Beads issue tracker installation and web UI
- GitHub Actions self-hosted runner setup

## Architecture

```
devbox.voidlabs.local (Host)
├── /srv/fast/claude-linux/    → Shared Claude config (~/.claude mount)
├── /srv/fast/codex/           → Shared Codex config
└── /srv/fast/code/
    ├── voidlabs-devtools/     → THIS REPO
    ├── dotfiles/              → Chezmoi dotfiles
    └── project-a/, project-b/ → Your projects...

Devcontainer mounts:
- /srv/fast           ← bind mount (same path)
- ~/.claude           ← bind from /srv/fast/claude-linux
- ~/.codex            ← bind from /srv/fast/codex
- /workspaces/...     ← your project
```

## Key Scripts

### new-project.sh

Scaffolds new devcontainer projects by fetching templates from GitHub using `gh` CLI.

```bash
./scripts/new-project.sh "My Project" /path/to/my-project
./scripts/new-project.sh --defaults "My Project" /path/to/my-project
```

Creates:
- `AGENTS.md` - AI agent guidance
- `.devcontainer/devcontainer.json` - Container configuration
- `.devcontainer/Dockerfile` - Base image
- `.devcontainer/voidlabs.conf` - Feature toggles
- `.devcontainer/scripts/post-start-*.sh` - Setup scripts

### install-beads.sh

Installs the beads issue tracker with full integration.

```bash
/srv/fast/code/voidlabs-devtools/scripts/install-beads.sh /workspaces/my-project
```

### post-start-common.sh

Runs on devcontainer start, handling:
1. Phase secrets injection
2. Beads installation
3. Factory droid setup
4. Chezmoi dotfiles
5. Worktrunk git worktree manager
6. Project-specific setup

## Shell Aliases

When in a devcontainer:

| Alias | Command | Description |
|-------|---------|-------------|
| `br` | `bd ready` | Show tasks ready to work |
| `bl` | `bd list` | List all issues |
| `bs` | `bd show` | Show issue details |
| `bst` | `bd stats` | Project statistics |
| `bweb` | `beads-webui.sh` | Start web UI |
| `cdf` | `cd /srv/fast/code` | Quick navigation |

## Beads Web UI

Real-time web dashboard for viewing beads issues:

```bash
# Build binary (one-time on devbox.voidlabs.local)
cd /srv/fast/code/voidlabs-devtools
docker build -t beads-webui-builder services/beads-webui
docker create --name tmp-beads beads-webui-builder
docker cp tmp-beads:/usr/local/bin/beads-webui bin/beads-webui
docker rm tmp-beads

# Start from any devcontainer
bweb  # Opens on port 8080
```

## GitHub Actions Self-Hosted Runner

Containerized runner for CI at `services/gh-runner/`:

```bash
cd /srv/fast/code/voidlabs-devtools/services/gh-runner
cp .env.example .env
# Edit .env for GH_PAT, PHASE_APP, etc.
docker compose up -d
```

Fetches `GH_PAT` from Phase at startup - no secrets on disk.

## Directory Structure

```
voidlabs-devtools/
├── devcontainers/template/     # New project template
├── ai-tooling/                 # AI agent configs
│   ├── claude/                 # Claude Code settings
│   ├── cursor/                 # Cursor integration
│   ├── factory/                # Factory/Droid config
│   └── hooks/                  # Claude Code hooks
├── services/
│   ├── gh-runner/              # GitHub Actions runner
│   └── beads-webui/            # Beads dashboard
├── scripts/                    # Setup utilities
├── shell/devcontainer.zsh     # Shell integration
└── bin/beads-webui            # Pre-built binary
```

## Related Entries

- [[Voidlabs Infrastructure Overview]]
- [[Beads Issue Tracker]]
