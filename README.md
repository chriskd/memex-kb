![Memex](https://raw.githubusercontent.com/chriskd/memex-kb/main/social-banner.png)

# Memex

CLI-first knowledge base for Markdown + YAML frontmatter across project and user scopes, with static publishing and agent-focused workflows.

## Start Here

- `kb/guides/index.md` (published docs landing page)
- `kb/guides/quick-start.md` (KB quick start)
- `kb/guides/ai-integration.md` (Claude Code, Codex, and skill-aware harnesses)
- `kb/reference/cli.md` (full command reference)

## Install

```bash
# Recommended
uv tool install memex-kb

# Optional semantic search deps
uv tool install 'memex-kb[search]'

# Or pip
pip install memex-kb
pip install 'memex-kb[search]'

# Verify
mx --version
```

Keyword search is available by default. Semantic search is optional (`[search]` extra).
If dependencies are missing, run `mx doctor`.

From a repo checkout, prefix with `uv run`:

```bash
uv sync --dev
uv run mx --version
```

## First Run

```bash
# Guided check; initialize if missing (non-interactive)
mx onboard --init --yes

# Same check, but only current directory context
mx onboard --init --yes --cwd-only

# Direct project KB initialization
mx init --sample
mx init --path docs/kb --sample

# User KB initialization
mx init --user --sample
```

Notes:
- `mx onboard --init --yes` can create a KB when none is configured.
- In onboarding, sample behavior is configurable with `--sample/--no-sample`.
- When both project and user KBs are active, paths may be scoped (`@project/...`, `@user/...`).

## Common Commands

```bash
# Search (keyword default; semantic optional)
mx search "deployment"
mx search "deployment" --mode=keyword
mx search "deployment" --mode=semantic
mx search "deployment" --scope=project
mx search "deployment" --include-neighbors --neighbor-depth=2

# Create/read/update
mx add --title="Setup" --tags="docs" --category=guides --content="..."
mx quick-add --content="Capture this note"
mx get @project/guides/setup.md
mx replace @project/guides/setup.md --tags="docs,setup"
mx delete @project/guides/setup.md --force

# Multi-KB visibility and context
mx info
mx context show
mx context validate

# Maintenance
mx health
mx doctor
mx doctor --timestamps
mx doctor --timestamps --fix
mx relations-lint --strict
```

## Publish

```bash
# Basic build
mx publish -o _site

# Source selection
mx publish --kb-root ./kb -o _site
mx publish --scope=project -o _site

# Landing page and base URL
mx publish --index guides/index --base-url /my-kb

# Content controls
mx publish --include-drafts --include-archived
mx publish --no-clean

# GitHub Pages workflow
mx publish --setup-github-actions
mx publish --setup-github-actions --dry-run
```

## Agents + Skills

Memex integrates with assistants in two layers:

- `mx` is the stable interface for any harness with shell access.
- `skills/memex-kb/` is the bundled reusable skill for harnesses that support `SKILL.md`.

```bash
# Claude Code: install the session hook into machine-local settings
mx session-context --install --install-path .claude/settings.local.json

# Claude Code native skill install from a repo checkout
mkdir -p .claude/skills
ln -s "$PWD/skills/memex-kb" .claude/skills/memex-kb

# Codex local skill install from a repo checkout (restart Codex after)
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
ln -s "$PWD/skills/memex-kb" "${CODEX_HOME:-$HOME/.codex}/skills/memex-kb"

# Installed package: locate the bundled skill first
python3 -c 'from importlib.resources import files; print(files("memex").joinpath("skills", "memex-kb"))'

# Portable helpers for any harness
mx prime
mx session-context
mx schema --compact
mx batch < commands.txt
mx --json-errors search "query"
```

For the full setup matrix, including Claude permissions, Codex skill conventions, and generic
skill installation guidance, see `kb/guides/ai-integration.md`.

## Config

Project config file: `.kbconfig` at repo root.

```yaml
# kb_path is canonical; project_kb still appears in some help/output
kb_path: ./kb
primary: inbox
warn_on_implicit_category: true
boost_paths:
  - inbox/*
  - guides/*
default_tags:
  - myproject
publish_base_url: /my-kb
publish_index_entry: guides/index
```

Environment variables:

| Variable | Description |
|----------|-------------|
| `MEMEX_USER_KB_ROOT` | Override user KB root (default `~/.memex/kb`) |
| `MEMEX_CONTEXT_NO_PARENT` | Only consider `.kbconfig` in current directory |
| `MEMEX_INDEX_ROOT` | Override index root |
| `MEMEX_QUIET` | Suppress warnings |

Useful global flags:
- `--json-errors` outputs structured errors with error codes.
- `-q, --quiet` suppresses warnings (same effect as `MEMEX_QUIET=1`).
