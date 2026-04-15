# Start Here

`mx` is the Memex CLI for Markdown knowledge bases with YAML frontmatter, in project scope (`./kb/`) and/or user scope (`~/.memex/kb/`).

## Install

```bash
uv tool install memex-kb
# optional semantic search:
uv tool install 'memex-kb[search]'
```

Keyword search works by default. Semantic search is optional.
Run `mx doctor` to verify dependencies.

## 60-Second First Run

```bash
# Guided setup + initialize if missing (safe in agents/CI)
mx onboard --init --yes

# Restrict context discovery to current directory only
mx onboard --init --yes --cwd-only

# Direct init options
mx init --sample
mx init --path docs/kb --sample
mx init --user --sample
```

Sample behavior:
- `mx onboard` supports `--sample/--no-sample`.
- `mx init` supports `--sample` for first-run content.

## Common Commands

```bash
# Search defaults to keyword (BM25)
mx search "first task"
mx search "first task" --mode=semantic
mx search "first task" --scope=project

# Read + write
mx list --limit=5
mx get @project/inbox/first-task.md
mx add --title="Setup" --tags="docs" --category=guides --content="..."
mx quick-add --content="capture this"

# Maintenance
mx doctor --timestamps
mx doctor --timestamps --fix
mx info
mx context show
```

When project and user KBs are both active, outputs may include scoped paths like `@project/...` and `@user/...`.

## Publish

```bash
mx publish -o _site
mx publish --kb-root ./kb -o _site
mx publish --scope=project -o _site
mx publish --index guides/index --base-url /my-kb
mx publish --include-drafts --include-archived
mx publish --no-clean
mx publish --setup-github-actions
mx publish --setup-github-actions --dry-run
```

## Agents

```bash
mx session-context
mx session-context --install
mx prime
mx schema --compact
mx batch < commands.txt
mx --json-errors search "query"
```

Memex works with assistants in two ways:
- direct `mx` commands in any shell-capable harness
- the bundled skill in `skills/memex-kb/` for harnesses that support `SKILL.md`

Common setups:

```bash
# Claude Code: machine-local hook install
mx session-context --install --install-path .claude/settings.local.json

# Codex: install the bundled skill, then restart Codex
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
ln -s "$PWD/skills/memex-kb" "${CODEX_HOME:-$HOME/.codex}/skills/memex-kb"
```

## Config + Env

`.kbconfig` (project root):

```yaml
kb_path: ./kb         # project_kb alias is still supported
primary: inbox
publish_base_url: /my-kb
publish_index_entry: guides/index
```

Environment variables:
- `MEMEX_USER_KB_ROOT`
- `MEMEX_CONTEXT_NO_PARENT`
- `MEMEX_INDEX_ROOT`
- `MEMEX_QUIET`

More detail: `kb/guides/quick-start.md` and `kb/reference/cli.md`.
Harness-specific detail: `kb/guides/ai-integration.md`.
