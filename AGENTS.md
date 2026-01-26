# Instructions for AI Agents

> This file provides operational guidance for AI agents working on this project.
> Source: [voidlabs-devtools](https://github.com/chriskd/voidlabs-devtools)

---

## 🚀 Maximize Parallelization

**This environment is designed for parallel work.** Always look for opportunities to parallelize:

### Use Subagents Aggressively

When you have multiple independent tasks, spawn subagents to work in parallel:

```
User: "Add auth, refactor the API, and write tests"

# BAD: Do sequentially
# GOOD: Spawn 3 subagents, each handling one task
```

**When to spawn subagents:**
- Multiple files need independent changes
- Research + implementation can happen concurrently
- Tests can run while you continue coding
- Multiple features/bugs in the same request

### Use Background Tasks

For long-running operations, use `run_in_background`:
- Test suites
- Builds
- Linting large codebases
- Any command >30 seconds

Continue working while they run, check results with `TaskOutput` when needed.

### Use Git Worktrees for Agent Isolation

When the user wants truly parallel agents (separate Claude instances), use worktrunk:

```bash
# Each agent gets isolated branch + directory
wt switch -c -x claude feat-auth     # Terminal 1
wt switch -c -x claude refactor-api  # Terminal 2

# Work independently, then merge
wt merge  # Squash + auto-generate commit message
```

See the **Parallel Work with Worktrunk** section below for details.

### Parallel Tool Calls

When making tool calls, batch independent calls in a single message:

```
# BAD: Sequential
Read file A → Read file B → Read file C

# GOOD: Parallel (single message with 3 Read calls)
[Read A, Read B, Read C]  # All execute concurrently
```

**The user's environment has capacity for parallel work. Use it.**

---

## Development Environment

### Infrastructure

Development happens in **devcontainers** running on `devbox.voidlabs.local` (a remote Docker host), not on the local Mac.

```
┌─────────────┐    Mutagen     ┌─────────────────────────┐
│    Mac      │ ──────sync───▶ │  devbox.voidlabs.local  │
│  (beta)     │                │       (alpha)           │
│             │                │                         │
│  Cursor ────┼── SSH remote ─▶│  ┌─────────────────┐    │
│  Claude     │                │  │  devcontainer   │    │
│  Codex      │                │  │  /srv/fast/code │    │
│  Factory    │                │  └─────────────────┘    │
└─────────────┘                └─────────────────────────┘
```

**Key implications for agents:**

- **"Local" means the container** - Commands run inside devcontainers on devbox.voidlabs.local
- **Code lives at `/srv/fast/code/`** - This is bind-mounted into containers
- **Docker builds are fast** - Images build on devbox.voidlabs.local, no network overhead for layers
- **Mutagen sync has slight delays** - If a file seems stale after editing, wait a moment
- **SSH agent forwarding works** - Git operations use forwarded keys through the SSH chain

**Shared resources across containers:**

| Path | Purpose |
|------|---------|
| `/srv/fast/code/voidlabs-devtools` | Shared scripts, hooks, templates |
| `~/.claude` | Claude Code settings (synced across containers) |
| `~/.ssh` | SSH keys (forwarded from Mac) |

**Development tools:**

- **Cursor** - Connects via SSH Remote extension, then attaches to containers
- **Claude Code** - CLI in terminal, uses shared settings
- **Factory Droid** - Local LLM agent (configured via `shared devtools setup`)
- **OpenAI Codex** - API-based agent

**Docker context on Mac:**
```bash
# Mac's docker CLI points to the remote
docker context use quasar  # quasar = ssh://chris@devbox.voidlabs.local
```

### Shared Tooling (voidlabs-devtools)

All projects share tooling from `/srv/fast/code/voidlabs-devtools`. This repository provides:

| Component | Purpose |
|-----------|---------|
| `AGENTS.md` | This file - agent guidance for this project |
| `scripts/new-project.sh` | Scaffolds new projects with devcontainer |
| `devcontainers/template/scripts/post-start-common.sh` | Shared container setup (Phase secrets, Factory droid, ticket tooling) |

**Session hooks run automatically** and inject:
- ticket reminders (ready/blocked when .tickets exists)
- Comments and documentation guidance

**You don't need to run these manually** - they execute on session start. But understanding where they come from helps if you want to suggest improvements.

**Project-local session hooks** should be created via `mx session-context --install` (writes `./hooks/session-context.sh`, which is gitignored).

**To improve shared tooling:**
- Edit files in `/srv/fast/code/voidlabs-devtools`
- Changes apply to all projects on next session start
- Consider creating a ticket for significant changes

**New projects are bootstrapped with:**
```bash
/srv/fast/code/voidlabs-devtools/scripts/new-project.sh <project-name> <target-dir>
```

This creates `.devcontainer/`, copies `AGENTS.md`, and sets up voidlabs-devtools integration.

---

### Python

**Use `uv` for all Python package management** - not pip, poetry, or pipenv:

```bash
# Create virtual environment
uv venv

# Install dependencies
uv pip install -e ".[dev]"

# Add new dependency
uv add fastapi

# Sync lockfile
uv sync
```

**Virtual environments are mandatory.** Always create `.venv/` in the project root. Never install packages globally.

**Project structure:**
```
project/
├── pyproject.toml      # Dependencies and project metadata
├── uv.lock             # Lockfile (commit this)
├── .venv/              # Virtual environment (gitignored)
└── src/                # Source code
```

### Web Applications

Most projects are **FastAPI/Starlette apps** running with **uvicorn**:

```bash
# Development
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production (via Dockerfile)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Dockerfile pattern:**
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY src/ ./src/
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Deployment

Apps are deployed to **Dokploy** (self-hosted PaaS):
- Push to main branch triggers auto-deploy
- Environment variables configured in Dokploy UI
- Secrets come from Phase (not hardcoded)

**Never commit secrets.** Use environment variables and Phase for all sensitive config.

### Code Quality

Before committing:
```bash
# Format and lint
ruff check --fix .
ruff format .

# Type check (if using types)
pyright

# Tests
pytest
```

---

## Clean Changes Over Backwards Compatibility

**When making design or architecture changes, lean into them fully.** Don't clutter code with backwards compatibility layers, fallbacks, or migration shims unless explicitly needed.

**Most projects here are:**
- Small and actively developed
- Not in production (or easily redeployable)
- Not consumed by external users
- Better served by clean code than compatibility

**Avoid these patterns unless explicitly requested:**

```python
# BAD - unnecessary compatibility layer
def get_user(user_id: str | int):  # Why support both?
    if isinstance(user_id, int):
        user_id = str(user_id)  # Legacy fallback
    ...

# BAD - keeping old code paths
def process(data, use_new_engine=True):  # Just use the new engine
    if use_new_engine:
        return new_process(data)
    return old_process(data)  # Dead code waiting to happen

# BAD - deprecation warnings in small projects
import warnings
warnings.warn("Use new_function instead", DeprecationWarning)  # Just delete old_function

# BAD - config fallbacks
config.get("new_key") or config.get("old_key") or DEFAULT  # Just use new_key
```

**Instead, make the clean change:**

```python
# GOOD - pick one type and use it
def get_user(user_id: str):
    ...

# GOOD - just use the new implementation
def process(data):
    return new_engine.process(data)

# GOOD - delete the old function, update callers
# (old_function is gone, callers updated)

# GOOD - use the new config key everywhere
config.get("new_key", DEFAULT)
```

**When backwards compatibility IS appropriate:**
- Public APIs with external consumers
- Deployed production systems with gradual rollout
- Shared libraries used by multiple projects
- User explicitly requests compatibility period

**When in doubt:** Ask "Is anyone actually using the old way?" If not, delete it.

---

## AI Planning Documents

AI assistants often create planning/design documents during development:
- PLAN.md, IMPLEMENTATION.md, ARCHITECTURE.md
- DESIGN.md, CODEBASE_SUMMARY.md, INTEGRATION_PLAN.md
- TESTING_GUIDE.md, TECHNICAL_DESIGN.md, etc.

**Best practice: Use a `history/` directory** for these ephemeral files:

```bash
project/
├── history/              # AI-generated planning docs (ephemeral)
│   ├── 2024-01-15-auth-design.md
│   └── 2024-01-20-api-refactor.md
├── src/                  # Actual code
└── README.md             # Permanent documentation
```

**Benefits:**
- Clean repository root
- Clear separation between ephemeral and permanent docs
- Easy to exclude from version control if desired
- Preserves planning history for archeological research

**Optional .gitignore entry:**
```
# AI planning documents (ephemeral)
history/
```

---

## Issue Tracking with ticket (tk)

We use **ticket (`tk`)** for issue tracking instead of Markdown TODOs or external tools. Tickets are Markdown files with YAML frontmatter stored in `.tickets/`.

### Quick Start

```bash
tk create "Short title" -d "Why/what" -t task -p 2
tk start <id>
tk add-note <id> "Progress update"
tk close <id>
```

### CLI Quick Reference

```bash
# Find work
tk ready
tk blocked
tk ls --status=open

# Create and manage tickets
tk create "Issue title" -d "Detailed context" -t bug|feature|task|epic|chore -p 0-4
tk start <id>
tk status <id> in_progress
tk close <id>

# Details and notes
tk show <id>
tk add-note <id> "Started investigating - found root cause in auth.py"
tk edit <id>

# Dependencies and links
tk dep <id> <dep-id>         # <id> depends on <dep-id>
tk dep tree <id>
tk link <id> <other-id>
```

### Workflow

1. **Check for ready work**: `tk ready`
2. **Claim your task**: `tk start <id>` (or `tk status <id> in_progress`)
3. **Work on it**: Implement, test, document
4. **Capture context**: `tk add-note <id> "..."` for discoveries and blockers
5. **Complete**: `tk close <id>`
6. **Commit**: Include `.tickets/*.md` with code changes

### Issue Quality

Always include meaningful descriptions and acceptance criteria.

**Good example:**

```bash
tk create "Fix auth bug in login handler"   -d "Login fails with 500 when password contains quotes. Found while testing feature X. Stack trace shows unescaped SQL in auth/login.go:45."   -t bug -p 1 --acceptance "- [ ] Tests added/updated
- [ ] README.md updated (if user-facing)
- [ ] Docs updated (if applicable)"
```

**Bad examples (missing context):**

```bash
tk create "Fix auth bug" -t bug -p 1
tk create "Add feature" -t feature
tk create "Refactor code" -t task
```

### Types and Priorities

- `bug` - Something broken
- `feature` - New functionality
- `task` - Tests/docs/refactors
- `epic` - Large feature with subtasks
- `chore` - Maintenance

Priorities:
- `0` Critical
- `1` High
- `2` Medium
- `3` Low
- `4` Backlog

### Dependencies and Links

- Use `tk dep` for blocking prerequisites.
- Use `tk link` for "see also"/related work.
- `tk blocked` shows tickets with unresolved deps.


## GitHub Issues and PRs

When asked to check GitHub issues or PRs, use `gh` CLI instead of browser tools:

```bash
# List open issues
gh issue list --limit 30

# List open PRs
gh pr list --limit 30

# View specific issue
gh issue view 201
```

**Why CLI over browser:**
- Browser tools consume more tokens and are slower
- CLI summaries are easier to scan and discuss
- Keeps the conversation focused and efficient

---

## Session Completion (Landing the Plane)

**When ending a work session**, you MUST complete ALL steps below. The plane is NOT landed until `git push` succeeds. NEVER stop before pushing. NEVER say "ready to push when you are!" - that is a FAILURE.

**MANDATORY WORKFLOW:**

1. **File tickets for remaining work** - Create tickets for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds (file P0 tickets if broken)
3. **Update ticket status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git add .tickets
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up git state**:
   ```bash
   git stash clear          # Remove old stashes
   git remote prune origin  # Clean up deleted remote branches
   ```
6. **Verify** - All changes committed AND pushed
7. **Choose follow-up ticket** - Pick next work and provide a prompt for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
- The user may be coordinating multiple agents - unpushed work breaks their workflow

**Example session:**

```bash
# 1. File remaining work
tk create "Add integration tests for sync" -d "..." -t task -p 2

# 2. Run quality gates (if code changed)
# [run your project's test/lint commands]

# 3. Close finished tickets
tk close <id>

# 4. PUSH TO REMOTE - MANDATORY
git pull --rebase
git add .tickets
git push
git status  # Verify "up to date with origin"

# 5. Clean up
git stash clear
git remote prune origin

# 6. Choose next work
tk ready
```

**Then provide the user with:**
- Summary of what was completed this session
- What tickets were filed for follow-up
- Confirmation that ALL changes have been pushed
- Recommended prompt for next session: "Continue work on ticket <id>: [title]. [Brief context]"


## Important Rules

- Use ticket for ALL task tracking
- Always include meaningful descriptions and acceptance criteria
- Link prerequisites with `tk dep` and related work with `tk link`
- Check `tk ready` before asking "what should I work on?"
- Store AI planning docs in `history/` directory, not repo root
- Do NOT create markdown TODO lists
- Do NOT duplicate tracking systems
- Do NOT clutter repo root with planning documents
