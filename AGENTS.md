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
- **Factory Droid** - Local LLM agent (configured via `bd setup factory`)
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
| `hooks/session-context.sh` | Claude Code SessionStart hook |
| `scripts/new-project.sh` | Scaffolds new projects with devcontainer |
| `devcontainers/template/scripts/post-start-common.sh` | Shared container setup (Phase secrets, Factory droid, bd hooks) |

**Session hooks run automatically** and inject:
- `bd prime` output (ready issues + workflow reminders)
- bd upgrade detection (notifies when bd version changed)
- Daemon health warnings (version mismatch detection)
- Stale issues check (>30 days untouched)
- Comments and documentation guidance

**You don't need to run these manually** - they execute on session start. But understanding where they come from helps if you want to suggest improvements.

**To improve shared tooling:**
- Edit files in `/srv/fast/code/voidlabs-devtools`
- Changes apply to all projects on next session start
- Consider creating a bd issue for significant changes

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

## Issue Tracking with bd (beads)

We use **bd (beads)** for issue tracking instead of Markdown TODOs or external tools.

### Why CLI over MCP?

bd offers both CLI and MCP server interfaces. **We use CLI** because:
- Minimizes context usage - Only injects ~1-2k tokens via `bd prime` vs MCP tool schemas
- Lower latency with direct CLI calls
- Works universally across any AI assistant

### Initializing bd

For new projects or first-time setup:

```bash
bd init --quiet  # Critical for agent environments - prevents interactive prompts
bd hooks install # Install git hooks for auto-sync
```

The `--quiet` flag is essential for non-interactive agent environments - it automatically installs git hooks and configures the merge driver without prompting.

### Getting Context with bd prime

Use `bd prime` to get AI-optimized workflow context at the start of a session:

```bash
bd prime              # Outputs ready issues, stale work, and workflow reminders
bd prime --json       # JSON format for programmatic use
```

This is more comprehensive than `bd ready` - it includes workflow guidance alongside the issue list.

### CLI Quick Reference

**Essential commands for AI agents:**

```bash
# Find work
bd ready --json                                    # Unblocked issues
bd stale --days 30 --json                          # Forgotten issues

# Create and manage issues
bd create "Issue title" --description="Detailed context about the issue" -t bug|feature|task -p 0-4 --json
bd create "Found bug" --description="What the bug is and how it was discovered" -p 1 --deps discovered-from:<parent-id> --json
bd create "Subtask" --parent <epic-id> --json     # Hierarchical subtask (gets ID like epic-id.1)
bd update <id> --status in_progress --json
bd close <id> --reason "Done" --json

# Search and filter
bd list --status open --priority 1 --json
bd list --label-any urgent,critical --json
bd show <id> --json

# Comments (preserve context across sessions)
bd comment <id> "Started investigating - found root cause in auth.py"
bd comment <id> "Blocked: waiting on API changes in #ABC-123"

# Graph links
bd relate <id1> <id2>                              # Bidirectional "see also"
bd duplicate <id> --of <canonical>                 # Mark as duplicate

# Sync (CRITICAL at end of session!)
bd sync  # Force immediate export/commit/push

# CLI help - discover all available flags
bd <command> --help                               # e.g., bd create --help
```

### Auto-Sync Behavior

bd automatically syncs with git:
- Exports to `.beads/issues.jsonl` after changes (5s debounce)
- Imports from JSONL when newer (e.g., after `git pull`)
- No manual export/import needed during normal work

**Always commit `.beads/issues.jsonl` with code changes** so issue state stays in sync with code state.

### Workflow

1. **Check for ready work**: Run `bd ready` to see what's unblocked (or `bd stale` to find forgotten issues)
2. **Claim your task**: `bd update <id> --status in_progress`
3. **Work on it**: Implement, test, document
4. **Discover new work**: If you find bugs or TODOs, create issues with `discovered-from` links
5. **Complete**: `bd close <id> --reason "Implemented"`
6. **Sync at end of session**: `bd sync`

---

## IMPORTANT: Always Include Issue Descriptions

**Issues without descriptions lack context for future work.** When creating issues, always include a meaningful description with:

- **Why** the issue exists (problem statement or need)
- **What** needs to be done (scope and approach)
- **How** you discovered it (if applicable during work)

**Good examples:**

```bash
# Bug discovered during work
bd create "Fix auth bug in login handler" \
  --description="Login fails with 500 error when password contains special characters like quotes. Found while testing feature X. Stack trace shows unescaped SQL in auth/login.go:45." \
  -t bug -p 1 --deps discovered-from:bd-abc --json

# Feature request
bd create "Add password reset flow" \
  --description="Users need ability to reset forgotten passwords via email. Should follow OAuth best practices and include rate limiting to prevent abuse." \
  -t feature -p 2 --json

# Technical debt
bd create "Refactor auth package for testability" \
  --description="Current auth code has tight DB coupling making unit tests difficult. Need to extract interfaces and add dependency injection. Blocks writing tests for bd-xyz." \
  -t task -p 3 --json
```

**Bad examples (missing context):**

```bash
bd create "Fix auth bug" -t bug -p 1 --json  # What bug? Where? Why?
bd create "Add feature" -t feature --json     # What feature? Why needed?
bd create "Refactor code" -t task --json      # What code? Why refactor?
```

---

## Issue Comments

**Leave comments on issues as you work** - not just open/close:

```bash
bd comment <issue-id> "Started investigating - found root cause in auth.py"
bd comment <issue-id> "Blocked: waiting on API changes in #ABC-123"
bd comment <issue-id> "Implemented approach 2, running tests now"
```

**When to comment:**
- Starting work on an issue (what you're trying first)
- Finding important context (root cause, blockers, related issues)
- Changing approach (why you pivoted)
- Partial progress before session end (what's done, what's left)
- Discoveries that future sessions need to know

Comments are for **future you** and **other agents** - they preserve context across sessions.

---

## Issue Types

- `bug` - Something broken that needs fixing
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature composed of multiple issues (supports hierarchical children)
- `chore` - Maintenance work (dependencies, tooling)

## Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (nice-to-have features, minor bugs)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

---

## Dependency Types

**Blocking dependencies:**
- `blocks` - Hard dependency (issue X blocks issue Y)

**Structural relationships:**
- `parent-child` - Epic/subtask relationship
- `discovered-from` - Track issues discovered during work
- `related` - Soft relationship (issues are connected)

**Graph links:**
- `relates_to` - Bidirectional "see also" links (`bd relate <id1> <id2>`)
- `duplicates` - Mark issue as duplicate (`bd duplicate <id> --of <canonical>`)
- `supersedes` - Version chains (`bd supersede <old> --with <new>`)

Only `blocks` dependencies affect the ready work queue.

---

## Planning Work with Dependencies

When breaking down large features into tasks, use **beads dependencies** to sequence work - NOT phases or numbered steps.

### Cognitive Trap: Temporal Language Inverts Dependencies

Words like "Phase 1", "Step 1", "first", "before" trigger temporal reasoning that **flips dependency direction**. Your brain thinks:
- "Phase 1 comes before Phase 2" → "Phase 1 blocks Phase 2" → `bd dep add phase1 phase2`

But that's **backwards**! The correct mental model:
- "Phase 2 **depends on** Phase 1" → `bd dep add phase2 phase1`

**Solution: Use requirement language, not temporal language**

Instead of phases, name tasks by what they ARE, and think about what they NEED:

```bash
# WRONG - temporal thinking leads to inverted deps
bd create "Phase 1: Create buffer layout" ...
bd create "Phase 2: Add message rendering" ...
bd dep add phase1 phase2  # WRONG! Says phase1 depends on phase2

# RIGHT - requirement thinking
bd create "Create buffer layout" ...
bd create "Add message rendering" ...
bd dep add msg-rendering buffer-layout  # msg-rendering NEEDS buffer-layout
```

**Verification**: After adding deps, run `bd blocked` - tasks should be blocked by their prerequisites, not their dependents.

---

## Duplicate Detection & Merging

AI agents should proactively detect and merge duplicate issues to keep the database clean:

**Detection strategies:**

1. **Before creating new issues**: Search for similar existing issues
   ```bash
   bd list --json | grep -i "authentication"
   bd show bd-41 bd-42 --json  # Compare candidates
   ```

2. **During work discovery**: Check for duplicates when filing discovered-from issues

**Merge workflow:**

```bash
# Step 1: Identify duplicates (bd-42 and bd-43 duplicate bd-41)
bd show bd-41 bd-42 bd-43 --json

# Step 2: Preview merge to verify
bd merge bd-42 bd-43 --into bd-41 --dry-run

# Step 3: Execute merge
bd merge bd-42 bd-43 --into bd-41 --json
```

**Best practices:**
- Merge early to prevent dependency fragmentation
- Choose the oldest or most complete issue as merge target

---

## Pro Tips for Agents

- **Start sessions with `bd prime`** to get comprehensive context (issues + workflow guidance)
- Always use `--json` flags for programmatic use
- **Always run `bd sync` at end of session** to flush/commit/push immediately
- Link discoveries with `discovered-from` to maintain context
- Check `bd ready` before asking "what next?"
- Use `bd dep tree` to understand complex dependencies
- Priority 0-1 issues are usually more important than 2-4
- Use `--dry-run` to preview changes before applying
- Run `bd info --whats-new` after upgrades to learn about new features

---

## Why `bd sync` Matters

When you finish making issue changes, always run:

```bash
bd sync
```

This immediately:
1. Exports pending changes to JSONL (bypasses 30s debounce)
2. Commits to git
3. Pulls from remote
4. Imports any updates
5. Pushes to remote

**Without `bd sync`**, changes sit in a 30-second debounce window. The user might think you pushed but the JSONL is still dirty.

---

## Git Hooks

**Install hooks for automatic sync** (prevents stale JSONL problems):

```bash
bd hooks install
```

This installs:
- **pre-commit** - Flushes pending changes before commit
- **post-merge** - Imports updated JSONL after pull/merge
- **pre-push** - Exports database before push (prevents stale JSONL)
- **post-checkout** - Imports JSONL after branch checkout

**Why hooks matter:** Without pre-push, you can have database changes committed locally but stale JSONL pushed to remote, causing multi-workspace divergence.

---

## Upgrading bd

After upgrading bd to a new version:

```bash
bd info --whats-new   # See what changed in this version
bd hooks install      # Regenerate hooks to match new version
```

**Always regenerate hooks after upgrading** - hook behavior may change between versions, and stale hooks can cause sync issues.

---

## Multi-Agent Coordination

When multiple agents work on the same repository, bd provides coordination features.

### Agent Identity

Set your agent identity for audit trails and messaging:

```bash
export BEADS_IDENTITY="worker-1"  # Or "claude-main", "codex-reviewer", etc.
```

This identity appears in issue history and enables inter-agent messaging.

### Inter-Agent Messaging (bd mail)

Beads includes a built-in messaging system for direct agent-to-agent communication. Messages are stored as beads issues, synced via git.

**Commands:**

```bash
# Send a message
bd mail send <recipient> -s "Subject" -m "Body"
bd mail send worker-2 -s "Handoff" -m "Your turn on bd-xyz" --urgent

# Check your inbox
bd mail inbox

# Read a specific message
bd mail read bd-a1b2

# Acknowledge (mark as read/close)
bd mail ack bd-a1b2

# Reply to a message (creates thread)
bd mail reply bd-a1b2 -m "Thanks, on it!"
```

**Use cases:**
- Task handoffs between agents
- Status updates to coordinator
- Blocking questions requiring response
- Priority signaling with `--urgent` flag

**Cleanup:** Messages are ephemeral. Run `bd cleanup --ephemeral --force` to delete closed messages.

See [docs/messaging.md](docs/messaging.md) for full documentation.

### Deletion Tracking

When issues are deleted, bd tracks them in `.beads/deletions.jsonl` to propagate deletions across clones. This ensures consistency when multiple agents or workspaces interact with the same repository.

---

## Daemon Management

bd can run background daemons for faster sync and event-driven updates.

### Checking Daemon Status

```bash
bd daemons list --json     # List running daemons
bd daemons health --json   # Check daemon health
bd daemons logs . -n 100   # View recent daemon logs
```

### Stopping Daemons

```bash
bd daemons killall --json  # Stop all daemons
```

### Event-Driven Mode

For better performance with multiple agents:

```bash
export BEADS_DAEMON_MODE=events  # Reduces CPU ~60%, latency <500ms
```

This mode uses filesystem events instead of polling, significantly reducing resource usage.

---

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

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds (file P0 issues if broken)
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up git state**:
   ```bash
   git stash clear          # Remove old stashes
   git remote prune origin  # Clean up deleted remote branches
   ```
6. **Verify** - All changes committed AND pushed
7. **Choose follow-up issue** - Pick next work and provide a prompt for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
- The user may be coordinating multiple agents - unpushed work breaks their workflow

**Example session:**

```bash
# 1. File remaining work
bd create "Add integration tests for sync" --description="..." -t task -p 2 --json

# 2. Run quality gates (if code changed)
# [run your project's test/lint commands]

# 3. Close finished issues
bd close bd-42 bd-43 --reason "Completed" --json

# 4. PUSH TO REMOTE - MANDATORY
git pull --rebase
bd sync
git push
git status  # Verify "up to date with origin"

# 5. Clean up
git stash clear
git remote prune origin

# 6. Choose next work
bd ready --json
```

**Then provide the user with:**
- Summary of what was completed this session
- What issues were filed for follow-up
- Confirmation that ALL changes have been pushed
- Recommended prompt for next session: "Continue work on bd-X: [title]. [Brief context]"

---

## Important Rules

- Use bd for ALL task tracking
- Always use `--json` flag for programmatic use
- Always include meaningful descriptions when creating issues
- Link discovered work with `discovered-from` dependencies
- Leave comments to preserve context across sessions
- Check `bd ready` before asking "what should I work on?"
- Store AI planning docs in `history/` directory, not repo root
- Run `bd <cmd> --help` to discover available flags
- Do NOT create markdown TODO lists
- Do NOT duplicate tracking systems
- Do NOT clutter repo root with planning documents
