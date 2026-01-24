# Memex

A knowledge base system with CLI interface (`mx`).

## Worktree Workflow

When working on features or tasks that modify code:

1. **Create worktree**: `wt switch --create <branch-name>`
2. **cd into it**: `cd .worktrees/<branch-name>`
3. **Verify location**: Run `pwd` before making edits
4. **ALL file edits must be in the worktree**, never in main

This preserves conversation history (tied to main) while keeping work isolated.

When done:
```bash
wt merge  # merges and cleans up worktree
```

## Project Structure

- `src/memex/` - Main package
- `src/memex/cli.py` - CLI entry point (`mx` command)
- `tests/` - pytest test suite

## Development

```bash
uv sync              # Install dependencies
uv run pytest        # Run tests
uv run ruff check    # Lint
uvx ty check src/    # Type check
```
