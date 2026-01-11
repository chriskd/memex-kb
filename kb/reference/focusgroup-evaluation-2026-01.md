---
title: Focusgroup Agent UX Evaluation (2026-01)
tags: [focusgroup, agent-ux, evaluation, cli]
created: 2026-01-11
---

# Focusgroup Agent UX Evaluation (2026-01)

Comprehensive evaluation of `mx` CLI agent usability using [focusgroup](https://github.com/chriskd/focusgroup) - 5 sessions with 12 total agents (8 Claude, 4 GPT).

## Overall Rating: 8-8.5/10

## Sessions

| Session ID | Focus | Agents | Key Finding |
|------------|-------|--------|-------------|
| 20260111-cb7d8f20 | General usability | 4 Claude | Validation crashes, JSON error gaps |
| 20260111-d86b9983 | General usability | 4 Claude | 8.5/10 rating |
| 20260111-2b993462 | Error messages | 2 Claude | 80% self-service, no command suggestions |
| 20260111-7c39a291 | Intuition test | 4 Claude | 100% command convergence |
| 20260111-6e6f778b | Intuition test | 4 GPT | More variance, same flag mistakes |

## Key Strengths

| Feature | Why It Works |
|---------|--------------|
| Token-conscious output | `--compact`, `--terse`, `--json` serve different context budgets |
| Schema introspection | `mx schema` - "agent gold" for self-discovery |
| Batch operations | `mx batch` reduces round-trips |
| Dry-run support | `mx add --dry-run` allows validation |
| Error handling | `--json-errors` with codes like `ENTRY_NOT_FOUND` (1002) |

## Issues Found

### P0: Crashes
- `mx search "test" --limit 0` - Python traceback
- `mx list --category=nonexistent` - Python traceback  
- `--json-errors` doesn't catch Click validation failures

### P1: Flag Discoverability
- All 8 agents guessed `--find`/`--replace` for patch (actual: `--old`/`--new`)
- Cross-model consensus = strong signal flags are unintuitive

### P2: Command Suggestions
- Flag typos get suggestions: `--jason` → "Possible options: --json"
- Command typos don't: `mx serach` → no "Did you mean 'search'?"

## Agent-Recommended Patterns

```bash
# Session start
mx prime

# Precise search (avoid semantic surprises)
mx search "query" --strict --terse --limit=1

# Check existence
mx get path.md --metadata  # Exit code 0/1

# Batch for efficiency
echo -e "search 'docker'\nget path/file.md" | mx batch

# Safe writes
mx add --title="..." --dry-run --json
```

## Cross-Model Comparison

| Metric | Claude | GPT |
|--------|--------|-----|
| Response variance | 0% (identical) | High (3+ variants) |
| Correct commands | 8/8 | 8/8 |
| Correct flags | ~6/8 | ~5/8 |

Both models guessed wrong on patch flags - validates this is a real UX gap.

## Related Beads Issues

- `voidlabs-kb-ktfu` - Epic: mx CLI Agent UX Improvements
- `voidlabs-kb-i24u` - Patch flag rename (--old/--new → --find/--replace)
- `voidlabs-kb-77q4` - Flag discoverability
- `voidlabs-kb-skeb` - Error message improvements
- `voidlabs-kb-pgml` - General agent usability
- `voidlabs-kb-jll6` - Append workflow confusion

## Session Logs

Retrieve full session data:
```bash
focusgroup logs show 20260111-cb7d8f20
focusgroup logs show 20260111-2b993462
focusgroup logs show 20260111-7c39a291
focusgroup logs show 20260111-6e6f778b
```
