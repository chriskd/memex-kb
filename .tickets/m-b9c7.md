# AI Agent Onboarding: Cold-Start UX Evaluation & Recommendations

**Status:** ready
**Priority:** high
**Tags:** ux, ai-agent, onboarding, cli, discoverability

## Summary

Three AI agents independently explored `mx` from zero context, each using a different strategy (sequential discovery, task-based discovery, and mental-model building). This ticket synthesizes their findings into prioritized, actionable improvements for the "zero to productive" agent experience.

## Methodology

- **Agent 1 (Sequential Discovery):** Ran 34 commands in sequence, documenting friction at each step
- **Agent 2 (Task-Based):** Tried to accomplish 5 specific tasks without reading documentation
- **Agent 3 (Mental Model):** Focused on building conceptual understanding through progressive exploration

All agents worked in a clean worktree with `memex-kb` installed without search dependencies (the common agent environment).

---

## Critical Bug Found

### `mx` (bare command) crashes with NameError

**All 3 agents** hit this as their very first command:

```
NameError: name 'has_keyword_search' is not defined
```

The status dashboard (`_output_status` in `cli.py`) references a variable that doesn't exist in the current code. This is the worst possible first impression — the most natural first command a user or agent would try produces a Python traceback.

**Fix:** The `_show_status()` function called when `ctx.invoked_subcommand is None` calls `_output_status()` which references `has_keyword_search`. This variable was likely removed during the search-optional refactor but the reference wasn't cleaned up.

**Location:** `src/memex/cli.py`, `_output_status` function

---

## Bugs Found During Exploration

### 1. `--json-errors` inconsistent across commands
- `mx --json-errors add --title="test"` → produces JSON errors correctly
- `mx --json-errors get nonexistent.md` → produces **plain text** error, not JSON
- Breaks programmatic agent error handling

### 2. `mx reindex` shows full traceback for missing deps
- `mx search` and `mx suggest-links` give clean one-line errors for missing whoosh
- `mx reindex` spews a 40-line Python traceback for the same condition
- Should catch the MemexError and present it cleanly

### 3. `mx info` entry count doesn't match `mx list`
- `mx info` reported 16 entries while `mx list --full-titles` showed 19
- Likely a scoping or caching difference, but confusing for verification

### 4. CRITICAL: `mx add` succeeds but returns exit code 1 (Agent 2 finding)
- `mx add` **creates the file successfully** but returns **exit code 1** because post-creation indexing fails (missing whoosh)
- An agent checking `$?` will think the add failed and may **retry, creating duplicates**
- **Fix:** Return exit code 0 with a warning when the file was created but indexing failed

### 5. CRITICAL: `mx delete` is entirely blocked by missing search deps (Agent 2 finding)
- `mx delete <path>` returns exit code 1 and **does NOT delete the file**
- The delete is blocked because de-indexing requires whoosh
- Combined with bug #4, this means: agents can create entries (partially) but **cannot remove them** through the CLI
- **Fix:** Allow file deletion even without search deps; skip de-indexing gracefully

### 6. `mx add --category=nonexistent` silently creates a new category directory (Agent 2 finding)
- No warning that a brand new category is being created
- Agent typo in `--category` creates a new directory with no undo path
- **Fix:** Warn when creating a new category, or list valid categories in the error

---

## Friction Points (Ranked by Impact)

### P0 — Blocks Agent Productivity

1. **`mx add`/`mx delete` asymmetry without search deps** (Agent 2 finding)
   - `add` creates file but returns failure exit code → agents retry → duplicates
   - `delete` is entirely blocked → agents cannot clean up
   - This is the most dangerous issue for unattended agent workflows

2. **Search unavailable without optional deps, no fallback**
   - `mx search` is the first command shown in `--help` and `mx prime`, but it fails out of the box
   - No built-in fallback (e.g., substring matching on titles/tags/content)
   - Agents that can't search can only browse by path, which is significantly less useful
   - **Recommendation:** Add lightweight grep-based fallback when whoosh is missing. Even matching against titles and tags would be valuable.

3. **`mx` bare command crash** (see Critical Bug above)

### P1 — Slows Agent Discovery

3. **`mx list` truncates paths and titles silently**
   - No visual indicator that truncation has occurred
   - `--full-titles` only un-truncates titles, not paths (misleading flag name)
   - **Recommendation:** Add footer when truncation occurs: `Tip: Use --full-titles to see full values`

4. **Path-based `mx get` has no "did you mean?" suggestions**
   - `mx get --title="Nonexistent"` gives fuzzy suggestions (great!)
   - `mx get nonexistent.md` just says "Entry not found" with no help
   - **Recommendation:** Add path-based fuzzy matching, or at minimum suggest `mx list` to find entries

5. **No `mx categories` command**
   - `--category` is required on `mx add` unless `.kbconfig` sets primary
   - Only way to discover available categories is `mx info` (which lists them as a side effect)
   - **Recommendation:** Either add `mx categories` or make the error message from `mx add` list available categories

### P2 — Minor Friction

6. **`mx relations` (no args) error message is vague**
   - Says: `Error: PATH is required unless --graph is specified.`
   - Should give an example: `Example: mx relations reference/cli.md`

7. **`mx search` (no deps) doesn't suggest alternatives**
   - When search fails for missing deps, could suggest: `Tip: Use 'mx list --tags=<tag>' for basic filtering`

8. **`mx prime` doesn't flag missing search deps**
   - An agent using `mx prime` for onboarding gets told to use `mx search` without knowing it's broken

---

## What Worked Well (Consistent Across All 3 Agents)

### Praised by all agents:
- **`mx --help`** — Grouped sections (Quick start, Create, Browse, Agent helpers) provide a natural learning path
- **`mx prime`** — Called "outstanding" by Agent 1. A single command gives agents everything they need
- **`mx tree`** — Instant spatial orientation of the KB
- **`mx doctor`** — Clear diagnosis with actionable fix instructions
- **`mx health`** — Useful health score with specific, actionable problems listed
- **`mx get --title` fuzzy matching** — Great UX when titles don't match exactly
- **Command typo correction** — `mx serach` → "Did you mean 'search'?" works well
- **Cross-command "See also"** — Help text for one command suggests related commands
- **Consistent `--json` flags** — Every command supports JSON output for programmatic use
- **`mx help <command>` alias** — Ergonomic alternative to `mx search --help`
- **`mx schema --compact`** — Agents can introspect the full command structure programmatically

---

## Ideal Agent Onboarding Sequence

Based on all 3 agents' experiences, the optimal "zero to productive" path:

| Step | Command | Purpose | Status |
|------|---------|---------|--------|
| 1 | `mx` | Status dashboard: KB root, entry count, suggested commands | BROKEN (crash) |
| 2 | `mx prime` | Full agent context: workflow, frontmatter rules, quick reference | Works great |
| 3 | `mx tree` | Spatial orientation of KB structure | Works great |
| 4 | `mx tags` | Discover tag vocabulary for filtering | Works great |
| 5 | `mx list --tags=<tag>` | Browse entries by topic | Works great |
| 6 | `mx get <path>` | Read an entry | Works great |
| 7 | `mx search "query"` | Find entries by content | BROKEN (no deps) |
| 8 | `mx health` | Understand KB quality/gaps | Works great |

**Key insight:** 6 of 8 steps work perfectly. Fixing the 2 broken steps (bare `mx` crash and search fallback) would make the onboarding path seamless.

---

## Recommended Changes (Priority Order)

### Must Fix
1. Fix `_output_status` NameError for bare `mx` command
2. Fix `mx add` exit code: return 0 when file created, even if indexing fails (warn instead)
3. Fix `mx delete` to work without search deps: delete file, skip de-indexing gracefully
4. Add search fallback for when whoosh/chromadb are missing (even simple title/tag substring matching)
5. Make `--json-errors` work consistently across all commands

### Should Fix
4. Clean up `mx reindex` traceback to use same one-line error as other commands
5. Add truncation hint footer to `mx list`
6. Add path fuzzy matching to `mx get <path>` (same as `--title` already does)
7. Add dependency status to `mx prime` output so agents know their limitations
8. Have `mx search` suggest `mx list --tags=<tag>` as fallback when deps are missing

### Nice to Have
9. Add `mx categories` command (or list categories in `mx add` error message)
10. Improve `mx relations` error to include an example
11. Make `--full-titles` also un-truncate paths, or rename to `--full-output`
12. Fix `mx info` entry count discrepancy with `mx list`

---

## Additional Insights from Agent 3 (Mental Model Approach)

Agent 3 took a unique "mental model building" approach and produced several insights not captured above:

### Command Discovery Graph

The discovery flow agents naturally follow radiates from `mx --help`:

```
mx (CRASHED) --> mx --help (primary discovery hub)
                    |
                    +--> mx prime (agent onboarding)
                    |       +--> mx info, mx context show, mx add,
                    |            mx list, mx get, mx health
                    |
                    +--> mx tree --> mx tags --> mx list --tags=X
                    |
                    +--> mx search (FAILED) --> mx doctor
                    |
                    +--> mx schema --compact (LLM integration)
                    +--> mx batch (multi-op)
                    |
                    +--> mx hubs --> mx relations <path>
```

**Key insight:** `mx --help` is the sole discovery hub. If bare `mx` crashes, agents must guess to try `--help`. Making `mx prime` more prominent (or the default bare-command behavior) would short-circuit the discovery graph.

### `mx session-context` vs `mx prime` are identical

Both commands produce the same output. The distinction is unclear. `session-context` is documented as being for hooks, `prime` for agent onboarding, but functionally they are the same.

### Truncated paths break agent workflows

`mx list` truncates paths like `reference/focusgroup-evaluation-mx-cli-dis...` — agents need exact paths for `mx get`, so truncated output forces a second call with `--json` or `--full-titles`. Default should show full paths since they are machine-readable identifiers.

### "Zoom In" Onboarding Pattern

The ideal mental model builds in layers:
1. **Tool overview** (mx --help) — what can I do?
2. **Agent recipe** (mx prime) — what should I do?
3. **KB structure** (mx info, mx tree, mx tags) — what exists?
4. **Reading entries** (mx get) — what do entries look like?
5. **Quality/capabilities** (mx health, mx doctor) — what works and what doesn't?

---

## Related Tickets

These existing tickets overlap with findings from this evaluation:

| Ticket | Title | Overlap |
|--------|-------|---------|
| m-08c6 | `mx search` crashes with ModuleNotFoundError | Same as finding #1 (search fallback) |
| m-fbc5 | `--json-errors` inconsistency | Same as bug #1 |
| m-6a8f | Entry count mismatch info vs health | Same as bug #3 |
| m-581f | `mx prime` should expose KB path + primary | Related to recommendation #7 |
| m-0022 | `mx prime` should list required frontmatter | Related to recommendation #7 |
| m-d451 | `mx prime` should mention `mx context show` | Related to recommendation #7 |
| m-f8d0 | Create 5-minute onboarding flow | Validated by this evaluation |
| m-0f9f | Quick-start should surface `--category` | Related to finding #5 |
| m-2fb0 | Warn on duplicate frontmatter with `--file` | Discovered by Agent 2 |
| m-f7be | Document orphan warnings in `mx health` | Noted by Agent 1 |

The **focusgroup evaluation** (`kb/reference/focusgroup-evaluation-mx-cli-discoverability-2026-01.md`) independently identified several of the same issues, particularly around `--json-errors` inconsistency and search dependency handling.

---

## Acceptance Criteria

- [ ] `mx` (bare) shows status dashboard without crash
- [ ] `mx add --title="..." --tags="..." --content="..."` returns exit code 0 when file created (even if indexing fails)
- [ ] `mx delete <path>` works without search deps (skips de-indexing gracefully)
- [ ] `mx search "query"` returns results even without whoosh (via fallback)
- [ ] `mx --json-errors get nonexistent.md` returns JSON error
- [ ] `mx reindex` without deps shows clean one-line error
- [ ] `mx prime` output includes dep status note when search is unavailable
- [ ] `mx add --category=nonexistent` warns before creating a new category
