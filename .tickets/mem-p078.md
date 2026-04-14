---
id: mem-p078
status: closed
deps: []
links: [mem-iodx]
created: 2026-04-14T01:04:08Z
type: task
priority: 2
assignee: chriskd
---
# Evaluate mtime/ctime fallback for KB freshness when agents bypass mx edits

Agents often edit kb markdown files directly instead of using mx patch/append/replace, even when the skill says not to. That bypasses mx's normal frontmatter update flow for fields like updated and last_edited_by, which can leave freshness sorting and audit metadata stale. Evaluate whether Memex should rely more on filesystem timestamps (mtime/ctime or birthtime) for freshness, ordering, or repair when frontmatter is missing/stale, and document the tradeoffs versus keeping frontmatter as the canonical source. Include the current behavior in core/publisher/doctor, the impact on mx whats-new and published recent views, and whether a hybrid model or explicit repair workflow is safer.

## Acceptance Criteria

- [x] If code changes are recommended, identify the main modules/tests to update

## Notes

**2026-04-13T19:34:00Z**

Evaluation summary:

- Current behavior in `src/memex/core.py` keeps frontmatter canonical for CLI recency:
  - `whats_new()` uses `updated` first, then `created`
  - entries without those fields are omitted from recency output
- Current behavior in `src/memex/publisher/templates.py` keeps frontmatter canonical for published recent views:
  - `_recent_sort_key()` sorts by `updated` or `created`
- Current behavior in `src/memex/doctor.py` already supports filesystem-aware audit/repair:
  - `mx doctor --timestamps` compares frontmatter against birthtime/ctime and mtime
  - `--fix` / `--force` can repair stale or missing timestamp fields explicitly

Recommendation:

- Do **not** make filesystem timestamps the default freshness source for `mx whats-new` or published recent views.
- Filesystem `mtime`/`ctime` are too unstable across git checkout, copy/rsync, archive extraction, and publish/build workflows; using them as the default source would cause false "recent" spikes and make published ordering nondeterministic.
- Keep frontmatter as the canonical source of truth, and use the explicit repair workflow from `mem-iodx` when agents bypass `mx` edit commands.

If a future hybrid mode is still desired, the main code paths to update are:

- `src/memex/core.py`
  - `whats_new()`
- `src/memex/publisher/generator.py`
  - carry effective freshness metadata into publish-time entry data if recency should consider filesystem state
- `src/memex/publisher/templates.py`
  - `_recent_sort_key()`
- `src/memex/session_context.py`
  - only if recent-entry summaries need to expose source/repair hints

Main tests to update/add for a future hybrid mode:

- `tests/test_core.py`
  - `TestWhatsNew`
- `tests/test_cli.py`
  - `mx whats-new` command coverage
- publisher recency ordering coverage
  - add a focused publisher test file or extend existing publisher tests to assert stable recent ordering with mixed frontmatter/filesystem timestamps
