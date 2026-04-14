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

- `src/memex/core.py`
  - `whats_new()` now uses filesystem timestamps as a recency fallback when frontmatter timestamps are missing
  - when frontmatter `updated` exists but is older than filesystem `mtime`, recency uses the newer filesystem timestamp for ordering/display
- `src/memex/publisher/generator.py` and `src/memex/publisher/templates.py`
  - published recent views now use the same effective recency rule for already-published entries
- `src/memex/doctor.py`
  - `mx doctor --timestamps` now reports stale `updated` values in addition to missing/invalid timestamps
  - `--fix` repairs stale `updated` values from filesystem `mtime` explicitly, without hidden write side effects in read commands

Recommendation / implemented approach:

- Keep frontmatter as the canonical stored metadata.
- Use effective recency (`max(frontmatter updated, filesystem mtime)`, plus filesystem fallback when timestamps are missing) for `mx whats-new` and recent views.
- Keep actual metadata repair explicit through `mx doctor --timestamps --fix`, rather than mutating files during `mx whats-new` or publish.

Main modules updated:

- `src/memex/core.py`
- `src/memex/doctor.py`
- `src/memex/publisher/generator.py`
- `src/memex/publisher/templates.py`
- `src/memex/timestamps.py`

Main tests updated/added:

- `tests/test_core.py`
  - `TestWhatsNew`
- `tests/test_cli.py`
  - timestamp doctor coverage
- `tests/test_publisher_relations.py`
  - recent ordering fallback coverage
