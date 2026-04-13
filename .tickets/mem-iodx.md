---
id: mem-iodx
status: closed
deps: []
links: []
created: 2026-02-08T08:00:19Z
type: feature
priority: 2
assignee: chriskd
---
# mx doctor: fix frontmatter timestamps from filesystem metadata

## Summary

Add support in `mx doctor` to detect and optionally correct entry frontmatter timestamps (`created`, `updated`) using filesystem metadata (best-effort).

Motivation: KBs often accumulate missing or incorrect timestamps (manual edits, imports, renames). That breaks recency-based workflows (`mx whats-new`, “recent entries” status display) and makes it harder for agents to trust timelines.

## Requirements / Constraints

- **Safety first**: default is report-only. Writing requires an explicit flag.
- **Best-effort creation time**:
  - Prefer filesystem birthtime when available (macOS / BSD `st_birthtime`).
  - Otherwise use `ctime` as a fallback (note: on Linux it is change-time, not true creation).
- **Updated time** should use filesystem `mtime`.
- Only modify entries that are missing timestamps or have invalid/non-parseable timestamps unless `--force` is provided.
- Must support both human output and `--json`.

## Proposed CLI UX

- Report: `mx doctor --timestamps`
- Apply: `mx doctor --timestamps --fix`
- Safety/controls:
  - `--dry-run` (compute and show changes without writing)
  - `--force` (overwrite even valid timestamps)
  - `--scope` / `--limit` consistent with other commands where applicable
  - `--json` for per-file before/after + source (birthtime|ctime|mtime)

## Acceptance Criteria

- [x] `mx doctor --timestamps` reports entries with missing/invalid `created` and/or `updated`.
- [x] `mx doctor --timestamps --fix` updates YAML frontmatter in-place using best-effort filesystem times (created from birthtime/ctime, updated from mtime).
- [x] Does not modify entries with valid timestamps unless `--force` is used.
- [x] Provides `--dry-run` and a clear summary (checked/changed/skipped); non-zero exit only on real errors.
- [x] `--json` includes per-file before/after timestamps and which source was used for each field.

## Notes

**2026-02-08T08:01:25Z**

Ticket body corrected (initial create command suffered shell backtick substitution).

**2026-04-07T00:27:09Z**

Implemented timestamp auditing/repair for `mx doctor` with new flags:
`--timestamps`, `--fix`, `--dry-run`, `--force`, `--scope`, and `--limit`.
Behavior:
- report mode surfaces only files with missing/invalid timestamps
- fix mode patches only `created` / `updated` lines in existing YAML frontmatter, preserving unknown keys/body content
- force mode recomputes even valid timestamps from filesystem metadata
- JSON includes per-file `before` / `after`, timestamp source, and changed/would-change state
- missing-frontmatter markdown files are skipped; malformed YAML/write failures are reported as real errors

Validation:
- `uv run pytest tests/test_cli.py -k doctor`
- `uv run pytest tests/test_cli.py`
- `uv run ruff check src/memex/doctor.py`
- `uv run mx doctor --timestamps --limit=3 --json`
