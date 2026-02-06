---
id: m-6a8f
status: closed
deps: []
links: []
created: 2026-01-29T20:39:01Z
type: bug
priority: 2
assignee: chriskd
---
# Entry count mismatch between mx info and mx health

Focusgroup agents observed mx info reporting 16 entries while mx health reported 15 on the same KB. Counts should align or be clearly labeled as different scopes.

## Acceptance Criteria

- [x] mx info and mx health counts match for the same KB scope
- [x] If counts intentionally differ, output labels explain why
- [x] Add a test or fixture that verifies count consistency

## Notes

**2026-02-06**

- `mx info` now counts parsed entries using the same rule as `mx health` (and reports parse errors per KB).
- Added CLI test to assert `mx info --json` primary KB entry count matches `mx health --json` `summary.total_entries`.
