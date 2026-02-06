---
id: m-6a8f
status: open
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

- [ ] mx info and mx health counts match for the same KB scope\n- [ ] If counts intentionally differ, output labels explain why\n- [ ] Add a test or fixture that verifies count consistency

