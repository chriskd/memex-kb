---
id: m-fbc5
status: closed
deps: []
links: []
created: 2026-01-29T20:38:55Z
type: bug
priority: 1
assignee: chriskd
---
# mx --json-errors should emit JSON consistently

Focusgroup report: running  still printed plain text. Need consistent JSON error payloads for programmatic use.

## Acceptance Criteria

- [ ] With --json-errors, all errors return JSON payloads (including missing entry)\n- [ ] Global flag usage is documented or the error message mentions correct placement\n- [ ] Tests cover JSON error output for a failing command

