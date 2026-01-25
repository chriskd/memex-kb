---
id: m-ed4d
status: closed
deps: []
links: []
created: 2026-01-24T22:13:42Z
type: task
priority: 4
assignee: chriskd
---
# Add relation type lint/taxonomy (optional)

Define a canonical set of relation types and provide a lint/check command to detect unknown or inconsistent types (non-blocking).

## Acceptance Criteria

- [ ] Document canonical relation types\n- [ ] Lint/check command identifies unknown types\n- [ ] Non-blocking by default (warnings)


## Notes

**2026-01-25T21:32:52Z**

Added canonical relation type taxonomy + linting: new relation_types module, core lint_relation_types, mx relations-lint (non-blocking w/ --strict), tests + docs updated.
