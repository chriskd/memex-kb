---
id: m-2ce2
status: in_progress
deps: []
links: []
created: 2026-01-27T01:08:29Z
type: task
priority: 1
assignee: chriskd
---
# Switch session context install to update .claude/settings.json

Replace hook script install with direct Claude settings update using mx session-context. Remove setup-remote.sh and update docs/settings accordingly.

## Acceptance Criteria

- [ ] mx session-context --install updates .claude/settings.json with command mx session-context
- [ ] setup-remote.sh removed and settings updated
- [ ] Docs updated to show direct hook command

