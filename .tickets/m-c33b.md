---
id: m-c33b
status: in_progress
deps: []
links: []
created: 2026-01-26T23:24:40Z
type: task
priority: 1
assignee: chriskd
---
# Move session context hook into CLI

User wants session-context hook output as CLI option with optional install to create local hook script in project. Also stop tracking private directories: hooks/, .config/, wt.toml, .devcontainer (add to .gitignore and remove from git). Update docs/AGENTS to reflect new hook flow.

## Acceptance Criteria

- [ ] mx session-context outputs dynamic project-relevant KB context
- [ ] mx session-context --install creates local hook script in project (ignored by git)
- [ ] hooks/, .config/, wt.toml, .devcontainer are gitignored and removed from repo
- [ ] Docs/AGENTS updated for new hook installation

