---
id: m-5b1a
status: closed
deps: []
links: []
created: 2026-01-25T06:39:53Z
type: task
priority: 3
assignee: chriskd
---
# Add perf sanity script for reindex/search/publish

No dedicated perf sanity tooling exists. Add a lightweight script to time mx reindex/search/publish with isolated indices/output.

## Acceptance Criteria

- [ ] scripts/perf-sanity.sh (or equivalent) runs reindex/search/publish and records timings\n- [ ] Script isolates indices/output under /tmp (or configurable)\n- [ ] README/docs mention how to run perf sanity check

