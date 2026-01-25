---
id: m-7ad1
status: closed
deps: []
links: []
created: 2026-01-25T05:55:52Z
type: chore
priority: 2
assignee: chriskd
---
# Add pyright to dev tooling

`uv run pyright` failed because the pyright binary is not available. Decide whether
to add pyright to dev dependencies or update docs/CI to remove the requirement.

## Acceptance Criteria

- [ ] `uv run pyright` succeeds (pyright added to deps) or docs updated to remove it
- [ ] CI/dev docs reflect the chosen approach

## Notes

**2026-01-25T06:27:31Z**

Added pyright to dev deps via uv add, documented in CONTRIBUTING, and addressed pyright errors in cli/indexer/test. uv run pyright now passes.
