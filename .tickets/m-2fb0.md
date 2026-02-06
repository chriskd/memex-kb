---
id: m-2fb0
status: closed
deps: []
links: []
created: 2026-01-29T16:14:30Z
type: task
priority: 2
assignee: chriskd
---
# mx add: warn on duplicate frontmatter in --file input

If --file content already has YAML frontmatter, warn or merge instead of writing duplicate frontmatter.

## Notes

**2026-02-06**

- `mx add` now strips leading YAML frontmatter from `--file`/`--stdin`/`--content` inputs and warns.
- Optional fields from the input frontmatter (description/status/aliases/etc.) are preserved via a metadata merge; frontmatter is not duplicated.
- Added a CLI test that asserts only one YAML frontmatter block is written when the input file already has one.
