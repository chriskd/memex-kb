---
id: m-dac5
status: closed
deps: []
links: []
created: 2026-01-25T20:06:42Z
type: task
priority: 2
assignee: chriskd
---
# Docs: remove .kbcontext references and fix CLI flag docs

README + KB docs still reference .kbcontext and mx context init; kb/reference/cli.md lists unsupported flags (mx add -c, mx search -c) and omits several commands. See kb/guides/ai-integration.md, kb/reference/cli.md, kb/reference/entry-format.md for mismatches.

## Acceptance Criteria

- [ ] Remove/replace .kbcontext + mx context init references with .kbconfig guidance\n- [ ] Fix CLI flags for mx add/search (no -c for add; no -c for search)\n- [ ] Add missing commands/flags to kb/reference/cli.md\n- [ ] Remove/update mx add --template example in kb/reference/entry-format.md\n- [ ] Validate examples against current mx --help output


## Notes

**2026-01-25T21:08:36Z**

Updated CLI docs to match current mx --help (search/add flags, new commands, global options), removed non-existent relations commands, and refreshed entry-format examples/templates. Removed .kbcontext references in AI integration + focusgroup eval docs.
