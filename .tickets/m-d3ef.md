---
id: m-d3ef
status: open
deps: []
links: []
created: 2026-01-27T05:08:11Z
type: task
priority: 2
assignee: chriskd
---
# Improve chunking + embedding context for search

Implement smarter chunking for search indexing: keep H2-based chunks but add token-based splitting for oversized sections/no-H2 docs with configurable max tokens + overlap. Include title + section heading in semantic embedding text to improve retrieval. Add config knobs and update indexing path accordingly. Avoid behavior changes for already well-structured docs.

## Acceptance Criteria

- [ ] H2-based chunking remains default for normal-sized sections
- [ ] Oversized sections and no-H2 docs are split by token window with overlap (configurable)
- [ ] New config options for max tokens/overlap are documented in config
- [ ] Embedding text includes document title + section heading
- [ ] Tests updated/added for chunking and embedding text

