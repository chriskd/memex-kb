---
title: Relations Graph Overview
tags:
  - memex
  - relations
  - graph
created: 2026-01-15T06:32:04.180518+00:00
updated: 2026-01-25T19:15:11+00:00
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
git_branch: relations-field
last_edited_by: chris
---


# Relations Graph Overview

This page connects wikilinks, typed relations, and semantic links into a single graph.

## Published UI

Published KB pages surface typed relations in two places:

- **Entry panel**: "Typed Relations" shows outgoing vs incoming edges with direction arrows and type labels.
- **Graph view**: Typed relations render as solid edges with arrowheads; semantic links and wikilinks appear with distinct styles, and the controls let you filter by origin and relation type.

## Search neighbors

`mx search --include-neighbors` expands results using semantic links, typed relations, and wikilinks.
Use `--neighbor-depth` to control hop count (default 1).

## Query the relations graph

Use the dedicated relations command when you want the graph directly:

- `mx relations path/to/entry.md`
- `mx relations path/to/entry.md --depth=2`
- `mx relations path/to/entry.md --origin=relations --type=documents`
- `mx relations path/to/entry.md --graph --json`

`mx search --include-neighbors` remains the quickest way to expand search results through semantic links, typed relations, and wikilinks.

## Editing typed relations

Use the dedicated helpers when possible, or edit frontmatter directly:

- `mx relations-add path/to/entry.md --relation "reference/cli.md=documents"`
- `mx relations-remove path/to/entry.md --relation "reference/cli.md=documents"`

```yaml
relations:
  - path: reference/cli.md
    type: documents
  - path: ref/other.md
    type: implements
```
