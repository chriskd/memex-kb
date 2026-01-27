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
relations:
  - path: design/relations-graph/wikilink-edge-behavior.md
    type: implements
  - path: design/relations-graph/frontmatter-edge-types.md
    type: depends_on
---


# Relations Graph Overview

This page connects wikilinks, typed relations, and semantic links into a single graph.

See [[design/relations-graph/wikilink-edge-behavior]] and [[design/relations-graph/frontmatter-edge-types]] for details.

## Published UI

Published KB pages surface typed relations in two places:

- **Entry panel**: "Typed Relations" shows outgoing vs incoming edges with direction arrows and type labels.
- **Graph view**: Typed relations render as solid edges with arrowheads; semantic links and wikilinks appear with distinct styles, and the controls let you filter by origin and relation type.

## Search neighbors

`mx search --include-neighbors` expands results using semantic links, typed relations, and wikilinks.
Use `--neighbor-depth` to control hop count (default 1).

## Query the relations graph

There isn't a dedicated CLI command yet. For now:
- Use `mx search --include-neighbors` to expand results through semantic links + typed relations + wikilinks
- Inspect `relations` in entry frontmatter (open the file or use `mx get path/to/entry.md --json`)

## Editing typed relations

Edit the `relations` frontmatter directly (or via `mx patch` / `mx replace`).

```yaml
relations:
  - path: reference/cli.md
    type: documents
  - path: ref/other.md
    type: implements
```
