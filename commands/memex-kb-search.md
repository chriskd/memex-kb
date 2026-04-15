---
name: memex-kb-search
description: Search the knowledge base
allowed-tools:
  - Bash
argument-hint: "<query>"
---

Search the knowledge base with the user’s query.

## Workflow

1. Run `mx search "<query>"`.
2. If the user wants narrower matching, retry with `--mode=keyword`, `--mode=semantic`, or `--scope=project`.
3. Present the top matches with title, path, score, and a short snippet when available.

## Useful Flags

- `--content` - include full entry content in results
- `--strict` - keep weak semantic matches out
- `--include-neighbors` - pull in linked entries around the matches
- `--neighbor-depth=N` - control neighbor hops, default `1`
- `--terse` - paths only
- `--full-titles` - avoid title truncation
- `--json` - machine-readable output

## Examples

```bash
mx search "deployment"
mx search "deployment" --mode=keyword
mx search "deployment" --mode=semantic --limit=5
mx search "config" --scope=project --include-neighbors --neighbor-depth=2
mx search "docker" --tags=infrastructure --content
```

If nothing useful comes back, try a broader query or switch modes before assuming the KB is missing the topic.
