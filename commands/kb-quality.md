---
name: kb-quality
description: Audit KB health and find problems
allowed-tools:
  - Bash
argument-hint: ""
---

Audit the knowledge base for health issues and problems.

## Workflow

1. Run `mx health` to audit the KB
2. Present findings:
   - Orphaned entries (no links to them)
   - Broken links (links to non-existent entries)
   - Stale content (old entries needing review)
   - Empty directories
3. Suggest fixes for any issues found

## Example

```bash
mx health           # Human-readable output
mx health --json    # JSON format for programmatic use
```

## What It Checks

- **Orphans**: Entries with no inbound links
- **Broken links**: `[[links]]` pointing to non-existent entries
- **Stale content**: Entries not updated in a long time
- **Empty directories**: Folders with no entries
