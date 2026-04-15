---
name: memex-kb-quality
description: Audit KB health and find problems
allowed-tools:
  - Bash
argument-hint: ""
---

Audit the live knowledge base for health problems.

## Workflow

1. Run `mx health`.
2. Summarize the live issues:
   - orphaned entries
   - broken links
   - stale content
   - empty directories
3. If relation types look suspicious, run `mx relations-lint --strict`.
4. Point to the smallest fix that clears the issue.

## Automation

```bash
mx health
mx health --json
mx relations-lint --strict
```

## Triage

- Treat orphans as a navigation problem unless the entry is intentionally standalone.
- Treat broken links as immediate cleanup.
- Treat stale content as a review queue, not a failure by itself.
- Treat empty directories as either cleanup or a signal that an index file is missing.
