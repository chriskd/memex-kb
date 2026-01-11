---
name: kb-quality
description: Evaluate KB search accuracy
allowed-tools:
  - mcp__memex__quality
argument-hint: "[limit] [cutoff]"
---

Run the KB quality checks to understand current search accuracy.

## Workflow

1. Call `mcp__memex__quality(limit=<limit>, cutoff=<cutoff>)`
2. Present:
   - Overall accuracy percentage
   - Total queries evaluated
   - Table of each query with expected documents, actual hits, and whether it passed within the cutoff rank
3. Highlight any failing queries so authors can improve coverage

Defaults: `limit=5`, `cutoff=3` (expected document must appear in top 3 results).
