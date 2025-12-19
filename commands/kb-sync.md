---
name: kb-sync
description: Commit and push knowledge base changes
allowed-tools:
  - mcp__plugin_voidlabs-kb_voidlabs-kb__sync
argument-hint: "[commit message]"
---

Commit and push knowledge base changes to git.

## Workflow

1. Call `mcp__plugin_voidlabs-kb_voidlabs-kb__sync` with the commit message
2. Report the result to the user

## Parameters

- **message**: Commit message (optional)
  - If provided: Use the user's message
  - If not provided: Default to "Update knowledge base"

## Example Usage

```
User: /kb-sync Added nginx proxy documentation

Claude: [calls sync with message "Added nginx proxy documentation"]
Successfully synced:
- Committed: "Added nginx proxy documentation"
- Pushed to remote
```

## Error Handling

If sync fails, report:
- What went wrong (no changes, git error, push rejected)
- Suggested fix if applicable

Common issues:
- "Nothing to commit" - No pending changes
- "Push rejected" - Remote has newer changes, may need pull first
