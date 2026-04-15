---
name: memex-kb-add
description: Add or update a knowledge base entry
allowed-tools:
  - Bash
argument-hint: "[title]"
---

Add KB content with the current Memex CLI. Prefer deterministic commands over interactive
questionnaires.

## Workflow

1. Resolve the active KB context:
   - `mx info` to confirm project/user KB roots and active scope
   - `mx context show` when you need the resolved project context
   - `mx context validate` if path resolution looks wrong

2. Check for duplicates before creating new content:
   - Search by title or obvious keywords: `mx search "<title>" --scope=project`
   - If both KBs are active, search the relevant scope explicitly and use `@project/...` or
     `@user/...` when reading or editing an existing entry

3. Choose the right command:
   - `mx add` when you already have the title and want to set metadata explicitly
   - `mx quick-add` when you have Markdown content and want Memex to infer the rest
   - `mx replace` when you meant to update an existing entry instead of creating a duplicate

4. Write to the correct scope:
   - Use `--scope=project` or `--scope=user` when scope matters
   - Use scoped paths like `@project/guides/setup.md` and `@user/notes/draft.md` when the target
     entry is already known
   - If the user did not specify a scope, follow the active scope from `mx info`

## Practical Rules

- Do not invent a category taxonomy. Use the repo's existing directory structure and configured
  write location instead of prompting for arbitrary buckets.
- Treat `--category` or path placement as a filesystem choice, not a content-classification form.
- Prefer updating an existing entry over creating a near-duplicate.
- Use `mx quick-add --stdin` when the source material is already Markdown or the assistant is
  capturing notes from another tool.

## Examples

```bash
mx info
mx context show
mx context validate

mx search "reverse proxy" --scope=project

mx add --scope=project --category=guides --title="Nginx reverse proxy" --tags="nginx,proxy" --content="..."
cat notes.md | mx quick-add --stdin --scope=user

mx replace @project/guides/setup.md --content="Updated content"
mx replace @user/notes/draft.md --tags="draft,ideas"
```

## Output Expectations

- Return the created or updated path, ideally in scoped form.
- If the KB is ambiguous, state which scope was used and why.
- If duplicate content was found, recommend updating the existing entry instead of adding a new one.
