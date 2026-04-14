# Memex Category Notes

Do not treat categories as global or static. Memex categories are just top-level directories in the active KB.

The canonical way to discover them is:

```bash
mx categories
mx info
mx tree --depth=2
```

## What To Do

1. Inspect the live KB before choosing a category.
2. Follow the repo's existing directory conventions.
3. Prefer the category where a future reader would look first.
4. Use tags and links for secondary topics instead of forcing category sprawl.

## Scope Matters

- A `project` KB often reflects repo-specific categories and conventions.
- A `user` KB may use a different structure entirely.
- When both are active, decide scope first, then category.

## Current Repo Example

In this repo today, `mx categories` reports:
- `design`
- `guides`
- `reference`

Treat that as an example of current state, not a rule for other Memex installs.

## Smells

Stop and re-check the live KB if you are about to:
- create a brand-new top-level directory without evidence that it is warranted
- force a topic into an old six-category taxonomy copied from historical docs
- choose a category before checking `.kbconfig`, `mx categories`, or `mx tree`
