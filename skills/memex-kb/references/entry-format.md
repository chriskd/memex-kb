# Memex Entry Format Notes

This file is intentionally thin. The canonical source of truth lives in the repo KB, not in this skill.

Read these files when writing or editing entries:
- `kb/reference/entry-format.md`
- `kb/reference/cli.md`

Use this note as a quick selector for what to open.

## Read `kb/reference/entry-format.md` for

- required and optional frontmatter fields
- `description`, `aliases`, `draft`, and `archived`
- `semantic_links`
- typed `relations`
- wikilink resolution rules
- template guidance and built-in templates

Current high-signal facts from the canonical doc:
- Required frontmatter fields are `title` and `tags`.
- `description` is useful because it improves search result summaries.
- Typed relations use canonical snake_case relation types such as `depends_on`, `implements`, `extends`, `documents`, `references`, `blocks`, and `related`.

## Read `kb/reference/cli.md` for

- exact `mx add`, `mx quick-add`, `mx ingest`, `mx patch`, `mx append`, and `mx replace` flags
- `--json` output shapes
- scope-aware path behavior such as `@project/...` and `@user/...`
- `mx templates`, `mx suggest-links`, and `mx relations-*`
- `mx context show` / `mx context validate` when choosing the right write target
- `mx doctor --timestamps --fix` when direct file edits left timestamps stale

## Practical Defaults

- Reuse existing tags from `mx tags`.
- Prefer adding a short `description` for durable entries.
- Prefer `mx patch` over `mx replace` unless rewriting the whole entry.
- Use explicit scoped paths in edits when both project and user KBs are active.
- Prefer `mx add` or `mx ingest` over `mx quick-add` when you need an explicit `project` or `user` target.
