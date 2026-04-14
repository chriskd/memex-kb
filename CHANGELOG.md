# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-04-14

### Fixed

- **Delete safety**: `mx delete` now blocks removal when other entries point at the target via typed frontmatter relations, and surfaces the actual blocking references in CLI output
- **Strict relation linting**: `mx relations-lint --strict` now exits non-zero consistently in both text and JSON modes when issues are found
- **Scoped search filtering**: `mx search` and hidden alias `mx find` now respect `--scope` correctly in single-KB and multi-KB flows
- **Quick-add machine mode**: `mx quick-add --json --confirm` now creates the entry instead of stopping at analysis/preview output
- **Structured mutation errors**: `--json-errors` now applies consistently to the reported `append` and `patch` validation/runtime failures
- **Publisher timestamp fallback**: `mx publish` now includes hand-authored entries that omit `created` by falling back to filesystem timestamps instead of silently skipping them
- **Onboarding exit codes**: `mx onboard` now returns guidance with exit `0` on fresh no-KB directories instead of looking like a failure to shells and hooks
- **Suggest-links dependency UX**: `mx suggest-links` now surfaces the semantic-search install hint instead of degrading to an empty result set
- **Context no-config guidance**: `mx context show --json` and `mx context validate --json` now include explicit next-step commands when `.kbconfig` is missing
- **Session-start context fallback**: `mx session-context` now returns a no-KB payload in both JSON and text modes instead of exiting successfully with no output

## [0.3.0] - 2026-04-13

### Added

- **Timestamp doctor**: `mx doctor --timestamps` can audit and repair `created` / `updated` frontmatter from filesystem metadata, with `--fix`, `--dry-run`, `--force`, `--scope`, `--limit`, and JSON output
- **Scoped multi-KB operations**: project/user scoped paths (`@project/...`, `@user/...`) now flow through multi-KB add/get/list/update/patch/delete/search workflows more consistently
- **Effective recency fallback**: `mx whats-new` and published recent views now fall back to filesystem timestamps when frontmatter timestamps are missing, while `mx doctor --timestamps --fix` can sync stale `updated` values explicitly
- **Onboarding and discoverability improvements**: stronger KB-not-configured guidance, `mx onboard --init --yes`, `mx init --sample`, docs landing/index pages, and better unknown-command suggestions
- **Agent-oriented compact JSON**: improved `mx prime` / `session-context` compact JSON output, bounded output via `--max-bytes`, and expanded machine-readable command metadata
- **Memex KB skill bundle**: renamed and expanded Codex skill docs under `skills/memex-kb/`

### Changed

- **Search packaging**: keyword search now ships by default; semantic search remains optional so cold-start installs are lighter
- **Publisher UX**: redesigned published KB theme, improved base URL handling, and configurable landing-page entry support
- **Project naming cleanup**: command/skill/docs naming standardized around `memex-kb`

### Fixed

- **Semantic fallback**: operations that can proceed without semantic search now fall back cleanly to keyword-only behavior when semantic model initialization fails
- **Multi-KB ambiguity**: `mx get --title` now reports ambiguous matches across project/user scopes instead of silently picking one
- **Scoped backlinking**: fixed scoped semantic backlink updates in multi-KB mode
- **Parse-broken reads**: `mx get` now degrades gracefully for entries with invalid or missing frontmatter, instead of returning an opaque failure
- **Compact JSON truncation**: bounded compact JSON output now truncates primary entries predictably under `--max-bytes`

## [0.2.1] - 2026-01-27

### Added

- **Embedding cache**: Persistent cache for semantic embeddings, avoiding recomputation on KB changes
- **Token-based chunking**: Smarter document splitting using tiktoken for better semantic search quality
- **Evaluation harness**: `mx eval` command with precision/recall/MRR metrics for search quality testing

## [0.2.0] - 2026-01-27

### Added

- **Typed relations**: Frontmatter relations with canonical relation types, plus CLI helpers to add/remove and inspect relations
- **Typed relation linting**: `mx relations-lint` warns on unknown or inconsistent types (with `--strict` for CI)
- **Publisher typed-relations UX**: Relation type labels + direction in entry panels, with relation metadata in graph output
- **CLI** (`mx`): Token-efficient command-line interface
  - Search, browse, and modify KB entries
  - Designed for automation and scripting
  - Minimal output format reduces context usage
- **Hybrid Search**: Best-of-both-worlds search approach
  - BM25 keyword search via Whoosh
  - Semantic search via ChromaDB + sentence-transformers
  - Reciprocal Rank Fusion (RRF) for result merging
  - Context-aware boosting for project relevance
- **Bidirectional Links**: Wiki-style `[[link]]` syntax
  - Auto-resolved to existing entries
  - Backlink tracking across the KB
  - Stale link detection in health checks

### Changed

- **Search neighbors**: `mx search --include-neighbors` now includes typed relations with relation metadata

### Technical

- Python 3.11+ with type hints throughout
- Pydantic models for all data structures
- Async-first architecture
- Configurable via environment variables
