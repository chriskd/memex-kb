---
id: m-db09
status: in_progress
deps: []
links: [m-3daa]
created: 2026-01-22T22:27:22Z
type: task
priority: 2
assignee: chriskd
---
# A-Mem component audit + keep/drop list

Inventory A-Mem parity components (LoCoMo adapter, eval runner, prompts, metadata extraction, evolution, linking) and decide keep/drop/repurpose for session memory vNext. Capture rationale and follow-up actions.

## Acceptance Criteria

- [ ] Audit doc lists components with keep/drop/repurpose\n- [ ] Rationale for each decision captured\n- [ ] Follow-up cleanup tickets created or linked\n- [ ] Roadmap KB entry updated with outcomes


## Notes

**2026-01-22T22:37:02Z**

Audited kb/a-mem-parity + tests/test_memory_amem.py + a-mem-init docs. Key callouts: a-mem-init CLI now does session backfill (backfill_session_entries), but kb/a-mem-parity/a-mem-init-command-specification.md still describes a-mem-init as general KB bootstrap (inventory/extract/link). Recommend either updating spec to match CLI or archiving as historical design and documenting amem_init_* as eval-only. Suggested archive candidates: a-mem-parity-audit-2026-01-15-paranoid.md, a-mem-parity-final-audit-2026-01-15.md, remaining-a-mem-implementation-gap.md (already status: archived), plus any worktree/branch-status snapshot sections. Keep living refs: a-mem-parity-analysis.md, a-mem-vs-mx-implementation-audit.md (post-2026-01-22 update), semantic-linking.md, entry-metadata-schema.md, keywords-and-embeddings.md, graph-aware-search.md, a-mem-strict-mode.md, memory-evolution-queue-architecture.md, strengthen-action-implementation.md, a-mem-evaluation-methodology.md, a-mem-test-cases-for-agent-evaluation.md. Cleanup tasks: fix mojibake/encoding (â/â/â¢ etc) in several docs, correct broken code blocks in keywords-and-embeddings.md, and align terminology (tags vs keywords) across parity docs.

**2026-01-22T22:38:33Z**

Audit focus: eval runner/LoCoMo adapter/prompts/CLI. Components: src/memex/eval/locomo.py adapter+enrichment, src/memex/eval/runner.py eval pipeline (temp KB, index, linking, keyword/tag extraction, evolution, LLM query+QA prompts), src/memex/eval/metrics.py LoCoMo QA metrics, src/memex/cli.py eval flags incl --amem defaults, scripts/evolution_ab_test.py, LoCoMo fixtures/tests. Rec: keep EvalRunner scaffolding + metadata/evolution pipelines; repurpose adapter/prompt/CLI to session-resume eval; drop/archive LoCoMo-specific categories/fixtures/QA metrics and evolution_ab_test after new eval harness lands.
