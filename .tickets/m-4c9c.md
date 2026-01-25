---
id: m-4c9c
status: closed
deps: []
links: []
created: 2026-01-25T06:39:57Z
type: task
priority: 2
assignee: chriskd
---
# Investigate semantic/hybrid search startup latency

Perf sanity: hybrid/semantic search ~5-6s on small (~15 file) KB, keyword search ~1.5s. Likely heavy startup cost (embedding/model load).

## Acceptance Criteria

- [ ] Identify root cause of per-invocation latency (profiling/notes)\n- [ ] Implement mitigation (cache/model reuse/daemon/batch)\n- [ ] Document before/after timings


## Notes

**2026-01-25T19:39:21Z**

Findings: semantic/hybrid latency comes from SentenceTransformer model load on each CLI invocation (ChromaIndex._get_model). Implemented hybrid fast-path: if keyword results >= limit and top score >= 0.7, skip semantic to avoid model load. Config: MEMEX_HYBRID_SEMANTIC_FASTPATH=1 (default), MEMEX_HYBRID_SEMANTIC_FASTPATH_MIN_SCORE=0.7, can set FASTPATH=0 to force full hybrid. Timings (tmp KB with 15 entries, local code via memex tool venv): keyword=1.501s, semantic=5.689s, hybrid_fastpath_off=5.271s, hybrid_fastpath_on=1.439s.
