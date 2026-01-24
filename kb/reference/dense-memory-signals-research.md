---
title: Dense Memory Signals Research
tags:
  - agent-memory
  - research
  - memory
  - signals
created: 2026-01-13T17:42:43.882297+00:00
updated: 2026-01-14T23:36:09.300430+00:00
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
semantic_links:
  - path: a-mem-parity/semantic-linking.md
    score: 0.62
    reason: bidirectional
  - path: a-mem-parity/a-mem-parity-analysis.md
    score: 0.768
    reason: bidirectional
---

# Dense Memory Signals Research

Research into what contextual signals can be captured and injected to help agents maintain context across sessions. Goal: identify high-value, low-token signals.

## What Existing Tools Capture

| Tool | Primary Signals | Token Cost |
|------|----------------|------------|
| **claude-mem** | Tool observations, semantic summaries, session data | 50-1000 tokens (progressive) |
| **basic-memory** | Structured notes, relations, knowledge graph links | Medium |
| **Mem0** | Facts extracted from conversations (not transcripts) | ~90% reduction vs full context |
| **Letta/MemGPT** | Message buffer, core memory blocks, recall history, archival knowledge | Variable |
| **A-Mem** (NeurIPS 2025) | Atomic notes with keywords, tags, context, embeddings, links | 1,200-2,500 tokens vs 16,900 baseline |

## Signal Tiers

### Tier 1: High Value / Low Token Cost (The Gold)

| Signal | Tokens | Value | Notes |
|--------|--------|-------|-------|
| Decisions made + rationale | ~15 | Critical | Claude can't infer this |
| Gotchas/warnings discovered | ~10 | Critical | "X looks right but fails because Y" |
| What user corrected | ~10 | Critical | Direct mistake prevention |
| Files in staging (git) | ~5 | High | Interrupted work signal |
| Stash contents | ~5 | High | Parked work context |
| Branch name | ~3 | High | Feature context in 3 tokens |
| Build status | ~3 | High | "FAILING" changes everything |
| Files read but NOT modified | ~10 | High | Research context |

### Tier 2: Medium Value / Medium Token Cost

| Signal | Tokens | Value |
|--------|--------|-------|
| Files touched (with context) | ~15 | High |
| Summary of what was done | ~25 | High |
| Questions Claude asked | ~15 | High |
| Tool usage patterns | ~10 | Medium |
| Errors encountered | ~15 | Medium |
| TODOs added/removed | ~10 | Medium |

### Tier 3: Novel/Non-Obvious Signals

| Signal | Tokens | Notes |
|--------|--------|-------|
| Causal chains | ~30 | "X led to Y which caused Z" |
| Semantic links | ~20 | Related memories by meaning |
| User intent shifts | ~15 | "Started on X, pivoted to Y" |
| Failed tool calls | ~10 | What approaches didn't work |
| Search queries with no results | ~10 | Dead ends to avoid |

## Key Research Insights

1. **Atomic > Monolithic**: A-Mem treats memories as single, self-contained units with rich metadata. This enables 85% token reduction.

2. **Causality matters**: claude-mem's innovation is that every observation includes what came before and after. The LLM sees causality, not snapshots.

3. **Facts > Transcripts**: Mem0 extracts relevant facts, not full transcripts—reducing tokens ~90%.

## Scaling Strategies

As agents become more capable and touch more files, signal count explodes. Strategies:

### 1. Hierarchical Summarization
- File-level → Directory-level → Project-level
- "Modified 12 files in src/api/" vs listing all 12

### 2. Recency Decay
- Recent signals: full detail
- Older signals: compressed summaries
- Ancient signals: existence only ("worked on caching 2 weeks ago")

### 3. Semantic Clustering
- Group related files by purpose, not path
- "Modified auth flow (5 files)" vs listing each

### 4. Activity Thresholds
- Below threshold: list all
- Above threshold: summarize + highlight outliers
- "Touched 47 files, notably: new cache.ts, heavily modified api.ts"

### 5. Delta-Only for Repeats
- First session: full context
- Subsequent: only what changed since last injection

### 6. User-Guided Salience
- Let user mark signals as important/irrelevant
- Learn what matters per-project

## Proposed Dense Injection Format

```
[Session: 2d ago | Branch: feat/caching | Build: PASSING]

RESUMED: Implementing Redis cache layer
TOUCHED: cache.ts(added), api.ts(modified), redis.ts(read-only)
STAGED: cache.ts (incomplete - needs TTL logic)

DECISIONS:
- Chose Redis over Memcached (team familiarity)

GOTCHAS:
- redis.ts exports are CJS not ESM

CORRECTIONS:
- User: "Use connection pooling, not single client"
```

Estimated: ~120 tokens with very high information density.

## Sources

- [[https://github.com/thedotmack/claude-mem|claude-mem GitHub]]
- [[https://docs.basicmemory.com/|basic-memory Docs]]
- [[https://github.com/mem0ai/mem0|Mem0 GitHub]]
- [[https://www.letta.com/blog/agent-memory|Letta Agent Memory]]
- [[https://arxiv.org/abs/2502.12110|A-Mem Paper (NeurIPS 2025)]]
- [[https://alok-mishra.com/2026/01/07/a-2026-memory-stack-for-enterprise-agents/|2026 Enterprise Memory Stack]]

## Related

- [[reference/agent-memory-comparison.md|Agent Memory Comparison]]
- [[guides/agent-memory.md|Agent Memory Guide]]