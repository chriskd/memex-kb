---
title: Chunking System Design
tags:
  - memex
  - architecture
  - chunking
  - a-mem
created: 2026-01-15T05:42:42.968730+00:00
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
git_branch: main
last_edited_by: chris
semantic_links:
  - path: a-mem-parity/a-mem-vs-mx-implementation-audit.md
    score: 0.632
    reason: embedding_similarity
  - path: a-mem-parity/a-mem-parity-analysis.md
    score: 0.632
    reason: embedding_similarity
  - path: a-mem-parity/a-mem-test-cases-for-agent-evaluation.md
    score: 0.622
    reason: embedding_similarity
---

# Chunking System for Memex

## Goal
Add configurable chunking that stores chunks **only in ChromaDB** while keeping markdown files whole. This enables:
1. **Precise snippet retrieval** (independent of A-Mem)
2. **A-Mem parity** when combined with evolution features
3. **LoCoMo-style evaluation** for testing efficacy

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Chunk storage | ChromaDB only | No file clutter, markdown stays human-readable |
| Default strategy | `headers` | Backward compatible; easy opt-in to `paragraph` |
| Priority strategy | `sentences` | Closest to A-Mem for parity testing |
| Frontmatter keywords | Document-level only | Chunk keywords live in ChromaDB metadata |
| Search default | Dedupe by file | Backward compatible, `--show-chunks` for granular |
| Eval suite | Specs now, impl Phase 2 | Ship chunking first, validate after |

## Configuration

```yaml
# .kbconfig
chunking:
  enabled: true
  strategy: paragraph    # headers, paragraph, semantic, sentences
  max_chunk_tokens: 256  # For semantic strategy
  overlap_tokens: 32     # Context continuity
  min_chunk_tokens: 20   # Avoid tiny chunks
```

## Implementation Phases

### Phase 1: Configuration (config.py)
- Add `ChunkingConfig` dataclass
- Add `get_chunking_config()` loader
- Update .kbconfig template in cli.py

### Phase 2: Chunking Strategies (parser/)
- Create `src/memex/parser/chunking.py` module
- Implement strategies:
  - `chunk_by_headers()` - refactor existing `_chunk_by_h2()`
  - `chunk_by_paragraph()` - split on double newlines
  - `chunk_by_semantic()` - sentence boundaries + token limits
  - `chunk_by_sentences()` - individual sentences (A-Mem-like)
- Add sentence splitting utility with offset tracking

### Phase 3: Index Schema Updates (indexer/)
- **ChromaDB ID format**: `path#chunk_{idx}` or `path#{section}#chunk_{idx}`
- **New metadata fields**: `chunk_idx`, `parent_section`, `chunk_strategy`, `start_offset`, `end_offset`
- **Update delete logic**: Query by path metadata, not ID pattern

### Phase 4: Models Update (models.py)
- Extend `DocumentChunk` with:
  - `chunk_idx: int`
  - `parent_section: str | None`
  - `chunk_strategy: str`
  - `start_offset: int`
  - `end_offset: int`

### Phase 5: Search & CLI Updates
- Add `--show-chunks` flag to search command
- Update SearchResult with chunk metadata
- JSON output includes full chunk info

### Phase 6: Migration & Reindex
- Store `chunking_strategy` in collection metadata
- Detect strategy mismatch, warn user
- Add `mx reindex --force-rechunk` command

## Critical Files

| File | Changes |
|------|---------|
| `src/memex/config.py` | Add ChunkingConfig dataclass |
| `src/memex/parser/chunking.py` | NEW - chunking strategies |
| `src/memex/parser/markdown.py` | Refactor to use chunking module |
| `src/memex/indexer/chroma_index.py` | New ID format, metadata fields |
| `src/memex/indexer/whoosh_index.py` | Match schema changes |
| `src/memex/models.py` | Extend DocumentChunk |
| `src/memex/cli.py` | Add --show-chunks, config template |
| `tests/test_chunking.py` | NEW - chunking tests |

## Semantic Chunking Algorithm

```
1. Split content into sentences with character offsets
2. Accumulate sentences until max_tokens reached
3. On overflow: create chunk, start new with overlap
4. Handle edge cases: code blocks, long sentences
```

## A-Mem Integration Points

When `amem.enabled` + `chunking.enabled`:
- Evolution updates `chunk_keywords` in ChromaDB (not frontmatter)
- `should_evolve` decision per chunk relationship
- Chunk-level semantic links possible

## Verification

1. **Unit tests**: Each strategy produces expected chunks
2. **Integration**: `mx reindex --force-rechunk` with each strategy
3. **Search**: `mx search "query" --show-chunks` returns chunk-level results
4. **Migration**: Strategy change detected, warning shown

## Phase 2: Evaluation Suite (Specification)

> Implementation deferred to Phase 2.

### Goal
Test A-Mem evolution efficacy using LoCoMo-style methodology, adapted for document-centric KB.

### Metrics (from A-Mem paper)
```python
# src/memex/eval/metrics.py
- exact_match: bool          # Exact string equality
- f1: float                  # Token overlap F1 score
- rouge1_f, rouge2_f, rougeL_f: float  # ROUGE scores
- bleu1-4: float             # BLEU scores
- bert_f1: float             # BERTScore F1
- meteor: float              # METEOR score
- sbert_similarity: float    # Sentence-BERT cosine similarity
```

### Test Categories (from LoCoMo)
| Category | Tests | Memex Equivalent |
|----------|-------|------------------|
| Multi-hop | Connecting multiple memories | Cross-document links via evolution |
| Temporal | Date/time reasoning | Entry created/updated metadata |
| Open-domain | General retrieval | Standard search |
| Single-hop | Direct retrieval | Exact keyword match |
| Adversarial | "Not mentioned" detection | No-result handling |

### Evaluation Flow
```
1. Create test KB with known entries
2. Add entries, let evolution run
3. Query with test questions
4. Compare retrieved content to expected answers
5. Calculate metrics across categories
```

### CLI Command (Phase 2)
```bash
mx eval --dataset tests/eval/locomo_adapted.json
mx eval --compare-strategies headers,paragraph,sentences
mx eval --with-evolution --without-evolution  # A/B test
```

### Adaptation for Document-Centric KB

**LoCoMo (conversational):**
- Each utterance = 1 memory note
- ~1-2 sentences per note

**Memex adaptation:**
- Each paragraph/section = 1 chunk (with `sentences` strategy)
- Test questions query for specific facts within chunks
- Evolution should improve retrieval by updating chunk keywords

### Test Dataset Format
```json
{
  "entries": [
    {
      "path": "test/doc1.md",
      "content": "...",
      "chunks_expected": 5
    }
  ],
  "qa": [
    {
      "question": "What configuration is needed for X?",
      "answer": "Set Y=Z in config",
      "category": 4,
      "expected_chunk": "test/doc1.md#chunk_2"
    }
  ]
}
```

