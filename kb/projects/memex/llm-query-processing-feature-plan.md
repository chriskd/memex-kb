---
title: LLM Query Processing Feature Plan
tags:
  - memex
  - llm
  - feature-plan
created: 2026-01-15T02:44:44.973987+00:00
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
git_branch: main
last_edited_by: chris
semantic_links:
  - path: reference/cli.md
    score: 0.794
    reason: embedding_similarity
  - path: reference/focusgroup-evaluation-mx-cli-discoverability-2026-01.md
    score: 0.782
    reason: embedding_similarity
  - path: guides/quick-start.md
    score: 0.769
    reason: embedding_similarity
  - path: a-mem-parity/a-mem-init-command-specification.md
    score: 0.731
    reason: embedding_similarity
  - path: a-mem-parity/keywords-and-embeddings.md
    score: 0.696
    reason: embedding_similarity
---

# Plan: LLM Query Processing for KB Entries

## Summary
Add opt-in LLM processing to KB entries with caching. Supports summarize, synthesize, and answer operations.

## CLI Interface

```bash
mx get path/entry.md --llm           # Summarize entry
mx get path/entry.md --llm --no-cache # Force fresh generation
mx search "query" --llm              # Synthesize results to answer query
mx ask "How do I deploy?"            # Shorthand for search + synthesize
mx ask "..." --show-sources          # Include source paths
mx llm-cache stats                   # Cache statistics
mx llm-cache clear                   # Clear cache
```

## Files to Create

### `src/memex/llm_query.py`
Core LLM operations:
- `content_hash(content: str) -> str` - SHA256 for cache keys
- `summarize_entry(entry: KBEntry) -> LLMResponse`
- `synthesize_entries(entries: list[KBEntry]) -> LLMResponse`
- `answer_query(query: str, entries: list[KBEntry]) -> LLMResponse`
- `_truncate_content()` - Handle token limits with tiktoken

### `src/memex/llm_cache.py`
Cache management (stored in `.indices/llm_cache.json`):
- `get_cached_response(operation, content_hashes, query) -> CachedLLMResponse | None`
- `cache_response(operation, paths, hashes, query, model, response, tokens)`
- `clear_cache() -> int`
- `cache_stats() -> dict`

Cache key: `sha256(operation + sorted(content_hashes) + query)[:16]`

## Files to Modify

### `src/memex/config.py`
Add `LLMQueryConfig` dataclass:
```python
@dataclass
class LLMQueryConfig:
    enabled: bool = False
    model: str = "anthropic/claude-3-5-haiku"
    max_input_tokens: int = 8000
    cache_enabled: bool = True
```

Add `get_llm_query_config()` (follow `get_memory_evolution_config()` pattern)

### `src/memex/models.py`
Add:
```python
class LLMResponse(BaseModel):
    operation: Literal["summarize", "synthesize", "answer"]
    input_paths: list[str]
    response: str
    model: str
    cached: bool = False
    token_count: int = 0
```

### `src/memex/cli.py`
1. Add to `get` command (line ~1189):
   - `--llm` flag
   - `--no-cache` flag

2. Add to `search` command (line ~973):
   - `--llm` flag
   - `--no-cache` flag

3. Add new `ask` command:
   - Search → load entries → call `answer_query()`
   - Options: `--limit`, `--show-sources`, `--no-cache`

4. Add `llm-cache` command group with `stats` and `clear` subcommands

## Config (.kbconfig)
```yaml
llm_query:
  enabled: true
  model: anthropic/claude-3-5-haiku
  max_input_tokens: 8000
  cache_enabled: true
```

## Cache Format (.indices/llm_cache.json)
```json
{
  "version": 1,
  "entries": {
    "cache_key_hash": {
      "operation": "summarize",
      "input_hashes": ["sha256..."],
      "input_paths": ["path.md"],
      "query": null,
      "model": "anthropic/claude-3-5-haiku",
      "response": "...",
      "created_at": "2024-01-14T12:00:00Z",
      "token_count": 150
    }
  }
}
```

## Implementation Order
1. Add `LLMQueryConfig` to `config.py`
2. Create `llm_cache.py`
3. Create `llm_query.py`
4. Add `LLMResponse` to `models.py`
5. Add `--llm` to `mx get`
6. Add `--llm` to `mx search`
7. Add `mx ask` command
8. Add `mx llm-cache` commands

## Verification
```bash
# Enable in config
echo 'llm_query:\n  enabled: true' >> .kbconfig

# Test summarize
mx get kb/tooling/beads.md --llm

# Test cached response
mx get kb/tooling/beads.md --llm  # Should say "(cached)"

# Test answer query
mx ask "How do I track issues?"

# Test cache management
mx llm-cache stats
mx llm-cache clear

# Run tests
uv run pytest tests/test_llm_query.py -v
```
