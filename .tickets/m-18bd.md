---
id: m-18bd
status: closed
deps: []
links: []
created: 2026-01-27T05:12:00Z
type: task
priority: 2
assignee: chriskd
---
# Cache embeddings + skip unchanged chunks on reindex

Add an embedding cache keyed by chunk hash (embedding text) to avoid re-embedding unchanged content. Store chunk hash in metadata or sidecar; during reindex, reuse cached embeddings and skip recomputation where possible. Ensure this integrates with the file watcher reindex path and full reindex.

## Acceptance Criteria

- [ ] Chunk hash computed from embedding text and stored
- [ ] Reindex reuses cached embeddings for unchanged chunks
- [ ] Full reindex path benefits from cache
- [ ] Cache location/versioning documented
- [ ] Tests updated/added for cache hit/miss behavior


## Notes

**2026-01-27T05:17:27Z**

Implemented SQLite embedding cache + hash in Chroma metadata; Chroma indexing reuses cached embeddings; added tests for cache roundtrip + reuse; README index storage note. Ran: .venv/bin/python -m pytest tests/test_embedding_cache.py
