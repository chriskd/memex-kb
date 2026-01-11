---
title: DocViewer (Epstein Project)
tags:
  - project
  - document-search
  - semantic-search
  - fastapi
  - react
created: 2025-12-20
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
---

# DocViewer (Epstein Project)

Document investigation platform for journalists with full-text + semantic search, hybrid ranking, and OCR support. Inspired by Google Pinpoint.

## Repository

- **Location:** `/srv/fast/code/epstein`

## Stack Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                 │
│              Vite + React + TypeScript + Tailwind                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         BACKEND                                  │
│                    FastAPI + SQLAlchemy + Pydantic               │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ SearchSvc   │  │ EmbeddingSvc│  │ VectorStoreSvc          │  │
│  │ (orchestr.) │  │ (Voyage)    │  │ (Qdrant)                │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│    PostgreSQL    │ │      Qdrant      │ │  File Storage    │
│  + ParadeDB BM25 │ │  (Vectors)       │ │  (Local/S3)      │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector store | Qdrant | Better filtered ANN than pgvector |
| FTS | ParadeDB BM25 | PostgreSQL-native, tuneable, IDF-aware |
| Hybrid fusion | App-side RRF | More control than in-DB fusion |
| Embedding | Voyage context-3 | 84.8% recall on gold set |

## CLI Commands

```bash
# Backend development
cd backend
uv sync
uv run uvicorn docviewer.main:app --reload --port 8000

# Frontend development
cd frontend
npm install && npm run dev

# Database migrations
uv run docviewer-migrate run

# Embedding operations
uv run docviewer-embed reindex --only-missing
uv run docviewer-embed query "describe the email..."

# OCR operations
uv run docviewer-ocr import-json ./payload.json

# Ingestion
uv run docviewer-ingest import-production --data-dir /path/to/data
```

## Configuration System

```
.env.defaults       # Base config (committed)
.env.development    # Dev overrides (committed)
.env.staging        # Staging overrides
.env.production     # Production overrides
.env                # Secrets only (git-ignored) - from Phase
```

Precedence (highest to lowest):
1. Environment variables (Phase shell export)
2. `.env` (secrets)
3. `.env.{ENVIRONMENT}` (environment-specific)
4. `.env.defaults` (base config)

## Deployment (Dokploy)

| Project | Contents | External Ports |
|---------|----------|----------------|
| docviewer-dev | Qdrant + Postgres | 5433, 6334 |
| docviewer-prod | Full stack | Internal only |

### URLs

- **Dev Postgres:** `postgresql://epstein:changeme_dev@dokploy.voidlabs.cc:5433/epstein`
- **Dev Qdrant:** `http://dokploy.voidlabs.cc:6334`

## GPU Processing (Hyperion)

### Local GPU Embeddings
- Qwen3-embeddings via llama.cpp on RTX 4080 Super
- Endpoint: `http://hyperion:8080/v1/embeddings`
- Provider: `EMBEDDING_PROVIDER=text_embeddings_inference`

### GPU PaddleOCR
- Typer command: `uv run docviewer-ocr paddle-worker`
- Docker image: `docker/ocr/Dockerfile.gpu`
- See: `docs/hyperion-ocr.md`

## Directory Structure

```
backend/
├── src/docviewer/
│   ├── api/v1/endpoints/   # FastAPI routes
│   ├── cli/                # Typer CLI
│   ├── services/           # Business logic
│   ├── schemas/            # Pydantic models
│   └── core/               # Config, DB setup
frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/           # API client
docs/
├── ARCHITECTURE.md
├── ROADMAP.md
└── project-goals.md
```

## Current Status

- ✅ Hybrid search (BM25 + semantic + RRF)
- ✅ Voyage context-3 embeddings
- ✅ Qdrant vector store
- 🔜 Query understanding (typo correction, entity expansion)
- 🔜 NER at ingest time
- 🔜 Multi-collection support

## Related Entries

- [[Voidlabs Infrastructure Overview]]
- [[Hyperion GPU Server]]
