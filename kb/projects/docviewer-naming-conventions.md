---
title: DocViewer Naming Conventions
tags:
  - docviewer
  - naming
  - conventions
  - documentation
created: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
---

# DocViewer Naming Conventions

## Project Directory

The project currently lives at `/srv/fast/code/epstein`, but the canonical name is **DocViewer**. The "epstein" name is a historical artifact (named after the Epstein documents dataset).

**Future plan**: Rename directory to `docviewer` for consistency.

## Compose Service Names

### Production (`docker-compose.prod.yml`)

| Service | Purpose |
|---------|---------|
| `db` | ParadeDB database |
| `backend` | FastAPI application |
| `frontend` | Nginx + React SPA |
| `migrations` | One-shot migration runner |

### Dokploy Generated Names

Dokploy adds random suffixes to avoid conflicts:
- Compose: `compose-bypass-open-source-pixel-szoiw5`
- Services: `docviewer-epsteindb-uouojr`

Use partial matching in CLI: `dokploy compose info docviewer`

## Database Naming

| Environment | Database | Host:Port |
|-------------|----------|-----------|
| Production | `epstein` (internal) | `db:5432` |
| Development | `epstein` | `dokploy.voidlabs.cc:5433` |

### Tables

| Table | Purpose |
|-------|---------|
| `documents` | Document metadata |
| `document_embeddings` | Vector embeddings per chunk |
| `images` | OCR page images |
| `tags` | Document tags |
| `datasets` | Evaluation dataset metadata |

## API Paths

| Path | Purpose |
|------|---------|
| `/api/*` | All API endpoints |
| `/health` | Health check (not under /api) |
| `/docs` | OpenAPI documentation |

## Environment Variables

### Database

```bash
DATABASE_URL=postgresql+psycopg://user:pass@host:port/dbname
POSTGRES_USER=epstein
POSTGRES_PASSWORD=<secret>
POSTGRES_DB=epstein
```

### Embedding

```bash
EMBEDDING_PROVIDER=voyage|openai|text_embeddings_inference
EMBEDDING_MODEL=voyage-context-3
EMBEDDING_DIMENSIONS=1024
```

### Vector Store

```bash
QDRANT_URL=http://host:6333
QDRANT_COLLECTION=epstein-document-embeddings
```

## File Naming

### Compose Files

| File | Purpose |
|------|---------|
| `docker-compose.prod.yml` | Production (Dokploy) |
| ~~`docker-compose.yml`~~ | Removed - use `scripts/dev.sh` |
| ~~`dokploy-stack.yml`~~ | Removed - replaced by prod.yml |

### Scripts

| File | Purpose |
|------|---------|
| `scripts/dev.sh` | Local development startup |

## Related

- [[docviewer-deployment-architecture]]
- [[Voidlabs Common Patterns]]
