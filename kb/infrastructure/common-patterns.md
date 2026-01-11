---
title: Voidlabs Common Patterns
tags:
  - infrastructure
  - patterns
  - best-practices
  - developer-guide
created: 2025-12-20
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
---

# Voidlabs Common Patterns

Standard patterns and conventions used across voidlabs projects.

## Project Structure Pattern

All voidlabs projects follow a consistent structure:

```
project-name/
├── AGENTS.md              # AI agent guidance (standard template)
├── README.md              # Project documentation
├── .beads/                # Beads issue tracker
├── .claude/               # Claude Code settings
├── .devcontainer/         # Devcontainer configuration
│   ├── devcontainer.json
│   ├── Dockerfile
│   ├── voidlabs.conf      # Feature toggles
│   └── scripts/
│       ├── post-start-common.sh
│       └── post-start-project.sh
├── backend/               # FastAPI backend (Python projects)
│   ├── pyproject.toml     # UV-managed dependencies
│   └── src/{project}/
├── frontend/              # Vite + React frontend
│   └── src/
└── docs/                  # Project documentation
    └── ARCHITECTURE.md
```

## Configuration Pattern

### Layered Configuration with Phase

All projects use a consistent configuration pattern:

```
.env.defaults       # Base config (committed) - non-secrets
.env.development    # Dev overrides (committed)
.env.staging        # Staging overrides (committed)
.env.production     # Production overrides (committed)
.env                # Secrets only (git-ignored) - from Phase
```

**Precedence (highest to lowest):**
1. Environment variables (Phase shell export)
2. `.env` (secrets file)
3. `.env.{ENVIRONMENT}` (environment-specific)
4. `.env.defaults` (base config)
5. Field defaults in code

### Phase Integration

```bash
# Environment setup
export PHASE_HOST="https://secrets.voidlabs.cc"
export PHASE_SERVICE_TOKEN="pss_user:v1:..."
export PHASE_ENV="development"  # or staging, production

# In devcontainers, Phase exports secrets automatically
# via post-start-common.sh
```

## Backend Stack Pattern

### FastAPI + SQLAlchemy + Pydantic

Standard backend structure:

```
backend/src/{project}/
├── api/v1/endpoints/    # FastAPI routes
├── cli/                 # Typer CLI commands
├── core/                # Config, DB setup, logging
├── schemas/             # Pydantic models
└── services/            # Business logic
```

**Dependencies via UV:**
```bash
cd backend
uv sync
uv run uvicorn {project}.main:app --reload
```

### Common CLI Pattern

Projects expose Typer CLIs as entry points:

```toml
# pyproject.toml
[project.scripts]
docviewer-migrate = "docviewer.cli.migrate:app"
docviewer-embed = "docviewer.cli.embed:app"
```

## Frontend Stack Pattern

### Vite + React + TypeScript + Tailwind

Standard frontend structure:

```
frontend/
├── src/
│   ├── components/      # Reusable UI components
│   ├── pages/           # Route pages
│   ├── hooks/           # Custom React hooks
│   ├── services/        # API client
│   └── types/           # TypeScript types
├── package.json
├── vite.config.ts
└── tailwind.config.js
```

**Development:**
```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

## Storage Pattern

### StorageService Abstraction

Projects use a unified storage service supporting multiple backends:

```python
# Supports both local:// and s3:// URIs
storage.store(file, "local://path/to/file.png")
storage.store(file, "s3://bucket/key.png")
storage.get("local://path/to/file.png")
```

**Configuration:**
```env
STORAGE_BACKEND=local  # or s3
STORAGE_LOCAL_PATH=/data/files
STORAGE_S3_BUCKET=my-bucket
STORAGE_S3_ENDPOINT=https://s3.amazonaws.com
```

## Deployment Pattern

### Dokploy

All production deployments use Dokploy at `dokploy.voidlabs.cc`:

- **Projects:** Each app gets its own Dokploy project
- **Dependencies:** Database, Qdrant, etc. as separate services
- **Secrets:** Injected via Phase environment variables

### Port Conventions

| Service | Dev Port | Container Port |
|---------|----------|----------------|
| Backend (FastAPI) | 8000 | 8000 |
| Frontend (Vite) | 5173 | 5173 |
| PostgreSQL | 5432/5433 | 5432 |
| Qdrant | 6333/6334 | 6333 |

## GPU Processing Pattern

### Hyperion Server

GPU workloads run on Hyperion (RTX 4080 Super):

```bash
# Embedding server (llama.cpp)
http://hyperion.voidlabs:8080/v1/embeddings

# OCR worker
CUDA_VISIBLE_DEVICES=0 uv run {project}-ocr paddle-worker
```

**Configuration:**
```env
EMBEDDING_PROVIDER=text_embeddings_inference
EMBEDDING_URL=http://hyperion:8080/v1/embeddings
```

## Issue Tracking Pattern

### Beads Workflow

All projects use beads for issue tracking:

```bash
# Initialize
bd init

# Workflow
bd ready               # Find unblocked work
bd update <id> --status in_progress
# ... do work ...
bd close <id> --reason "Completed"
bd sync                # Commit and push
```

**Session start:** `bd prime` provides context to AI agents.

## Testing Pattern

### Backend (pytest)
```bash
uv run pytest
uv run pytest --cov
```

### Frontend (Vitest/Jest)
```bash
npm run test
npm run lint
```

### CI Commands
```bash
uv run ruff check
npm run build
```

## Related Entries

- [[Voidlabs Infrastructure Overview]]
- [[Voidlabs Devtools]]
- [[Phase Secrets Management]]
- [[Beads Issue Tracker]]
