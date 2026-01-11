---
title: DocViewer Deployment Architecture
tags:
  - docviewer
  - deployment
  - docker
  - dokploy
  - architecture
created: 2025-12-22
updated: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
---

# DocViewer Deployment Architecture

## Overview

DocViewer uses two distinct deployment modes:

| Mode | Tool | Database | When to Use |
|------|------|----------|-------------|
| **Production** | `docker-compose.prod.yml` via Dokploy | Internal ParadeDB | Real deployments |
| **Local Dev** | Native processes | Dev DB on Dokploy | Active development |

## Production Architecture

```
┌─────────────────────────────────────────────────────┐
│  dokploy.voidlabs.cc                                │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ docviewer-prod compose                      │   │
│  │                                             │   │
│  │  frontend:80 ──▶ backend:8000 ──▶ db:5432  │   │
│  │  (nginx+SPA)     (FastAPI)       (ParadeDB) │   │
│  │                                             │   │
│  │  All internal except frontend              │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Key Points

- **Git-based deployment**: Push to main triggers deploy via Dokploy webhook
- **Internal networking**: All services internal, only frontend exposes port 80
- **API proxy**: Frontend nginx proxies `/api/*` to backend
- **Database**: ParadeDB (Postgres + pgvector + BM25) runs inside compose
- **Migrations**: One-shot migration service runs on deploy

### Files

| File | Purpose |
|------|---------|
| `docker-compose.prod.yml` | Production compose stack |
| `frontend/nginx.conf` | Nginx config with API proxy |
| `Dockerfile` | Backend build |
| `frontend/Dockerfile` | Frontend build (nginx + SPA) |

### Deployment Commands

```bash
# View status
dokploy compose info docviewer-prod

# Redeploy (pulls from git)
dokploy compose deploy docviewer-prod

# View logs
dokploy compose logs docviewer-prod
```

## Development Architecture

```
┌───────────────────────────────────────────────────────────────┐
│  devbox.voidlabs.local (devcontainer)                         │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Native processes via scripts/dev.sh                     │ │
│  │                                                         │ │
│  │  Frontend (Vite)  Backend (uvicorn)                    │ │
│  │  localhost:5173   localhost:8000                        │ │
│  └─────────────────────────────────────────────────────────┘ │
│                   │                                           │
│                   ▼                                           │
│    dokploy.voidlabs.cc:5433 (postgres-dev)                   │
└───────────────────────────────────────────────────────────────┘
```

### Why Native Processes?

- **Faster iteration**: No container rebuild on code changes
- **Shared database**: Uses postgres-dev on Dokploy (port 5433)
- **Hot reload**: Both Vite and uvicorn have hot reload

### Development Commands

```bash
# Start everything
./scripts/dev.sh

# Start only frontend
./scripts/dev.sh frontend

# Start only backend
./scripts/dev.sh backend

# Run migrations only
./scripts/dev.sh migrate
```

## Troubleshooting

### Migration failures on deploy

If migrations fail with column/table errors, the database may be in a corrupt state from a previous failed deploy. Clean up:

```bash
ssh chris@dokploy.voidlabs.cc
docker stop <db-container>
docker rm <db-container>
docker volume rm <pgdata-volume>
dokploy compose deploy docviewer-prod
```

### Service naming in Dokploy

Dokploy generates unique service names with random suffixes. Use `docker service ls` to find actual names.

## Related

- [[dokploy-deployment-guide]]
- [[Voidlabs Common Patterns]]

## ## Production Architecture

## Production Architecture

```
┌─────────────────────────────────────────────────────┐
│  dokploy.voidlabs.cc                                │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ docviewer-prod compose                      │   │
│  │                                             │   │
│  │  frontend:80 ──▶ backend:8000 ──▶ db:5432  │   │
│  │  (nginx+SPA)     (FastAPI)       (ParadeDB) │   │
│  │       │                                     │   │
│  │       └──────────▶ Garage S3               │   │
│  │                    (efta-images)            │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Key Points

- **Git-based deployment**: Push to main triggers deploy via Dokploy webhook
- **Internal networking**: All services internal, only frontend exposes port 80
- **API proxy**: Frontend nginx proxies `/api/*` to backend
- **Database**: ParadeDB (Postgres + pgvector + BM25) runs inside compose
- **Migrations**: One-shot migration service runs on deploy
- **Image storage**: S3-compatible storage via Garage (`efta-images` bucket)

### S3 Storage Configuration

Production uses Garage S3 for document images:

| Variable | Value |
|----------|-------|
| `STORAGE_DRIVER` | `s3` |
| `STORAGE_S3_ENDPOINT` | `https://garage-fast.voidlabs.cc` |
| `STORAGE_S3_BUCKET` | `efta-images` |
| `STORAGE_S3_REGION` | `garage` |

Access keys are configured in Dokploy's environment variables (not in git).

See [[garage-s3-object-storage]] for Garage administration.
