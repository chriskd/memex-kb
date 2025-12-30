---
title: DocViewer Local Development Workflow
tags:
  - docviewer
  - development
  - workflow
  - local-dev
  - s3
created: 2025-12-22
updated: 2025-12-23
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
---

# DocViewer Local Development Workflow

## Quick Start

```bash
cd /srv/fast/code/epstein
./scripts/dev.sh
```

This starts:
- **Frontend** (Vite): http://localhost:5173
- **Backend** (uvicorn): http://localhost:8000
- **Database**: Uses postgres-dev on dokploy.voidlabs.cc:5433
- **Storage**: Uses S3 (Garage) - same bucket as production

## Prerequisites

| Dependency | Required For |
|------------|--------------|
| `node` + `npm` | Frontend (Vite + React) |
| `uv` | Python package management |
| Network access | Database on dokploy.voidlabs.cc:5433, S3 on garage-fast.voidlabs.cc |

## Script Options

```bash
# Start everything (default)
./scripts/dev.sh

# Start only frontend
./scripts/dev.sh frontend

# Start only backend  
./scripts/dev.sh backend

# Run migrations only
./scripts/dev.sh migrate

# Show help
./scripts/dev.sh help
```

## Environment Configuration

The script loads environment from:
1. `.env` - Secrets (API keys, S3 credentials, DB password)
2. `.env.development` - Dev-specific overrides

### Storage Configuration

Dev mirrors production by using S3 storage:

```bash
# In .env.development
STORAGE_DRIVER="s3"
STORAGE_S3_BUCKET="efta-images"
STORAGE_S3_ENDPOINT="https://garage-fast.voidlabs.cc"
STORAGE_S3_REGION="garage"

# In .env (secrets, gitignored)
STORAGE_S3_ACCESS_KEY=<your-key>
STORAGE_S3_SECRET_KEY=<your-secret>
```

This ensures dev tests the exact same S3 code paths that production uses.

### Database Configuration

```bash
DATABASE_URL=postgresql+psycopg://epstein:changeme_dev@dokploy.voidlabs.cc:5433/epstein
```

## Why Native Processes?

We use native processes instead of Docker Compose for development because:

1. **Faster iteration** - No container rebuilds on code changes
2. **Hot reload works** - Vite and uvicorn both hot-reload
3. **Shared dev database** - postgres-dev on Dokploy is persistent
4. **Simpler debugging** - No container layer between you and the code

## Common Tasks

### Adding a Python dependency

```bash
cd backend
uv add <package>
uv sync
```

### Adding a JS dependency

```bash
cd frontend
npm install <package>
```

### Running migrations

```bash
./scripts/dev.sh migrate
# or
cd backend && uv run docviewer-migrate
```

### Regenerating OpenAPI client

```bash
cd frontend
npm run generate
```

## Troubleshooting

### "Connection refused" to database

The dev database runs on dokploy.voidlabs.cc:5433. Check:
- Network connectivity to dokploy.voidlabs.cc
- postgres-dev compose is running in Dokploy

### S3 Storage Errors

If images fail to load, verify S3 configuration:

```bash
# Test S3 connectivity
cd backend
source ../.env && source ../.env.development
uv run python -c "
from docviewer.core.config import get_settings
import boto3
settings = get_settings()
s3 = boto3.client('s3',
    endpoint_url=settings.storage.s3_endpoint,
    aws_access_key_id=settings.storage.s3_access_key,
    aws_secret_access_key=settings.storage.s3_secret_key,
    region_name=settings.storage.s3_region
)
response = s3.list_objects_v2(Bucket=settings.storage.s3_bucket, MaxKeys=3)
print(f'✓ Connected. Found {response.get(\"KeyCount\", 0)} objects')
"
```

### Backend won't start

```bash
cd backend
uv sync  # Ensure deps are installed
uv run docviewer-migrate  # Run migrations
```

### Frontend build errors

```bash
cd frontend
rm -rf node_modules
npm ci
```

## Related

- [[docviewer-deployment-architecture]]
- [[Voidlabs Devtools]]
- [[dataset-ingestion-workflow]]