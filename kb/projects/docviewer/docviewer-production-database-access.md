---
title: DocViewer Production Database Access
tags:
  - docviewer
  - database
  - production
  - dokploy
  - postgresql
  - troubleshooting
created: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
---

# DocViewer Production Database Access

DocViewer has separate dev and production databases. CLI tools need different access patterns for each.

## Database Topology

| Environment | Host | Port | Access |
|-------------|------|------|--------|
| Dev | dokploy.voidlabs.cc | 5433 | External (postgres-dev) |
| Production | db (internal) | 5432 | Docker internal only |

## Dev Database Access (External)

From local machine or devcontainer:
```bash
# Via psql
PGPASSWORD=changeme_dev psql -h dokploy.voidlabs.cc -p 5433 -U epstein -d epstein

# Via CLI tools
source .env  # Contains DATABASE_URL pointing to :5433
docviewer-ingest import-production --dataset-id UUID ...
```

## Production Database Access

Production DB has no external port. Options:

### Option 1: Exec into Backend Container (Recommended)
```bash
# Copy files to server first
scp data.dat dokploy.voidlabs.cc:/tmp/

# Copy into container
ssh dokploy.voidlabs.cc "docker cp /tmp/data.dat docviewer-prod-xxx-backend-1:/tmp/"

# Fix permissions
ssh dokploy.voidlabs.cc "docker exec --user root docviewer-prod-xxx-backend-1 chown -R appuser:appuser /tmp/data.dat"

# Run CLI
ssh dokploy.voidlabs.cc "docker exec docviewer-prod-xxx-backend-1 docviewer-ingest import-production ..."
```

### Option 2: Direct DB Access via Container
```bash
ssh dokploy.voidlabs.cc "docker exec docviewer-prod-xxx-db-1 psql -U epstein -d epstein -c 'SELECT ...'"
```

### Option 3: Execute SQL File
```bash
scp script.sql dokploy.voidlabs.cc:/tmp/
ssh dokploy.voidlabs.cc "docker cp /tmp/script.sql docviewer-prod-xxx-db-1:/tmp/"
ssh dokploy.voidlabs.cc "docker exec docviewer-prod-xxx-db-1 psql -U epstein -d epstein -f /tmp/script.sql"
```

## Finding Container Names

Container names include random suffixes:
```bash
ssh dokploy.voidlabs.cc "docker ps | grep docviewer"
# docviewer-prod-rwlm59-backend-1
# docviewer-prod-rwlm59-db-1
# docviewer-prod-rwlm59-frontend-1
```

## Common Gotcha

**API creates data in production, CLI reads from dev!**

If you create datasets via `https://docviewer.voidlabs.cc/api/...` but run CLI locally, you'll get "Dataset not found" because CLI connects to dev DB.

## Related

- [[docviewer-deployment-architecture]] - Full architecture overview
- [[docker-cp-permission-patterns-in-dokploy]] - Permission fixes