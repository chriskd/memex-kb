---
title: Dokploy Service Naming and DATABASE_URL
tags:
  - dokploy
  - database
  - docker-swarm
  - configuration
  - troubleshooting
  - postgresql
  - database-url
  - service-discovery
  - dns
  - connection-error
  - host-not-found
  - name-resolution
created: 2025-12-22
updated: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
---

# Dokploy Service Naming and DATABASE_URL

## Problem

When connecting services in Dokploy (e.g., backend to database), using the wrong hostname in `DATABASE_URL` causes connection failures:

```
failed to resolve host 'epstein-db': Name or service not known
```

## Root Cause

Dokploy creates **unique service names** by appending a random suffix to avoid conflicts:

- Compose defines: `db` or `epstein-db`
- Dokploy creates: `docviewer-epsteindb-uouojr`

The Docker DNS name is the **Dokploy-generated name**, not the compose service name.

## Solution

### Find the correct service name

```bash
# List all swarm services
docker service ls | grep -i <keyword>

# Example output:
# m4mlqzp72zxh   docviewer-epsteindb-uouojr   replicated   1/1   pgvector/pgvector:pg16
```

### Update DATABASE_URL

Use the full Dokploy service name:

```bash
# Wrong
DATABASE_URL=postgresql+psycopg://user:pass@epstein-db:5432/dbname

# Correct  
DATABASE_URL=postgresql+psycopg://user:pass@docviewer-epsteindb-uouojr:5432/dbname
```

### Update via Docker Swarm

```bash
docker service update --env-add 'DATABASE_URL=postgresql+psycopg://user:pass@docviewer-epsteindb-uouojr:5432/dbname' <service-name>
```

## Best Practice

When setting up DATABASE_URL in Dokploy environment variables, always verify the actual service name first with `docker service ls`.

## Related

- [[dokploy-deployment-guide]]
- [[dokploy-network-recovery]]

## ## Search Keywords

host not found, name or service not known, connection refused, database connection, postgres connection, service name, docker dns, swarm service name
