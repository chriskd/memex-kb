---
title: Dokploy Deployment Guide
tags:
  - dokploy
  - deployment
  - api
  - docker
  - infrastructure
created: 2025-12-22
updated: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
---

# Dokploy Deployment Guide

## Overview
Dokploy is our self-hosted PaaS at `dokploy.voidlabs.cc` for deploying Docker applications and compose stacks.

## Authentication

### API Key
- Stored in Phase: `DOKPLOY_API_KEY` in the docviewer app (Development environment)
- Use header: `x-api-key: <token>` (NOT `Authorization: Bearer`)

```bash
# Get API key from Phase
cd /srv/fast/code/epstein && phase secrets get DOKPLOY_API_KEY --plain

# Set in environment
export DOKPLOY_API_KEY="..."
```

## Dokploy CLI (Recommended)

A CLI tool is available at `/srv/fast/code/dokploy-python-client`. This is the **recommended way** for AI agents to interact with Dokploy.

### Installation

```bash
cd /srv/fast/code/dokploy-python-client
uv sync

# Set API key
export DOKPLOY_API_KEY="..."

# Test connection
dokploy
```

### Complete Command Reference

#### Application Commands (`dokploy app`)

```bash
dokploy app list [-v]                    # List all applications
dokploy app status <id-or-name>          # Get application details
dokploy app redeploy <id-or-name> [-t "reason"]  # Trigger redeploy
dokploy app logs <id-or-name> [-n 5]     # Show deployment history
dokploy app start <id-or-name>           # Start application
dokploy app stop <id-or-name>            # Stop application
```

#### Compose Commands (`dokploy compose`)

```bash
dokploy compose list [-v]                # List all compose stacks
dokploy compose info <id-or-name>        # Get compose details
dokploy compose deploy <id-or-name>      # Initial deployment
dokploy compose redeploy <id-or-name> [-t "reason"]  # Redeploy
dokploy compose start <id-or-name>       # Start compose
dokploy compose stop <id-or-name>        # Stop compose
dokploy compose logs <id-or-name> [-n 5] # Show deployment history
dokploy compose services <id-or-name>    # List services in stack
```

#### Domain Commands (`dokploy domains`)

```bash
dokploy domains list <compose>           # List domains for compose
dokploy domains add <compose> <host> <service> <port> [--path /api] [--strip-path]
dokploy domains update <domain_id> [--host x] [--port y] [--service z]
dokploy domains rm <domain_id>           # Delete domain
```

#### Docker Commands (`dokploy docker`)

```bash
dokploy docker containers <app-name>     # List containers (debugging)
dokploy docker restart <container-id>    # Restart container
dokploy docker config <container-id>     # Get container config
```

#### Project Commands (`dokploy projects`)

```bash
dokploy projects list                    # List all projects
```

### Finding Resource IDs

Run `dokploy` without arguments to discover IDs:

```
docviewer-dev (NcwCKurRv7tXNlctzRyJR)
  Applications:
    ● docviewer-backend (Ufjj6UaPE9tC4S2DYsVYh)
  Composes:
    ● postgres-dev (wcaBZPm67rCP8JCRADH2U)
    ● qdrant-dev (beeo6OVKAACLHeoLH5B4Q)

Total: 1 application(s), 2 compose(s)
```

**Partial name matching** is supported:

```bash
dokploy app status docviewer     # Matches "docviewer-backend"
dokploy compose redeploy postgres # Matches "postgres-dev"
```

### Common Workflows

#### Redeploy an Application

```bash
dokploy app list
dokploy app redeploy docviewer-backend -t "Updated config"
dokploy app logs docviewer-backend
```

#### Debug a Non-Running Service

```bash
# List containers to see what's actually running
dokploy docker containers docviewer

# If no containers found, check status
dokploy app status docviewer
dokploy compose info postgres-dev

# Check deployment logs for errors
dokploy app logs docviewer -n 10
```

#### Add a Domain to a Compose

```bash
dokploy compose list
dokploy domains add postgres-dev db.example.com postgres 5432
dokploy domains list postgres-dev
```

## Raw API Operations (curl)

For operations not covered by the CLI:

### List All Projects and Composes
```bash
curl -s -H "x-api-key: $DOKPLOY_API_KEY" \
  https://dokploy.voidlabs.cc/api/project.all
```

### Get Application Details
```bash
curl -s -H "x-api-key: $DOKPLOY_API_KEY" \
  "https://dokploy.voidlabs.cc/api/application.one?applicationId=<APP_ID>"
```

### Trigger Application Redeploy
```bash
curl -s -X POST \
  -H "x-api-key: $DOKPLOY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"applicationId":"<APP_ID>","title":"Deployment reason"}' \
  https://dokploy.voidlabs.cc/api/application.redeploy
```

### Get Deployment History
```bash
curl -s -H "x-api-key: $DOKPLOY_API_KEY" \
  "https://dokploy.voidlabs.cc/api/deployment.all?applicationId=<APP_ID>"
```

### Get Running Containers
```bash
curl -s -H "x-api-key: $DOKPLOY_API_KEY" \
  "https://dokploy.voidlabs.cc/api/docker.getContainersByAppNameMatch?appName=<NAME>"
```

## Viewing Build Logs

Logs are stored on the Dokploy server. Get the log path from `dokploy app logs`:

```bash
ssh dokploy.voidlabs.cc "cat /etc/dokploy/logs/<app-name>/<log-file>.log"
```

## Python API Usage

For programmatic access beyond the CLI:

```python
from dokploy_api_client import AuthenticatedClient
from dokploy_api_client.api.application import application_redeploy
from dokploy_api_client.models import ApplicationRedeployBody
import os

# IMPORTANT: Use x-api-key header, not Bearer token
client = AuthenticatedClient(
    base_url="https://dokploy.voidlabs.cc/api",
    token=os.environ["DOKPLOY_API_KEY"],
    prefix="",  # No "Bearer" prefix
    auth_header_name="x-api-key",  # Custom header name
)

with client as c:
    body = ApplicationRedeployBody(application_id="Ufjj6UaPE9tC4S2DYsVYh")
    result = application_redeploy.sync_detailed(client=c, body=body)
    print(f"Status: {result.status_code}")
```

## API Modules Reference

| Module | Purpose |
|--------|---------|
| `api.project` | Project CRUD, listing |
| `api.application` | App deploy, redeploy, start, stop, env vars |
| `api.compose` | Compose deploy, redeploy, services |
| `api.deployment` | Deployment history, logs |
| `api.docker` | Container management |
| `api.domain` | Domain/routing configuration |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DOKPLOY_API_KEY` | (required) | API authentication key |
| `DOKPLOY_URL` | `https://dokploy.voidlabs.cc/api` | API base URL |

## Troubleshooting

### "DOKPLOY_API_KEY not found"
Set the environment variable or create a `.env` file with `DOKPLOY_API_KEY=...`

### 401 Unauthorized
- Verify the API key is correct
- The client uses `x-api-key` header (not Bearer token)

### "Multiple matches" error
When partial name matching finds multiple results, use the full ID instead.

### Container Fails to Start

If the application shows "running" status in Dokploy but no container exists:

```bash
# Debug with CLI
dokploy docker containers <app-name>

# Or check Docker Swarm directly
docker service ps <app-name> --no-trunc
```

Common issues:
- **Missing bind mount path**: Container configured with a path that doesn't exist on the server
- **Port mismatch**: Domain configured with wrong port

### Server Unreachable

If `dokploy.voidlabs.cc` is unreachable:
1. Check if it's a network issue vs server down
2. The server IP is 192.168.51.87
3. Contact infrastructure team if persistent

## Related

- [[dokploy-python-cli-reference]] - Complete CLI command reference
- [[Voidlabs Common Patterns]] - Deployment patterns
- [[DocViewer Project]] - Main docviewer documentation

## ## SSH Access

## SSH Access

To SSH to the Dokploy VM:

```bash
# Use the ansible key with chris user
ssh -i ~/.ssh/id_ed25519-ansible chris@dokploy.voidlabs.cc

# Or with the ansible key loaded
ssh chris@dokploy.voidlabs.cc
```

**Key points:**
- IP: `192.168.51.87`
- User: `chris` (not root)
- Key: `~/.ssh/id_ed25519-ansible`

### Direct Container Access

```bash
# Run docker commands
ssh -i ~/.ssh/id_ed25519-ansible chris@dokploy.voidlabs.cc "docker ps"

# Check logs
ssh -i ~/.ssh/id_ed25519-ansible chris@dokploy.voidlabs.cc "docker logs <container>"

# Exec into container
ssh -i ~/.ssh/id_ed25519-ansible chris@dokploy.voidlabs.cc "docker exec -it <container> sh"
```

## ### Common Workflows

### Common Workflows

#### Deploy vs Redeploy

**CRITICAL DIFFERENCE:**

| Command | Git Pull? | Container Recreate? | Use When |
|---------|-----------|---------------------|----------|
| `deploy` | ✅ Yes | ✅ Yes | Code changed in git |
| `redeploy` | ❌ No | ⚠️ Only if config changed | Just restart containers |

**If you pushed code changes to git, you MUST use `deploy`, not `redeploy`!**

```bash
# After pushing code changes to git
dokploy compose deploy <name>      # Pulls from git, recreates containers

# Just restart existing containers (no git pull)
dokploy compose redeploy <name>    # Only restarts, does NOT pull from git
```

#### Update Code and Redeploy

```bash
# 1. Make changes to compose file in repo
cd /srv/fast/code/voidlabs-ansible
vim dokploy/garage/docker-compose.yml

# 2. Commit and push
git add . && git commit -m "fix: update config" && git push

# 3. Deploy (NOT redeploy) to pull changes
dokploy compose deploy garage-with-ui
```

**WARNING:** Never edit files directly in `/etc/dokploy/compose/*/code/` - this bypasses Dokploy's workflow and will cause issues on next deploy.

## ## Raw API Operations (curl)

## Raw API Operations (curl)

For operations not covered by the CLI:

### Create Compose
```bash
curl -s -X POST \
  -H "x-api-key: $DOKPLOY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-compose",
    "projectId": "<PROJECT_ID>",
    "environmentId": "<ENVIRONMENT_ID>",
    "description": "My compose stack"
  }' \
  https://dokploy.voidlabs.cc/api/compose.create
```

### Configure Compose GitHub Source
```bash
curl -s -X POST \
  -H "x-api-key: $DOKPLOY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "composeId": "<COMPOSE_ID>",
    "sourceType": "github",
    "repository": "repo-name",
    "owner": "owner",
    "branch": "main",
    "composePath": "./docker-compose.prod.yml",
    "githubId": "<GITHUB_ID>"
  }' \
  https://dokploy.voidlabs.cc/api/compose.update
```

### Set Compose Environment Variables
```bash
curl -s -X POST \
  -H "x-api-key: $DOKPLOY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "composeId": "<COMPOSE_ID>",
    "env": "KEY1=value1\nKEY2=value2"
  }' \
  https://dokploy.voidlabs.cc/api/compose.update
```

### Deploy Compose (pulls from git)
```bash
curl -s -X POST \
  -H "x-api-key: $DOKPLOY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"composeId": "<COMPOSE_ID>"}' \
  https://dokploy.voidlabs.cc/api/compose.deploy
```

### Delete Resources
```bash
# Delete application
curl -s -X POST \
  -H "x-api-key: $DOKPLOY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"applicationId": "<APP_ID>"}' \
  https://dokploy.voidlabs.cc/api/application.delete

# Delete postgres
curl -s -X POST \
  -H "x-api-key: $DOKPLOY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"postgresId": "<POSTGRES_ID>"}' \
  https://dokploy.voidlabs.cc/api/postgres.remove

# Delete compose
curl -s -X POST \
  -H "x-api-key: $DOKPLOY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"composeId": "<COMPOSE_ID>"}' \
  https://dokploy.voidlabs.cc/api/compose.delete
```

### List All Projects and Composes

## Complete Command Reference

### Complete Command Reference

#### Application Commands (`dokploy app`)

```bash
dokploy app list [-v]                    # List all applications
dokploy app status <id-or-name>          # Get application details
dokploy app redeploy <id-or-name> [-t "reason"]  # Trigger redeploy
dokploy app logs <id-or-name> [-n 5]     # Show deployment history
dokploy app start <id-or-name>           # Start application
dokploy app stop <id-or-name>            # Stop application
dokploy app create <name> -p <project>   # Create new application
dokploy app delete <id-or-name> [-f]     # Delete application
```

#### Compose Commands (`dokploy compose`)

```bash
dokploy compose list [-v]                # List all compose stacks
dokploy compose info <id-or-name>        # Get compose details
dokploy compose deploy <id-or-name>      # Initial deployment (pulls from git)
dokploy compose redeploy <id-or-name> [-t "reason"]  # Redeploy (no git pull)
dokploy compose start <id-or-name>       # Start compose
dokploy compose stop <id-or-name>        # Stop compose
dokploy compose logs <id-or-name> [-n 5] # Show deployment history
dokploy compose services <id-or-name>    # List services in stack
dokploy compose create <name> -p <project>  # Create new compose stack
dokploy compose update <id-or-name> [options]  # Update compose settings
dokploy compose delete <id-or-name> [-f] # Delete compose stack
```

#### Compose Environment Variables (`dokploy compose env`)

```bash
dokploy compose env list <id-or-name>           # List env vars
dokploy compose env set <id-or-name> KEY=value  # Set env vars (merges)
dokploy compose env set <id-or-name> -r KEY=val # Replace all vars
dokploy compose env unset <id-or-name> KEY      # Remove env var
```

#### Domain Commands (`dokploy domains`)

```bash
dokploy domains list <compose>           # List domains for compose
dokploy domains add <compose> <host> <service> <port> [--path /api] [--strip-path]
dokploy domains update <domain_id> [--host x] [--port y] [--service z]
dokploy domains rm <domain_id>           # Delete domain
```

#### Docker Commands (`dokploy docker`)

```bash
dokploy docker containers <app-name>     # List containers (debugging)
dokploy docker restart <container-id>    # Restart container
dokploy docker config <container-id>     # Get container config
```

#### Project Commands (`dokploy projects`)

```bash
dokploy projects list                    # List all projects
```