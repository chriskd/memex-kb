---
title: Dokploy Python CLI Reference
tags:
  - dokploy
  - cli
  - python
  - api
  - deployment
  - reference
created: 2025-12-22
updated: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
---

# Dokploy Python CLI Reference

Complete reference for the Dokploy Python CLI at `/srv/fast/code/dokploy-python-client`.

## Quick Start

```bash
cd /srv/fast/code/dokploy-python-client
export DOKPLOY_API_KEY="..."  # Get from Phase
uv run dokploy
```

## Application Commands (`dokploy app`)

### Listing & Status

```bash
dokploy app list [-v]              # List all applications
dokploy app status <id-or-name>    # Get application details
dokploy app logs <id-or-name> [-n] # Show deployment history
```

### Lifecycle

```bash
dokploy app start <id-or-name>     # Start application
dokploy app stop <id-or-name>      # Stop application
dokploy app deploy <id-or-name> [-t "reason"]    # Pull from git & deploy
dokploy app redeploy <id-or-name> [-t "reason"]  # Restart containers (no git pull)
```

**CRITICAL:** `deploy` pulls from git, `redeploy` does not. After pushing code, use `deploy`.

### CRUD

```bash
dokploy app create <name> -p <project> [-d "description"] [-e env-name]
dokploy app delete <id-or-name> [-f]
```

### Update Settings

```bash
dokploy app update <id-or-name> [OPTIONS]
```

**Options:**
| Option | Description | Example |
|--------|-------------|---------|
| `--name, -n` | Display name | `--name "My App"` |
| `--description, -d` | Description | `--description "Production API"` |
| `--branch, -b` | Git branch | `--branch main` |
| `--dockerfile` | Dockerfile path | `--dockerfile ./Dockerfile.prod` |
| `--docker-image` | Docker image | `--docker-image nginx:latest` |
| `--command, -c` | Run command | `--command "npm start"` |
| `--replicas, -r` | Replica count | `--replicas 3` |
| `--auto-deploy/--no-auto-deploy` | Auto deploy on push | `--auto-deploy` |
| `--memory-limit` | Memory limit | `--memory-limit 1g` |
| `--cpu-limit` | CPU limit | `--cpu-limit 0.5` |
| `--build-path` | Build path in repo | `--build-path ./backend` |

**Examples:**

```bash
# Update git branch
dokploy app update my-app --branch main

# Configure build settings
dokploy app update my-app --dockerfile ./Dockerfile --build-path ./backend

# Scale and set resource limits
dokploy app update my-app --replicas 3 --memory-limit 1g --cpu-limit 0.5

# Enable auto-deploy
dokploy app update my-app --auto-deploy
```

### Environment Variables (`dokploy app env`)

```bash
dokploy app env list <id-or-name>           # List env vars
dokploy app env set <id-or-name> KEY=value [KEY2=value2...]  # Set (merges)
dokploy app env set <id-or-name> -r KEY=val # Replace all vars
dokploy app env unset <id-or-name> KEY [KEY2...]  # Remove env vars
```

**Examples:**

```bash
# View current environment
dokploy app env list my-app

# Add/update variables (preserves existing)
dokploy app env set my-app DATABASE_URL=postgres://... REDIS_URL=redis://...

# Replace ALL variables
dokploy app env set my-app --replace NEW_VAR=value

# Remove variables
dokploy app env unset my-app OLD_VAR DEPRECATED_VAR
```

## Compose Commands (`dokploy compose`)

### Listing & Info

```bash
dokploy compose list [-v]              # List all compose stacks
dokploy compose info <id-or-name>      # Get compose details
dokploy compose services <id-or-name>  # List services in stack
dokploy compose logs <id-or-name> [-n] # Show deployment history
```

### Lifecycle

```bash
dokploy compose start <id-or-name>     # Start compose
dokploy compose stop <id-or-name>      # Stop compose
dokploy compose deploy <id-or-name>    # Pull from git & deploy
dokploy compose redeploy <id-or-name> [-t "reason"]  # Restart (no git pull)
```

### CRUD

```bash
dokploy compose create <name> -p <project> [-d "description"]
dokploy compose update <id-or-name> [OPTIONS]
dokploy compose delete <id-or-name> [-f]
```

### Environment Variables (`dokploy compose env`)

```bash
dokploy compose env list <id-or-name>
dokploy compose env set <id-or-name> KEY=value
dokploy compose env set <id-or-name> -r KEY=val  # Replace all
dokploy compose env unset <id-or-name> KEY
```

## Postgres Commands (`dokploy postgres`)

```bash
dokploy postgres list              # List all Postgres instances
dokploy postgres status <id>       # Get Postgres details
dokploy postgres start <id>        # Start instance
dokploy postgres stop <id>         # Stop instance
dokploy postgres create <name> -p <project> [--database-name db] [--database-user user]
dokploy postgres delete <id> [-f]
```

## Domain Commands (`dokploy domains`)

```bash
dokploy domains list <compose>
dokploy domains add <compose> <host> <service> <port> [--path /api] [--strip-path]
dokploy domains update <domain_id> [--host x] [--port y] [--service z]
dokploy domains rm <domain_id>
```

## Docker Commands (`dokploy docker`)

For debugging container issues:

```bash
dokploy docker containers <app-name>  # List containers
dokploy docker restart <container-id> # Restart container
dokploy docker config <container-id>  # Get container config
```

## Project Commands (`dokploy projects`)

## Project Commands (`dokploy projects`)

### Listing & Info

```bash
dokploy projects list [-v]         # List all projects
dokploy projects info <id-or-name> # Get project details with environments
```

### CRUD

```bash
dokploy projects create <name> [-d "description"]
dokploy projects update <id-or-name> [--name new-name] [--description "desc"]
dokploy projects delete <id-or-name> [-f]
```

**Examples:**

```bash
# Create a project
dokploy projects create my-project -d "My awesome project"

# Update project name
dokploy projects update my-project --name new-project-name

# Delete with confirmation skip
dokploy projects delete old-project -f
```

## Environment Commands (`dokploy env`)

Environments are namespaces within projects for organizing services (e.g., staging, production).

### Listing & Info

```bash
dokploy env list [-p project] [-v]     # List all environments
dokploy env info <id-or-name> [-p project]  # Get environment details
```

### CRUD

```bash
dokploy env create <name> -p <project> [-d "description"]
dokploy env update <id-or-name> [--name new-name] [--description "desc"] [-p project]
dokploy env delete <id-or-name> [-f] [-p project]
```

**Examples:**

```bash
# Create staging environment
dokploy env create staging -p my-project -d "Staging environment"

# List environments in a project
dokploy env list -p my-project -v

# Get environment info
dokploy env info staging -p my-project

# Delete environment
dokploy env delete old-env -f
```

## Name Matching

All commands support **partial name matching**:

```bash
dokploy app status docviewer     # Matches "docviewer-backend"
dokploy compose redeploy postgres # Matches "postgres-dev"
```

If multiple matches found, use the full ID instead.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DOKPLOY_API_KEY` | (required) | API key from Phase |
| `DOKPLOY_URL` | `https://dokploy.voidlabs.cc/api` | API base URL |

## Python API Usage

For programmatic access:

```python
from dokploy_api_client import AuthenticatedClient
from dokploy_api_client.api.application import application_update
from dokploy_api_client.models import ApplicationUpdateBody
import os

client = AuthenticatedClient(
    base_url="https://dokploy.voidlabs.cc/api",
    token=os.environ["DOKPLOY_API_KEY"],
    prefix="",
    auth_header_name="x-api-key",
)

with client as c:
    body = ApplicationUpdateBody(
        application_id="<APP_ID>",
        branch="main",
        auto_deploy=True,
    )
    result = application_update.sync_detailed(client=c, body=body)
    print(f"Status: {result.status_code}")
```

## Common Workflows

### Deploy Code Changes

```bash
# After pushing to git
dokploy app deploy my-app -t "Deploy v1.2.0"
dokploy app logs my-app
```

### Configure New Application

```bash
dokploy app create my-app -p my-project
dokploy app update my-app --branch main --dockerfile ./Dockerfile
dokploy app env set my-app DATABASE_URL=... API_KEY=...
dokploy app deploy my-app
```

### Scale Application

```bash
dokploy app update my-app --replicas 3 --memory-limit 2g
dokploy app redeploy my-app -t "Scale to 3 replicas"
```

## Related

- [[dokploy-deployment-guide]] - General deployment guide with SSH access, troubleshooting
- [[Voidlabs Common Patterns]] - Infrastructure patterns