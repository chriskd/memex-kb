---
title: Dokploy Container Management
tags:
  - dokploy
  - docker
  - containers
  - debugging
  - infrastructure
created: 2025-12-22
updated: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
---

# Dokploy Container Management

## Overview

Dokploy uses Docker Swarm to manage containers. Understanding the container naming conventions and management tools helps with debugging and operations.

## Container Naming Conventions

Dokploy generates container names following this pattern:

```
<project>-<service>-<random-suffix>
```

Examples:
- `docviewer-docviewer-pvxtnc` - Application container
- `compose-parse-auxiliary-port-cgcdph` - Auto-generated compose name
- `voidlabssharedservices-garagewithui-vo0uq2-garage-fast-1` - Compose service

## Finding Containers

### Using the Dokploy CLI

```bash
# List containers matching an app name
dokploy docker containers <app-name>

# Example output:
#   ● docviewer-backend-abc123
#       State: running (Up 2 hours)
#       Image: ghcr.io/user/docviewer:latest
#       ID: a1b2c3d4e5f6
#       Ports: 8000
```

### Using the API

```bash
curl -s -H "x-api-key: $DOKPLOY_API_KEY" \
  "https://dokploy.voidlabs.cc/api/docker.getContainersByAppNameMatch?appName=docviewer"
```

### Direct Docker Commands (on server)

```bash
# SSH to dokploy server
ssh dokploy.voidlabs.cc

# List all containers
docker ps

# Filter by name
docker ps --filter "name=docviewer"

# Check swarm services
docker service ls
docker service ps <service-name> --no-trunc
```

## Common Container Operations

### Restart a Container

```bash
# Via CLI
dokploy docker restart <container-id>

# Via API
curl -X POST -H "x-api-key: $DOKPLOY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"containerId":"<container-id>"}' \
  https://dokploy.voidlabs.cc/api/docker.restartContainer
```

### Get Container Config

```bash
dokploy docker config <container-id>
```

### View Container Logs

Container logs must be accessed directly on the server:

```bash
ssh dokploy.voidlabs.cc "docker logs <container-id> --tail 100"
```

## Debugging Non-Running Services

When an application shows "running" in Dokploy but no container exists:

### 1. Check Swarm Service Status

```bash
ssh dokploy.voidlabs.cc "docker service ps <app-name> --no-trunc"
```

This shows why containers failed to start.

### 2. Common Failure Reasons

| Error | Cause | Solution |
|-------|-------|----------|
| `bind source path does not exist` | Missing mount directory | Create directory or remove mount |
| `port already allocated` | Port conflict | Change port or stop conflicting service |
| `image not found` | Registry auth or missing image | Check registry config, rebuild |
| `OOM killed` | Out of memory | Increase memory limit |

### 3. Check Deployment Logs

```bash
# Get log path from deployment history
dokploy app logs <app-name>

# Read the log file
ssh dokploy.voidlabs.cc "cat /etc/dokploy/logs/<app-name>/<log-file>.log"
```

## Known Container Names

For admin operations on shared services:

| Service | Container Name |
|---------|----------------|
| Garage Fast | `voidlabssharedservices-garagewithui-vo0uq2-garage-fast-1` |
| Garage Slow | `voidlabssharedservices-garagewithui-vo0uq2-garage-slow-1` |
| Garage WebUI Fast | `voidlabssharedservices-garagewithui-vo0uq2-garage-webui-fast-1` |
| Garage WebUI Slow | `voidlabssharedservices-garagewithui-vo0uq2-garage-webui-slow-1` |

## Related

- [[Dokploy Deployment Guide]] - CLI and API reference
- [[Garage S3 Object Storage]] - Garage container operations
- [[Voidlabs Infrastructure Overview]] - Server details