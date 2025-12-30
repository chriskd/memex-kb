---
title: Dokploy Network Recovery After VM Restart
tags:
  - dokploy
  - traefik
  - networking
  - troubleshooting
  - docker
  - recovery
  - 504-error
  - gateway-timeout
  - isolated-network
  - vm-restart
  - docker-compose
  - bad-gateway
created: 2025-12-22
updated: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
---

# Dokploy Network Recovery After VM Restart

## Problem

After a VM restart, crash, or filesystem recovery, Docker Compose services deployed with Dokploy's **isolated network** feature may lose their Traefik routing. Symptoms include:

- 504 Gateway Timeout on previously working domains
- Traefik labels exist on containers but routing doesn't work
- Containers are on isolated networks (e.g., `voidlabssharedservices-phase-kdkgev`) separate from `dokploy-network`

## Root Cause

From [Dokploy docs](https://docs.dokploy.com/docs/core/docker-compose/utilities):

> If using a custom installation with Traefik as a Docker service (rather than a standalone container), system restarts can cause services to "lose their network references to Traefik."

Dokploy's isolated deployments create app-specific networks and Traefik is supposed to connect to each. After restarts, these connections can break.

## Solution

**DO NOT manually connect networks** with `docker network connect`. This is a band-aid that doesn't persist and can cause other issues.

**DO redeploy through Dokploy**:
1. Go to Dokploy UI → find the affected compose/application
2. Click "Redeploy" 
3. This reconnects the isolated network to Traefik properly

## Prevention

- Use Dokploy's official installation with standalone Traefik container (not Docker Swarm service)
- After any VM recovery, check all isolated apps and redeploy if needed

## Related

- [[dokploy-deployment-guide]]
- Dokploy uses Docker labels for Traefik routing (e.g., `traefik.http.routers.*.rule`)

## ## Search Keywords

504, bad gateway, gateway timeout, traefik not routing, container unreachable, isolated network, docker network connect, redeploy, vm crash, system restart
