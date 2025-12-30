---
title: Dokploy Compose Deployment Lessons
tags:
  - dokploy
  - networking
  - compose
  - traefik
  - isolated-deployments
  - troubleshooting
  - reference
created: 2025-12-22
updated: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
---

# Dokploy Compose Networking - Definitive Guide

**Last verified:** December 2025  
**Source code reference:** `/srv/fast/code/dokploy-for-reference/dokploy`  
**Docs reference:** `/srv/fast/code/dokploy-for-reference/website/apps/docs/content/docs/core/docker-compose/`

## TL;DR - The Rule

**If your compose has inter-service communication (frontend→backend, app→db), you MUST either:**

1. ✅ Enable **Isolated Deployments** in Dokploy UI (recommended), OR
2. ✅ Manually add ALL services to `dokploy-network` in your compose file

**If you don't do one of these, services can't talk to each other.**

---

## How Dokploy Networking Actually Works

### The Core Mechanism

When you add a domain to a compose service in Dokploy:

**WITHOUT Isolated Deployments (`isolatedDeployment: false`):**
- Dokploy adds ONLY the service with the domain to `dokploy-network`
- Other services stay on compose's auto-created default network
- **Result: Service with domain can't reach other services!**

**WITH Isolated Deployments (`isolatedDeployment: true`):**
- Dokploy creates a network named after your `appName`
- Dokploy connects Traefik TO that network
- All services communicate via compose's network
- **Result: Everything works!**

### Source Code Proof

File: `dokploy/packages/server/src/utils/builders/compose.ts`

```typescript
// Lines 56-58: What Isolated Deployments does
${compose.isolatedDeployment ? 
  `docker network create --attachable ${compose.appName}` : ""}

// After deploy, connects Traefik to the compose network:
${compose.isolatedDeployment ? 
  `docker network connect ${compose.appName} $(docker ps --filter "name=dokploy-traefik" -q)` : ""}
```

File: `dokploy/packages/server/src/utils/docker/domain.ts`

```typescript
// Lines 217-222: Without isolation, only domain service gets network
if (!compose.isolatedDeployment) {
  result.services[serviceName].networks = addDokployNetworkToService(
    result.services[serviceName].networks,
  );
}
```

### Documentation Quote

From `website/apps/docs/content/docs/core/docker-compose/domains.mdx` (line 88):

> "If you're not using Isolated Deployments, Dokploy will add the `dokploy-network` to the service you selected, **however you need to add `dokploy-network` to the other services to maintain connectivity**."

---

## How to Check Current Settings

```bash
# Get compose isolation setting
curl -s -H "x-api-key: $DOKPLOY_API_KEY" \
  "https://dokploy.voidlabs.cc/api/compose.one?composeId=<ID>" | \
  jq '{appName, isolatedDeployment, randomize}'
```

## How to Enable Isolated Deployments

### Via API:
```bash
curl -s -X POST \
  -H "x-api-key: $DOKPLOY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"composeId": "<ID>", "isolatedDeployment": true}' \
  https://dokploy.voidlabs.cc/api/compose.update
```

### Via UI:
1. Go to compose → General tab
2. Toggle "Isolated Deployments" ON
3. Redeploy

---

## Network Debugging Commands

```bash
# Check what networks a container is on
docker inspect <container> --format '{{json .NetworkSettings.Networks}}' | jq -r 'keys[]'

# Check what networks Traefik is connected to
docker inspect dokploy-traefik --format '{{json .NetworkSettings.Networks}}' | jq -r 'keys[]'

# Test DNS resolution from inside a container
docker exec <container> nslookup <service-name>

# List all containers with their networks
docker ps --format 'table {{.Names}}\t{{.Networks}}'
```

---

## Common Mistakes We Made

### Mistake 1: Blaming "networking" generically
**Reality:** The issue was always that Dokploy only adds the domain service to dokploy-network.

### Mistake 2: Creating custom networks as workaround
**Reality:** Works, but the proper fix is enabling Isolated Deployments.

### Mistake 3: Looking at other deployed apps as "correct" examples
**Reality:** Other apps (Phase, etc.) may also be misconfigured. Always compare against Dokploy source.

---

## Reference Files

| What | Path |
|------|------|
| Compose builder (isolation logic) | `dokploy/packages/server/src/utils/builders/compose.ts` |
| Domain/network injection | `dokploy/packages/server/src/utils/docker/domain.ts` |
| Traefik setup | `dokploy/packages/server/src/setup/traefik-setup.ts` |
| Docs: Compose utilities | `website/apps/docs/content/docs/core/docker-compose/utilities.mdx` |
| Docs: Compose domains | `website/apps/docs/content/docs/core/docker-compose/domains.mdx` |

---

## Related
- [[dokploy-deployment-guide]] - CLI and API reference
- [[docviewer-deployment-architecture]] - DocViewer specific setup