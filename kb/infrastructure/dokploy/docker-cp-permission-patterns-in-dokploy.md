---
title: Docker CP Permission Patterns in Dokploy
tags:
  - dokploy
  - docker
  - permissions
  - troubleshooting
  - containers
created: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
---

# Docker CP Permission Patterns in Dokploy

When copying files into running Dokploy containers, permission issues commonly occur.

## The Problem

Files copied via `docker cp` retain the host user's UID:
```
-rw-r--r-- 1 1001 1001 110561 Dec 22 22:21 VOL00001.DAT
```

But the container runs as a different user (e.g., `appuser` with UID 1000), causing:
```
PermissionError: [Errno 13] Permission denied: '/tmp/doj-fixed/VOL00001.DAT'
```

## Solution Pattern

```bash
# 1. Copy files to container
docker cp /local/path container:/tmp/destination

# 2. Fix ownership (requires root)
docker exec --user root container chown -R appuser:appuser /tmp/destination

# 3. Verify permissions
docker exec container ls -la /tmp/destination
```

## Alternative: Pre-fix on Host

If you know the target UID before copying:
```bash
# On host, before docker cp
sudo chown -R 1000:1000 /local/path
docker cp /local/path container:/tmp/destination
```

## DocViewer Specifics

DocViewer backend container runs as `appuser` (UID 1000):
```bash
docker exec docviewer-prod-xxx-backend-1 id
# uid=1000(appuser) gid=1000(appuser) groups=1000(appuser)
```

## Related

- [[dokploy-deployment-guide]] - General Dokploy patterns
- [[docviewer-deployment-architecture]] - DocViewer container structure