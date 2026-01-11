---
title: Dokploy VM Crash Recovery and Traefik Troubleshooting
tags:
  - dokploy
  - traefik
  - vm-crash
  - troubleshooting
  - docker
  - nfs
  - systemd
  - proxmox
  - 503-error
  - connection-refused
  - container-not-running
  - restart-policy
  - persistent-logging
  - journald
  - boot-loop
  - recovery
  - infrastructure
  - vm-restart
created: 2025-12-22
updated: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
git_branch: main
last_edited_by: chris
---

# Dokploy VM Crash Recovery and Traefik Troubleshooting

## Problem Summary

After VM crashes or restarts, dokploy.voidlabs.cc becomes unreachable with "connection refused" on ports 80/443, even though the VM is running and pingable.

**Symptoms:**
- VM responds to ping but HTTPS fails with "connection refused"
- Port 3000 (Dokploy internal) still accessible directly
- Multiple quick reboots in boot history
- `systemctl --failed` shows mount failures

## Root Cause Analysis

### Primary Issue: Traefik Not Auto-Restarting

The `dokploy-traefik` container had restart policy set to `no`, meaning it doesn't restart after VM reboot:

```bash
# Check restart policy
docker inspect dokploy-traefik --format '{{.HostConfig.RestartPolicy.Name}}'
# Output: no  <-- This is the problem
```

### Secondary Issue: Stale NFS Mount

A stale `/srv/shares` entry in `/etc/fstab` was causing systemd mount failures:

```
mount.nfs: access denied by server while mounting 192.168.50.2:/srv/shares
```

This mount was never exported on quasar - only `/srv/fast-nfs` and `/srv/slow-nfs` exist.

## Diagnostic Commands

### Check if Traefik is running
```bash
# Via PVE guest agent (when SSH unavailable)
ssh root@192.168.50.2 "qm guest exec 118 -- docker ps --filter name=traefik"

# Direct (when SSH works)
ssh root@dokploy.voidlabs.cc "docker ps --filter name=traefik"
```

### Check port connectivity
```bash
nc -zv 192.168.51.87 22 80 443 3000
# Expected: 22 open, 80/443 open (if Traefik running), 3000 open
```

### Check boot history for crash patterns
```bash
ssh root@192.168.50.2 "qm guest exec 118 -- journalctl --list-boots"
```

### Check failed systemd units
```bash
ssh root@192.168.50.2 "qm guest exec 118 -- systemctl --failed"
```

## Resolution Steps

### 1. Start Traefik Container
```bash
ssh root@192.168.50.2 "qm guest exec 118 -- docker start dokploy-traefik"
```

### 2. Set Traefik Restart Policy
```bash
ssh root@192.168.50.2 "qm guest exec 118 -- docker update --restart=unless-stopped dokploy-traefik"
```

### 3. Remove Stale NFS Mount (if applicable)
```bash
# Check fstab for stale mounts
ssh root@192.168.50.2 "qm guest exec 118 -- cat /etc/fstab"

# Remove stale entry
ssh root@192.168.50.2 "qm guest exec 118 -- sed -i '/192.168.50.2:\\/srv\\/shares/d' /etc/fstab"

# Reload systemd and clear failed units
ssh root@192.168.50.2 "qm guest exec 118 -- systemctl daemon-reload"
ssh root@192.168.50.2 "qm guest exec 118 -- systemctl reset-failed"
```

### 4. Start Other Stopped Containers
Docker Compose containers don't auto-restart like Swarm services:
```bash
ssh root@192.168.50.2 "qm guest exec 118 -- docker start <container-name>"
```

## Logging Configuration

Persistent journald logging was enabled to survive reboots:

```bash
# Create config
mkdir -p /etc/systemd/journald.conf.d
cat > /etc/systemd/journald.conf.d/persistent.conf << 'EOF'
[Journal]
Storage=persistent
SystemMaxUse=500M
EOF

# Restart journald
systemctl restart systemd-journald
```

**To check crash logs after next incident:**
```bash
# Check previous boot logs
journalctl -b -1 --no-pager | tail -100

# Check all error-level messages from last boot
journalctl -b -1 -p err --no-pager
```

## Prevention

1. **Traefik restart policy** - Now set to `unless-stopped`
2. **Clean fstab** - Removed stale mounts that don't have corresponding NFS exports
3. **Persistent logging** - Journal survives reboots for post-mortem analysis

## VM Access Methods

When direct SSH fails, use PVE guest agent:
```bash
# Run command via guest agent
ssh root@192.168.50.2 "qm guest exec 118 -- <command>"

# Check VM status
ssh root@192.168.50.2 "qm list | grep dokploy"
```

## Related

- [[infrastructure/dokploy/dokploy-network-recovery-after-vm-restart.md|Dokploy Network Recovery After VM Restart]] - For 504 errors after restart
- [[infrastructure/dokploy/dokploy-container-management.md|Dokploy Container Management]] - Container naming and operations
- [[infrastructure/proxmox-vm-filesystem-recovery-with-fsck.md|Proxmox VM Filesystem Recovery]] - For filesystem corruption issues
- [[infrastructure/dokploy/dokploy-compose-deployment-lessons.md|Dokploy Compose Deployment Lessons]] - Networking gotchas

## Search Keywords

dokploy down, traefik not running, connection refused 443, connection refused 80, dokploy unreachable, vm crash, boot loop, container not starting, docker restart policy, nfs mount failed, access denied by server, systemd failed units, journald persistent, qm guest exec, proxmox vm troubleshooting