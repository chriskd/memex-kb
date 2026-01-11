---
title: Voidlabs Storage Architecture
tags:
  - storage
  - zfs
  - nfs
  - virtiofs
  - infrastructure
created: 2025-12-20
updated: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
edit_sources:
  - voidlabs-kb
---

# Voidlabs Storage Architecture

The voidlabs infrastructure uses a multi-tier ZFS storage model with different access patterns for containers vs VMs.

## Storage Tiers

| Tier | Host Path | Purpose | Hardware |
|------|-----------|---------|----------|
| **fast** | `/srv/fast` | Performance-sensitive data | NVMe SSDs |
| **slow** | `/srv/slow` | Bulk data, archives, object storage | HDDs |

## Access Patterns

### LXC Containers
Containers use **bind mounts** for direct filesystem access:
```yaml
mounts:
  - host: /srv/fast/appdata/myapp
    container: /data
```

### QEMU VMs
VMs have dual access methods:

1. **NFS Mounts** - Traditional file sharing
   - Export path: `/srv/fast-nfs`, `/srv/slow-nfs`
   - Uses bindfs overlay for UID mapping
   - **Recommended for workloads requiring full POSIX compliance** (Garage S3, databases, applications using xattrs)

2. **virtiofs** - High-performance shared folders
   - Proxmox native feature
   - Better performance than NFS for local VMs
   - **Limitations:** Does not support extended attributes (xattrs), certain ioctls, or advanced file locking
   - See [[VirtioFS Limitations and Troubleshooting]] for known issues and solutions

## User/Permission Mapping

```yaml
# LXC unprivileged container range
container_uid: 101000
container_gid: 101000

# VM access via storage group
storage_group_gid: 101000
```

The `chris` user is automatically added to the `storage` group on VMs for seamless access.

## Bindfs Overlays

Bindfs creates FUSE-based overlay mounts that handle UID/GID translation:

```
/srv/fast (host UIDs) → bindfs → /srv/fast-nfs (mapped UIDs) → NFS export
```

This allows VMs with different UID schemes to access the same underlying storage.

## Mount Configuration

Storage mounts are defined in `infrastructure.yml`:

```yaml
managed_hosts:
  myapp:
    mounts:
      - host: /srv/fast/appdata/myapp
        container: /data
        readonly: false
    nfs_mounts:  # For VMs
      - source: quasar:/srv/fast-nfs/appdata
        target: /mnt/fast
```

## Related Playbooks

- `bindfs_overlays.yml` - Create FUSE overlay mounts
- `nfs_mounts.yml` - Configure VM NFS access
- `virtiofs_mounts.yml` - Configure VM shared folders

## Related Entries

- [[Voidlabs Infrastructure Overview]]
- [[Voidlabs Provisioning Workflow]]
- [[VirtioFS Limitations and Troubleshooting]]
- [[Garage S3 Object Storage]]