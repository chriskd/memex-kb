---
title: VirtioFS Limitations and Troubleshooting
tags:
  - infrastructure
  - virtiofs
  - nfs
  - troubleshooting
  - proxmox
  - storage
created: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
---

# VirtioFS Limitations and Troubleshooting

VirtioFS is Proxmox's high-performance shared folder solution for QEMU VMs, but it has limitations that can cause failures for certain workloads.

## Known Issues

### IO Error 95: Operation Not Supported

**Symptom:**
```
Error: IO error: Not supported (os error 95)
```

Files and directories appear to exist (visible with `ls`, `find`, etc.) but operations fail with "not supported" errors.

**Root Cause:**

VirtioFS implements a limited subset of POSIX filesystem features. It lacks support for:
- Extended attributes (xattrs)
- Certain ioctls
- File locking mechanisms required by some applications
- Some advanced filesystem operations

**Affected Workloads:**

- **Garage S3 Storage** - Requires full POSIX semantics for its storage backend
- **Database engines** - May use file locking or xattrs
- **Container runtimes** - Docker/Podman may need xattrs for overlay filesystems
- **Any application using `setxattr()`/`getxattr()`** system calls

**Example Error Pattern:**

```bash
# Directory appears to exist
$ ls /mnt/virtiofs/garage/data
metadata/  data/

# But operations fail
$ garage server
Error: IO error: Not supported (os error 95)
```

## Solutions

### Solution 1: Use NFS Mounts Instead

For workloads requiring full POSIX capabilities, use NFS instead of virtiofs.

**Configuration in `infrastructure.yml`:**

```yaml
managed_hosts:
  myvm:
    type: vm
    nfs_mounts:
      - source: quasar:/srv/fast-nfs/appdata/myapp
        target: /mnt/fast-nfs
        options: "rw,soft,intr,timeo=30"
```

**Manual Mount:**

```bash
# Create mount point
sudo mkdir -p /mnt/fast-nfs

# Mount NFS share
sudo mount -t nfs 192.168.50.2:/srv/fast-nfs /mnt/fast-nfs \
  -o rw,soft,intr,timeo=30

# Verify
df -h | grep nfs
```

**Make Permanent:**

Add to `/etc/fstab`:
```
192.168.50.2:/srv/fast-nfs /mnt/fast-nfs nfs rw,soft,intr,timeo=30 0 0
```

### Solution 2: Use Local VM Storage

For truly critical workloads, consider using local VM disk images instead of shared storage.

## When to Use Each Option

| Storage Type | Best For | Avoid For |
|--------------|----------|-----------|
| **VirtioFS** | Read-heavy workloads, code repositories, general file sharing | Databases, object storage, applications using xattrs |
| **NFS** | Full POSIX compliance needed, Garage S3, databases | Extremely latency-sensitive workloads |
| **Local Disk** | Maximum performance and isolation | Scenarios requiring shared access from host |

## Available NFS Mounts at Voidlabs

| Mount Point | Source | Purpose |
|-------------|--------|---------|
| `/mnt/fast-nfs` | `192.168.50.2:/srv/fast-nfs` | Performance-sensitive shared data |
| `/mnt/slow-nfs` | `192.168.50.2:/srv/slow-nfs` | Bulk/archival shared data |

Both NFS exports are created via bindfs overlays with proper UID/GID mapping for the `storage` group.

## Debugging Steps

1. **Check if virtiofs is mounted:**
   ```bash
   mount | grep virtiofs
   ```

2. **Test file operations:**
   ```bash
   # Try to create a file with xattrs
   touch /mnt/virtiofs/test.txt
   setfattr -n user.test -v "value" /mnt/virtiofs/test.txt
   ```
   If this fails with "Operation not supported", the application likely needs NFS.

3. **Check application logs:**
   Look for error messages mentioning:
   - `ENOTSUP` (operation not supported)
   - `error 95`
   - `xattr`, `ioctl`, or file locking failures

4. **Compare with NFS:**
   ```bash
   # Test same operation on NFS mount
   touch /mnt/fast-nfs/test.txt
   setfattr -n user.test -v "value" /mnt/fast-nfs/test.txt
   ```

## Migration Pattern: VirtioFS to NFS

Example: Migrating Garage S3 from virtiofs to NFS.

**1. Stop the application:**
```bash
cd /path/to/docker-compose
docker-compose down
```

**2. Ensure NFS is mounted:**
```bash
sudo mount -t nfs 192.168.50.2:/srv/fast-nfs /mnt/fast-nfs -o rw,soft,intr,timeo=30
```

**3. Update docker-compose.yml:**
```yaml
# Before (virtiofs)
volumes:
  - /srv/fast/garage/data:/data

# After (NFS)
volumes:
  - /mnt/fast-nfs/garage/data:/data
```

**4. Migrate data if needed:**
```bash
# Copy from virtiofs to NFS location
sudo rsync -av /srv/fast/garage/ /mnt/fast-nfs/garage/
```

**5. Restart application:**
```bash
docker-compose up -d
```

## Performance Considerations

**VirtioFS advantages:**
- Lower latency for metadata operations
- Better small file performance
- No network overhead

**NFS advantages:**
- Full POSIX compliance
- Better large file streaming
- Easier to debug (standard protocol)
- Can be accessed from multiple VMs simultaneously

For most applications, the performance difference is negligible compared to the compatibility benefits of NFS.

## Related Entries

- [[Voidlabs Storage Architecture]]
- [[Garage S3 Object Storage]]
- [[Voidlabs Infrastructure Overview]]