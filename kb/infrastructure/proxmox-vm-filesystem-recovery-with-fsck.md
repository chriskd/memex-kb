---
title: Proxmox VM Filesystem Recovery with fsck
tags:
  - proxmox
  - filesystem
  - recovery
  - fsck
  - ext4
  - troubleshooting
  - vm
  - read-only
  - journal
  - disk-repair
  - qm
  - zfs
  - zvol
  - e2fsck
  - corruption
created: 2025-12-22
updated: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
---

# Proxmox VM Filesystem Recovery with fsck

## Symptoms

VM shows EXT4 filesystem errors and goes read-only:

```
EXT4-fs error (device sda1): ext4_journal_check_start:83: Detected aborted journal
EXT4-fs (sda1): Remounting filesystem read-only
```

Services fail with "read-only file system" errors.

## Recovery Steps

### 1. Stop the VM

```bash
ssh root@<proxmox-host>
qm stop <vmid>
```

### 2. Find the disk device

```bash
# Check VM config for disk
qm config <vmid> | grep -E '(scsi|virtio|ide|sata).*disk'
# Example: scsi0: guests:vm-118-disk-0,discard=on,size=150G,ssd=1

# Find the partition devices
ls -la /dev/zvol/guests/ | grep <vmid>
# Look for: vm-118-disk-0-part1 (root), vm-118-disk-0-part14 (BIOS), vm-118-disk-0-part15 (EFI)
```

### 3. Run fsck

```bash
# Force full check with auto-repair
sudo e2fsck -f -y /dev/zvol/guests/vm-<vmid>-disk-0-part1
```

**Important**: Use `-f` to force full check even if filesystem appears clean. A basic `fsck -y` may not catch all issues.

### 4. Start the VM

```bash
qm start <vmid>
```

### 5. Verify

```bash
ssh user@<vm>
mount | grep ' / '  # Should show 'rw' not 'ro'
dmesg | grep -i 'ext4\|error'  # Should be clean
```

## Common Issues

### "Device is mounted" error
If fsck says device is mounted, check for stale mounts from previous recovery attempts:
```bash
mount | grep <vmid>
umount /mnt/vm<vmid>
```

### Services fail after recovery
Docker Swarm may need reinitialization:
```bash
docker swarm init --force-new-cluster
```

Dokploy isolated apps may need redeployment (see [[dokploy-network-recovery]]).

## Related

- [[dokploy-network-recovery]]

## ## Search Keywords

read-only file system, aborted journal, filesystem corruption, ext4 error, remounting read-only, disk repair, vm won't boot, emergency mode
