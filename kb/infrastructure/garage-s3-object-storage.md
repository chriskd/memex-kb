---
title: Garage S3 Object Storage
tags:
  - infrastructure
  - s3
  - garage
  - object-storage
  - dokploy
created: 2025-12-22
updated: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
---

# Garage S3 Object Storage

Garage is a self-hosted, S3-compatible distributed object storage system running on dokploy.voidlabs.cc.

## Instances

| Instance | Endpoint | Storage | Use Case |
|----------|----------|---------|----------|
| garage-fast | `https://garage-fast.voidlabs.cc` | NVMe pool | High-performance workloads |
| garage-slow | `https://garage-slow.voidlabs.cc` | HDD pool | Archival/cold storage |

Both are managed via Dokploy with WebUI access available.

## Architecture

Garage separates concerns into two API layers:

1. **Admin API** - Bucket/key management (requires admin token or CLI access)
2. **S3 API** - Object operations (uses standard S3 access keys)

### Storage Backend Requirements

Garage requires **full POSIX filesystem semantics** for its data directory:
- Extended attributes (xattrs) support
- Proper file locking
- Complete ioctl support

**Important:** VirtioFS does not support these features and will fail with `IO error: Not supported (os error 95)`. Use NFS mounts instead.

See [[VirtioFS Limitations and Troubleshooting]] for details.

## AWS CLI Configuration

```bash
# Configure a profile for Garage
aws configure set aws_access_key_id "YOUR_ACCESS_KEY" --profile garage
aws configure set aws_secret_access_key "YOUR_SECRET_KEY" --profile garage
aws configure set region "garage" --profile garage

# Use with --endpoint-url
aws --profile garage --endpoint-url https://garage-fast.voidlabs.cc s3 ls
```

## Bucket Management (Admin Operations)

Access keys cannot create buckets. Use the Garage CLI inside the container:

```bash
# Create bucket
ssh dokploy.voidlabs.cc "docker exec voidlabssharedservices-garagewithui-vo0uq2-garage-fast-1 \
  /garage bucket create my-bucket"

# Grant access to a key
ssh dokploy.voidlabs.cc "docker exec voidlabssharedservices-garagewithui-vo0uq2-garage-fast-1 \
  /garage bucket allow --read --write my-bucket --key GK..."

# List buckets
ssh dokploy.voidlabs.cc "docker exec voidlabssharedservices-garagewithui-vo0uq2-garage-fast-1 \
  /garage bucket list"

# Show bucket info
ssh dokploy.voidlabs.cc "docker exec voidlabssharedservices-garagewithui-vo0uq2-garage-fast-1 \
  /garage bucket info my-bucket"
```

## Object Operations (S3 API)

Standard S3 operations work with the endpoint URL:

```bash
# Upload file
aws --profile garage --endpoint-url https://garage-fast.voidlabs.cc \
  s3 cp myfile.pdf s3://my-bucket/

# Sync directory
aws --profile garage --endpoint-url https://garage-fast.voidlabs.cc \
  s3 sync ./local-dir s3://my-bucket/prefix/

# List objects
aws --profile garage --endpoint-url https://garage-fast.voidlabs.cc \
  s3 ls s3://my-bucket/
```

## Python/boto3 Usage

```python
import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="https://garage-fast.voidlabs.cc",
    aws_access_key_id="GK...",
    aws_secret_access_key="...",
    region_name="garage"
)

# Upload
s3.upload_file("local.pdf", "my-bucket", "remote.pdf")

# Download
s3.download_file("my-bucket", "remote.pdf", "local.pdf")

# Generate presigned URL (for web access)
url = s3.generate_presigned_url(
    "get_object",
    Params={"Bucket": "my-bucket", "Key": "file.pdf"},
    ExpiresIn=3600
)
```

## Container Names

For admin operations, the relevant containers on dokploy are:

- `voidlabssharedservices-garagewithui-vo0uq2-garage-fast-1` - Fast tier Garage
- `voidlabssharedservices-garagewithui-vo0uq2-garage-slow-1` - Slow tier Garage
- `voidlabssharedservices-garagewithui-vo0uq2-garage-webui-fast-1` - Fast WebUI
- `voidlabssharedservices-garagewithui-vo0uq2-garage-webui-slow-1` - Slow WebUI

### Volume Configuration

Garage data directories must be on **NFS mounts**, not virtiofs:

```yaml
# docker-compose.yml
services:
  garage-fast:
    volumes:
      # Use NFS mount path
      - /mnt/fast-nfs/garage/data:/data
      # NOT virtiofs path like /srv/fast/garage/data
```

**Setup NFS mount on dokploy VM:**

```bash
# Mount fast tier storage
sudo mount -t nfs 192.168.50.2:/srv/fast-nfs /mnt/fast-nfs \
  -o rw,soft,intr,timeo=30

# Mount slow tier storage
sudo mount -t nfs 192.168.50.2:/srv/slow-nfs /mnt/slow-nfs \
  -o rw,soft,intr,timeo=30
```

Make permanent by adding to `/etc/fstab` or configuring via `infrastructure.yml`.

## Existing Buckets

| Bucket | Purpose | Access Key |
|--------|---------|------------|
| `efta-images` | EFTA case document images | docviewer key |

## See Also

- [[Voidlabs Infrastructure Overview]]
- [[Voidlabs Storage Architecture]]
- [[VirtioFS Limitations and Troubleshooting]]
- [[Dokploy Container Management]]

## ### Storage Backend Requirements

### Storage Backend Requirements

Garage requires **full POSIX filesystem semantics** for its data directory:
- Extended attributes (xattrs) support
- Proper file locking
- Complete ioctl support

**CRITICAL:** VirtioFS does not support these features and will fail with:
- `IO error: Not supported (os error 95)` on xattr operations
- `Could not find expected marker file 'garage-marker'` after container restart

**Always use NFS mounts**, not virtiofs:

| Mount Type | Path Pattern | Works? |
|------------|--------------|--------|
| NFS | `/mnt/fast-nfs/...` | ✅ Yes |
| VirtioFS | `/mnt/fast/...` | ❌ No |

See [[VirtioFS Limitations and Troubleshooting]] for details.

## ### Volume Configuration

### Volume Configuration

Garage data directories **MUST** be on NFS mounts, not virtiofs:

```yaml
# docker-compose.yml
services:
  garage-fast:
    volumes:
      # ✅ CORRECT: Use NFS mount path
      - /mnt/fast-nfs/dokploy.voidlabs.cc/voidlabs-shared-services/garage/data:/var/lib/garage/data
      # ❌ WRONG: VirtioFS path will fail
      # - /mnt/fast/dokploy.voidlabs.cc/.../garage/data:/var/lib/garage/data
```

**Verify NFS is mounted on dokploy VM:**

```bash
# Check if NFS mounts exist
ssh -i ~/.ssh/id_ed25519-ansible chris@dokploy.voidlabs.cc "mount | grep nfs"

# Mount if missing
ssh -i ~/.ssh/id_ed25519-ansible chris@dokploy.voidlabs.cc "
  sudo mount -t nfs 192.168.50.2:/srv/fast-nfs /mnt/fast-nfs -o rw,soft,intr,timeo=30
  sudo mount -t nfs 192.168.50.2:/srv/slow-nfs /mnt/slow-nfs -o rw,soft,intr,timeo=30
"
```

NFS mounts should be persistent in `/etc/fstab`:

```
192.168.50.2:/srv/fast-nfs /mnt/fast-nfs nfs rw,soft,intr,timeo=30 0 0
192.168.50.2:/srv/slow-nfs /mnt/slow-nfs nfs rw,soft,intr,timeo=30 0 0
```

## ## Existing Buckets

## Existing Buckets

| Bucket | Purpose | Access Keys |
|--------|---------|-------------|
| `efta-images` | EFTA/DOJ document images | `docviewer` (dev), `docviewer-prod` (production) |

### Key Details

| Key Name | Key ID | Buckets | Environment |
|----------|--------|---------|-------------|
| `docviewer` | `GKfdbcb1fcd0f2f1e7c36479e8` | efta-images, docviewer-images | Development |
| `docviewer-prod` | `GK812ac25f8b505735494080f2` | efta-images | Production (Dokploy) |
