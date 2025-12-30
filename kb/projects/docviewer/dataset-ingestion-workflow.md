---
title: DocViewer Dataset Ingestion Workflow
tags:
  - docviewer
  - ingestion
  - workflow
  - datasets
  - s3
created: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
---

# DocViewer Dataset Ingestion Workflow

## Overview
Standardized process to ingest new document datasets into DocViewer.

## Prerequisites
- Backend API running
- Garage S3 bucket accessible (`efta-images`)
- PostgreSQL database accessible
- Dataset files extracted to `/srv/fast/datashare/efta/`

## Directory Structure

### Expected Layout
```
/srv/fast/datashare/efta/doj-{N}/
├── DATA/
│   ├── VOL000XX.DAT    # Concordance metadata
│   └── VOL000XX.OPT    # Image path mappings
└── IMAGES/
    └── *.pdf           # Document images
```

### DOJ Dataset Reference
| Dataset | Slug | PDFs | Size |
|---------|------|------|------|
| 1 | doj-1 | 3,142 | 2.5GB |
| 2 | doj-2 | 574 | 1.3GB |
| 3 | doj-3 | 67 | 1.2GB |
| 4 | doj-4 | 152 | 708MB |
| 5 | doj-5 | 120 | 125MB |
| 6 | doj-6 | 13 | 104MB |
| 7 | doj-7 | 17 | 195MB |

## Ingestion Steps

### Step 1: Create Dataset via API
```bash
curl -X POST https://docviewer.voidlabs.cc/api/v1/datasets/ \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "DOJ Dataset 1",
    "slug": "doj-1",
    "description": "DOJ EFTA Disclosure - Dataset 1 (3,142 documents)"
  }'
```

Save the returned UUID as `DATASET_ID`.

### Step 2: Upload Images to S3
```bash
aws s3 sync /srv/fast/datashare/efta/doj-1/IMAGES/ \
  s3://efta-images/doj-1/ \
  --endpoint-url=https://garage-fast.voidlabs.cc
```

### Step 3: Ingest Metadata
```bash
docviewer-ingest import-production \
  --dataset-id="$DATASET_ID" \
  --data-dir=/srv/fast/datashare/efta/doj-1
```

This imports:
- Document metadata from DAT file
- Image paths from OPT file
- Text content (if TEXT files present)

### Step 4: Verify
```bash
# Check document count
curl https://docviewer.voidlabs.cc/api/v1/datasets/$DATASET_ID

# Check S3 file count
aws s3 ls s3://efta-images/doj-1/ --recursive \
  --endpoint-url=https://garage-fast.voidlabs.cc | wc -l
```

## Post-Ingestion

### Queue OCR
```bash
docviewer-ocr queue --dataset=doj-1
```

### Generate Embeddings
```bash
docviewer-embed run --dataset=doj-1
```

## Automation Script
See `scripts/ingest-dataset.sh` (to be created).

## DAT/OPT File Format

### DAT (Concordance)
- Delimiter: `þ` (thorn character)
- Fields: Bates Begin, Bates End, Author, Date Created, Email fields, etc.

### OPT
- CSV format
- Fields: Bates Number, Volume, Path, Is First Page, Page Count

## Troubleshooting

### VirtioFS Errors (error 95)
- Use NFS mounts instead: `/mnt/fast-nfs/`
- Or run operations from devcontainer

### Image Path Mismatch
- Database stores absolute paths
- Ensure `EPSTEIN_DATA_ROOT` matches storage driver config

## Related
- [[docviewer-project-status-recovery-plan]] - Current project status
- [[garage-s3-object-storage]] - S3 configuration
- [[voidlabs-storage-architecture]] - Storage tiers