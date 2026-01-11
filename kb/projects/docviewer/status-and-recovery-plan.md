---
title: DocViewer Project Status & Recovery Plan
tags:
  - docviewer
  - efta
  - epstein
  - deployment
  - status
created: 2025-12-22
updated: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
beads_issues:
  - epstein-rqh
  - epstein-so4
  - voidlabs-kb-y2u
---

# DocViewer Project Status & Recovery Plan

## Overview
DocViewer is a document analysis tool for investigative reporters to browse, search, and analyze EFTA (Epstein Files Transparency Act) documents.

## Current State (as of 2025-12-22)

## Current State (as of 2025-12-22)

### Infrastructure Status
| Component | Status | Details |
|-----------|--------|---------|
| Dokploy Server | ⚠️ UNSTABLE | Proxmox container keeps crashing |
| Backend API | 🔄 CONFIGURED | Image built, env fixed, waiting for server stability |
| Frontend | ⚠️ NOT DEPLOYED | Code exists, needs deploy after backend is stable |
| PostgreSQL | ✅ CONFIGURED | Service started, 1/1 replicas |
| Garage S3 | ⚠️ PARTIAL | Bucket exists, only 2 files uploaded |
| Qdrant Vector | ⚠️ UNKNOWN | Not tested |

### Fixes Applied (2025-12-22)
1. **Domain port**: 5001 → 8000 ✅
2. **Mount directory**: Created `/srv/shares/epstein-documents` ✅
3. **DATABASE_URL**: Fixed to use `docviewer-epsteindb-uouojr` ✅
4. **Backend image**: Rebuilt with uvicorn on port 8000 ✅
5. **Database service**: Started successfully ✅

### Pending
- Investigate Proxmox container stability
- Verify backend responds after server is stable
- Deploy frontend

### Data Status
| Metric | Current | Expected | Completion |
|--------|---------|----------|------------|
| Datasets | 5 | 7+ | Partial |
| Documents | 6,274 | ~4,085 new | Some overlap |
| Images (S3) | 2 | 30,833 | 0% |
| OCR Complete | 1,454 | 30,833 | 4.7% |
| Embeddings | 0 | 6,274 | 0% |

## Critical Issues

### 1. Backend Not Running
- `docker-compose.yml:62` has wrong module path
- `dokploy-stack.yml:48` references missing script
- Port mismatch between container and exposure

### 2. Frontend Not Deployed
- No production Dockerfile
- Bug in `DocumentPage.tsx:163` - image URLs missing datasetId

### 3. Images Not in S3
- Upload interrupted by VirtioFS errors
- Switched to NFS but full sync never completed

## Recovery Plan

### Phase 1: Fix Deployment (P0)
- Fix docker-compose.yml module path
- Fix dokploy-stack.yml port and migration
- Create frontend Dockerfile
- Fix image URL bug

### Phase 2: Projects Feature (Required for reporters)
- Database migration for projects table
- Backend API endpoints
- Frontend project selector

### Phase 3: Ingestion Pipeline
- Normalize directory structure
- Create automation script
- Ingest DOJ datasets 1-7

### Phase 4: OCR (GPU on hyperion)
- Process ~4,085 documents

### Phase 5: Embeddings
- Generate vectors for semantic search

## Tracking
- **Epic**: epstein-rqh
- **Backend Bug**: epstein-crq
- **Frontend Bug**: epstein-qms
- **S3 Sync**: epstein-zxd
- **OCR**: epstein-tds
- **Embeddings**: epstein-k9i

## Related Documentation
- [[docviewer-architecture]] - System architecture
- [[garage-s3-object-storage]] - S3 storage configuration
- [[voidlabs-storage-architecture]] - Storage tiers