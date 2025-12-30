---
title: DocViewer Project
tags:
  - index
  - docviewer
  - project
created: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
---

# DocViewer Project

Document investigation platform for journalists analyzing EFTA (Epstein Files Transparency Act) documents. Features full-text + semantic search, hybrid ranking, OCR support, and multi-dataset isolation.

## Documentation

### Core Documentation
- [[overview]] - Project overview, architecture, and technology stack
- [[status-and-recovery-plan]] - Current deployment status and recovery roadmap
- [[dataset-ingestion-workflow]] - How to ingest new document datasets

### Related Infrastructure
- [[garage-s3-object-storage]] - S3 storage for document images
- [[virtiofs-limitations-and-troubleshooting]] - Storage access patterns

## Quick Links

| Resource | Location |
|----------|----------|
| Code | `/srv/fast/code/epstein/` |
| Worktree | `.worktrees/multiple-data-sets/` |
| Data | `/srv/fast/datashare/2025-12-19-Epstein/` |
| Production | https://docviewer.voidlabs.cc |

## Beads Tracking

| Issue | Description |
|-------|-------------|
| epstein-rqh | Production Recovery Epic |
| epstein-crq | Backend deployment fix |
| epstein-qms | Frontend deployment fix |
| epstein-4ak | Projects feature |

## Status Summary

| Component | Status |
|-----------|--------|
| Backend | ❌ Down (Bad Gateway) |
| Frontend | ❌ Not deployed |
| Database | ✅ Up |
| S3 Storage | ⚠️ Partial |
| OCR | 5% complete |
| Embeddings | 0% complete |