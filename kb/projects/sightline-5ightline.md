---
title: SightLine (5ightline Project)
tags:
  - project
  - facial-recognition
  - aws-rekognition
  - fastapi
  - react
  - lambda
created: 2025-12-20
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
---

# SightLine (5ightline Project)

Web-based facial recognition investigation tool for investigative journalism. Leverages AWS Rekognition for facial analysis.

## Repository

- **Location:** `/srv/fast/code/5ightline`

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React SPA)                      │
│  ┌─────────────┐ ┌─────────────┐ ┌───────────────┐ ┌───────────┐ │
│  │ Collections │ │    Faces    │ │ Rekognition   │ │   Jobs    │ │
│  │ Management  │ │  Management │ │   Users       │ │ Processing│ │
│  └─────────────┘ └─────────────┘ └───────────────┘ └───────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTPS/REST
┌────────────────────────────┴────────────────────────────────────┐
│                    Backend API (FastAPI)                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Service Layer                          │   │
│  │ ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐ │   │
│  │ │Rekognition │ │   Image    │ │   Job    │ │  Health  │ │   │
│  │ │  Service   │ │ Processor  │ │Processor │ │  Check   │ │   │
│  │ └────────────┘ └────────────┘ └──────────┘ └──────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────┬──────────────────┬──────────────────┬────────────────┘
          │                  │                  │
    ┌─────┴─────┐     ┌──────┴──────┐    ┌─────┴─────┐
    │  SQLite   │     │ AWS         │    │   AWS S3  │
    │ Database  │     │ Rekognition │    │  Storage  │
    └───────────┘     └─────────────┘    └───────────┘
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18 + Material-UI v5 + React Router v6 |
| Backend | FastAPI + SQLAlchemy 2.0 + Pydantic |
| Database | SQLite (dev) / PostgreSQL (prod) |
| File Storage | AWS S3 |
| Facial Recognition | AWS Rekognition |
| Package Management | UV (Python), npm (JS) |

## Lambda Functions

Located in `lambdas/`:

| Function | Purpose |
|----------|---------|
| `job-processor` | Starts Rekognition StartFaceSearch on S3 videos |
| `results-to-ddb` | Consumes Rekognition SNS messages via SQS |
| `extract-frame` | Extracts video frames for matches |

### Lambda Environment Variables

```
AWS_REGION
REK_ROLE_ARN
REK_SNS_TOPIC_ARN
SUBMIT_QUEUE_URL
RESULTS_QUEUE_URL
EXTRACT_QUEUE_URL
OUTPUT_BUCKET
OUTPUT_PREFIX
DDB_TABLE_JOBS
DDB_TABLE_MATCHES
```

### Deploy (SAM)

```bash
cd lambdas
sam build --use-container --parallel
sam deploy --guided
```

## API Endpoints

### Collections
```
GET/POST   /api/v1/collections
GET/PUT/DELETE /api/v1/collections/{id}
POST       /api/v1/collections/sync-aws
```

### Faces
```
GET/POST   /api/v1/faces
GET/DELETE /api/v1/faces/{id}
```

### Rekognition Users
```
GET/POST   /api/v1/rekognition-users
GET/PUT/DELETE /api/v1/rekognition-users/{id}
PUT        /api/v1/rekognition-users/{id}/faces
```

### Jobs
```
GET/POST   /api/v1/jobs
GET/DELETE /api/v1/jobs/{id}
POST       /api/v1/jobs/{id}/cancel
GET        /api/v1/jobs/{id}/results
```

## S3 Bucket Structure

```
rekognition-investigation-images/
├── collections/{collection_id}/
│   ├── source-images/
│   └── face-crops/
├── media-sets/{media_set_id}/
│   ├── images/
│   └── thumbnails/
├── temp/processing/{job_id}/
└── exports/{export_id}/
```

## Development Workflow

```bash
# Backend
cd backend
source venv/bin/activate
uvicorn app.main:app --reload

# Frontend
cd frontend
npm run dev

# Access
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

## Database Schema (Core Tables)

- `collections` - Face collections in AWS Rekognition
- `media_sets` - Collections of images/videos to search
- `media_set_files` - Individual files within media sets
- `detection_sessions` - Temporary face detection for user selection
- `rekognition_users` - User profiles for organizing faces
- `faces` - Indexed faces in collections
- `jobs` - Analysis/search jobs
- `results` - Job match results

## Key Features

- **Two-Step Face Selection:** Detection sessions allow users to select which faces to index
- **Job Progress:** Server-Sent Events (SSE) for real-time progress tracking
- **Batch Processing:** Queue-based architecture for large media sets

## Related Entries

- [[Voidlabs Infrastructure Overview]]
- [[AWS Rekognition Integration]]
