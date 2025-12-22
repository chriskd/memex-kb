# syntax=docker/dockerfile:1
# Voidlabs KB Explorer - Production Dockerfile
# Multi-stage build with aggressive caching for fast rebuilds

# Stage 1: Build
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy ONLY dependency specification first (layer caching optimization)
# This layer is cached until pyproject.toml changes
COPY pyproject.toml .

# Create stub source structure for hatchling to find the package
RUN mkdir -p src/voidlabs_kb && \
    echo '__version__ = "0.0.0"' > src/voidlabs_kb/__init__.py

# Create virtual environment and install dependencies with BuildKit cache
# The mount caches persist across builds, so repeated builds reuse downloaded packages
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install -e .

ENV PATH="/app/.venv/bin:$PATH"

# NOW copy real source code (changes here don't invalidate dependency layer)
COPY src/ src/

# Reinstall package with actual source (deps already installed, this is fast)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install . --reinstall-package voidlabs-kb

# Pre-download the embedding model with BuildKit cache
# Cache persists across builds - model only downloads once
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" && \
    cp -r /root/.cache/huggingface /tmp/huggingface-cache

# Stage 2: Runtime
FROM python:3.12-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code
COPY --from=builder /app/src /app/src

# Copy pre-downloaded model cache (from temp location due to BuildKit cache mount)
COPY --from=builder /tmp/huggingface-cache /home/appuser/.cache/huggingface
RUN chown -R appuser:appuser /home/appuser/.cache || true

# Create directories for indices and views
RUN mkdir -p /data/indices /data/views && chown -R appuser:appuser /data

# Switch to non-root user
USER appuser

# Environment
ENV KB_ROOT=/kb
ENV INDEX_ROOT=/data/indices
ENV VIEWS_ROOT=/data/views
ENV HOST=0.0.0.0
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/stats')" || exit 1

# Use tini as init process
ENTRYPOINT ["/usr/bin/tini", "--"]

# Run the web server
CMD ["python", "-m", "voidlabs_kb.webapp.api"]
