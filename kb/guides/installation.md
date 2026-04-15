---
title: Installation Guide
tags: [installation, setup, getting-started]
created: 2026-01-06
updated: 2026-04-15
description: Install memex, choose a KB scope, and verify your setup
---

# Installation Guide

Memex installs with keyword search enabled by default (Whoosh). Semantic search
(ChromaDB + sentence-transformers) is optional via the `search` extra.

## Install (Recommended)

```bash
# With uv (recommended)
uv tool install memex-kb

# With pip
pip install memex-kb

# Verify installation
mx --version
```

This includes:
- Keyword search (Whoosh)
- KB health checks and graph tooling
- Static site publishing (`mx publish`)

## Choose Your First KB

If you are starting a new repo, the fastest path is:

```bash
mx onboard --init --yes
```

If you want to avoid walking up parent directories while testing in a sandbox:

```bash
mx onboard --cwd-only --init --yes
```

For personal or custom KBs:

```bash
mx init --user --sample
mx init --path docs/kb --no-sample
mx init --force --sample
```

`mx init --user` creates a KB in `~/.memex/kb/`. `--sample` creates a starter entry; `--no-sample`
leaves the KB empty.

If you cloned an existing KB or copied docs from another system, run:

```bash
mx doctor --timestamps
mx doctor --timestamps --fix
```

That repairs missing, invalid, or stale frontmatter timestamps before publishing or browsing.

## Semantic Search (Optional)

If you want semantic search, install the `search` extra (larger footprint; may download models on first use):

```bash
uv tool install 'memex-kb[search]'
# or
pip install 'memex-kb[search]'
```

## From Source

For development or customization:

```bash
git clone https://github.com/chriskd/memex-kb.git
cd memex

# Runtime dependencies
uv sync

# Dev dependencies
uv sync --dev
```

## GPU Support (Optional)

If you have an NVIDIA GPU and want CUDA acceleration:

```bash
uv sync --index pytorch-gpu=https://download.pytorch.org/whl/cu124
```

## Platform Notes

- **ARM64 (Apple Silicon)**: ChromaDB capped at <1.0.0 for onnxruntime compatibility
- **CPU-only default**: PyTorch configured for CPU to minimize install size
- **Python requirement**: 3.11 or higher
- `MEMEX_CONTEXT_NO_PARENT=1` disables parent `.kbconfig` discovery for the current shell
- `MEMEX_USER_KB_ROOT` overrides the user KB location
- `MEMEX_INDEX_ROOT` moves search indices out of the KB directory
- `MEMEX_QUIET=1` suppresses non-fatal warnings

## Next Steps

After installation:
1. [[guides/quick-start|Quick Start Guide]] - Create your first KB
2. [[guides/ai-integration|AI Integration]] - Configure AI assistants
3. [[reference/index|Reference Index]] - Command and format details
