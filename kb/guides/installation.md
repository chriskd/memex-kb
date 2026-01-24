---
title: Installation Guide
tags: [installation, setup, getting-started]
created: 2026-01-06
description: How to install memex with different configurations
---

# Installation Guide

Memex can be installed in two configurations: minimal (keyword search only) or full (with semantic search).

## Minimal Install (Recommended Start)

Fast, lightweight installation with keyword search only:

```bash
# With uv (recommended)
uv tool install memex-kb

# With pip
pip install memex-kb

# Verify installation
mx --version
```

This gives you:
- Full CLI functionality (`mx` command)
- MCP server for Claude Desktop
- BM25 keyword search via Whoosh
- ~100MB footprint

## Full Install (Semantic Search)

Add semantic search for meaning-based queries:

```bash
# With uv
uv tool install "memex-kb[semantic]"

# With pip
pip install "memex-kb[semantic]"
```

This adds:
- ChromaDB for vector storage
- sentence-transformers for embeddings
- CPU-only PyTorch (~500MB additional)
- First search downloads embedding model (~100MB)

## From Source

For development or customization:

```bash
git clone https://github.com/chriskd/memex.git
cd memex

# Core only (~100MB)
uv sync

# With semantic search (~600MB)
uv sync --all-extras
```

## GPU Support (Optional)

If you have an NVIDIA GPU and want CUDA acceleration:

```bash
uv sync --all-extras --index pytorch-gpu=https://download.pytorch.org/whl/cu124
```

## Platform Notes

- **ARM64 (Apple Silicon)**: ChromaDB capped at <1.0.0 for onnxruntime compatibility
- **CPU-only default**: PyTorch configured for CPU to minimize install size
- **Python requirement**: 3.11 or higher

## Next Steps

After installation:
1. [[guides/quick-start|Quick Start Guide]] - Create your first KB
2. [[reference/cli|CLI Reference]] - Full command documentation
3. [[guides/mcp-setup|MCP Server Setup]] - Configure for Claude Desktop
