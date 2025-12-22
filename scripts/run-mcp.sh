#!/bin/bash
# MCP server wrapper for voidlabs-kb
set -e

# Use canonical source location for KB content (shared across all installs)
export KB_ROOT="/srv/fast/code/voidlabs-kb/kb"
export INDEX_ROOT="/srv/fast/code/voidlabs-kb/.indices"

exec uv --directory "${CLAUDE_PLUGIN_ROOT}" run python -m voidlabs_kb
