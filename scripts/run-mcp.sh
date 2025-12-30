#!/bin/bash
# MCP server wrapper for memex
set -e

# Use plugin's own KB content directory (portable across installations)
export KB_ROOT="${CLAUDE_PLUGIN_ROOT}/kb"
export INDEX_ROOT="${CLAUDE_PLUGIN_ROOT}/.indices"

exec uv --directory "${CLAUDE_PLUGIN_ROOT}" run python -m memex
