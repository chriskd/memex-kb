#!/bin/bash
# MCP server wrapper that sets up environment from CLAUDE_PLUGIN_ROOT
set -e

export KB_ROOT="${CLAUDE_PLUGIN_ROOT}/kb"
export INDEX_ROOT="${CLAUDE_PLUGIN_ROOT}/.indices"

exec uv --directory "${CLAUDE_PLUGIN_ROOT}" run python -m voidlabs_kb
