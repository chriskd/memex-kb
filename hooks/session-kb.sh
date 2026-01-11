#!/usr/bin/env bash
# SessionStart hook: Show KB quick reference and relevant entries
# Runs with uv to ensure memex module is available

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PLUGIN_ROOT" && uv run python hooks/session-context.py
