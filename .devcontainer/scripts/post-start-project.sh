#!/usr/bin/env bash
# Project-specific post-start setup for voidlabs-kb
# This runs AFTER the common post-start script (called by post-start-common.sh)
set -euo pipefail

workspace="${1:-$(pwd)}"

log() { printf '[voidlabs-kb:post-start] %s\n' "$*"; }

# --- Beads Initialization ---
# Initialize beads if bd is available but .beads doesn't exist yet
# (bd is installed by post-start-common.sh if VOIDLABS_BEADS=true)
if command -v bd &>/dev/null; then
    if [[ ! -d "$workspace/.beads" ]]; then
        log "Initializing bd (beads)..."
        (cd "$workspace" && bd init --quiet) || log "bd init skipped (may already be initialized)"
    fi
fi

# --- Project-specific setup ---
# Add your recurring post-start tasks here

log "Project-specific setup complete."
