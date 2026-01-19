#!/usr/bin/env bash
# Project-specific post-start setup for memex
# This runs AFTER the common post-start script (called by post-start-common.sh)
set -euo pipefail

workspace="${1:-$(pwd)}"

log() { printf '[memex:post-start] %s\n' "$*"; }

# --- Ticket Initialization ---
# Ensure .tickets exists if tk is available
# (tk is installed by post-start-common.sh if VOIDLABS_TICKET=true)
if command -v tk &>/dev/null; then
    if [[ ! -d "$workspace/.tickets" ]]; then
        log "Initializing tk (.tickets)..."
        mkdir -p "$workspace/.tickets"
    fi
fi


# --- Project-specific setup ---
# Add your recurring post-start tasks here

log "Project-specific setup complete."
