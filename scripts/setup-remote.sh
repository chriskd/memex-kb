#!/bin/bash
# Setup script for Claude Code remote cloud sessions
# Installs memex (mx) and beads (bd) CLIs

set -e

# Only run in remote environments
if [ "$CLAUDE_CODE_REMOTE" != "true" ]; then
    echo "Skipping remote setup (local environment)"
    exit 0
fi

echo "Setting up remote environment..."

# Install memex CLI via pip
echo "Installing memex (mx)..."
pip install --quiet memex

# Install beads CLI via npm
echo "Installing beads (bd)..."
npm install -g @beads/bd --silent

# Persist environment variables for subsequent commands
if [ -n "$CLAUDE_ENV_FILE" ]; then
    echo "Configuring environment variables..."

    # Set KB root to project-local kb/ directory
    echo "export MEMEX_KB_ROOT=\"\$CLAUDE_PROJECT_DIR/kb\"" >> "$CLAUDE_ENV_FILE"

    # Ensure npm global bin is in PATH
    echo 'export PATH="$PATH:$(npm prefix -g)/bin"' >> "$CLAUDE_ENV_FILE"
fi

echo "Remote setup complete!"
echo "  - mx $(mx --version 2>/dev/null || echo 'installed')"
echo "  - bd $(bd --version 2>/dev/null || echo 'installed')"

exit 0
