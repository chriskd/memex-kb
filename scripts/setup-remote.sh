#!/bin/bash
# Setup script for Claude Code remote cloud sessions
# Installs memex (mx) and ticket (tk) CLIs

set -e

# Only run in remote environments
if [ "$CLAUDE_CODE_REMOTE" != "true" ]; then
    echo "Skipping remote setup (local environment)"
    exit 0
fi

echo "Setting up remote environment..."

# Install memex CLI via pip
echo "Installing memex (mx)..."
pip install --quiet memex-kb

# Install ticket CLI via direct download
echo "Installing ticket (tk)..."
install_dir="${HOME}/.local/bin"
mkdir -p "$install_dir"
curl -fsSL https://raw.githubusercontent.com/wedow/ticket/master/ticket -o "$install_dir/tk"
chmod +x "$install_dir/tk"

# Persist environment variables for subsequent commands
if [ -n "$CLAUDE_ENV_FILE" ]; then
    echo "Configuring environment variables..."

    # Set KB root to project-local kb/ directory
    echo "export MEMEX_KB_ROOT=\"\$CLAUDE_PROJECT_DIR/kb\"" >> "$CLAUDE_ENV_FILE"

    # Ensure local bin is in PATH
    echo 'export PATH="$PATH:$HOME/.local/bin"' >> "$CLAUDE_ENV_FILE"
fi

echo "Remote setup complete!"
echo "  - mx $(mx --version 2>/dev/null || echo 'installed')"
echo "  - tk $(tk --version 2>/dev/null || echo 'installed')"

exit 0
