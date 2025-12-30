#!/usr/bin/env bash
# Shared post-start setup for all voidlabs devcontainers
# Managed by copier - updates via `copier update`
#
# Features are controlled by .devcontainer/voidlabs.conf
# Configs are in .devcontainer/configs/
set -euo pipefail

workspace="${1:-$(pwd)}"

# --- Logging ---
log() { printf '[voidlabs-devtools] %s\n' "$*"; }
log_feature() { printf '[voidlabs-devtools:%s] %s\n' "$1" "$2"; }

# --- Configuration ---
# Defaults (all features enabled)
VOIDLABS_BEADS="${VOIDLABS_BEADS:-true}"
VOIDLABS_VLMAIL="${VOIDLABS_VLMAIL:-true}"
VOIDLABS_MEMEX="${VOIDLABS_MEMEX:-true}"
VOIDLABS_FACTORY="${VOIDLABS_FACTORY:-true}"
VOIDLABS_CHEZMOI="${VOIDLABS_CHEZMOI:-true}"
VOIDLABS_WORKTRUNK="${VOIDLABS_WORKTRUNK:-true}"

load_config() {
    local config_file="$workspace/.devcontainer/voidlabs.conf"
    if [[ -f "$config_file" ]]; then
        # shellcheck source=/dev/null
        source "$config_file"
        log "Loaded config from $config_file"
    else
        log "No voidlabs.conf found, using defaults (all features enabled)"
    fi
}

# --- Phase Secrets (Always Enabled) ---
# Load Phase secrets into current environment (for use by other setup functions)
export_phase_secrets() {
    if command -v phase &>/dev/null && [[ -n "${PHASE_SERVICE_TOKEN:-}" ]]; then
        local phase_env="${PHASE_ENV:-development}"
        local phase_app="${PHASE_APP:-}"
        local app_flag=""

        # Build app flag if PHASE_APP is set
        if [[ -n "$phase_app" ]]; then
            app_flag="--app=$phase_app"
            log_feature "phase" "Exporting secrets from Phase (app=$phase_app, env=$phase_env)..."
        else
            log_feature "phase" "Exporting secrets from Phase (env=$phase_env)..."
        fi

        # shellcheck disable=SC2086
        local secrets
        secrets=$(phase secrets export --env "$phase_env" $app_flag --format=kv 2>&1)
        local exit_code=$?

        if [[ $exit_code -ne 0 ]]; then
            log_feature "phase" "Failed to export secrets: $secrets"
            return 1
        fi

        # Eval the secrets to export them
        eval "$secrets"

        # Verify we got ANTHROPIC_API_KEY (expected key)
        if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
            log_feature "phase" "ANTHROPIC_API_KEY loaded successfully"
        else
            log_feature "phase" "Warning: ANTHROPIC_API_KEY not found in Phase secrets"
        fi
    fi
}

setup_phase_secrets() {
    local SNIPPET_MARKER="# >>> phase-secrets >>>"
    local PHASE_SOURCE_BLOCK="
$SNIPPET_MARKER
# Export Phase secrets to environment (requires PHASE_SERVICE_TOKEN, PHASE_HOST set)
# Uses PHASE_ENV (default: development) and optionally PHASE_APP
if command -v phase &>/dev/null && [ -n \"\${PHASE_SERVICE_TOKEN:-}\" ]; then
  _phase_env=\"\${PHASE_ENV:-development}\"
  _phase_app_flag=\"\"
  [ -n \"\${PHASE_APP:-}\" ] && _phase_app_flag=\"--app=\$PHASE_APP\"
  eval \$(phase secrets export --env \"\$_phase_env\" \$_phase_app_flag --format=kv 2>/dev/null) || true
  unset _phase_env _phase_app_flag
fi
# <<< phase-secrets <<<"

    # Add to profile.d for login shells
    local PROFILE_SNIPPET="/etc/profile.d/90-phase-secrets.sh"
    if command -v sudo &>/dev/null; then
        cat << 'EOF' | sudo tee "$PROFILE_SNIPPET" >/dev/null
# Export Phase secrets to environment
# Uses PHASE_ENV (default: development) and optionally PHASE_APP
if command -v phase &>/dev/null && [ -n "${PHASE_SERVICE_TOKEN:-}" ]; then
  _phase_env="${PHASE_ENV:-development}"
  _phase_app_flag=""
  [ -n "${PHASE_APP:-}" ] && _phase_app_flag="--app=$PHASE_APP"
  eval $(phase secrets export --env "$_phase_env" $_phase_app_flag --format=kv 2>/dev/null) || true
  unset _phase_env _phase_app_flag
fi
EOF
        sudo chmod 755 "$PROFILE_SNIPPET"
        log_feature "phase" "Installed secrets loader at $PROFILE_SNIPPET"
    fi

    # Add to bashrc for interactive non-login shells
    if [[ -f "$HOME/.bashrc" ]] && ! grep -q "$SNIPPET_MARKER" "$HOME/.bashrc" 2>/dev/null; then
        echo "$PHASE_SOURCE_BLOCK" >> "$HOME/.bashrc"
        log_feature "phase" "Added secrets export to ~/.bashrc"
    fi

    # Add to zshrc for zsh shells
    if [[ -f "$HOME/.zshrc" ]] && ! grep -q "$SNIPPET_MARKER" "$HOME/.zshrc" 2>/dev/null; then
        echo "$PHASE_SOURCE_BLOCK" >> "$HOME/.zshrc"
        log_feature "phase" "Added secrets export to ~/.zshrc"
    fi
}

# --- Beads Issue Tracking ---
setup_beads() {
    # Install bd if not present (uses official installer which downloads pre-built binaries)
    if ! command -v bd &>/dev/null; then
        log_feature "beads" "Installing bd from GitHub releases..."
        if curl -fsSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash; then
            # Ensure ~/.local/bin is in PATH for this session (where installer puts it)
            export PATH="$HOME/.local/bin:$PATH"
            log_feature "beads" "Installed bd v$(bd version 2>/dev/null | awk '{print $3}' || echo 'unknown')"
        else
            log_feature "beads" "Failed to install bd, skipping"
            return 0
        fi
    fi

    # Verify bd is available
    if ! command -v bd &>/dev/null; then
        log_feature "beads" "bd not available after install attempt, skipping"
        return 0
    fi

    # Skip if not in a git repository
    if ! git rev-parse --git-dir &>/dev/null 2>&1; then
        log_feature "beads" "Not in a git repository, skipping hooks"
        return 0
    fi

    # If .beads directory exists, ensure hooks are installed and current
    if [[ -d "$workspace/.beads" ]]; then
        log_feature "beads" "Installing/updating git hooks..."
        (cd "$workspace" && bd --quiet hooks install 2>/dev/null) || true
    fi
}

# --- vl-mail Agent Messaging ---
setup_vlmail() {
    local VLMAIL_BIN="$HOME/.local/bin/vl-mail"
    local DEVTOOLS_DIR="/srv/fast/code/voidlabs-devtools"
    local VLMAIL_SRC="$DEVTOOLS_DIR/cmd/vl-mail"

    # Check if voidlabs-devtools is available
    if [[ ! -d "$VLMAIL_SRC" ]]; then
        log_feature "vl-mail" "voidlabs-devtools not found at $DEVTOOLS_DIR, skipping"
        return 0
    fi

    # Check if Go is available
    local GO_BIN="/usr/local/go/bin/go"
    if [[ ! -x "$GO_BIN" ]]; then
        log_feature "vl-mail" "Go not found at $GO_BIN, skipping"
        return 0
    fi

    # Build vl-mail if binary doesn't exist or source is newer
    mkdir -p "$(dirname "$VLMAIL_BIN")"
    if [[ ! -f "$VLMAIL_BIN" ]] || [[ "$VLMAIL_SRC/main.go" -nt "$VLMAIL_BIN" ]]; then
        log_feature "vl-mail" "Building vl-mail..."
        if (cd "$DEVTOOLS_DIR" && $GO_BIN build -o "$VLMAIL_BIN" ./cmd/vl-mail/); then
            log_feature "vl-mail" "Installed vl-mail to $VLMAIL_BIN"
        else
            log_feature "vl-mail" "Failed to build vl-mail"
            return 0
        fi
    else
        log_feature "vl-mail" "vl-mail already up to date"
    fi

    # Configure BEADS_MAIL_DELEGATE env var for bd mail integration
    local ENV_MARKER="# >>> vl-mail-delegate >>>"
    local ENV_BLOCK="
$ENV_MARKER
# vl-mail as bd mail delegate
export BEADS_MAIL_DELEGATE=\"vl-mail\"
# <<< vl-mail-delegate <<<"

    # Add to shell profiles if not present
    for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
        if [[ -f "$rc" ]] && ! grep -q "$ENV_MARKER" "$rc" 2>/dev/null; then
            echo "$ENV_BLOCK" >> "$rc"
            log_feature "vl-mail" "Added BEADS_MAIL_DELEGATE to $rc"
        fi
    done

    # Export for current session
    export BEADS_MAIL_DELEGATE="vl-mail"
    log_feature "vl-mail" "bd mail delegate configured"
}

# --- Memex Knowledge Base CLI (mx) ---
setup_memex() {
    local MEMEX_REPO="/srv/fast/code/memex"

    # Check if memex repo is available
    if [[ ! -d "$MEMEX_REPO" ]]; then
        log_feature "memex" "memex repo not found at $MEMEX_REPO, skipping"
        return 0
    fi

    # Check if uv is available
    if ! command -v uv &>/dev/null; then
        log_feature "memex" "uv not found, skipping"
        return 0
    fi

    # Ensure ~/.local/bin is in PATH (where uv tool installs binaries)
    export PATH="$HOME/.local/bin:$PATH"

    # Install/update memex if not present or if source changed
    # Use a marker file to track installation timestamp
    local MARKER="$HOME/.local/share/memex-installed"
    local NEEDS_INSTALL=false

    if ! command -v mx &>/dev/null; then
        NEEDS_INSTALL=true
    elif [[ ! -f "$MARKER" ]]; then
        NEEDS_INSTALL=true
    elif [[ "$MEMEX_REPO/src/memex/cli.py" -nt "$MARKER" ]] || \
         [[ "$MEMEX_REPO/src/memex/core.py" -nt "$MARKER" ]] || \
         [[ "$MEMEX_REPO/pyproject.toml" -nt "$MARKER" ]]; then
        NEEDS_INSTALL=true
    fi

    if [[ "$NEEDS_INSTALL" == "true" ]]; then
        log_feature "memex" "Installing memex (mx) from $MEMEX_REPO..."
        # Use uv tool install for isolated environment with CLI in PATH
        # --python 3.11: Pin to Python 3.11 (onnxruntime lacks wheels for newer versions like 3.14)
        # --refresh: Force fresh package metadata
        local install_output
        if install_output=$(uv tool install --force --refresh --python 3.11 -e "$MEMEX_REPO" 2>&1); then
            mkdir -p "$(dirname "$MARKER")"
            touch "$MARKER"
            log_feature "memex" "Installed memex CLI (mx)"
        else
            log_feature "memex" "Failed to install memex:"
            echo "$install_output" | head -20
            return 0
        fi
    else
        log_feature "memex" "memex (mx) already up to date"
    fi

    # Verify installation
    if command -v mx &>/dev/null; then
        log_feature "memex" "mx ready ($(mx --version 2>/dev/null || echo 'version unknown'))"
    fi
}

# --- Factory Droid Integration ---
setup_factory() {
    local FACTORY_CONFIG_DIR="$HOME/.factory"
    local FACTORY_CONFIG="$FACTORY_CONFIG_DIR/config.json"

    # Install droid if not present
    if ! command -v droid &>/dev/null; then
        log_feature "factory" "Installing droid CLI..."
        if curl -fsSL https://app.factory.ai/cli | sh; then
            # Add to PATH for current session
            export PATH="$HOME/.factory/bin:$PATH"
            log_feature "factory" "Installed droid CLI"
        else
            log_feature "factory" "Failed to install droid CLI"
            return 0
        fi
    fi

    # Install config from bundled template if not present or outdated
    mkdir -p "$FACTORY_CONFIG_DIR"
    local BUNDLED_CONFIG="$workspace/.devcontainer/configs/factory-config.json"
    if [[ -f "$BUNDLED_CONFIG" ]]; then
        if [[ ! -f "$FACTORY_CONFIG" ]] || ! diff -q "$BUNDLED_CONFIG" "$FACTORY_CONFIG" &>/dev/null; then
            cp "$BUNDLED_CONFIG" "$FACTORY_CONFIG"
            log_feature "factory" "Installed config from bundled template to $FACTORY_CONFIG"
        fi
    else
        log_feature "factory" "No bundled config found at $BUNDLED_CONFIG, skipping"
    fi

    # Configure beads to use Factory droid (if bd is available)
    if command -v bd &>/dev/null && command -v droid &>/dev/null; then
        log_feature "factory" "Configuring beads for Factory droid..."
        bd setup factory 2>/dev/null || true
    fi
}

# --- Chezmoi Dotfiles ---
setup_chezmoi() {
    local DOTFILES_REPO="git@github.com:chriskd/dotfiles.git"

    # Install chezmoi if not present
    if ! command -v chezmoi &>/dev/null; then
        log_feature "chezmoi" "Installing chezmoi..."
        if sh -c "$(curl -fsLS get.chezmoi.io)" -- -b "$HOME/.local/bin"; then
            export PATH="$HOME/.local/bin:$PATH"
            log_feature "chezmoi" "Installed chezmoi"
        else
            log_feature "chezmoi" "Failed to install chezmoi"
            return 0
        fi
    fi

    # Initialize chezmoi from git repo if not already done
    if [[ ! -d "$HOME/.local/share/chezmoi" ]]; then
        # Ensure GitHub's SSH host key is in known_hosts (prevents interactive prompt)
        mkdir -p "$HOME/.ssh"
        if ! grep -q "github.com" "$HOME/.ssh/known_hosts" 2>/dev/null; then
            log_feature "chezmoi" "Adding GitHub to known_hosts..."
            ssh-keyscan -t ed25519,rsa github.com >> "$HOME/.ssh/known_hosts" 2>/dev/null
        fi

        # Check if SSH agent is available for passphrase-protected keys
        if [[ -z "${SSH_AUTH_SOCK:-}" ]]; then
            log_feature "chezmoi" "Warning: SSH agent not available, keys with passphrases won't work"
        fi

        log_feature "chezmoi" "Initializing from $DOTFILES_REPO..."
        # Use timeout to prevent indefinite hang (30 seconds should be plenty for clone)
        if ! timeout 30 chezmoi init "$DOTFILES_REPO" 2>&1; then
            log_feature "chezmoi" "Failed to initialize chezmoi (SSH auth issue?), skipping"
            return 0
        fi
    fi

    # Unset 1Password service token to avoid mode conflict with chezmoi
    # (chezmoi expects interactive mode but service token triggers service mode)
    unset OP_SERVICE_ACCOUNT_TOKEN

    # Apply dotfiles (IN_DEVCONTAINER is already set, so 1Password calls will be skipped)
    log_feature "chezmoi" "Applying dotfiles..."
    chezmoi apply --force

    log_feature "chezmoi" "Dotfiles applied"
}

# --- Worktrunk Git Worktree Manager ---
setup_worktrunk() {
    # Skip if wt not installed
    if ! command -v wt &>/dev/null; then
        log_feature "worktrunk" "wt not installed, skipping"
        return 0
    fi

    # Install shell integration (enables directory switching)
    log_feature "worktrunk" "Installing shell integration..."
    wt config shell install 2>/dev/null || true

    # Set Anthropic API key for llm CLI (from Phase secrets)
    if command -v llm &>/dev/null && [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
        log_feature "worktrunk" "Setting Anthropic API key for llm..."
        echo "$ANTHROPIC_API_KEY" | llm keys set anthropic 2>/dev/null || true
    fi

    # Create worktrunk user config if not present (use bundled config)
    local WT_CONFIG_DIR="$HOME/.config/worktrunk"
    local WT_CONFIG="$WT_CONFIG_DIR/config.toml"
    local BUNDLED_CONFIG="$workspace/.devcontainer/configs/worktrunk-user-config.toml"
    if [[ ! -f "$WT_CONFIG" ]]; then
        mkdir -p "$WT_CONFIG_DIR"
        if [[ -f "$BUNDLED_CONFIG" ]]; then
            cp "$BUNDLED_CONFIG" "$WT_CONFIG"
            log_feature "worktrunk" "Installed config from bundled template to $WT_CONFIG"
        else
            log_feature "worktrunk" "No bundled config found at $BUNDLED_CONFIG, skipping user config"
        fi
    fi

    # Add wsc alias for quick worktree + Claude creation
    local ALIAS_MARKER="# >>> worktrunk-alias >>>"
    local ALIAS_BLOCK="
$ALIAS_MARKER
# Worktrunk alias: create worktree and launch Claude Code
alias wsc='wt switch --create --execute=claude'
# <<< worktrunk-alias <<<"

    # Add to zshrc if not present
    if [[ -f "$HOME/.zshrc" ]] && ! grep -q "$ALIAS_MARKER" "$HOME/.zshrc" 2>/dev/null; then
        echo "$ALIAS_BLOCK" >> "$HOME/.zshrc"
        log_feature "worktrunk" "Added wsc alias to ~/.zshrc"
    fi

    # Add to bashrc if not present
    if [[ -f "$HOME/.bashrc" ]] && ! grep -q "$ALIAS_MARKER" "$HOME/.bashrc" 2>/dev/null; then
        echo "$ALIAS_BLOCK" >> "$HOME/.bashrc"
        log_feature "worktrunk" "Added wsc alias to ~/.bashrc"
    fi

    # Install Claude Code plugin for status tracking (if claude is available)
    # Check installed_plugins.json directly since .claude is shared across containers
    local PLUGINS_FILE="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/plugins/installed_plugins.json"
    if command -v claude &>/dev/null; then
        if [[ ! -f "$PLUGINS_FILE" ]] || ! grep -q '"worktrunk@worktrunk"' "$PLUGINS_FILE" 2>/dev/null; then
            log_feature "worktrunk" "Installing Claude Code plugin for status tracking..."
            claude plugin marketplace add max-sixty/worktrunk 2>/dev/null || true
            claude plugin install worktrunk@worktrunk 2>/dev/null || true
        else
            log_feature "worktrunk" "Claude Code plugin already installed"
        fi
    fi

    log_feature "worktrunk" "Setup complete"
}

# --- pbs Environment ---
setup_pbs_env() {
    local DEVTOOLS_DIR="/srv/fast/code/voidlabs-devtools"

    # Skip if devtools not available
    if [[ ! -d "$DEVTOOLS_DIR" ]]; then
        return 0
    fi

    local ENV_MARKER="# >>> pbs-env >>>"
    local ENV_BLOCK="$ENV_MARKER
export PBS_TEMPLATE_DIR=\"$DEVTOOLS_DIR/devcontainers/template/.devcontainer\"
export PBS_PROJECTS_ROOT=\"/srv/fast/code\"
# <<< pbs-env <<<"

    # Add to shell profiles if not present
    for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
        if [[ -f "$rc" ]] && ! grep -q "$ENV_MARKER" "$rc" 2>/dev/null; then
            echo "$ENV_BLOCK" >> "$rc"
            log_feature "pbs" "Added PBS env vars to $rc"
        fi
    done

    # Export for current session
    export PBS_TEMPLATE_DIR="$DEVTOOLS_DIR/devcontainers/template/.devcontainer"
    export PBS_PROJECTS_ROOT="/srv/fast/code"
    log_feature "pbs" "Environment configured"
}

# --- Project-Specific Post-Start ---
run_project_post_start() {
    local PROJECT_POST_START="$workspace/.devcontainer/scripts/post-start-project.sh"
    if [[ -x "$PROJECT_POST_START" ]]; then
        log "Running project-specific post-start..."
        "$PROJECT_POST_START" "$workspace"
    fi
}

# --- Main ---
main() {
    log "Running shared post-start setup..."
    log "Workspace: $workspace"

    # Load project config
    load_config

    # Phase secrets (always enabled)
    if [[ -n "${PHASE_SERVICE_TOKEN:-}" ]]; then
        # First, load secrets into current environment (for use by setup functions)
        export_phase_secrets
        # Then, install shell integration for future sessions
        setup_phase_secrets
    else
        log_feature "phase" "PHASE_SERVICE_TOKEN not set, skipping"
    fi

    # Optional features (controlled by config)
    [[ "$VOIDLABS_BEADS" == "true" ]] && setup_beads
    [[ "$VOIDLABS_VLMAIL" == "true" ]] && setup_vlmail
    [[ "$VOIDLABS_MEMEX" == "true" ]] && setup_memex
    [[ "$VOIDLABS_FACTORY" == "true" ]] && setup_factory
    [[ "$VOIDLABS_CHEZMOI" == "true" ]] && setup_chezmoi
    [[ "$VOIDLABS_WORKTRUNK" == "true" ]] && setup_worktrunk

    # pbs environment (for template sync)
    setup_pbs_env

    # Project-specific setup
    run_project_post_start

    log "Done."
}

main "$@"
