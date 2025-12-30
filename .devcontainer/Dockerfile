# memex Development Container
# Uses prebuilt base image from local registry for fast startup
# Base includes: Go, Node, Phase CLI, git, zsh, tmux, Homebrew (with fzf, neovim, uv, worktrunk)
#
# To rebuild the base image manually:
#   cd /srv/fast/code/voidlabs-devtools
#   docker build -t localhost:5000/devcontainer-base:latest -f devcontainers/base/Dockerfile .
#   docker push localhost:5000/devcontainer-base:latest
#
# Or trigger via GitHub Actions (push to main with Dockerfile changes)

FROM localhost:5000/devcontainer-base:latest

# Add project-specific packages here if needed:
# RUN apt-get update && apt-get install -y \
#     your-package \
#     && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspaces/memex
