---
title: Voidlabs CI/CD Automation
tags:
  - cicd
  - github-actions
  - automation
  - testing
  - releases
created: 2025-12-20
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
---

# Voidlabs CI/CD Automation

CI/CD patterns and automation used across voidlabs projects, with examples from the beads CLI tool.

## GitHub Actions Workflows

### Standard Pipeline Structure

```yaml
# Typical workflow triggers
on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    branches: [main]
```

### Common Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| CI | Push/PR | Version check, tests, lint, coverage |
| Release | Tag push (v*) | Multi-platform builds, package publishing |
| Nightly | Schedule | Full integration tests |
| Test PyPI | Manual | Staging package verification |

## Testing Strategy

### Coverage Thresholds
```yaml
coverage:
  minimum: 45%   # Build fails below this
  warning: 55%   # Warning issued below this
```

### Test Modes
- **Short tests** (`-short` flag) - Fast CI feedback
- **Full tests** - Nightly integration runs (30-min timeout)
- **Skip patterns** - Custom test runner with exclusion support

### Linting
- **Go**: golangci-lint with security-focused rules
- **Python**: ruff, mypy
- **General**: pre-commit hooks

## Release Automation

### Version Synchronization

The `bump-version.sh` script updates version across all files:
```bash
./scripts/bump-version.sh 1.2.3
# Updates: version.go, pyproject.toml, package.json, etc.
```

### Release Flow
```bash
# Local release workflow
./scripts/release.sh patch  # or minor, major

# Automated on tag push:
# 1. GoReleaser builds binaries (5 platforms)
# 2. PyPI package published
# 3. npm package published  
# 4. Homebrew formula updated
```

### Build Targets
| OS | Architecture |
|----|-------------|
| Linux | amd64, arm64 |
| macOS | amd64, arm64 |
| Windows | amd64 |

## Secrets Management

### GitHub Secrets
```
PYPI_API_TOKEN        # PyPI publishing
TEST_PYPI_API_TOKEN   # Staging publishing
HOMEBREW_TAP_TOKEN    # Homebrew formula updates
```

### Infrastructure Secrets
1Password integration with `op://` references:
```yaml
# In .env or ansible vars
api_key: op://vault/item/field
```

## Dependabot Configuration

```yaml
# .github/dependabot.yml
updates:
  - package-ecosystem: "gomod"
  - package-ecosystem: "github-actions"  
  - package-ecosystem: "pip"
```

## Ansible Deployment

### Playbook-Based Deployments
```bash
# Bootstrap new host
ansible-playbook playbooks/bootstrap.yml -l target

# Deploy application
ansible-playbook playbooks/software/myapp.yml -l target

# Sync networking
ansible-playbook playbooks/opnsense_dhcp.yml
```

### Dynamic Inventory
Terraform state → `terraform_inventory.py` → Ansible groups

## Best Practices

1. **Version Check First** - CI validates version consistency early
2. **Short Tests for PRs** - Full tests run nightly
3. **Tag-Based Releases** - `v*` tags trigger full release
4. **Multi-Package Sync** - Single version across all package managers
5. **Secrets via 1Password** - No plaintext secrets in repos

## Related Entries

- [[Voidlabs Infrastructure Overview]]
- [[Voidlabs Provisioning Workflow]]
- [[Voidlabs Knowledge Base Plugin]]