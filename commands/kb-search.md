---
name: kb-search
description: Search the knowledge base
allowed-tools:
  - Bash
argument-hint: "<query>"
---

Search the knowledge base using the provided query.

## Workflow

1. Run `mx search "<query>"` with the user's query
2. Display results in a readable format:
   - Title
   - Path
   - Relevance snippet (if available)

## Search Modes

- **hybrid** (default): Combines keyword and semantic search
- **keyword**: Traditional text matching (`--mode=keyword`)
- **semantic**: Vector similarity search (`--mode=semantic`)

## Example

```bash
mx search "kubernetes deployment"
mx search "cloudflare" --mode=semantic
mx search "docker" --tags=infrastructure
```

## Example Output

```
Found 3 results for "kubernetes deployment":

1. **Kubernetes Deployment Strategies**
   Path: infrastructure/kubernetes/deployments.md
   "...rolling updates and blue-green deployment patterns..."

2. **ArgoCD Setup Guide**
   Path: devops/argocd/setup.md
   "...GitOps deployment workflow for Kubernetes..."
```

If no results found, suggest:
- Try different keywords
- Use a broader search term
- Check available categories with `mx tree`
