---
title: Kubernetes Basics
tags:
  - infrastructure
  - kubernetes
  - containers
  - orchestration
created: 2024-12-19
---

# Kubernetes Basics

Container orchestration fundamentals for voidlabs infrastructure.

## Core Concepts

### Pods
Smallest deployable unit. Contains one or more containers sharing network/storage.

### Deployments
Manage pod replicas, rolling updates, and rollbacks.

### Services
Expose pods via stable network endpoints. Types: ClusterIP, NodePort, LoadBalancer.

### ConfigMaps and Secrets
Externalize configuration from container images.

## Common Commands

```bash
# Get cluster info
kubectl cluster-info

# List pods
kubectl get pods -n namespace

# Describe pod
kubectl describe pod my-pod

# View logs
kubectl logs -f my-pod

# Apply manifest
kubectl apply -f deployment.yaml
```

## Helm Charts

Package manager for Kubernetes. We use Helm for standardized deployments.

## Related

- [[devops/docker-patterns]]
- [[infrastructure/devcontainers]]
