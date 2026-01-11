---
title: Voidlabs Infrastructure Overview
tags:
  - infrastructure
  - proxmox
  - homelab
  - ansible
  - terraform
  - architecture
created: 2025-12-20
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
---

# Voidlabs Infrastructure Overview

The voidlabs homelab is a production-grade infrastructure running on Proxmox VE, managed entirely through Infrastructure as Code (IaC) principles.

## Technology Stack

| Layer | Technology |
|-------|------------|
| Hypervisor | Proxmox VE (host: `quasar`) |
| Provisioning | Terraform (Proxmox provider) |
| Configuration | Ansible |
| Firewall/Gateway | OPNsense |
| Reverse Proxy | Caddy (on OPNsense) |
| DNS | AdGuard Home + Unbound |
| Storage | Multi-tier ZFS |

## Architecture Flow

```
Infrastructure as Code → Terraform → Proxmox Guests → Ansible → Configured Services
                                  ↓
                              Terraform State
                                  ↓
                            Dynamic Inventory (terraform_inventory.py)
```

## Network Layout

| Network | CIDR | Purpose |
|---------|------|---------|
| Management VLAN | 192.168.50.0/24 | Proxmox host, Windows PC |
| Lab VLAN 51 | 192.168.51.0/24 | All containers and VMs |

**Key IPs:**
- Gateway: OPNsense (192.168.51.1)
- DNS: AdGuard (192.168.51.53)
- Reverse Proxy: Caddy (192.168.51.80)
- NFS Server: Proxmox (192.168.50.2)

## Managed Guests

### LXC Containers
- `adguard` (192.168.51.53) - DNS server
- `docker` (192.168.51.81) - Docker host with Portainer
- `media` (192.168.51.50) - Jellyfin, Arr* stack
- `n8n` (192.168.51.89) - Workflow automation
- `searxng` (192.168.51.56) - Private search engine
- `qbittorrent` (192.168.51.83) - Torrent client
- `typingmind` (192.168.51.55) - AI chat interface
- `homebridge` (192.168.51.51) - HomeKit bridge

### QEMU VMs
- `dokploy` (192.168.51.87) - Self-hosted PaaS
- `nebula` (192.168.51.59) - Home Assistant OS

## Key Design Principles

1. **Single Source of Truth** - `infrastructure.yml` defines all state
2. **Automation-First** - MAC addresses auto-extracted from Terraform
3. **Non-Destructive** - Playbooks only manage "Ansible managed" entries
4. **API-First** - OPNsense managed via REST API
5. **State Preservation** - Terraform ignores externally-changed attributes

## Related Entries

- [[Voidlabs Storage Architecture]]
- [[Voidlabs Networking and DNS]]
- [[Voidlabs Provisioning Workflow]]
- [[Voidlabs CI/CD Automation]]