---
title: Voidlabs Networking and DNS
tags:
  - networking
  - dns
  - dhcp
  - caddy
  - opnsense
  - infrastructure
created: 2025-12-20
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
---

# Voidlabs Networking and DNS

The voidlabs network is managed through OPNsense with automated DHCP, DNS, and reverse proxy configuration.

## DNS Architecture

```
External: *.voidlabs.cc → Caddy (192.168.51.80) → Backend services
Internal: *.voidlabs.local → Unbound on OPNsense
```

### Components
- **AdGuard Home** (192.168.51.53) - DNS blocking and rewrites
- **Unbound** (on OPNsense) - Internal DNS resolution
- **Caddy** (on OPNsense) - Reverse proxy with automatic TLS

## Service Domain Configuration

Domains are defined in `infrastructure.yml`:

```yaml
service_domains:
  # Route through Caddy reverse proxy
  - domain: app.voidlabs.cc
    target: caddy
    
  # Direct to host IP
  - domain: admin.voidlabs.cc
    target: host
    
  # Reference another managed host
  - domain: api.voidlabs.cc
    target: other_host
    
  # Literal IP override
  - domain: custom.voidlabs.cc
    target_ip: 192.168.51.100
```

## DHCP Configuration

Static DHCP reservations are managed via Ansible:

```yaml
managed_hosts:
  myapp:
    reserved_ip: 192.168.51.XX
    dhcp: true  # or "net0" | "net1" for multi-interface VMs
    mac: auto   # Extracted from Terraform state
```

The `opnsense_dhcp.yml` playbook:
1. Reads `infrastructure.yml`
2. Extracts MACs from Terraform state
3. Creates static DHCP reservations via OPNsense API
4. Creates Unbound DNS host overrides
5. Removes orphaned "Ansible managed" entries

## Caddy Reverse Proxy

Managed via OPNsense API integration:

```yaml
managed_hosts:
  myapp:
    caddy:
      - name: myapp.voidlabs.cc
        port: 8080
        skip_tls_verify: false  # For self-signed backend certs
```

### TLS Configuration
- Wildcard subjects: `*.voidlabs.cc`, `*.pt.voidlabs.cc`
- DNS provider: Cloudflare (for ACME DNS challenges)
- Automatic certificate renewal

## Custom Ansible Filters

```python
# opnsense_filters.py
terraform_mac_lookup(state, hostname, interface)  # Extract MACs
split_fqdn(domain)                                 # Parse for Unbound
resolve_dns_target(hosts, target, ...)            # DNS resolution
```

## Related Playbooks

- `opnsense_dhcp.yml` - Sync DHCP and DNS
- `opnsense_caddy.yml` - Sync reverse proxy config
- `adguard_dns.yml` - Configure AdGuard rewrites

## Related Entries

- [[Voidlabs Infrastructure Overview]]
- [[Voidlabs Provisioning Workflow]]