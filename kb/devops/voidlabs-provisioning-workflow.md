---
title: Voidlabs Provisioning Workflow
tags:
  - provisioning
  - ansible
  - terraform
  - workflow
  - infrastructure
created: 2025-12-20
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
---

# Voidlabs Provisioning Workflow

Step-by-step guide for provisioning new infrastructure in the voidlabs homelab.

## Quick Reference

```bash
# 1. Create infrastructure (LXC)
./scripts/provision-container.sh myapp --cpu 2 --ram 2048 --clone 9001

# 2. Configure in infrastructure.yml
# 3. Sync networking
cd ansible && ansible-playbook playbooks/opnsense_dhcp.yml
ansible-playbook playbooks/opnsense_caddy.yml

# 4. Bootstrap and install
ansible-playbook playbooks/bootstrap.yml -l myapp
ansible-playbook playbooks/software/myapp.yml -l myapp
```

## Available Templates

| VMID | Type | Name | Features |
|------|------|------|----------|
| 9000 | LXC | `voidlabs-debian12-base` | SSH, chris user, base packages |
| 9001 | LXC | `voidlabs-debian12-docker` | Base + Docker, fuse enabled |
| 9100 | QEMU | `voidlabs-debian12-cloud` | Cloud-init, QEMU agent, serial console |

## Detailed Workflow

### Step 1: Provision the Guest

**For LXC containers:**
```bash
./scripts/provision-container.sh myapp \
  --cpu 2 \
  --ram 2048 \
  --clone 9001 \  # Docker template
  --full          # Full clone (not linked)
```

**For VMs:**
```bash
./scripts/provision-vm.sh myapp \
  --cpu 4 \
  --ram 4096 \
  --clone 9100
```

### Step 2: Configure infrastructure.yml

```yaml
managed_hosts:
  myapp:
    reserved_ip: 192.168.51.XX
    dhcp: true
    description: "My new service"
    
    # Storage mounts
    mounts:
      - host: /srv/fast/appdata/myapp
        container: /data
    
    # Reverse proxy
    caddy:
      - name: myapp.voidlabs.cc
        port: 8080
    
    # DNS entries
    service_domains:
      - domain: myapp.voidlabs.local
        target: host
```

### Step 3: Sync Networking

```bash
# Create DHCP reservation and DNS entries
ansible-playbook playbooks/opnsense_dhcp.yml

# Configure reverse proxy
ansible-playbook playbooks/opnsense_caddy.yml
```

### Step 4: Bootstrap

The bootstrap playbook configures:
- SSH ED25519 key deployment
- `chris` user with passwordless sudo
- Console autologin (for Proxmox web console)
- Base packages (curl, wget, vim, git, etc.)
- Timezone: America/New_York

```bash
ansible-playbook playbooks/bootstrap.yml -l myapp
```

### Step 5: Install Application

Create a software playbook or run an existing one:
```bash
ansible-playbook playbooks/software/myapp.yml -l myapp
```

## Dynamic Inventory

The `terraform_inventory.py` script auto-generates host information:

```python
# Automatic groups created:
- lxc_containers
- vms  
- proxmox_guests

# Per-host variables:
- ansible_host (IP from DHCP)
- ansible_user
- proxmox_type (lxc/qemu)
- proxmox_vmid
```

## Maintenance Playbooks

| Playbook | Purpose |
|----------|---------|
| `pve_create_template.yml` | Build LXC templates |
| `pve_create_vm_template.yml` | Build cloud-init VM templates |
| `pve_rpool_migration.yml` | Online OS disk migration |
| `pve_zpool_stripe.yml` | Add disk to pool |

## Related Entries

- [[Voidlabs Infrastructure Overview]]
- [[Voidlabs Storage Architecture]]
- [[Voidlabs Networking and DNS]]