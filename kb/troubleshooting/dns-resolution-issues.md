---
title: DNS Resolution Issues
tags:
  - troubleshooting
  - networking
  - dns
created: 2024-12-19
---

# DNS Resolution Issues

Common problems when domain names fail to resolve.

## Symptoms

- Connection timeouts to external services
- "Name or service not known" errors
- Intermittent connectivity issues

## Common Causes

1. **Misconfigured resolv.conf** - Check /etc/resolv.conf for correct nameservers
2. **Firewall blocking port 53** - DNS uses UDP/TCP port 53
3. **DHCP not updating DNS** - Static vs dynamic configuration conflicts

## Debugging Steps

```bash
# Test DNS resolution
nslookup example.com
dig example.com

# Check current DNS servers
cat /etc/resolv.conf

# Test specific DNS server
dig @8.8.8.8 example.com
```

## Related

- [[infrastructure/networking]]
