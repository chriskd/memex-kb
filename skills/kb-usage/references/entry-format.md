# KB Entry Format Specification

Every knowledge base entry is a Markdown file with YAML frontmatter.

## File Location

Entries live in category directories under `kb/`:

```
kb/
  infrastructure/
  devops/
  development/
  troubleshooting/
  architecture/
  patterns/
```

Use lowercase, hyphenated filenames: `kubernetes-pod-networking.md`

## Frontmatter Schema

### Required Fields

```yaml
---
title: Human-readable title for the entry
tags: [tag1, tag2, tag3]
created: YYYY-MM-DD
---
```

| Field | Type | Description |
|-------|------|-------------|
| `title` | string | Clear, descriptive title (used in search results) |
| `tags` | list | Lowercase tags for categorization and search |
| `created` | date | ISO 8601 date (YYYY-MM-DD) |

### Optional Fields

```yaml
---
title: Kubernetes Pod Networking
tags: [kubernetes, networking, infrastructure]
created: 2024-01-15
updated: 2024-03-20
author: agent-name
status: active
related:
  - devops/kubernetes-basics
  - troubleshooting/network-debugging
---
```

| Field | Type | Description |
|-------|------|-------------|
| `updated` | date | Last significant update date |
| `author` | string | Who created/owns this entry |
| `status` | enum | `active`, `draft`, `deprecated`, `archived` |
| `related` | list | Explicit list of related entry paths |

## Date Formats

Always use ISO 8601: `YYYY-MM-DD`

```yaml
created: 2024-01-15    # Correct
created: 01/15/2024    # Wrong
created: Jan 15, 2024  # Wrong
```

## Tag Conventions

- Lowercase only: `kubernetes` not `Kubernetes`
- Singular form: `container` not `containers`
- Hyphenate multi-word: `code-review` not `code_review`
- Be specific: `python-testing` not just `testing` when relevant

**Reserved tags (auto-applied by system):**
- Category names are implicit based on file location
- Don't manually tag with category names

## Link Syntax

Use double-bracket wiki-style links:

```markdown
See [[devops/kubernetes-basics]] for cluster setup.
Check [[troubleshooting/dns-issues]] if resolution fails.
```

**Link rules:**
- Path is relative to `kb/` directory
- Omit the `.md` extension
- Links are case-sensitive
- Broken links are detected during indexing

**Linking within text:**
```markdown
When configuring [[infrastructure/cloudflare-dns|Cloudflare DNS]],
remember to set the TTL appropriately.
```

The `|alias` syntax displays "Cloudflare DNS" while linking to the full path.

## Section Structure

Recommended structure for entries:

```markdown
---
title: Entry Title
tags: [relevant, tags]
created: 2024-01-15
---

# Entry Title

Brief overview (1-2 sentences) of what this entry covers.

## Context

When and why you'd need this information.

## Steps / Details

Main content. Use subheadings for organization.

### Subsection

More detail as needed.

## Examples

Concrete examples with code blocks or commands.

## Troubleshooting

Common issues and solutions (if applicable).

## See Also

- [[related/entry-one]]
- [[related/entry-two]]
```

## Code Blocks

Use fenced code blocks with language identifiers:

````markdown
```bash
kubectl get pods -n production
```

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: example
```

```python
def example():
    return "use language hints"
```
````

## Complete Example

```markdown
---
title: Configuring Cloudflare DNS for New Domains
tags: [dns, cloudflare, infrastructure, domains]
created: 2024-02-10
updated: 2024-03-15
status: active
---

# Configuring Cloudflare DNS for New Domains

How to add and configure a new domain in our Cloudflare account.

## Prerequisites

- Access to Cloudflare dashboard (see [[infrastructure/cloudflare-access]])
- Domain registrar credentials
- Nameserver update permissions

## Steps

1. Log into Cloudflare and select "Add Site"
2. Enter the domain name
3. Select the appropriate plan (usually Free for internal)
4. Copy the assigned nameservers

```
ns1.cloudflare.com
ns2.cloudflare.com
```

5. Update nameservers at your registrar
6. Wait for propagation (usually 1-24 hours)

## Common DNS Records

| Type | Name | Value | TTL |
|------|------|-------|-----|
| A | @ | 192.0.2.1 | Auto |
| CNAME | www | @ | Auto |
| MX | @ | mail.example.com | Auto |

## Troubleshooting

**Nameserver propagation taking too long:**
Check with `dig NS example.com` - if old nameservers show, wait longer.

**SSL certificate not issued:**
Ensure DNS records are proxied (orange cloud) for automatic SSL.

## See Also

- [[infrastructure/ssl-certificates]]
- [[troubleshooting/dns-issues]]
```
