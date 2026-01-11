---
name: kb-usage
description: Use this skill when working with the memex knowledge base. Triggers on queries about "organizational docs", "check the knowledge base", "add to KB", or when an agent discovers reusable knowledge worth documenting.
---

# Using the Memex Knowledge Base

Memex is a knowledge base for documenting patterns, infrastructure, troubleshooting guides, and operational knowledge. This skill teaches you how to search, contribute, and maintain the KB effectively.

## When to Search the KB

Search the knowledge base BEFORE asking questions about:
- Voidlabs infrastructure (DNS, cloud, networking, servers)
- Development conventions and patterns
- CI/CD pipelines and deployment procedures
- Known issues and their solutions
- Architectural decisions and trade-offs

```
# Use the MCP search tool
search "kubernetes deployment"
search "how to configure cloudflare"
```

If you don't find relevant entries, that's valuable information - consider contributing what you learn.

## When to Contribute

Add new entries when you discover:
- Solutions to problems that took significant debugging
- Patterns that should be reused across projects
- Infrastructure configurations worth documenting
- Operational procedures that aren't obvious
- Architectural decisions with their rationale

**Do NOT create entries for:**
- Project-specific details (use project docs instead)
- Temporary workarounds (unless they're long-term)
- Information already in upstream documentation

## Entry Format

Every KB entry requires YAML frontmatter:

```markdown
---
title: Clear, Descriptive Title
tags: [infrastructure, kubernetes, networking]
created: 2024-01-15
---

# Title

Content goes here...
```

See [[references/entry-format]] for the complete specification.

## Linking Best Practices

Bidirectional links help readers discover related knowledge:

```markdown
See also [[devops/kubernetes-basics]] for cluster setup.
Related: [[troubleshooting/dns-issues]]
```

**Guidelines for linking:**
- Link when entries genuinely relate (encouraged, not required)
- Think about what a reader might want to explore next
- Don't force links just to have them
- Update existing entries to link to your new content when relevant

## Tag Taxonomy

Use existing tags when possible. Check what tags exist before inventing new ones.

**Common tags by category:**
- infrastructure: `cloud`, `dns`, `networking`, `servers`, `storage`
- devops: `ci`, `cd`, `monitoring`, `deployment`, `docker`, `kubernetes`
- development: `python`, `rust`, `go`, `testing`, `tooling`
- troubleshooting: `debugging`, `performance`, `errors`

See [[references/categories]] for the full category taxonomy.

## Quality Guidelines

**Titles:** Be specific. "Kubernetes Pod Networking" not "K8s Stuff".

**Content:**
- Keep entries focused on one topic
- Include actionable information (commands, configs, steps)
- Add examples when they clarify usage
- Explain the "why" not just the "what"

**Maintenance:**
- Update existing entries rather than creating duplicates
- Mark outdated information clearly
- Remove entries that are no longer relevant

## Quick Reference

| Action | Tool | Example |
|--------|------|---------|
| Find knowledge | `search` | `search "cloudflare dns"` |
| Add new entry | `add` | `add infrastructure/dns-setup.md` |
| Update entry | `update` | `update path="devops/docker.md"` |

## Anti-patterns

- Creating an entry without searching first (may duplicate)
- Linking everything to everything (dilutes link value)
- Using project-specific tags in org-wide KB
- Leaving entries without tags or with wrong category
