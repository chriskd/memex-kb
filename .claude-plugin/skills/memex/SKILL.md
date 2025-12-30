---
name: memex
description: >
  Search and contribute to the organizational knowledge base. Use for finding patterns,
  troubleshooting guides, infrastructure docs, and preserving discoveries. Triggers on
  "search KB", "check knowledge base", "add to KB", "document this for future".
allowed-tools: "Read,Bash(mx:*)"
version: "0.1.0"
author: "memex"
license: "MIT"
---

# Memex - Organizational Knowledge Base

Personal knowledge base with semantic search for storing and retrieving organizational patterns, guides, and discoveries.

## Overview

**memex** provides persistent storage for organizational knowledge that survives across sessions. Unlike conversation history, KB entries are searchable, linkable, and shareable.

**Key Distinction**:
- **memex**: Organizational patterns, infrastructure docs, troubleshooting guides
- **Conversation**: Ephemeral context for current task

**Core Capabilities**:
- **Hybrid Search**: Keyword + semantic search finds relevant entries
- **Bidirectional Links**: `[[path/to/entry.md]]` creates connected knowledge
- **Project Context**: `.kbcontext` files scope searches to project-relevant entries
- **Tag System**: Categorize entries for easy filtering

## Prerequisites

**Required**:
- **mx CLI**: Must be installed and in PATH
- **KB Root**: `MEMEX_KB_ROOT` environment variable set

**Verify Installation**:
```bash
mx --version  # Should return version info
mx tree       # Should show KB structure
```

## Instructions

### Session Start Protocol

**Every session, check KB first:**

#### Step 1: Search for Relevant Patterns

```bash
mx search "your-topic"
```

Or check recent project-specific entries:

```bash
mx whats-new --project=<project-name>
```

#### Step 2: Read Relevant Entries

```bash
mx get path/to/entry.md
```

### Session End Protocol

**Before completing, consider contributing:**

If you discovered:
- Reusable patterns or solutions
- Troubleshooting steps worth preserving
- Infrastructure or deployment knowledge

Add it to the KB:

```bash
mx add --title="Pattern: My Discovery" --tags="patterns,topic" --content="..."
```

### CLI Quick Reference

| Command | Purpose | Example |
|---------|---------|---------|
| `mx search` | Find entries | `mx search "deployment"` |
| `mx get` | Read an entry | `mx get tooling/beads.md` |
| `mx add` | Create entry | `mx add --title="..." --tags="..."` |
| `mx tree` | Browse structure | `mx tree` |
| `mx whats-new` | Recent changes | `mx whats-new --project=myapp` |
| `mx health` | Audit KB | `mx health` |
| `mx suggest-links` | Find related | `mx suggest-links path.md` |

### Entry Format

Entries are Markdown with YAML frontmatter:

```markdown
---
title: Entry Title
tags: [tag1, tag2]
created: 2024-01-15
---

# Entry Title

Content with [[bidirectional links]] to other entries.
```

Use `[[path/to/entry.md|Display Text]]` for links.

## When to Search KB

- **Before implementing**: Check if pattern already documented
- **When troubleshooting**: Search for known issues and fixes
- **Understanding infrastructure**: Find deployment guides, architecture docs

## When to Contribute

- **Discovered reusable pattern**: Document for future agents
- **Solved tricky problem**: Save troubleshooting steps
- **Learned infrastructure details**: Preserve deployment knowledge

**Decision Rule**: If you'd want to find this again in 2 weeks, add it to KB.

## Output

This skill produces:

**Search Results**:
```
PATH                                      TITLE                               SCORE
----------------------------------------  ----------------------------------  -----
tooling/beads-issue-tracker.md            Beads Issue Tracker                 0.85
infrastructure/deployment-guide.md        Deployment Guide                    0.72
```

**Entry Content**:
```
# Entry Title
Tags: tag1, tag2
------------------------------------------------------------
[Full markdown content]
```

## Error Handling

### Common Failures

#### 1. `mx: command not found`
**Solution**: Ensure memex is installed: `pip install -e /path/to/memex`

#### 2. `MEMEX_KB_ROOT not set`
**Solution**: Set environment variable pointing to KB directory

#### 3. `No results found`
**Solution**: Try different keywords, use semantic mode: `mx search "query" --mode=semantic`

## Resources

- **Full CLI Reference**: `mx --help`
- **Project Context**: Create `.kbcontext` file in project root
- **Health Check**: `mx health` audits for orphans, broken links, stale content
