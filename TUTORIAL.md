# First 5 Minutes with Memex

This tutorial walks you through a realistic workflow to get productive with Memex quickly.

## 1. Install Memex

```bash
# Recommended: minimal install with keyword search
uv tool install memex

# Or with pip
pip install memex

# Verify installation
mx --version
```

> **Want semantic search?** Install with `uv tool install "memex[semantic]"` instead. This adds ~500MB of ML dependencies but enables meaning-based search.

## 2. Create Your Knowledge Base

```bash
# Create a directory for your KB
mkdir -p ~/kb

# Set the environment variable (add to your shell profile for persistence)
export MEMEX_KB_ROOT=~/kb

# Initialize the KB
mx init
```

You should see output confirming your KB is ready.

## 3. Add Your First Entry

Let's add a note about a useful command you just learned:

```bash
mx add \
  --title="Git Stash Workflow" \
  --tags="git,workflow,cli" \
  --category=tooling \
  --content="# Git Stash Workflow

Quick save work in progress:
\`\`\`bash
git stash push -m 'WIP: feature X'
git stash list
git stash pop
\`\`\`

Use \`git stash apply\` to keep the stash after applying."
```

This creates `tooling/git-stash-workflow.md` in your KB.

## 4. Search for It

```bash
# Find entries matching 'stash'
mx search "stash"
```

You'll see results ranked by relevance:

```
tooling/git-stash-workflow.md (0.85)
  Git Stash Workflow
  Tags: git, workflow, cli
```

Try different queries:
```bash
mx search "save work in progress"   # Semantic match (if installed)
mx search "git" --tags=workflow     # Filter by tag
```

## 5. Read It Back

```bash
# View the full entry
mx get tooling/git-stash-workflow.md

# Just the metadata
mx get tooling/git-stash-workflow.md --metadata
```

## 6. Check KB Health

```bash
mx health
```

This audits your KB for:
- Missing frontmatter
- Broken links
- Orphaned entries
- Index sync issues

## What's Next?

### Browse your KB
```bash
mx tree              # Directory structure
mx list              # All entries
mx tags              # All tags with counts
mx whats-new         # Recent changes
```

### Add more entries
```bash
# From a file
mx add --title="Meeting Notes" --tags="meetings" --file=notes.md

# Quick add with auto-generated metadata
echo "Learned about Docker volumes today..." | mx quick-add --stdin
```

### Update entries
```bash
# Change tags
mx update tooling/git-stash-workflow.md --tags="git,tips"

# Append content
mx update tooling/git-stash-workflow.md \
  --content="Also useful: git stash branch <name>" \
  --append --timestamp
```

### Find connections
```bash
mx hubs                                    # Most connected entries
mx suggest-links tooling/git-stash-workflow.md  # Related entries
```

### Use with AI agents
```bash
# Claude Code: add to .claude/settings.local.json
{ "permissions": { "allow": ["Bash(mx:*)"] } }

# Or use hooks for automatic context
{ "hooks": { "SessionStart": [{ "command": "mx prime" }] } }
```

## Tips

1. **Use categories** - Organize entries into directories: `tooling/`, `projects/`, `reference/`
2. **Tag consistently** - Use `mx tags` to see existing tags before creating new ones
3. **Link entries** - Use `[[path/to/entry]]` in content to create bidirectional links
4. **Search often** - Check `mx search` before adding duplicates

---

See the [README](README.md) for the full CLI reference and configuration options.
