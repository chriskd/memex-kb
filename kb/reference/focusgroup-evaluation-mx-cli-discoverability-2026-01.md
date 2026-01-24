---
title: 'Focusgroup Evaluation: mx CLI Discoverability (2026-01)'
description: 'Focusgroup Evaluation: mx CLI Discoverability Date: 2026-01-12 Agents: Opus
  4.5 (Claude), Codex (OpenAI) Mode:...'
tags:
  - focusgroup
  - evaluation
  - mx
  - cli
created: 2026-01-12T06:52:04.320941+00:00
updated: 2026-01-15T03:22:55.904510+00:00
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
semantic_links:
  - path: a-mem-parity/a-mem-test-cases-for-agent-evaluation.md
    score: 0.609
    reason: bidirectional
---

# Focusgroup Evaluation: mx CLI Discoverability

**Date:** 2026-01-12
**Agents:** Opus 4.5 (Claude), Codex (OpenAI)
**Mode:** Exploration (agents ran commands interactively)

## Summary

Two AI agents evaluated the `mx` CLI for discoverability and agent-friendliness. They ran `mx prime`, then attempted common operations without reading `--help`.

## Key Findings

### What Works Well

| Feature | Assessment |
|---------|------------|
| `mx prime` | Excellent - provides quick workflow context |
| `mx batch` | Outstanding - JSON output, multiple ops, exactly what agents need |
| `mx schema` | Very useful for programmatic discovery |
| `mx search` | Clean tabular output with scores and confidence |
| `mx health` | Good actionable diagnostics |
| Tag suggestions | Helpful after `mx add` |

### Bugs Found

| Priority | Issue | Impact |
|----------|-------|--------|
| P0 | `mx info` shows 0 entries when entries exist | Misleading stats |
| P0 | `mx publish` crashes with AttributeError | Feature broken |
| P1 | `--json-errors` doesn't produce JSON | Blocks programmatic error handling |
| P1 | `mx history` always empty | Feature broken |
| P1 | `mx tags/hubs` throw tracebacks without KB | Inconsistent errors |
| P2 | `mx replace --tags` requires `--content` | Unnecessary friction |
| P2 | `\n` in `--content` stored literally | Escape handling missing |
| P2 | `mx whats-new` datetime crash (intermittent) | Fragile datetime handling |

### Documentation Issues

- `mx add` requires `--category` but docs show it as optional
- `--tag` vs `--tags` inconsistency in examples (fixed 2026-01-12)
- `.kbcontext` vs `.kbconfig` naming confusion

## Agent-Friendliness Recommendations

1. **Fix `--json-errors`** - Agents need structured errors
2. **Make error handling consistent** - All commands should give friendly errors or structured JSON
3. **`mx batch` is the gold standard** - Other commands should follow its patterns
4. **Document `--stdin` for formatted content** - Since `--content` doesn't interpret escapes

## Best Practices for Agents

```bash
# Use batch for multiple operations (structured JSON output)
mx batch << 'EOF'
add --title='...' --tags='...' --category=tooling --content='...'
search 'query'
EOF

# Use --stdin for content with formatting
echo -e "Line 1\n\nLine 2" | mx add --title="..." --tags="..." --category=... --stdin

# Check for KB first
mx info >/dev/null 2>&1 || mx init
```

## Session Details

- Session ID: `218cdae5`
- Duration: ~11 minutes
- Log: `~/.local/share/focusgroup/logs/20260112-218cdae5.json`