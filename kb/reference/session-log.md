---
title: Session Log Feature
tags:
  - memex
  - sessions
  - logging
  - ai-integration
created: '2026-01-11'
---

# Session Log Feature

The session-log feature provides persistent logging of work sessions to knowledge base entries. It's designed for AI agents and developers to track progress across sessions, recover context after interruptions, and create a searchable history of work.

## Quick Start

```bash
# Log a message to the project's session entry
mx session-log -m "Implemented OAuth2 flow"

# With tags and links
mx session-log -m "Fixed auth bug" --tags="auth,bugfix" --links="patterns/oauth2.md"

# Explicit entry path
mx session-log -m "..." --entry=projects/myapp/devlog.md
```

## How It Works

### Entry Resolution

Session-log automatically finds or creates the appropriate session entry:

1. **Explicit `--entry` flag**: Uses exact path specified
2. **`.kbcontext` `session_entry`**: Uses configured path from project context
3. **`.kbcontext` `primary`**: Derives path as `{primary}/sessions.md`
4. **Error**: If no context available and no explicit path

### Entry Format

**New entries** are created with this structure:

```markdown
---
title: MyProject Sessions
tags:
  - myproject
  - sessions
created: '2026-01-11T10:30:00Z'
---

# Session Log

Session notes for myproject.

## 2026-01-11 10:30 UTC

Initial session message here.
```

**Appended entries** add new timestamped sections:

```markdown
## 2026-01-11 14:45 UTC

#auth #progress

Fixed authentication module, added OAuth2 support.

Related: [[patterns/oauth2.md]], [[projects/myapp/tests.md]]
```

## CLI Reference

```bash
mx session-log [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `-m, --message TEXT` | Session summary message |
| `-f, --file PATH` | Read message from file |
| `--stdin` | Read message from stdin |
| `-e, --entry TEXT` | Explicit entry path (overrides context) |
| `--tags TEXT` | Additional tags (comma-separated, added as `#tag`) |
| `--links TEXT` | Wiki-style links to include (comma-separated) |
| `--no-timestamp` | Don't add timestamp header |
| `--json` | Output as JSON |

### JSON Output

```json
{
  "path": "projects/myapp/sessions.md",
  "action": "appended",
  "project": "myapp",
  "context_source": ".kbcontext (session_entry)"
}
```

## Configuration

### .kbcontext File

Configure session logging in your project's `.kbcontext`:

```yaml
# Explicit session entry path
session_entry: projects/myapp/sessions.md

# Or derive from primary directory
primary: projects/myapp
# → Uses projects/myapp/sessions.md

# Full example
project: myapp
primary: projects/myapp
session_entry: projects/myapp/sessions.md
paths:
  - projects/myapp
  - tooling/*
default_tags:
  - myapp
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `MEMEX_KB_ROOT` | Knowledge base root directory (required) |
| `VL_KB_CONTEXT` | Override `.kbcontext` discovery |

## AI Agent Integration

Session logging is designed for AI agent workflows:

### Session Start
```bash
mx session-log -m 'Started work on feature X' --entry=projects/myapp/sessions.md
```

### During Work
```bash
# Log progress with context links
mx session-log -m 'Implemented auth module' \
  --tags=auth,progress \
  --links='patterns/oauth2.md'
```

### Session End
```bash
# Summarize completed and remaining work
mx session-log -m 'Completed: auth module. TODO: tests' \
  --links='tooling/testing.md'
```

### Context Recovery

After session interruptions or context compaction:

```bash
# Check previous progress
mx get projects/myapp/sessions.md

# Or search session content
mx search "auth module" --tags=sessions
```

## Project Detection

When creating new session entries, the project name is detected:

1. **Git remote URL**: Extracts repository name
2. **Git root directory**: Uses directory name
3. **Current directory**: Falls back to cwd name
4. **"Unknown"**: Final fallback

This determines the entry title and default tags.

## Implementation Details

| Component | Location |
|-----------|----------|
| Core function | `src/memex/core.py:log_session()` |
| CLI command | `src/memex/cli.py` |
| Result model | `src/memex/models.py:SessionLogResult` |
| Context resolution | `src/memex/context.py:get_session_entry_path()` |
| Tests | `tests/test_session_log.py` |

## Related

- [[reference/cli.md|CLI Reference]] - Full command documentation
- [[guides/ai-integration.md|AI Integration Guide]] - Agent workflow patterns
