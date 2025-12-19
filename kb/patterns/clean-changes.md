---
title: Clean Changes Over Backwards Compatibility
tags:
  - philosophy
  - patterns
  - refactoring
created: 2024-12-19
contributors:
  - agent
---

# Clean Changes Over Backwards Compatibility

**When making design or architecture changes, lean into them fully.** Do not clutter code with backwards compatibility layers, fallbacks, or migration shims unless explicitly needed.

## Why This Matters

Most projects here are:
- Small and actively developed
- Not in production (or easily redeployable)
- Not consumed by external users
- Better served by clean code than compatibility

## Anti-Patterns to Avoid

Unless explicitly requested, avoid:

```python
# BAD - unnecessary compatibility layer
def get_user(user_id: str | int):  # Why support both?
    if isinstance(user_id, int):
        user_id = str(user_id)  # Legacy fallback
    ...

# BAD - keeping old code paths
def process(data, use_new_engine=True):  # Just use the new engine
    if use_new_engine:
        return new_process(data)
    return old_process(data)  # Dead code waiting to happen

# BAD - deprecation warnings in small projects
import warnings
warnings.warn("Use new_function instead", DeprecationWarning)  # Just delete old_function

# BAD - config fallbacks
config.get("new_key") or config.get("old_key") or DEFAULT  # Just use new_key
```

## Preferred Patterns

Make the clean change:

```python
# GOOD - pick one type and use it
def get_user(user_id: str):
    ...

# GOOD - just use the new implementation
def process(data):
    return new_engine.process(data)

# GOOD - delete the old function, update callers
# (old_function is gone, callers updated)

# GOOD - use the new config key everywhere
config.get("new_key", DEFAULT)
```

## When Backwards Compatibility IS Appropriate

- Public APIs with external consumers
- Deployed production systems with gradual rollout
- Shared libraries used by multiple projects
- User explicitly requests compatibility period

## The Test

When in doubt, ask: **"Is anyone actually using the old way?"**

If not, delete it.
