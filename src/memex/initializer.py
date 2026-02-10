"""KB initialization helpers used by CLI commands.

Keep this logic non-interactive and free of Click dependencies so it can be reused by:
- `mx init` (CLI wrapper for humans)
- `mx onboard` (one-shot setup + validation for agents/humans)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InitResult:
    kb_path: Path
    config_path: Path
    scope: str  # "project" or "user"
    files: list[str]


def initialize_kb(
    *,
    cwd: Path,
    path: str | None,
    user: bool,
    force: bool,
    sample: bool = False,
) -> InitResult:
    """Initialize a knowledge base.

    Args:
        cwd: Current working directory (project root for project-scope KBs).
        path: Custom KB path (project scope only).
        user: If True, create user-scope KB at ~/.memex/kb/.
        force: If True, allow initializing even if KB path already exists.
        sample: If True, create a small sample entry under inbox/ for first-run success.

    Returns:
        InitResult describing the created KB.

    Raises:
        ValueError: For invalid option combinations.
        FileExistsError: If KB already exists and force=False.
        OSError: For filesystem errors.
    """
    from .context import LOCAL_KB_CONFIG_FILENAME, USER_KB_DIR
    from .frontmatter import build_frontmatter, create_new_metadata

    if user and path:
        raise ValueError("--user and --path are mutually exclusive")

    # Determine target directory based on scope
    if user:
        kb_path = USER_KB_DIR
    else:
        kb_path = Path(path) if path else (cwd / "kb")

    # Check if already exists
    if kb_path.exists() and not force:
        scope_label = "User" if user else "Project"
        raise FileExistsError(f"{scope_label} KB already exists at {kb_path}")

    kb_path.mkdir(parents=True, exist_ok=True)

    # Create a couple of common top-level categories so the first-run "prime" flow
    # doesn't immediately require directory creation.
    default_primary = "inbox"
    (kb_path / default_primary).mkdir(parents=True, exist_ok=True)
    (kb_path / "guides").mkdir(parents=True, exist_ok=True)

    # Create README entry with scope-appropriate content
    readme_path = kb_path / "README.md"
    if user:
        readme_metadata = create_new_metadata(
            title="User Knowledge Base",
            tags=["kb", "meta", "user"],
        )
        readme_frontmatter = build_frontmatter(readme_metadata)
        readme_body = """# User Knowledge Base

This directory contains your personal knowledge base entries managed by `mx`.
This KB is available everywhere and is not shared with collaborators.

## Usage

```bash
mx add --title="Entry" --tags="tag1,tag2" --content="..." --scope=user
mx search "query" --scope=user
mx list --scope=user
```

## Structure

Entries are Markdown files with YAML frontmatter:

```markdown
---
title: Entry Title
tags: [tag1, tag2]
created: 2024-01-15T10:30:00
---

# Entry Title

Your content here.
```

## Scope

User KB entries are personal and available in all projects.
They are stored at ~/.memex/kb/ and are not committed to git.
"""
        readme_content = f"{readme_frontmatter}{readme_body}"
    else:
        readme_metadata = create_new_metadata(
            title="Project Knowledge Base",
            tags=["kb", "meta", "project"],
        )
        readme_frontmatter = build_frontmatter(readme_metadata)
        readme_body = """# Project Knowledge Base

This directory contains project-specific knowledge base entries managed by `mx`.
Commit this directory to share knowledge with collaborators.

## Usage

```bash
mx add --title="Entry" --tags="tag1,tag2" --content="..." --scope=project
mx search "query" --scope=project
mx list --scope=project
```

## Structure

Entries are Markdown files with YAML frontmatter:

```markdown
---
title: Entry Title
tags: [tag1, tag2]
created: 2024-01-15T10:30:00
---

# Entry Title

Your content here.
```

## Integration

Project KB entries take precedence over global KB entries in search results.
This keeps project-specific knowledge close to the code.
"""
        readme_content = f"{readme_frontmatter}{readme_body}"

    readme_path.write_text(readme_content, encoding="utf-8")

    # Create config file with scope-appropriate defaults
    if user:
        config_path = kb_path / LOCAL_KB_CONFIG_FILENAME
        config_content = f"""# User KB Configuration
# This file marks this directory as your personal memex knowledge base

# Default write directory for new entries (relative to KB root)
primary: {default_primary}

# Optional: warn when `mx add` omits --category and no primary is set
# (defaults to KB root (.)). Set to false to silence that warning.
# warn_on_implicit_category: true

# Optional: default tags for entries created here
# default_tags:
#   - personal

# Optional: exclude patterns (glob)
# exclude:
#   - "*.draft.md"
"""
    else:
        config_path = cwd / ".kbconfig"
        try:
            relative_kb_path = kb_path.relative_to(cwd)
        except ValueError:
            relative_kb_path = kb_path

        config_content = f"""# Project KB Configuration
# This file configures the project knowledge base

# Path to the KB directory (required for project-scope KBs)
kb_path: ./{relative_kb_path}

# Optional: default tags for entries created here
# default_tags:
#   - {cwd.name}

# Default write directory for new entries (relative to KB root)
primary: {default_primary}

# Optional: warn when `mx add` omits --category and no primary is set
# (defaults to KB root (.)). Set to false to silence that warning.
# warn_on_implicit_category: true

# Optional: boost these paths in search (glob patterns)
# boost_paths:
#   - {default_primary}/*
#   - reference/*

# Optional: exclude patterns from indexing (glob)
# exclude:
#   - "*.draft.md"
"""

    config_path.write_text(config_content, encoding="utf-8")

    scope_label = "user" if user else "project"
    if user:
        files_created = ["kb/README.md", f"kb/{LOCAL_KB_CONFIG_FILENAME}"]
    else:
        files_created = [f"{kb_path.name}/README.md", ".kbconfig"]

    if sample:
        sample_entry_path = kb_path / default_primary / "first-task.md"
        if force or not sample_entry_path.exists():
            sample_metadata = create_new_metadata(
                title="First Task",
                tags=["onboarding"],
            )
            sample_frontmatter = build_frontmatter(sample_metadata)
            sample_body = """# First Task

Write a note, link it, and try search.

Next:

- `mx list --limit=5`
- `mx get inbox/first-task.md`
- `mx add --title=\"What I learned\" --tags=\"notes\" --category=inbox --content=\"...\"`
- `mx search \"First Task\"`
"""
            sample_entry_path.write_text(
                f"{sample_frontmatter}{sample_body}",
                encoding="utf-8",
            )
            rel = f"{kb_path.name}/{default_primary}/first-task.md"
            files_created.append(rel if not user else f"kb/{default_primary}/first-task.md")

    return InitResult(kb_path=kb_path, config_path=config_path, scope=scope_label, files=files_created)
