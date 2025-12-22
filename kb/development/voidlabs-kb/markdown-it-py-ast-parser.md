---
title: Markdown-it-py AST Parser
tags:
  - voidlabs-kb
  - markdown
  - parsing
  - ast
  - markdown-it-py
created: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: voidlabs-kb
---

# Markdown-it-py AST Parser

The voidlabs-kb uses [markdown-it-py](https://markdown-it-py.readthedocs.io/) for AST-based markdown parsing, replacing the original regex-based approach.

## Why markdown-it-py?

The original implementation used hand-rolled regex patterns for:
- Link extraction (`\[\[([^\]]+)\]\]`)
- HTML conversion (headers, code blocks, lists, etc.)

This caused issues:
1. **Missing table support** - GFM tables weren't rendered
2. **False positives** - Links inside code blocks were incorrectly extracted
3. **Maintenance burden** - Each markdown feature needed its own regex

markdown-it-py provides:
- Proper AST parsing with token stream
- GFM table support out of the box
- Plugin architecture for custom syntax
- Battle-tested CommonMark compliance

## Wikilink Plugin

Custom inline rule for `[[wikilink]]` syntax:

```python
# Pattern: [[target]] or [[target|alias]]
WIKILINK_PATTERN = re.compile(r"\[\[([^|\]\n]+)(?:\|([^\]\n]+))?\]\]")

def _wikilink_rule(state: StateInline, silent: bool) -> bool:
    # Quick rejection test
    if state.src[state.pos : state.pos + 2] != "[[":
        return False
    
    match = WIKILINK_PATTERN.match(state.src, state.pos)
    if not match:
        return False
    
    if not silent:
        token = state.push("wikilink", "", 0)
        token.meta = {"target": match.group(1), "alias": match.group(2)}
    
    state.pos = match.end()
    return True
```

Key design choices:
- **Quick rejection** - Check for `[[` before running regex
- **Token metadata** - Store target and alias separately
- **Silent mode** - Support validation without token emission

## Single-Pass Architecture

The `render_markdown()` function returns both HTML and links:

```python
@dataclass
class MarkdownResult:
    html: str
    links: list[str]

def render_markdown(content: str) -> MarkdownResult:
    tokens = md.parse(content)
    links = _extract_wikilinks(tokens)
    html = md.render(content)
    return MarkdownResult(html=html, links=links)
```

Benefits:
- Parse once, extract multiple outputs
- Links in code blocks aren't extracted (AST knows context)
- Consistent behavior between HTML and link extraction

## Usage

```python
from voidlabs_kb.parser import render_markdown, extract_links

# Full rendering with links
result = render_markdown("See [[foo/bar|docs]] for details.")
print(result.html)   # <p>See <a class="wikilink" data-path="foo/bar">docs</a>...</p>
print(result.links)  # ['foo/bar']

# Link extraction only (delegates to render_markdown internally)
links = extract_links("Check [[overview]] and [[details]].")
# ['overview', 'details']
```

## Files

| File | Purpose |
|------|---------|
| `parser/md_renderer.py` | markdown-it-py wrapper, wikilink plugin |
| `parser/links.py` | `extract_links()` delegates to AST parser |
| `webapp/api.py` | Uses `render_markdown()` for entry HTML |

## Related

- [[Voidlabs KB MCP Server]] - The MCP server using this parser
