"""Publisher tests for typed relation rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memex.publisher.generator import PublishConfig, SiteGenerator


def _write_entry(
    kb_root: Path,
    path: str,
    title: str,
    tags: list[str],
    relations: list[dict] | None = None,
) -> None:
    entry_path = kb_root / path
    entry_path.parent.mkdir(parents=True, exist_ok=True)

    tags_str = f"[{', '.join(tags)}]"

    relations_yaml = ""
    if relations:
        relations_yaml = "relations:\n"
        for relation in relations:
            relations_yaml += f"  - path: {relation['path']}\n"
            relations_yaml += f"    type: {relation['type']}\n"

    frontmatter = f"""---
title: {title}
tags: {tags_str}
created: 2024-01-15
{relations_yaml}---

# {title}

Content for {title}.
"""

    entry_path.write_text(frontmatter)


@pytest.mark.asyncio
async def test_publish_graph_includes_relation_type(tmp_kb: Path, tmp_path: Path) -> None:
    _write_entry(
        tmp_kb,
        "a.md",
        "Entry A",
        ["test"],
        relations=[{"path": "b.md", "type": "depends_on"}],
    )
    _write_entry(tmp_kb, "b.md", "Entry B", ["test"])

    output_dir = tmp_path / "site"
    config = PublishConfig(output_dir=output_dir, clean=True)
    generator = SiteGenerator(config, tmp_kb)
    await generator.generate()

    graph_data = json.loads((output_dir / "graph.json").read_text())
    edges = graph_data["edges"]

    assert any(
        edge["origin"] == "relations"
        and edge["type"] == "depends_on"
        and edge["source"] == "a"
        and edge["target"] == "b"
        for edge in edges
    )


@pytest.mark.asyncio
async def test_publish_entry_includes_typed_relations_panel(tmp_kb: Path, tmp_path: Path) -> None:
    _write_entry(
        tmp_kb,
        "a.md",
        "Entry A",
        ["test"],
        relations=[{"path": "b.md", "type": "depends_on"}],
    )
    _write_entry(tmp_kb, "b.md", "Entry B", ["test"])

    output_dir = tmp_path / "site"
    config = PublishConfig(output_dir=output_dir, clean=True)
    generator = SiteGenerator(config, tmp_kb)
    await generator.generate()

    entry_a_html = (output_dir / "a.html").read_text()
    entry_b_html = (output_dir / "b.html").read_text()

    assert "Typed Relations" in entry_a_html
    assert "Outgoing" in entry_a_html
    assert "depends_on" in entry_a_html

    assert "Typed Relations" in entry_b_html
    assert "Incoming" in entry_b_html
    assert "depends_on" in entry_b_html


@pytest.mark.asyncio
async def test_publish_skips_mismatched_scoped_relations(tmp_kb: Path, tmp_path: Path) -> None:
    _write_entry(
        tmp_kb,
        "a.md",
        "Entry A",
        ["test"],
        relations=[{"path": "@project/b.md", "type": "depends_on"}],
    )
    _write_entry(tmp_kb, "b.md", "Entry B", ["test"])

    output_dir = tmp_path / "site"
    config = PublishConfig(output_dir=output_dir, clean=True)
    generator = SiteGenerator(config, tmp_kb)
    await generator.generate()

    graph_data = json.loads((output_dir / "graph.json").read_text())
    edges = graph_data["edges"]

    assert not any(edge["origin"] == "relations" for edge in edges)
