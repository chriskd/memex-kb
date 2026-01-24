"""Integration tests for memex end-to-end workflows.

These tests verify system behavior across components:
- Add -> Search -> Find flow
- Update -> Effects propagate (search, backlinks, tags)
- Delete -> Cleanup (search, backlinks)
- Complex workflows with links and batch operations

Design:
- Uses real filesystem with tmp_kb fixture
- Tests realistic user workflows
- Marked slow tests with @pytest.mark.slow
- Target: run in <10 seconds
"""

import asyncio

import pytest

from memex import core
from memex.backlinks_cache import ensure_backlink_cache, rebuild_backlink_cache
from memex.tags_cache import ensure_tags_cache, rebuild_tags_cache
from memex.parser import parse_entry, extract_links
from conftest import create_entry


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def run_async(coro):
    """Run async coroutine synchronously."""
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────────────
# Add -> Search -> Find Flow
# ─────────────────────────────────────────────────────────────────────────────


class TestAddSearchFlow:
    """Test the add -> search -> get workflow."""

    def test_added_entry_is_searchable_by_title(self, tmp_kb):
        """Entry added via core.add_entry is findable by title."""
        result = run_async(core.add_entry(
            title="Unique Integration Test Entry",
            content="# Unique Integration Test Entry\n\nThis tests the add->search flow.",
            tags=["integration", "testing"],
            category="general",
        ))

        assert "path" in result
        assert result["path"].endswith(".md")

        # Search should find it by title
        search_results = run_async(core.search(query="Unique Integration Test", limit=10))
        paths = [r.path for r in search_results.results]
        assert result["path"] in paths

    def test_added_entry_is_searchable_by_content(self, tmp_kb):
        """Entry is searchable by content keywords."""
        result = run_async(core.add_entry(
            title="Content Search Test",
            content="# Content Search Test\n\nThis contains xylophone and zzzunique keywords.",
            tags=["search"],
            category="general",
        ))

        # Search by unique content keyword
        search_results = run_async(core.search(query="xylophone zzzunique", limit=10))
        paths = [r.path for r in search_results.results]
        assert result["path"] in paths

    def test_get_retrieves_added_entry(self, tmp_kb):
        """get_entry returns the added entry with correct content."""
        original_content = "# Get Test\n\nThis content should be retrievable."
        result = run_async(core.add_entry(
            title="Get Test Entry",
            content=original_content,
            tags=["retrieve"],
            category="general",
        ))

        entry = run_async(core.get_entry(result["path"]))

        assert entry.metadata.title == "Get Test Entry"
        assert "This content should be retrievable" in entry.content
        assert "retrieve" in entry.metadata.tags

    def test_added_entry_has_correct_metadata(self, tmp_kb):
        """Added entry has correct frontmatter metadata."""
        result = run_async(core.add_entry(
            title="Metadata Test",
            content="# Metadata Test\n\nTesting metadata.",
            tags=["meta", "test"],
            category="general",
        ))

        entry = run_async(core.get_entry(result["path"]))

        assert entry.metadata.title == "Metadata Test"
        assert set(entry.metadata.tags) == {"meta", "test"}
        assert entry.metadata.created is not None

    def test_search_with_tag_filter(self, tmp_kb):
        """Search can filter by tags."""
        run_async(core.add_entry(
            title="Tag Filter Entry One",
            content="# Entry One\n\nFirst entry.",
            tags=["filter-tag", "alpha"],
            category="general",
        ))
        run_async(core.add_entry(
            title="Tag Filter Entry Two",
            content="# Entry Two\n\nSecond entry.",
            tags=["filter-tag", "beta"],
            category="general",
        ))
        run_async(core.add_entry(
            title="Tag Filter Entry Three",
            content="# Entry Three\n\nThird entry no filter tag.",
            tags=["other"],
            category="general",
        ))

        # Search with tag filter
        results = run_async(core.search(query="Entry", tags=["filter-tag"], limit=10))
        paths = [r.path for r in results.results]

        assert len(results.results) == 2
        assert all("filter-tag" in r.tags for r in results.results)


# ─────────────────────────────────────────────────────────────────────────────
# Update -> Effects Propagate
# ─────────────────────────────────────────────────────────────────────────────


class TestUpdatePropagation:
    """Test that updates propagate to search, backlinks, and tags."""

    def test_update_content_reflects_in_search(self, tmp_kb):
        """Updated content is searchable."""
        # Create entry
        result = run_async(core.add_entry(
            title="Update Search Test",
            content="# Update Search Test\n\nOriginal content here.",
            tags=["update"],
            category="general",
        ))

        # Update with new keyword
        run_async(core.update_entry(
            path=result["path"],
            content="# Update Search Test\n\nUpdated with elephant keyword.",
        ))

        # Search should find new content
        results = run_async(core.search(query="elephant", limit=10))
        paths = [r.path for r in results.results]
        assert result["path"] in paths

    def test_update_tags_reflects_in_cache(self, tmp_kb):
        """Updated tags appear in tags cache."""
        result = run_async(core.add_entry(
            title="Tag Update Test",
            content="# Tag Update Test\n\nContent.",
            tags=["original-tag"],
            category="general",
        ))

        # Update with new tags (must also provide content or section_updates)
        run_async(core.update_entry(
            path=result["path"],
            content="# Tag Update Test\n\nContent with updated tags.",
            tags=["new-tag-one", "new-tag-two"],
        ))

        # Tags cache should include new tags
        tags_counts = ensure_tags_cache(tmp_kb)
        assert "new-tag-one" in tags_counts
        assert "new-tag-two" in tags_counts
        # Original tag should be removed (only this entry had it)
        assert "original-tag" not in tags_counts

    def test_update_with_link_creates_backlink(self, tmp_kb):
        """Adding a link via update creates a backlink."""
        # Create target entry
        target = run_async(core.add_entry(
            title="Backlink Target",
            content="# Backlink Target\n\nThis will be linked to.",
            tags=["target"],
            category="general",
        ))

        # Create source entry
        source = run_async(core.add_entry(
            title="Backlink Source",
            content="# Backlink Source\n\nNo links yet.",
            tags=["source"],
            category="general",
        ))

        # Update source to link to target
        target_link = target["path"].replace(".md", "")
        run_async(core.update_entry(
            path=source["path"],
            content=f"# Backlink Source\n\nNow links to [[{target_link}]].",
        ))

        # Check backlinks (backlinks are stored without .md extension)
        backlinks = ensure_backlink_cache(tmp_kb)
        target_key = target["path"].replace(".md", "")
        assert target_key in backlinks
        source_key = source["path"].replace(".md", "")
        assert source_key in backlinks[target_key] or source["path"] in backlinks[target_key]

    def test_update_removing_link_clears_backlink(self, tmp_kb):
        """Removing a link via update clears the backlink."""
        # Create target entry
        target = run_async(core.add_entry(
            title="Link Target",
            content="# Link Target\n\nTarget entry.",
            tags=["target"],
            category="general",
        ))

        target_link = target["path"].replace(".md", "")

        # Create source entry with link
        source = run_async(core.add_entry(
            title="Link Source",
            content=f"# Link Source\n\nLinks to [[{target_link}]].",
            tags=["source"],
            category="general",
        ))

        # Verify backlink exists (backlinks are stored without .md extension)
        backlinks = ensure_backlink_cache(tmp_kb)
        target_key = target["path"].replace(".md", "")
        source_key = source["path"].replace(".md", "")
        assert source_key in backlinks.get(target_key, []) or source["path"] in backlinks.get(target_key, [])

        # Update source to remove link
        run_async(core.update_entry(
            path=source["path"],
            content="# Link Source\n\nNo more links.",
        ))

        # Backlink should be cleared
        backlinks = rebuild_backlink_cache(tmp_kb)
        assert source_key not in backlinks.get(target_key, [])
        assert source["path"] not in backlinks.get(target_key, [])

    def test_section_update_preserves_other_sections(self, tmp_kb):
        """Section updates only modify the target section."""
        result = run_async(core.add_entry(
            title="Section Test",
            content="# Section Test\n\n## Section One\n\nOriginal one.\n\n## Section Two\n\nOriginal two.",
            tags=["sections"],
            category="general",
        ))

        # Update only Section One
        run_async(core.update_entry(
            path=result["path"],
            section_updates={"Section One": "Updated one."},
        ))

        entry = run_async(core.get_entry(result["path"]))
        assert "Updated one" in entry.content
        assert "Original two" in entry.content


# ─────────────────────────────────────────────────────────────────────────────
# Delete -> Cleanup
# ─────────────────────────────────────────────────────────────────────────────


class TestDeleteCleanup:
    """Test that delete cleans up search, backlinks, and tracking."""

    def test_deleted_entry_not_in_search(self, tmp_kb):
        """Deleted entry no longer appears in search results."""
        result = run_async(core.add_entry(
            title="Delete Search Test",
            content="# Delete Search Test\n\nUnique deleteme content.",
            tags=["delete"],
            category="general",
        ))

        # Verify it's searchable
        pre_results = run_async(core.search(query="deleteme", limit=10))
        assert any(r.path == result["path"] for r in pre_results.results)

        # Delete it
        run_async(core.delete_entry(result["path"], force=True))

        # Should not be in search
        post_results = run_async(core.search(query="deleteme", limit=10))
        assert all(r.path != result["path"] for r in post_results.results)

    def test_deleted_entry_not_retrievable(self, tmp_kb):
        """Deleted entry cannot be retrieved."""
        result = run_async(core.add_entry(
            title="Delete Get Test",
            content="# Delete Get Test\n\nContent.",
            tags=["delete"],
            category="general",
        ))

        run_async(core.delete_entry(result["path"], force=True))

        with pytest.raises(ValueError, match="not found"):
            run_async(core.get_entry(result["path"]))

    def test_delete_with_backlinks_requires_force(self, tmp_kb):
        """Deleting entry with backlinks requires force=True."""
        # Create target and source
        target = run_async(core.add_entry(
            title="Delete Target",
            content="# Delete Target\n\nTarget.",
            tags=["target"],
            category="general",
        ))

        target_link = target["path"].replace(".md", "")
        run_async(core.add_entry(
            title="Delete Source",
            content=f"# Delete Source\n\nLinks to [[{target_link}]].",
            tags=["source"],
            category="general",
        ))

        # Delete without force should fail
        with pytest.raises(ValueError, match="backlink"):
            run_async(core.delete_entry(target["path"], force=False))

        # Delete with force should succeed
        run_async(core.delete_entry(target["path"], force=True))

    def test_delete_updates_tags_cache(self, tmp_kb):
        """Deleted entry's tags are removed from cache."""
        result = run_async(core.add_entry(
            title="Tag Delete Test",
            content="# Tag Delete Test\n\nContent.",
            tags=["unique-delete-tag"],
            category="general",
        ))

        # Verify tag exists
        tags_before = ensure_tags_cache(tmp_kb)
        assert "unique-delete-tag" in tags_before

        # Delete entry
        run_async(core.delete_entry(result["path"], force=True))

        # Tag should be gone (was only used by this entry)
        tags_after = rebuild_tags_cache(tmp_kb)
        assert "unique-delete-tag" not in tags_after

    def test_delete_returns_had_backlinks(self, tmp_kb):
        """Delete returns list of entries that linked to deleted entry."""
        target = run_async(core.add_entry(
            title="Backlink Delete Target",
            content="# Backlink Delete Target\n\nTarget.",
            tags=["target"],
            category="general",
        ))

        target_link = target["path"].replace(".md", "")
        source = run_async(core.add_entry(
            title="Backlink Delete Source",
            content=f"# Backlink Delete Source\n\nLinks to [[{target_link}]].",
            tags=["source"],
            category="general",
        ))

        result = run_async(core.delete_entry(target["path"], force=True))

        assert "had_backlinks" in result
        source_key = source["path"].replace(".md", "")
        # had_backlinks may be stored without .md extension
        assert source["path"] in result["had_backlinks"] or source_key in result["had_backlinks"]


# ─────────────────────────────────────────────────────────────────────────────
# Complex Workflows
# ─────────────────────────────────────────────────────────────────────────────


class TestComplexWorkflows:
    """Test complex multi-step workflows."""

    def test_create_linked_entries_verify_backlinks(self, tmp_kb):
        """Create A linking to B, verify backlinks on B."""
        entry_b = run_async(core.add_entry(
            title="Entry B",
            content="# Entry B\n\nThis is entry B.",
            tags=["workflow"],
            category="general",
        ))

        b_link = entry_b["path"].replace(".md", "")
        entry_a = run_async(core.add_entry(
            title="Entry A",
            content=f"# Entry A\n\nThis links to [[{b_link}]].",
            tags=["workflow"],
            category="general",
        ))

        # Verify backlink on B (backlinks stored without .md extension)
        entry_b_data = run_async(core.get_entry(entry_b["path"]))
        entry_a_key = entry_a["path"].replace(".md", "")
        assert entry_a["path"] in entry_b_data.backlinks or entry_a_key in entry_b_data.backlinks

    def test_multiple_entries_linking_same_target(self, tmp_kb):
        """Multiple entries can link to the same target."""
        target = run_async(core.add_entry(
            title="Multi Link Target",
            content="# Multi Link Target\n\nTarget.",
            tags=["multi"],
            category="general",
        ))

        target_link = target["path"].replace(".md", "")

        source1 = run_async(core.add_entry(
            title="Source One",
            content=f"# Source One\n\nLinks to [[{target_link}]].",
            tags=["source"],
            category="general",
        ))

        source2 = run_async(core.add_entry(
            title="Source Two",
            content=f"# Source Two\n\nAlso links to [[{target_link}]].",
            tags=["source"],
            category="general",
        ))

        target_data = run_async(core.get_entry(target["path"]))
        # Backlinks may be stored without .md extension
        source1_key = source1["path"].replace(".md", "")
        source2_key = source2["path"].replace(".md", "")
        assert source1["path"] in target_data.backlinks or source1_key in target_data.backlinks
        assert source2["path"] in target_data.backlinks or source2_key in target_data.backlinks

    def test_circular_links(self, tmp_kb):
        """Entries can have circular links."""
        entry_a = run_async(core.add_entry(
            title="Circular A",
            content="# Circular A\n\nWill link to B.",
            tags=["circular"],
            category="general",
        ))

        a_link = entry_a["path"].replace(".md", "")
        entry_b = run_async(core.add_entry(
            title="Circular B",
            content=f"# Circular B\n\nLinks back to [[{a_link}]].",
            tags=["circular"],
            category="general",
        ))

        # Update A to link to B
        b_link = entry_b["path"].replace(".md", "")
        run_async(core.update_entry(
            path=entry_a["path"],
            content=f"# Circular A\n\nNow links to [[{b_link}]].",
        ))

        # Both should have backlinks (stored without .md extension)
        a_data = run_async(core.get_entry(entry_a["path"]))
        b_data = run_async(core.get_entry(entry_b["path"]))

        b_key = entry_b["path"].replace(".md", "")
        a_key = entry_a["path"].replace(".md", "")
        assert entry_b["path"] in a_data.backlinks or b_key in a_data.backlinks
        assert entry_a["path"] in b_data.backlinks or a_key in b_data.backlinks

    def test_add_with_links_parameter(self, tmp_kb):
        """add_entry with links parameter creates proper [[links]]."""
        target = run_async(core.add_entry(
            title="Links Param Target",
            content="# Links Param Target\n\nTarget.",
            tags=["links"],
            category="general",
        ))

        source = run_async(core.add_entry(
            title="Links Param Source",
            content="# Links Param Source\n\nSource.",
            tags=["links"],
            category="general",
            links=[target["path"]],
        ))

        source_data = run_async(core.get_entry(source["path"]))
        assert target["path"] in source_data.links or target["path"].replace(".md", "") in source_data.links

    def test_list_entries_by_category(self, tmp_kb):
        """list_entries filters by category correctly."""
        # Create entries in different directories
        (tmp_kb / "testcat").mkdir(exist_ok=True)
        create_entry(tmp_kb, "testcat/entry1.md", "Entry One", "Content one", ["cat"])
        create_entry(tmp_kb, "testcat/entry2.md", "Entry Two", "Content two", ["cat"])
        create_entry(tmp_kb, "general/entry3.md", "Entry Three", "Content three", ["gen"])

        results = run_async(core.list_entries(category="testcat", limit=10))

        paths = [r["path"] for r in results]
        assert "testcat/entry1.md" in paths
        assert "testcat/entry2.md" in paths
        assert all("testcat" in p for p in paths)

    def test_search_with_content_hydration(self, tmp_kb):
        """Search with include_content returns full content."""
        result = run_async(core.add_entry(
            title="Content Hydration Test",
            content="# Content Hydration Test\n\nFull content should be here.",
            tags=["hydrate"],
            category="general",
        ))

        search_results = run_async(core.search(
            query="Hydration Test",
            limit=5,
            include_content=True,
        ))

        matched = next((r for r in search_results.results if r.path == result["path"]), None)
        assert matched is not None
        assert matched.content is not None
        assert "Full content should be here" in matched.content


# ─────────────────────────────────────────────────────────────────────────────
# Edge Cases and Error Handling
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_add_duplicate_title_different_path(self, tmp_kb):
        """Can add entries with same title in different directories."""
        (tmp_kb / "dir1").mkdir(exist_ok=True)
        (tmp_kb / "dir2").mkdir(exist_ok=True)

        result1 = run_async(core.add_entry(
            title="Duplicate Title",
            content="# Duplicate Title\n\nFirst entry.",
            tags=["dup"],
            directory="dir1",
        ))

        result2 = run_async(core.add_entry(
            title="Duplicate Title",
            content="# Duplicate Title\n\nSecond entry.",
            tags=["dup"],
            directory="dir2",
        ))

        assert result1["path"] != result2["path"]
        assert "dir1" in result1["path"]
        assert "dir2" in result2["path"]

    def test_add_without_category_fails(self, tmp_kb):
        """add_entry without category or directory fails."""
        with pytest.raises(ValueError, match="category.*directory"):
            run_async(core.add_entry(
                title="No Category",
                content="# No Category\n\nContent.",
                tags=["test"],
            ))

    def test_add_without_tags_fails(self, tmp_kb):
        """add_entry without tags fails."""
        with pytest.raises(ValueError, match="tag"):
            run_async(core.add_entry(
                title="No Tags",
                content="# No Tags\n\nContent.",
                tags=[],
                category="general",
            ))

    def test_update_nonexistent_fails(self, tmp_kb):
        """update_entry on nonexistent path fails."""
        with pytest.raises(ValueError, match="not found"):
            run_async(core.update_entry(
                path="nonexistent.md",
                content="# New Content",
            ))

    def test_get_nonexistent_fails(self, tmp_kb):
        """get_entry on nonexistent path fails."""
        with pytest.raises(ValueError, match="not found"):
            run_async(core.get_entry("nonexistent.md"))

    def test_delete_nonexistent_fails(self, tmp_kb):
        """delete_entry on nonexistent path fails."""
        with pytest.raises(ValueError, match="not found"):
            run_async(core.delete_entry("nonexistent.md"))

    def test_special_characters_in_title(self, tmp_kb):
        """Titles with special characters are slugified correctly."""
        result = run_async(core.add_entry(
            title="Special: Characters! Are @#$ Here",
            content="# Special Characters\n\nContent.",
            tags=["special"],
            category="general",
        ))

        # Path should be slugified
        assert ".md" in result["path"]
        assert ":" not in result["path"]
        assert "!" not in result["path"]
        assert "@" not in result["path"]


# ─────────────────────────────────────────────────────────────────────────────
# File System Consistency
# ─────────────────────────────────────────────────────────────────────────────


class TestFileSystemConsistency:
    """Test that file system state matches expected state."""

    def test_add_creates_file(self, tmp_kb):
        """add_entry creates actual file on disk."""
        result = run_async(core.add_entry(
            title="File Creation Test",
            content="# File Creation Test\n\nContent.",
            tags=["file"],
            category="general",
        ))

        file_path = tmp_kb / result["path"]
        assert file_path.exists()
        assert file_path.is_file()

    def test_file_content_matches_entry(self, tmp_kb):
        """File content matches what parse_entry returns."""
        result = run_async(core.add_entry(
            title="Content Match Test",
            content="# Content Match Test\n\nSpecific content here.",
            tags=["match"],
            category="general",
        ))

        file_path = tmp_kb / result["path"]
        metadata, content, _ = parse_entry(file_path)

        assert metadata.title == "Content Match Test"
        assert "Specific content here" in content
        assert "match" in metadata.tags

    def test_update_modifies_file(self, tmp_kb):
        """update_entry modifies actual file on disk."""
        result = run_async(core.add_entry(
            title="Update File Test",
            content="# Update File Test\n\nOriginal.",
            tags=["update"],
            category="general",
        ))

        run_async(core.update_entry(
            path=result["path"],
            content="# Update File Test\n\nModified content.",
        ))

        file_path = tmp_kb / result["path"]
        _, content, _ = parse_entry(file_path)
        assert "Modified content" in content
        assert "Original" not in content

    def test_delete_removes_file(self, tmp_kb):
        """delete_entry removes actual file from disk."""
        result = run_async(core.add_entry(
            title="Delete File Test",
            content="# Delete File Test\n\nContent.",
            tags=["delete"],
            category="general",
        ))

        file_path = tmp_kb / result["path"]
        assert file_path.exists()

        run_async(core.delete_entry(result["path"], force=True))

        assert not file_path.exists()

    def test_add_creates_nested_directories(self, tmp_kb):
        """add_entry creates nested directories if needed."""
        result = run_async(core.add_entry(
            title="Nested Dir Test",
            content="# Nested Dir Test\n\nContent.",
            tags=["nested"],
            directory="deep/nested/path",
        ))

        assert "deep/nested/path" in result["path"]
        assert (tmp_kb / "deep" / "nested" / "path").is_dir()
