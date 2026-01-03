"""Tests for memex.frontmatter module."""

from datetime import date

import pytest

from memex.frontmatter import build_frontmatter, create_new_metadata, update_metadata_for_edit
from memex.models import EntryMetadata


class TestBuildFrontmatter:
    """Tests for build_frontmatter function."""

    def test_minimal_metadata(self):
        """Builds frontmatter with only required fields."""
        metadata = EntryMetadata(
            title="Test Entry",
            tags=["python"],
            created=date(2024, 1, 15),
        )

        result = build_frontmatter(metadata)

        assert "---" in result
        assert "title: Test Entry" in result
        assert "tags:" in result
        assert "  - python" in result
        assert "created: 2024-01-15" in result

    def test_with_updated_date(self):
        """Includes updated date when present."""
        metadata = EntryMetadata(
            title="Updated Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            updated=date(2024, 1, 15),
        )

        result = build_frontmatter(metadata)

        assert "updated: 2024-01-15" in result

    def test_with_contributors(self):
        """Includes contributors when present."""
        metadata = EntryMetadata(
            title="Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            contributors=["Alice", "Bob"],
        )

        result = build_frontmatter(metadata)

        assert "contributors:" in result
        assert "  - Alice" in result
        assert "  - Bob" in result

    def test_with_aliases(self):
        """Includes aliases when present."""
        metadata = EntryMetadata(
            title="Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            aliases=["old-name", "alternate"],
        )

        result = build_frontmatter(metadata)

        assert "aliases:" in result
        assert "  - old-name" in result
        assert "  - alternate" in result

    def test_status_not_published(self):
        """Includes status when not default 'published'."""
        metadata = EntryMetadata(
            title="Draft Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            status="draft",
        )

        result = build_frontmatter(metadata)

        assert "status: draft" in result

    def test_status_published_omitted(self):
        """Omits status when it's the default 'published'."""
        metadata = EntryMetadata(
            title="Published Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            status="published",
        )

        result = build_frontmatter(metadata)

        assert "status:" not in result

    def test_with_source_project(self):
        """Includes source_project when present."""
        metadata = EntryMetadata(
            title="Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            source_project="my-project",
        )

        result = build_frontmatter(metadata)

        assert "source_project: my-project" in result

    def test_with_edit_sources(self):
        """Includes edit_sources when present."""
        metadata = EntryMetadata(
            title="Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            edit_sources=["project-a", "project-b"],
        )

        result = build_frontmatter(metadata)

        assert "edit_sources:" in result
        assert "  - project-a" in result
        assert "  - project-b" in result

    def test_with_model(self):
        """Includes model when present."""
        metadata = EntryMetadata(
            title="AI Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            model="claude-3-opus",
        )

        result = build_frontmatter(metadata)

        assert "model: claude-3-opus" in result

    def test_with_git_branch(self):
        """Includes git_branch when present."""
        metadata = EntryMetadata(
            title="Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            git_branch="feature/new-feature",
        )

        result = build_frontmatter(metadata)

        assert "git_branch: feature/new-feature" in result

    def test_with_last_edited_by(self):
        """Includes last_edited_by when present."""
        metadata = EntryMetadata(
            title="Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            last_edited_by="claude-agent",
        )

        result = build_frontmatter(metadata)

        assert "last_edited_by: claude-agent" in result

    def test_with_beads_issues(self):
        """Includes beads_issues when present."""
        metadata = EntryMetadata(
            title="Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            beads_issues=["issue-123", "issue-456"],
        )

        result = build_frontmatter(metadata)

        assert "beads_issues:" in result
        assert "  - issue-123" in result
        assert "  - issue-456" in result

    def test_with_beads_project(self):
        """Includes beads_project when present."""
        metadata = EntryMetadata(
            title="Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            beads_project="my-beads-project",
        )

        result = build_frontmatter(metadata)

        assert "beads_project: my-beads-project" in result

    def test_full_metadata(self):
        """Builds complete frontmatter with all fields populated."""
        metadata = EntryMetadata(
            title="Complete Entry",
            tags=["python", "testing"],
            created=date(2024, 1, 1),
            updated=date(2024, 1, 15),
            contributors=["Alice"],
            aliases=["complete", "full"],
            status="draft",
            source_project="main-project",
            edit_sources=["other-project"],
            model="claude-opus-4",
            git_branch="main",
            last_edited_by="ci-agent",
            beads_issues=["beads-001"],
            beads_project="tracker",
        )

        result = build_frontmatter(metadata)

        # Check all fields are present
        assert "title: Complete Entry" in result
        assert "  - python" in result
        assert "  - testing" in result
        assert "created: 2024-01-01" in result
        assert "updated: 2024-01-15" in result
        assert "contributors:" in result
        assert "aliases:" in result
        assert "status: draft" in result
        assert "source_project: main-project" in result
        assert "edit_sources:" in result
        assert "model: claude-opus-4" in result
        assert "git_branch: main" in result
        assert "last_edited_by: ci-agent" in result
        assert "beads_issues:" in result
        assert "beads_project: tracker" in result


class TestCreateNewMetadata:
    """Tests for create_new_metadata function."""

    def test_minimal_metadata(self):
        """Creates metadata with only required fields."""
        metadata = create_new_metadata(
            title="New Entry",
            tags=["test"],
        )

        assert metadata.title == "New Entry"
        assert metadata.tags == ["test"]
        assert metadata.created == date.today()
        assert metadata.updated is None
        assert metadata.contributors == []

    def test_with_source_project(self):
        """Sets source_project when provided."""
        metadata = create_new_metadata(
            title="Entry",
            tags=["test"],
            source_project="my-project",
        )

        assert metadata.source_project == "my-project"

    def test_with_contributor(self):
        """Adds contributor to list."""
        metadata = create_new_metadata(
            title="Entry",
            tags=["test"],
            contributor="Alice",
        )

        assert metadata.contributors == ["Alice"]

    def test_with_model(self):
        """Sets model for agent entries."""
        metadata = create_new_metadata(
            title="Entry",
            tags=["test"],
            model="claude-opus-4",
        )

        assert metadata.model == "claude-opus-4"

    def test_with_git_branch(self):
        """Sets git_branch for provenance."""
        metadata = create_new_metadata(
            title="Entry",
            tags=["test"],
            git_branch="feature/branch",
        )

        assert metadata.git_branch == "feature/branch"

    def test_with_actor(self):
        """Sets last_edited_by from actor."""
        metadata = create_new_metadata(
            title="Entry",
            tags=["test"],
            actor="claude-agent",
        )

        assert metadata.last_edited_by == "claude-agent"


class TestUpdateMetadataForEdit:
    """Tests for update_metadata_for_edit function."""

    @pytest.fixture
    def original_metadata(self):
        """Create sample original metadata."""
        return EntryMetadata(
            title="Original",
            tags=["original"],
            created=date(2024, 1, 1),
            contributors=["Alice"],
            source_project="project-a",
        )

    def test_preserves_immutable_fields(self, original_metadata):
        """Preserves title, created, and source_project."""
        updated = update_metadata_for_edit(original_metadata)

        assert updated.title == "Original"
        assert updated.created == date(2024, 1, 1)
        assert updated.source_project == "project-a"

    def test_sets_updated_date(self, original_metadata):
        """Sets updated to today's date."""
        updated = update_metadata_for_edit(original_metadata)

        assert updated.updated == date.today()

    def test_updates_tags(self, original_metadata):
        """Replaces tags when new_tags provided."""
        updated = update_metadata_for_edit(
            original_metadata,
            new_tags=["new", "tags"],
        )

        assert updated.tags == ["new", "tags"]

    def test_preserves_tags_when_none(self, original_metadata):
        """Preserves original tags when new_tags is None."""
        updated = update_metadata_for_edit(original_metadata)

        assert updated.tags == ["original"]

    def test_adds_new_contributor(self, original_metadata):
        """Adds new contributor to existing list."""
        updated = update_metadata_for_edit(
            original_metadata,
            new_contributor="Bob",
        )

        assert updated.contributors == ["Alice", "Bob"]

    def test_skips_duplicate_contributor(self, original_metadata):
        """Doesn't add contributor if already present."""
        updated = update_metadata_for_edit(
            original_metadata,
            new_contributor="Alice",
        )

        assert updated.contributors == ["Alice"]

    def test_adds_edit_source(self, original_metadata):
        """Adds edit_source from different project."""
        updated = update_metadata_for_edit(
            original_metadata,
            edit_source="project-b",
        )

        assert "project-b" in updated.edit_sources

    def test_skips_edit_source_same_as_origin(self, original_metadata):
        """Doesn't add edit_source if same as source_project."""
        updated = update_metadata_for_edit(
            original_metadata,
            edit_source="project-a",  # Same as source_project
        )

        assert "project-a" not in updated.edit_sources

    def test_skips_duplicate_edit_source(self):
        """Doesn't add edit_source if already present."""
        metadata = EntryMetadata(
            title="Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            source_project="project-a",
            edit_sources=["project-b"],
        )

        updated = update_metadata_for_edit(
            metadata,
            edit_source="project-b",
        )

        assert updated.edit_sources.count("project-b") == 1

    def test_sets_breadcrumb_fields(self, original_metadata):
        """Sets model, git_branch, and actor."""
        updated = update_metadata_for_edit(
            original_metadata,
            model="claude-opus-4",
            git_branch="main",
            actor="agent",
        )

        assert updated.model == "claude-opus-4"
        assert updated.git_branch == "main"
        assert updated.last_edited_by == "agent"

    def test_preserves_beads_fields(self):
        """Preserves beads_issues and beads_project."""
        metadata = EntryMetadata(
            title="Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            beads_issues=["issue-1"],
            beads_project="tracker",
        )

        updated = update_metadata_for_edit(metadata)

        assert updated.beads_issues == ["issue-1"]
        assert updated.beads_project == "tracker"

    def test_preserves_aliases(self):
        """Preserves existing aliases."""
        metadata = EntryMetadata(
            title="Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            aliases=["alias-1", "alias-2"],
        )

        updated = update_metadata_for_edit(metadata)

        assert updated.aliases == ["alias-1", "alias-2"]

    def test_preserves_status(self):
        """Preserves existing status."""
        metadata = EntryMetadata(
            title="Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            status="draft",
        )

        updated = update_metadata_for_edit(metadata)

        assert updated.status == "draft"
