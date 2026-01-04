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
        assert "- python" in result
        assert "created: 2024-01-15" in result

    def test_title_with_colon(self):
        """Titles with colons are properly quoted for valid YAML."""
        metadata = EntryMetadata(
            title="vl-mail: Lightweight Agent Mail CLI",
            tags=["test"],
            created=date(2024, 1, 15),
        )

        result = build_frontmatter(metadata)

        # yaml.safe_dump uses single quotes for strings with colons
        assert "'vl-mail: Lightweight Agent Mail CLI'" in result

    def test_title_with_quotes(self):
        """Titles with quotes are properly escaped."""
        metadata = EntryMetadata(
            title='Entry with "quoted" text',
            tags=["test"],
            created=date(2024, 1, 15),
        )

        result = build_frontmatter(metadata)

        # yaml.safe_dump handles quote escaping
        assert "Entry with" in result
        assert "quoted" in result

    def test_tags_with_colons(self):
        """Tags containing colons are properly quoted."""
        metadata = EntryMetadata(
            title="Entry",
            tags=["simple", "category: subcategory", "foo:bar"],
            created=date(2024, 1, 15),
        )

        result = build_frontmatter(metadata)

        # Simple tags stay unquoted
        assert "- simple" in result
        # Tags with ": " are quoted by yaml.safe_dump
        assert "'category: subcategory'" in result
        # Tags without space after colon may not need quoting
        assert "foo:bar" in result

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
        assert "- Alice" in result
        assert "- Bob" in result

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
        assert "- old-name" in result
        assert "- alternate" in result

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
        assert "- project-a" in result
        assert "- project-b" in result

    def test_with_model(self):
        """Includes model when present."""
        metadata = EntryMetadata(
            title="Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            model="claude-opus-4",
        )

        result = build_frontmatter(metadata)

        assert "model: claude-opus-4" in result

    def test_with_git_branch(self):
        """Includes git_branch when present."""
        metadata = EntryMetadata(
            title="Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            git_branch="feature/new-thing",
        )

        result = build_frontmatter(metadata)

        assert "git_branch: feature/new-thing" in result

    def test_with_last_edited_by(self):
        """Includes last_edited_by when present."""
        metadata = EntryMetadata(
            title="Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            last_edited_by="ci-agent",
        )

        result = build_frontmatter(metadata)

        assert "last_edited_by: ci-agent" in result

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
        assert "- issue-123" in result
        assert "- issue-456" in result

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
        assert "- python" in result
        assert "- testing" in result
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

    def test_roundtrip_valid_yaml(self):
        """Frontmatter can be parsed back as valid YAML."""
        import yaml

        metadata = EntryMetadata(
            title='vl-mail: Test with "quotes" and colons',
            tags=["simple", "category: subcategory"],
            created=date(2024, 1, 15),
            aliases=["old: name"],
            contributors=['User <email: local>'],
        )

        fm = build_frontmatter(metadata)
        # Extract YAML content between --- markers
        content = fm.split("---")[1]
        parsed = yaml.safe_load(content)

        assert parsed["title"] == 'vl-mail: Test with "quotes" and colons'
        assert parsed["tags"] == ["simple", "category: subcategory"]
        assert parsed["aliases"] == ["old: name"]
        assert parsed["contributors"] == ['User <email: local>']


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
        """Creates metadata with source_project."""
        metadata = create_new_metadata(
            title="Entry",
            tags=["test"],
            source_project="my-project",
        )

        assert metadata.source_project == "my-project"

    def test_with_contributor(self):
        """Creates metadata with contributor."""
        metadata = create_new_metadata(
            title="Entry",
            tags=["test"],
            contributor="Alice",
        )

        assert metadata.contributors == ["Alice"]

    def test_with_model(self):
        """Creates metadata with model."""
        metadata = create_new_metadata(
            title="Entry",
            tags=["test"],
            model="claude-opus-4",
        )

        assert metadata.model == "claude-opus-4"

    def test_with_git_branch(self):
        """Creates metadata with git_branch."""
        metadata = create_new_metadata(
            title="Entry",
            tags=["test"],
            git_branch="main",
        )

        assert metadata.git_branch == "main"

    def test_with_actor(self):
        """Creates metadata with actor."""
        metadata = create_new_metadata(
            title="Entry",
            tags=["test"],
            actor="ci-agent",
        )

        assert metadata.last_edited_by == "ci-agent"


class TestUpdateMetadataForEdit:
    """Tests for update_metadata_for_edit function."""

    @pytest.fixture
    def base_metadata(self):
        """Create base metadata for update tests."""
        return EntryMetadata(
            title="Original Title",
            tags=["original"],
            created=date(2024, 1, 1),
            source_project="original-project",
        )

    def test_preserves_immutable_fields(self, base_metadata):
        """Preserves title, created, and source_project."""
        updated = update_metadata_for_edit(base_metadata)

        assert updated.title == "Original Title"
        assert updated.created == date(2024, 1, 1)
        assert updated.source_project == "original-project"

    def test_sets_updated_date(self, base_metadata):
        """Sets updated date to today."""
        updated = update_metadata_for_edit(base_metadata)

        assert updated.updated == date.today()

    def test_updates_tags(self, base_metadata):
        """Updates tags when provided."""
        updated = update_metadata_for_edit(
            base_metadata,
            new_tags=["new", "tags"],
        )

        assert updated.tags == ["new", "tags"]

    def test_preserves_tags_when_none(self, base_metadata):
        """Preserves original tags when new_tags is None."""
        updated = update_metadata_for_edit(base_metadata, new_tags=None)

        assert updated.tags == ["original"]

    def test_adds_new_contributor(self, base_metadata):
        """Adds new contributor to list."""
        updated = update_metadata_for_edit(
            base_metadata,
            new_contributor="Alice",
        )

        assert "Alice" in updated.contributors

    def test_skips_duplicate_contributor(self):
        """Doesn't duplicate existing contributors."""
        metadata = EntryMetadata(
            title="Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            contributors=["Alice"],
        )

        updated = update_metadata_for_edit(
            metadata,
            new_contributor="Alice",
        )

        assert updated.contributors.count("Alice") == 1

    def test_adds_edit_source(self, base_metadata):
        """Adds edit_source to list."""
        updated = update_metadata_for_edit(
            base_metadata,
            edit_source="other-project",
        )

        assert "other-project" in updated.edit_sources

    def test_skips_edit_source_same_as_origin(self, base_metadata):
        """Doesn't add edit_source if same as source_project."""
        updated = update_metadata_for_edit(
            base_metadata,
            edit_source="original-project",
        )

        assert "original-project" not in updated.edit_sources

    def test_skips_duplicate_edit_source(self):
        """Doesn't duplicate existing edit_sources."""
        metadata = EntryMetadata(
            title="Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            edit_sources=["other-project"],
        )

        updated = update_metadata_for_edit(
            metadata,
            edit_source="other-project",
        )

        assert updated.edit_sources.count("other-project") == 1

    def test_sets_breadcrumb_fields(self, base_metadata):
        """Sets model, git_branch, and last_edited_by."""
        updated = update_metadata_for_edit(
            base_metadata,
            model="claude-opus-4",
            git_branch="feature/update",
            actor="ci-agent",
        )

        assert updated.model == "claude-opus-4"
        assert updated.git_branch == "feature/update"
        assert updated.last_edited_by == "ci-agent"

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
        """Preserves aliases list."""
        metadata = EntryMetadata(
            title="Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            aliases=["alias1", "alias2"],
        )

        updated = update_metadata_for_edit(metadata)

        assert updated.aliases == ["alias1", "alias2"]

    def test_preserves_status(self):
        """Preserves status."""
        metadata = EntryMetadata(
            title="Entry",
            tags=["test"],
            created=date(2024, 1, 1),
            status="draft",
        )

        updated = update_metadata_for_edit(metadata)

        assert updated.status == "draft"
