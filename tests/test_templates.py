"""Tests for the templates module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from memex.templates import (
    Template,
    BUILTIN_TEMPLATES,
    get_template,
    list_templates,
    apply_template,
    get_all_templates,
    _load_user_template,
    USER_TEMPLATES_DIR,
)


class TestBuiltinTemplates:
    """Tests for built-in templates."""

    def test_builtin_templates_exist(self):
        """All expected built-in templates exist."""
        expected = {
            "troubleshooting", "project", "pattern",
            "decision", "runbook", "api", "meeting", "blank",
        }
        assert set(BUILTIN_TEMPLATES.keys()) == expected

    def test_builtin_templates_have_required_fields(self):
        """Built-in templates have name, description, and content."""
        for name, template in BUILTIN_TEMPLATES.items():
            assert template.name == name
            assert template.description
            assert template.source == "builtin"
            # All except blank should have content
            if name != "blank":
                assert template.content

    def test_troubleshooting_template_structure(self):
        """Troubleshooting template has expected sections."""
        template = BUILTIN_TEMPLATES["troubleshooting"]
        assert "## Problem" in template.content
        assert "## Cause" in template.content
        assert "## Solution" in template.content
        assert "## Related" in template.content
        assert "troubleshooting" in template.suggested_tags
        assert "fix" in template.suggested_tags

    def test_decision_template_has_adr_sections(self):
        """Decision template follows ADR format."""
        template = BUILTIN_TEMPLATES["decision"]
        assert "## Context" in template.content
        assert "## Decision" in template.content
        assert "## Rationale" in template.content
        assert "## Consequences" in template.content
        assert "adr" in template.suggested_tags


class TestGetTemplate:
    """Tests for get_template function."""

    def test_get_builtin_template(self):
        """Can retrieve built-in templates by name."""
        template = get_template("troubleshooting")
        assert template is not None
        assert template.name == "troubleshooting"

    def test_get_unknown_template(self):
        """Returns None for unknown template names."""
        template = get_template("nonexistent")
        assert template is None

    def test_get_template_case_sensitive(self):
        """Template names are case-sensitive."""
        assert get_template("troubleshooting") is not None
        assert get_template("Troubleshooting") is None


class TestListTemplates:
    """Tests for list_templates function."""

    def test_list_templates_includes_builtins(self):
        """List includes all built-in templates."""
        templates = list_templates()
        names = {t.name for t in templates}
        assert "troubleshooting" in names
        assert "project" in names
        assert "pattern" in names

    def test_list_templates_sorted(self):
        """Templates are sorted by source priority then name."""
        templates = list_templates()
        # With no user/project templates, should be sorted by name
        names = [t.name for t in templates]
        assert names == sorted(names)


class TestApplyTemplate:
    """Tests for apply_template function."""

    def test_apply_template_adds_title(self):
        """Applied template includes title as H1."""
        template = BUILTIN_TEMPLATES["troubleshooting"]
        content = apply_template(template, "Fix login timeout")

        assert content.startswith("# Fix login timeout\n")

    def test_apply_template_includes_sections(self):
        """Applied template includes template sections."""
        template = BUILTIN_TEMPLATES["troubleshooting"]
        content = apply_template(template, "My Issue")

        assert "## Problem" in content
        assert "## Solution" in content

    def test_apply_blank_template(self):
        """Blank template just adds the title."""
        template = BUILTIN_TEMPLATES["blank"]
        content = apply_template(template, "My Entry")

        assert content == "# My Entry\n\n"

    def test_apply_template_strips_content(self):
        """Template content is stripped to avoid trailing whitespace."""
        template = Template(
            name="test",
            description="Test",
            content="\n\n## Section\n\n\n",
        )
        content = apply_template(template, "Title")
        assert content.endswith("## Section\n")


class TestUserTemplates:
    """Tests for user-defined template loading."""

    def test_load_yaml_template(self, tmp_path):
        """Can load template from YAML file."""
        yaml_content = """
name: custom
description: My custom template
content: |
  ## Custom Section
  Content here
suggested_tags:
  - custom
  - test
"""
        yaml_file = tmp_path / "custom.yaml"
        yaml_file.write_text(yaml_content)

        template = _load_user_template(yaml_file)

        assert template is not None
        assert template.name == "custom"
        assert template.description == "My custom template"
        assert "## Custom Section" in template.content
        assert template.suggested_tags == ["custom", "test"]
        assert template.source == "user"

    def test_load_markdown_template(self, tmp_path):
        """Can load template from Markdown file."""
        md_content = """<!-- My markdown template -->
## Section 1

Content here

## Section 2

More content
"""
        md_file = tmp_path / "mytemplate.md"
        md_file.write_text(md_content)

        template = _load_user_template(md_file)

        assert template is not None
        assert template.name == "mytemplate"  # From filename
        assert template.description == "My markdown template"  # From comment
        assert "## Section 1" in template.content
        assert template.source == "user"

    def test_load_invalid_yaml_returns_none(self, tmp_path):
        """Invalid YAML returns None."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("not: valid: yaml: here")

        template = _load_user_template(yaml_file)
        assert template is None

    def test_load_nonexistent_file_returns_none(self, tmp_path):
        """Nonexistent file returns None."""
        template = _load_user_template(tmp_path / "missing.yaml")
        assert template is None


class TestProjectTemplates:
    """Tests for project-specific template loading."""

    @patch("memex.templates.get_kb_context")
    def test_project_templates_override_builtin(self, mock_context, tmp_path):
        """Project templates take priority over builtins."""
        # Create a .kbcontext with custom troubleshooting template
        kbcontext = tmp_path / ".kbcontext"
        kbcontext.write_text("""
primary: projects/test
templates:
  troubleshooting:
    description: Custom troubleshooting
    content: |
      ## My Custom Problem
      ...
""")
        mock_ctx = MagicMock()
        mock_ctx.source_file = kbcontext
        mock_context.return_value = mock_ctx

        templates = get_all_templates()

        assert templates["troubleshooting"].source == "project"
        assert "My Custom Problem" in templates["troubleshooting"].content

    @patch("memex.templates.get_kb_context")
    def test_no_project_context(self, mock_context):
        """Works when no project context exists."""
        mock_context.return_value = None

        templates = get_all_templates()

        # Should still have builtins
        assert "troubleshooting" in templates


class TestTemplateDataclass:
    """Tests for Template dataclass."""

    def test_template_defaults(self):
        """Template has sensible defaults."""
        template = Template(
            name="test",
            description="Test template",
            content="Content",
        )

        assert template.suggested_tags == []
        assert template.source == "builtin"

    def test_template_with_all_fields(self):
        """Template accepts all fields."""
        template = Template(
            name="full",
            description="Full template",
            content="## Section\n",
            suggested_tags=["tag1", "tag2"],
            source="user",
        )

        assert template.name == "full"
        assert template.suggested_tags == ["tag1", "tag2"]
        assert template.source == "user"
