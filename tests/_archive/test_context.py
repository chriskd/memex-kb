"""Tests for project context (.kbcontext) and KB config (.kbconfig)."""

import os
from pathlib import Path

import pytest

from memex.context import (
    CONTEXT_FILENAME,
    LOCAL_KB_CONFIG_FILENAME,
    KBConfig,
    KBContext,
    clear_context_cache,
    clear_kbconfig_cache,
    create_default_context,
    discover_kb_context,
    get_kb_context,
    get_kbconfig,
    load_kbconfig,
    matches_glob,
    validate_context,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear context and kbconfig caches before and after each test."""
    clear_context_cache()
    clear_kbconfig_cache()
    yield
    clear_context_cache()
    clear_kbconfig_cache()


@pytest.fixture
def project_dir(tmp_path) -> Path:
    """Create a temporary project directory."""
    project = tmp_path / "myproject"
    project.mkdir()
    return project


@pytest.fixture
def kb_root(tmp_path) -> Path:
    """Create a temporary KB root with standard categories."""
    root = tmp_path / "kb"
    root.mkdir()
    for category in ("projects", "tooling", "infrastructure"):
        (root / category).mkdir()
    (root / "projects" / "myproject").mkdir()
    return root


# ─────────────────────────────────────────────────────────────────────────────
# matches_glob tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMatchesGlob:
    """Tests for glob pattern matching."""

    def test_exact_match(self):
        """Test exact path matching."""
        assert matches_glob("tooling/beads.md", "tooling/beads")
        assert matches_glob("tooling/beads", "tooling/beads")

    def test_prefix_match(self):
        """Test prefix matching without wildcards."""
        assert matches_glob("projects/memex/docs.md", "projects/memex")
        assert matches_glob("projects/memex/subdir/file.md", "projects/memex")

    def test_no_match(self):
        """Test non-matching paths."""
        assert not matches_glob("tooling/beads.md", "projects")
        assert not matches_glob("infrastructure/docker.md", "tooling")

    def test_single_wildcard(self):
        """Test single-level wildcard (*)."""
        assert matches_glob("projects/foo/bar.md", "projects/*")
        assert matches_glob("projects/bar.md", "projects/*")
        # Should match nested paths too (pattern prefix)
        assert matches_glob("infrastructure/docker/guide.md", "infrastructure/*")

    def test_recursive_wildcard(self):
        """Test recursive wildcard (**)."""
        assert matches_glob("projects/a/b/c.md", "projects/**")
        assert matches_glob("projects/deep/nested/file.md", "projects/**")
        assert matches_glob("projects/file.md", "projects/**")

    def test_md_extension_handling(self):
        """Test that .md extension is handled correctly."""
        assert matches_glob("tooling/beads.md", "tooling/beads")
        assert matches_glob("tooling/beads.md", "tooling/beads.md")

    def test_trailing_slash_handling(self):
        """Test trailing slash normalization."""
        assert matches_glob("projects/foo/bar.md", "projects/foo/")
        assert matches_glob("projects/foo/bar.md", "projects/foo")


# ─────────────────────────────────────────────────────────────────────────────
# KBContext tests
# ─────────────────────────────────────────────────────────────────────────────


class TestKBContext:
    """Tests for KBContext dataclass."""

    def test_from_dict(self):
        """Test creating context from dict."""
        data = {
            "primary": "projects/myapp",
            "paths": ["projects/myapp", "tooling/*"],
            "default_tags": ["myapp", "python"],
            "project": "my-app",
        }
        ctx = KBContext.from_dict(data)

        assert ctx.primary == "projects/myapp"
        assert ctx.paths == ["projects/myapp", "tooling/*"]
        assert ctx.default_tags == ["myapp", "python"]
        assert ctx.project == "my-app"

    def test_from_dict_minimal(self):
        """Test creating context from minimal dict."""
        ctx = KBContext.from_dict({})

        assert ctx.primary is None
        assert ctx.paths == []
        assert ctx.default_tags == []
        assert ctx.project is None

    def test_get_project_name_from_config(self):
        """Test project name from explicit config."""
        ctx = KBContext(project="explicit-name")
        assert ctx.get_project_name() == "explicit-name"

    def test_get_project_name_from_source_file(self, project_dir):
        """Test project name from source file directory."""
        ctx = KBContext(source_file=project_dir / ".kbcontext")
        assert ctx.get_project_name() == "myproject"

    def test_get_all_boost_paths_includes_primary(self):
        """Test that primary is included in boost paths."""
        ctx = KBContext(
            primary="projects/myapp",
            paths=["tooling/beads"],
        )
        boost_paths = ctx.get_all_boost_paths()

        assert "projects/myapp" in boost_paths
        assert "tooling/beads" in boost_paths

    def test_get_all_boost_paths_no_duplicate_primary(self):
        """Test that primary is not duplicated if already in paths."""
        ctx = KBContext(
            primary="projects/myapp",
            paths=["projects/myapp", "tooling/beads"],
        )
        boost_paths = ctx.get_all_boost_paths()

        assert boost_paths.count("projects/myapp") == 1


# ─────────────────────────────────────────────────────────────────────────────
# Context discovery tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDiscoverKBContext:
    """Tests for context file discovery."""

    def test_discover_from_cwd(self, project_dir, monkeypatch):
        """Test finding context file from current directory."""
        # Create .kbcontext file
        context_file = project_dir / CONTEXT_FILENAME
        context_file.write_text("""
primary: projects/myproject
paths:
  - projects/myproject
default_tags:
  - myproject
""")

        monkeypatch.chdir(project_dir)
        ctx = discover_kb_context()

        assert ctx is not None
        assert ctx.primary == "projects/myproject"
        assert ctx.source_file == context_file

    def test_discover_from_subdirectory(self, project_dir, monkeypatch):
        """Test finding context file walking up from subdirectory."""
        # Create .kbcontext in project root
        context_file = project_dir / CONTEXT_FILENAME
        context_file.write_text("primary: projects/myproject")

        # Create and cd to subdirectory
        subdir = project_dir / "src" / "deep"
        subdir.mkdir(parents=True)

        monkeypatch.chdir(subdir)
        ctx = discover_kb_context()

        assert ctx is not None
        assert ctx.source_file == context_file

    def test_discover_with_env_override(self, project_dir, tmp_path, monkeypatch):
        """Test VL_KB_CONTEXT env var override."""
        # Create context file in a different location
        alt_context = tmp_path / "custom" / ".kbcontext"
        alt_context.parent.mkdir()
        alt_context.write_text("primary: custom/path")

        # Set env var to override
        monkeypatch.setenv("VL_KB_CONTEXT", str(alt_context))
        monkeypatch.chdir(project_dir)

        ctx = discover_kb_context()

        assert ctx is not None
        assert ctx.primary == "custom/path"
        assert ctx.source_file == alt_context

    def test_discover_no_context(self, tmp_path, monkeypatch):
        """Test when no context file exists."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        monkeypatch.chdir(empty_dir)
        ctx = discover_kb_context()

        assert ctx is None

    def test_discover_invalid_yaml(self, project_dir, monkeypatch):
        """Test handling of invalid YAML in context file."""
        context_file = project_dir / CONTEXT_FILENAME
        context_file.write_text("invalid: yaml: content: [")

        monkeypatch.chdir(project_dir)
        ctx = discover_kb_context()

        assert ctx is None


class TestGetKBContext:
    """Tests for cached context getter."""

    def test_caches_result(self, project_dir, monkeypatch):
        """Test that results are cached per directory."""
        context_file = project_dir / CONTEXT_FILENAME
        context_file.write_text("primary: test")

        monkeypatch.chdir(project_dir)

        ctx1 = get_kb_context()
        ctx2 = get_kb_context()

        assert ctx1 is ctx2  # Same instance (cached)

    def test_cache_cleared(self, project_dir, monkeypatch):
        """Test that cache can be cleared."""
        context_file = project_dir / CONTEXT_FILENAME
        context_file.write_text("primary: test")

        monkeypatch.chdir(project_dir)

        ctx1 = get_kb_context()
        clear_context_cache()
        ctx2 = get_kb_context()

        assert ctx1 is not ctx2  # Different instances after cache clear


# ─────────────────────────────────────────────────────────────────────────────
# Validation tests
# ─────────────────────────────────────────────────────────────────────────────


class TestValidateContext:
    """Tests for context validation."""

    def test_valid_context(self, kb_root):
        """Test validation of valid context."""
        ctx = KBContext(
            primary="projects/myproject",
            paths=["projects/myproject", "tooling"],
        )
        warnings = validate_context(ctx, kb_root)

        assert warnings == []

    def test_missing_primary(self, kb_root):
        """Test warning for missing primary directory."""
        ctx = KBContext(
            primary="projects/nonexistent",
            paths=[],
        )
        warnings = validate_context(ctx, kb_root)

        assert len(warnings) == 1
        assert "nonexistent" in warnings[0]

    def test_missing_path(self, kb_root):
        """Test warning for missing path."""
        ctx = KBContext(
            primary="projects/myproject",
            paths=["nonexistent/path"],
        )
        warnings = validate_context(ctx, kb_root)

        assert len(warnings) == 1
        assert "nonexistent/path" in warnings[0]

    def test_glob_patterns_skipped(self, kb_root):
        """Test that glob patterns are not validated."""
        ctx = KBContext(
            primary="projects/myproject",
            paths=["projects/*", "tooling/**"],
        )
        warnings = validate_context(ctx, kb_root)

        assert warnings == []


# ─────────────────────────────────────────────────────────────────────────────
# Default content generation tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCreateDefaultContext:
    """Tests for default context content generation."""

    def test_basic_content(self):
        """Test basic content generation."""
        content = create_default_context("myapp")

        assert "primary: projects/myapp" in content
        assert "default_tags:" in content
        assert "- myapp" in content

    def test_custom_directory(self):
        """Test content with custom directory."""
        content = create_default_context("myapp", "custom/path")

        assert "primary: custom/path" in content
        assert "- custom/path" in content


# ─────────────────────────────────────────────────────────────────────────────
# KBConfig tests
# ─────────────────────────────────────────────────────────────────────────────


class TestKBConfig:
    """Tests for KBConfig dataclass."""

    def test_from_dict_full(self):
        """Creates KBConfig from dict with all fields."""
        data = {
            "default_tags": ["personal", "notes"],
            "exclude": ["*.draft.md", "private/**"],
        }
        config = KBConfig.from_dict(data)

        assert config.default_tags == ["personal", "notes"]
        assert config.exclude == ["*.draft.md", "private/**"]

    def test_from_dict_empty(self):
        """Creates KBConfig from empty dict."""
        config = KBConfig.from_dict({})

        assert config.default_tags == []
        assert config.exclude == []

    def test_from_dict_partial(self):
        """Creates KBConfig with only some fields."""
        data = {"default_tags": ["work"]}
        config = KBConfig.from_dict(data)

        assert config.default_tags == ["work"]
        assert config.exclude == []

    def test_source_file_tracking(self, tmp_path):
        """Tracks source file path."""
        source = tmp_path / ".kbconfig"
        config = KBConfig.from_dict({}, source_file=source)

        assert config.source_file == source


class TestLoadKBConfig:
    """Tests for load_kbconfig function."""

    def test_load_full_config(self, tmp_path):
        """Loads config file with all fields."""
        config_file = tmp_path / LOCAL_KB_CONFIG_FILENAME
        config_file.write_text("""
default_tags:
  - personal
  - notes
exclude:
  - "*.draft.md"
""")
        config = load_kbconfig(tmp_path)

        assert config is not None
        assert config.default_tags == ["personal", "notes"]
        assert config.exclude == ["*.draft.md"]
        assert config.source_file == config_file

    def test_load_empty_config(self, tmp_path):
        """Handles empty config file (just comments)."""
        config_file = tmp_path / LOCAL_KB_CONFIG_FILENAME
        config_file.write_text("""# Empty config
# Only comments here
""")
        config = load_kbconfig(tmp_path)

        assert config is not None
        assert config.default_tags == []
        assert config.exclude == []

    def test_load_missing_config(self, tmp_path):
        """Returns None for missing config file."""
        config = load_kbconfig(tmp_path)

        assert config is None

    def test_load_invalid_yaml(self, tmp_path):
        """Returns None for invalid YAML."""
        config_file = tmp_path / LOCAL_KB_CONFIG_FILENAME
        config_file.write_text("not: valid: yaml: [[")

        config = load_kbconfig(tmp_path)

        assert config is None

    def test_load_non_dict_yaml(self, tmp_path):
        """Returns None for non-dict YAML."""
        config_file = tmp_path / LOCAL_KB_CONFIG_FILENAME
        config_file.write_text("- just\n- a\n- list")

        config = load_kbconfig(tmp_path)

        assert config is None


class TestGetKBConfig:
    """Tests for get_kbconfig caching function."""

    def test_caches_result(self, tmp_path):
        """Caches config to avoid repeated reads."""
        config_file = tmp_path / LOCAL_KB_CONFIG_FILENAME
        config_file.write_text("default_tags: [cached]")

        # First call
        config1 = get_kbconfig(tmp_path)
        assert config1 is not None
        assert config1.default_tags == ["cached"]

        # Modify file (should not affect cached result)
        config_file.write_text("default_tags: [modified]")

        # Second call should return cached
        config2 = get_kbconfig(tmp_path)
        assert config2 is config1  # Same object
        assert config2.default_tags == ["cached"]

    def test_cache_cleared(self, tmp_path):
        """Cache can be cleared to reload config."""
        config_file = tmp_path / LOCAL_KB_CONFIG_FILENAME
        config_file.write_text("default_tags: [original]")

        config1 = get_kbconfig(tmp_path)
        assert config1.default_tags == ["original"]

        # Modify and clear cache
        config_file.write_text("default_tags: [updated]")
        clear_kbconfig_cache()

        config2 = get_kbconfig(tmp_path)
        assert config2.default_tags == ["updated"]
