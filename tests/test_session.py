"""Tests for session context persistence."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from memex.cli import cli
from memex.session import (
    SESSION_FILENAME,
    SessionContext,
    clear_session,
    get_session,
    load_session,
    save_session,
)


@pytest.fixture
def index_root(tmp_path) -> Path:
    """Create a temporary index root."""
    root = tmp_path / ".indices"
    root.mkdir()
    return root


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


# ─────────────────────────────────────────────────────────────────────────────
# SessionContext Model Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSessionContext:
    """Tests for SessionContext dataclass."""

    def test_is_empty_true(self):
        """Empty session reports as empty."""
        ctx = SessionContext()
        assert ctx.is_empty()

    def test_is_empty_false_with_tags(self):
        """Session with tags is not empty."""
        ctx = SessionContext(tags=["infra"])
        assert not ctx.is_empty()

    def test_is_empty_false_with_project(self):
        """Session with project is not empty."""
        ctx = SessionContext(project="myapp")
        assert not ctx.is_empty()

    def test_is_empty_false_with_both(self):
        """Session with both tags and project is not empty."""
        ctx = SessionContext(tags=["infra"], project="myapp")
        assert not ctx.is_empty()

    def test_merge_tags_both_empty(self):
        """Merging empty tags returns None."""
        ctx = SessionContext()
        assert ctx.merge_tags(None) is None

    def test_merge_tags_session_only(self):
        """Session tags returned when CLI tags empty."""
        ctx = SessionContext(tags=["a", "b"])
        assert set(ctx.merge_tags(None)) == {"a", "b"}

    def test_merge_tags_cli_only(self):
        """CLI tags returned when session tags empty."""
        ctx = SessionContext()
        assert set(ctx.merge_tags(["x", "y"])) == {"x", "y"}

    def test_merge_tags_union(self):
        """Tags are merged as union."""
        ctx = SessionContext(tags=["a", "b"])
        assert set(ctx.merge_tags(["b", "c"])) == {"a", "b", "c"}

    def test_merge_tags_empty_list(self):
        """Merging empty list with session tags returns session tags."""
        ctx = SessionContext(tags=["a"])
        result = ctx.merge_tags([])
        assert set(result) == {"a"}

    def test_to_dict(self):
        """Session can be serialized to dict."""
        ctx = SessionContext(tags=["infra", "docker"], project="api")
        result = ctx.to_dict()
        assert result == {"tags": ["infra", "docker"], "project": "api"}

    def test_from_dict(self):
        """Session can be deserialized from dict."""
        data = {"tags": ["python", "web"], "project": "backend"}
        ctx = SessionContext.from_dict(data)
        assert ctx.tags == ["python", "web"]
        assert ctx.project == "backend"

    def test_from_dict_empty(self):
        """from_dict handles empty dict."""
        ctx = SessionContext.from_dict({})
        assert ctx.tags == []
        assert ctx.project is None

    def test_from_dict_partial(self):
        """from_dict handles partial data."""
        ctx = SessionContext.from_dict({"tags": ["test"]})
        assert ctx.tags == ["test"]
        assert ctx.project is None


# ─────────────────────────────────────────────────────────────────────────────
# Session Persistence Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSessionPersistence:
    """Tests for session load/save."""

    def test_load_empty(self, index_root):
        """load_session returns empty when no file exists."""
        ctx = load_session(index_root)
        assert ctx.is_empty()

    def test_save_and_load_round_trip(self, index_root):
        """Session can be saved and loaded."""
        ctx = SessionContext(tags=["infra", "docker"], project="api")
        save_session(ctx, index_root)

        loaded = load_session(index_root)
        assert loaded.tags == ["infra", "docker"]
        assert loaded.project == "api"

    def test_save_creates_file(self, index_root):
        """save_session creates the session file."""
        ctx = SessionContext(tags=["test"])
        save_session(ctx, index_root)

        session_file = index_root / SESSION_FILENAME
        assert session_file.exists()

    def test_save_overwrites_existing(self, index_root):
        """save_session overwrites existing session."""
        ctx1 = SessionContext(tags=["first"], project="proj1")
        save_session(ctx1, index_root)

        ctx2 = SessionContext(tags=["second"], project="proj2")
        save_session(ctx2, index_root)

        loaded = load_session(index_root)
        assert loaded.tags == ["second"]
        assert loaded.project == "proj2"

    def test_clear_session_returns_true(self, index_root):
        """clear_session returns True when session existed."""
        ctx = SessionContext(tags=["test"], project="proj")
        save_session(ctx, index_root)

        result = clear_session(index_root)
        assert result is True
        assert load_session(index_root).is_empty()

    def test_clear_session_returns_false_when_empty(self, index_root):
        """clear_session returns False when no active session."""
        # No session file exists
        result = clear_session(index_root)
        assert result is False

    def test_clear_session_with_empty_session_file(self, index_root):
        """clear_session returns False when file exists but session is empty."""
        save_session(SessionContext(), index_root)
        result = clear_session(index_root)
        assert result is False

    def test_get_session_convenience(self, index_root):
        """get_session is equivalent to load_session."""
        ctx = SessionContext(tags=["test"])
        save_session(ctx, index_root)

        loaded = get_session(index_root)
        assert loaded.tags == ["test"]

    def test_load_handles_malformed_json(self, index_root):
        """load_session returns empty for malformed JSON."""
        session_file = index_root / SESSION_FILENAME
        session_file.write_text("not valid json {{{")

        ctx = load_session(index_root)
        assert ctx.is_empty()

    def test_load_handles_schema_mismatch(self, index_root):
        """load_session returns empty for schema version mismatch."""
        session_file = index_root / SESSION_FILENAME
        session_file.write_text(json.dumps({
            "schema_version": 999,
            "session": {"tags": ["test"], "project": "proj"}
        }))

        ctx = load_session(index_root)
        assert ctx.is_empty()


# ─────────────────────────────────────────────────────────────────────────────
# CLI Command Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSessionCLI:
    """Tests for session CLI commands."""

    def test_session_show_empty(self, runner, index_root, monkeypatch):
        """mx session show with no session."""
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))
        result = runner.invoke(cli, ["session", "show"])
        assert result.exit_code == 0
        assert "No active session" in result.output

    def test_session_show_with_context(self, runner, index_root, monkeypatch):
        """mx session show displays active session."""
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))
        save_session(SessionContext(tags=["infra"], project="api"), index_root)

        result = runner.invoke(cli, ["session", "show"])
        assert result.exit_code == 0
        assert "infra" in result.output
        assert "api" in result.output

    def test_session_show_json(self, runner, index_root, monkeypatch):
        """mx session show --json outputs JSON."""
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))
        save_session(SessionContext(tags=["test"]), index_root)

        result = runner.invoke(cli, ["session", "show", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["active"] is True
        assert data["tags"] == ["test"]

    def test_session_start_tags_only(self, runner, index_root, monkeypatch):
        """mx session start --tags creates session."""
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))
        result = runner.invoke(cli, ["session", "start", "--tags=infra,docker"])
        assert result.exit_code == 0
        assert "Session started" in result.output

        ctx = load_session(index_root)
        assert set(ctx.tags) == {"infra", "docker"}

    def test_session_start_project_only(self, runner, index_root, monkeypatch):
        """mx session start --project creates session."""
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))
        result = runner.invoke(cli, ["session", "start", "--project=myapp"])
        assert result.exit_code == 0

        ctx = load_session(index_root)
        assert ctx.project == "myapp"

    def test_session_start_both(self, runner, index_root, monkeypatch):
        """mx session start with both tags and project."""
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))
        result = runner.invoke(cli, ["session", "start", "--tags=api", "--project=backend"])
        assert result.exit_code == 0

        ctx = load_session(index_root)
        assert ctx.tags == ["api"]
        assert ctx.project == "backend"

    def test_session_start_requires_argument(self, runner, index_root, monkeypatch):
        """mx session start requires at least one of --tags or --project."""
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))
        result = runner.invoke(cli, ["session", "start"])
        assert result.exit_code == 1
        assert "At least one of --tags or --project required" in result.output

    def test_session_set_replace_tags(self, runner, index_root, monkeypatch):
        """mx session set --tags replaces existing tags."""
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))
        save_session(SessionContext(tags=["old1", "old2"]), index_root)

        result = runner.invoke(cli, ["session", "set", "--tags=new1,new2"])
        assert result.exit_code == 0

        ctx = load_session(index_root)
        assert set(ctx.tags) == {"new1", "new2"}

    def test_session_set_add_tags(self, runner, index_root, monkeypatch):
        """mx session set --add-tags appends to existing."""
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))
        save_session(SessionContext(tags=["existing"]), index_root)

        result = runner.invoke(cli, ["session", "set", "--add-tags=new1,new2"])
        assert result.exit_code == 0

        ctx = load_session(index_root)
        assert set(ctx.tags) == {"existing", "new1", "new2"}

    def test_session_set_project(self, runner, index_root, monkeypatch):
        """mx session set --project updates project."""
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))
        save_session(SessionContext(project="old"), index_root)

        result = runner.invoke(cli, ["session", "set", "--project=new"])
        assert result.exit_code == 0

        ctx = load_session(index_root)
        assert ctx.project == "new"

    def test_session_clear_active(self, runner, index_root, monkeypatch):
        """mx session clear removes active session."""
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))
        save_session(SessionContext(tags=["test"]), index_root)

        result = runner.invoke(cli, ["session", "clear"])
        assert result.exit_code == 0
        assert "cleared" in result.output.lower()

        ctx = load_session(index_root)
        assert ctx.is_empty()

    def test_session_clear_empty(self, runner, index_root, monkeypatch):
        """mx session clear with no active session."""
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))
        result = runner.invoke(cli, ["session", "clear"])
        assert result.exit_code == 0
        assert "No active session" in result.output

    def test_session_default_shows_status(self, runner, index_root, monkeypatch):
        """mx session (no subcommand) shows status."""
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))
        result = runner.invoke(cli, ["session"])
        assert result.exit_code == 0
        # Should show same output as 'mx session show'
        assert "session" in result.output.lower()

    def test_session_json_flag_on_commands(self, runner, index_root, monkeypatch):
        """All session commands support --json."""
        monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

        # start with --json
        result = runner.invoke(cli, ["session", "start", "--tags=test", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["action"] == "started"

        # set with --json
        result = runner.invoke(cli, ["session", "set", "--project=proj", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["action"] == "updated"

        # clear with --json
        result = runner.invoke(cli, ["session", "clear", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["action"] == "cleared"


# ─────────────────────────────────────────────────────────────────────────────
# Search Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSearchSessionIntegration:
    """Tests for session integration with search command."""

    def test_no_session_flag_exists(self, runner):
        """Search command has --no-session flag."""
        result = runner.invoke(cli, ["search", "--help"])
        assert "--no-session" in result.output
        assert "Ignore session context" in result.output
