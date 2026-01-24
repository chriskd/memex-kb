"""Tests for session context persistence."""

import json
from pathlib import Path

import pytest

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


# NOTE: CLI session command tests removed as the 'mx session' subcommand group
# was removed from the CLI. The session module itself is still used internally.
