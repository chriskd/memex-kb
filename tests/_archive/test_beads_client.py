"""Tests for beads_client CLI wrapper."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from memex import beads_client


def test_find_beads_db_with_project_root(tmp_path: Path) -> None:
    """Finds beads DB from a project root."""
    project_root = tmp_path / "project"
    beads_dir = project_root / ".beads"
    beads_dir.mkdir(parents=True)
    db_path = beads_dir / "beads.db"
    db_path.write_text("db")

    result = beads_client.find_beads_db(project_root)

    assert result is not None
    assert result.path == project_root
    assert result.db_path == db_path


def test_find_beads_db_with_beads_dir(tmp_path: Path) -> None:
    """Finds beads DB when pointed at a .beads directory."""
    project_root = tmp_path / "project"
    beads_dir = project_root / ".beads"
    beads_dir.mkdir(parents=True)
    db_path = beads_dir / "beads.db"
    db_path.write_text("db")

    result = beads_client.find_beads_db(beads_dir)

    assert result is not None
    assert result.path == project_root
    assert result.db_path == db_path


def test_find_beads_db_missing_returns_none(tmp_path: Path) -> None:
    """Returns None when no beads DB exists."""
    project_root = tmp_path / "project"
    project_root.mkdir()

    result = beads_client.find_beads_db(project_root)

    assert result is None


def test_run_bd_command_success_parses_json(tmp_path: Path) -> None:
    """Parses JSON when bd returns success."""
    db_path = tmp_path / "beads.db"

    fake_result = SimpleNamespace(returncode=0, stdout='{"ok": true}', stderr="")
    with patch("memex.beads_client.subprocess.run", return_value=fake_result) as run:
        result = beads_client.run_bd_command(db_path, ["list"])

    assert result == {"ok": True}
    run.assert_called_once()
    called_args = run.call_args[0][0]
    assert called_args[:4] == ["bd", "--db", str(db_path), "--json"]
    assert "--readonly" in called_args


def test_run_bd_command_empty_output_returns_list(tmp_path: Path) -> None:
    """Empty stdout yields empty list."""
    db_path = tmp_path / "beads.db"
    fake_result = SimpleNamespace(returncode=0, stdout="  \n", stderr="")
    with patch("memex.beads_client.subprocess.run", return_value=fake_result):
        result = beads_client.run_bd_command(db_path, ["list"])

    assert result == []


def test_run_bd_command_nonzero_returncode_returns_none(tmp_path: Path) -> None:
    """Non-zero return codes return None."""
    db_path = tmp_path / "beads.db"
    fake_result = SimpleNamespace(returncode=1, stdout="", stderr="boom")
    with patch("memex.beads_client.subprocess.run", return_value=fake_result):
        result = beads_client.run_bd_command(db_path, ["list"])

    assert result is None


def test_run_bd_command_timeout_returns_none(tmp_path: Path) -> None:
    """Timeouts return None."""
    db_path = tmp_path / "beads.db"
    with patch(
        "memex.beads_client.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="bd", timeout=1),
    ):
        result = beads_client.run_bd_command(db_path, ["list"])

    assert result is None


def test_run_bd_command_invalid_json_returns_none(tmp_path: Path) -> None:
    """Invalid JSON returns None."""
    db_path = tmp_path / "beads.db"
    fake_result = SimpleNamespace(returncode=0, stdout="{bad json", stderr="")
    with patch("memex.beads_client.subprocess.run", return_value=fake_result):
        result = beads_client.run_bd_command(db_path, ["list"])

    assert result is None


def test_run_bd_command_missing_cli_returns_none(tmp_path: Path) -> None:
    """Missing bd CLI returns None."""
    db_path = tmp_path / "beads.db"
    with patch("memex.beads_client.subprocess.run", side_effect=FileNotFoundError):
        result = beads_client.run_bd_command(db_path, ["list"])

    assert result is None


def test_list_show_status_comments_wrappers(tmp_path: Path) -> None:
    """Wrapper functions coerce return types."""
    db_path = tmp_path / "beads.db"

    with patch("memex.beads_client.run_bd_command", return_value=[{"id": "x"}]):
        assert beads_client.list_issues(db_path) == [{"id": "x"}]

    with patch("memex.beads_client.run_bd_command", return_value={"id": "x"}):
        assert beads_client.show_issue(db_path, "x") == {"id": "x"}

    with patch("memex.beads_client.run_bd_command", return_value=[{"id": "y"}]):
        assert beads_client.show_issue(db_path, "y") == {"id": "y"}

    with patch("memex.beads_client.run_bd_command", return_value=None):
        assert beads_client.show_issue(db_path, "z") is None

    with patch("memex.beads_client.run_bd_command", return_value={"ok": True}):
        assert beads_client.get_status(db_path) == {"ok": True}

    with patch("memex.beads_client.run_bd_command", return_value=None):
        assert beads_client.get_status(db_path) is None

    with patch("memex.beads_client.run_bd_command", return_value=[{"body": "hi"}]):
        assert beads_client.get_comments(db_path, "x") == [{"body": "hi"}]

    with patch("memex.beads_client.run_bd_command", return_value=None):
        assert beads_client.get_comments(db_path, "x") == []
