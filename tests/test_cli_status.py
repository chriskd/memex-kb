"""Tests for CLI default status output.

Verifies that running `mx` without arguments shows status instead of help,
and that the status output includes correct information.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from memex.cli import cli, _show_status, _get_recent_entries_for_status, _output_status


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_kb_root(tmp_path):
    """Create a temporary KB root directory."""
    kb_root = tmp_path / "kb"
    kb_root.mkdir()
    return kb_root


@pytest.fixture
def mock_context():
    """Create a mock KBContext object."""
    context = Mock()
    context.source_file = Path(".kbcontext")
    context.primary = "projects/myproject"
    context.default_tags = ["python", "testing"]
    context.get_project_name.return_value = "myproject"
    return context


@pytest.fixture
def mock_detected_context():
    """Create a mock DetectedContext object."""
    detected = Mock()
    detected.project_name = "detected-project"
    return detected


@pytest.fixture
def mock_recent_entries():
    """Create sample recent entries."""
    return [
        {
            "path": "projects/myproject/feature.md",
            "title": "New Feature Implementation",
            "activity_date": "2026-01-05T10:30:00",
            "activity_type": "created",
            "tags": ["myproject", "feature"],
        },
        {
            "path": "tooling/testing.md",
            "title": "Testing Guidelines",
            "activity_date": "2026-01-04T15:20:00",
            "activity_type": "updated",
            "tags": ["testing"],
        },
        {
            "path": "notes/meeting-2026-01-03.md",
            "title": "Team Meeting Notes",
            "activity_date": "2026-01-03T09:00:00",
            "activity_type": "created",
            "tags": ["meeting"],
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Happy Path Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNoArgsShowsStatus:
    """Test that `mx` with no arguments shows status, not help."""

    @patch("memex.cli._show_status")
    def test_no_args_calls_show_status(self, mock_show_status, runner):
        """Running mx without arguments calls _show_status() instead of showing help.

        Category: Happy path
        Purpose: Verify the new default behavior is invoked
        """
        result = runner.invoke(cli, [])

        # Should call _show_status
        mock_show_status.assert_called_once()
        assert result.exit_code == 0

    @patch("memex.config.get_kb_root")
    @patch("memex.context.get_kb_context")
    @patch("memex.cli._get_recent_entries_for_status")
    def test_no_args_shows_status_not_help(
        self, mock_get_entries, mock_get_context, mock_get_root, runner, mock_kb_root
    ):
        """Running mx without arguments shows status output, not help text.

        Category: Happy path
        Purpose: Verify output is status, not help
        """
        mock_get_root.return_value = mock_kb_root
        mock_get_context.return_value = None
        mock_get_entries.return_value = []

        result = runner.invoke(cli, [])

        # Should show status output
        assert "Memex Knowledge Base" in result.output
        assert result.exit_code == 0

        # Should NOT show help
        assert "Usage:" not in result.output


class TestStatusWithKBConfigured:
    """Test status output when KB is properly configured."""

    @patch("memex.config.get_kb_root")
    @patch("memex.context.get_kb_context")
    @patch("memex.cli._get_recent_entries_for_status")
    def test_status_shows_kb_root(
        self, mock_get_entries, mock_get_context, mock_get_root, runner, mock_kb_root
    ):
        """Status displays KB root path when configured.

        Category: Happy path
        Purpose: Verify KB root is shown in output
        """
        mock_get_root.return_value = mock_kb_root
        mock_get_context.return_value = None
        mock_get_entries.return_value = []

        result = runner.invoke(cli, [])

        assert f"KB Root: {mock_kb_root}" in result.output
        assert result.exit_code == 0

    @patch("memex.config.get_kb_root")
    @patch("memex.context.get_kb_context")
    @patch("memex.cli._get_recent_entries_for_status")
    def test_status_shows_context_info(
        self, mock_get_entries, mock_get_context, mock_get_root,
        runner, mock_kb_root, mock_context
    ):
        """Status displays context information when .kbcontext exists.

        Category: Happy path
        Purpose: Verify context details are displayed
        """
        mock_get_root.return_value = mock_kb_root
        mock_get_context.return_value = mock_context
        mock_get_entries.return_value = []

        result = runner.invoke(cli, [])

        assert "Context: .kbcontext" in result.output
        assert "Primary: projects/myproject" in result.output
        assert "Tags:    python, testing" in result.output
        assert result.exit_code == 0

    @patch("memex.config.get_kb_root")
    @patch("memex.context.get_kb_context")
    @patch("memex.context.detect_project_context")
    @patch("memex.cli._get_recent_entries_for_status")
    def test_status_shows_detected_project(
        self, mock_get_entries, mock_detect, mock_get_context,
        mock_get_root, runner, mock_kb_root, mock_detected_context
    ):
        """Status displays auto-detected project when no .kbcontext exists.

        Category: Happy path
        Purpose: Verify auto-detection is shown
        """
        mock_get_root.return_value = mock_kb_root
        mock_get_context.return_value = None
        mock_detect.return_value = mock_detected_context
        mock_get_entries.return_value = []

        result = runner.invoke(cli, [])

        assert "Project: detected-project (auto-detected)" in result.output
        assert "Run 'mx context init' to configure" in result.output
        assert result.exit_code == 0

    @patch("memex.config.get_kb_root")
    @patch("memex.context.get_kb_context")
    @patch("memex.cli._get_recent_entries_for_status")
    def test_status_shows_recent_entries(
        self, mock_get_entries, mock_get_context, mock_get_root,
        runner, mock_kb_root, mock_recent_entries
    ):
        """Status displays recent entries section with correct formatting.

        Category: Happy path
        Purpose: Verify recent entries are displayed correctly
        """
        mock_get_root.return_value = mock_kb_root
        mock_get_context.return_value = None
        mock_get_entries.return_value = mock_recent_entries

        result = runner.invoke(cli, [])

        assert "Recent Entries" in result.output
        assert "NEW 2026-01-05" in result.output
        assert "projects/myproject/feature.md" in result.output
        assert "New Feature Implementation" in result.output
        assert "UPD 2026-01-04" in result.output
        assert "tooling/testing.md" in result.output
        assert result.exit_code == 0

    @patch("memex.config.get_kb_root")
    @patch("memex.context.get_kb_context")
    @patch("memex.cli._get_recent_entries_for_status")
    def test_status_shows_suggested_commands_with_content(
        self, mock_get_entries, mock_get_context, mock_get_root,
        runner, mock_kb_root, mock_recent_entries
    ):
        """Status shows relevant commands when KB has content.

        Category: Happy path
        Purpose: Verify suggested commands for populated KB
        """
        mock_get_root.return_value = mock_kb_root
        mock_get_context.return_value = None
        mock_get_entries.return_value = mock_recent_entries

        result = runner.invoke(cli, [])

        assert "Commands" in result.output
        assert 'mx search "query"' in result.output
        assert "mx whats-new" in result.output
        assert "mx tree" in result.output
        assert "mx --help" in result.output
        assert result.exit_code == 0


# ─────────────────────────────────────────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────────────────────────────────────────


class TestStatusEdgeCases:
    """Test status output edge cases."""

    @patch("memex.config.get_kb_root")
    @patch("memex.context.get_kb_context")
    @patch("memex.cli._get_recent_entries_for_status")
    def test_status_with_empty_kb(
        self, mock_get_entries, mock_get_context, mock_get_root, runner, mock_kb_root
    ):
        """Status shows helpful commands when KB exists but has no entries.

        Category: Edge case
        Purpose: Verify output for empty KB
        """
        mock_get_root.return_value = mock_kb_root
        mock_get_context.return_value = None
        mock_get_entries.return_value = []

        result = runner.invoke(cli, [])

        assert "Memex Knowledge Base" in result.output
        assert f"KB Root: {mock_kb_root}" in result.output
        # Should suggest add command for empty KB
        assert 'mx add --title="..."' in result.output
        assert result.exit_code == 0

    @patch("memex.config.get_kb_root")
    @patch("memex.context.get_kb_context")
    @patch("memex.context.detect_project_context")
    @patch("memex.cli._get_recent_entries_for_status")
    def test_status_with_no_context_no_detection(
        self, mock_get_entries, mock_detect, mock_get_context,
        mock_get_root, runner, mock_kb_root
    ):
        """Status handles case where there's no context and no auto-detection.

        Category: Edge case
        Purpose: Verify behavior when project detection fails
        """
        mock_get_root.return_value = mock_kb_root
        mock_get_context.return_value = None
        mock_detect.return_value = None
        mock_get_entries.return_value = []

        result = runner.invoke(cli, [])

        assert "Context: (none)" in result.output
        assert "mx context init" in result.output
        assert result.exit_code == 0

    def test_status_truncates_long_paths(self, runner):
        """Status truncates long entry paths to fit display.

        Category: Edge case
        Purpose: Verify path truncation for readability
        """
        long_path = "projects/very/deep/nested/structure/with/many/levels/file.md"
        entries = [
            {
                "path": long_path,
                "title": "Test Entry",
                "activity_date": "2026-01-05T10:30:00",
                "activity_type": "created",
            }
        ]

        with patch("memex.config.get_kb_root") as mock_get_root, \
             patch("memex.context.get_kb_context") as mock_get_context, \
             patch("memex.cli._get_recent_entries_for_status") as mock_get_entries:

            mock_get_root.return_value = Path("/tmp/kb")
            mock_get_context.return_value = None
            mock_get_entries.return_value = entries

            result = runner.invoke(cli, [])

            # Should show truncated path with ellipsis
            assert "..." in result.output
            assert result.exit_code == 0

    def test_status_limits_entries_to_five(self, runner):
        """Status limits recent entries display to 5 items.

        Category: Edge case
        Purpose: Verify entry count limit
        """
        # Create 10 entries
        entries = [
            {
                "path": f"entry{i}.md",
                "title": f"Entry {i}",
                "activity_date": f"2026-01-0{i % 9 + 1}T10:30:00",
                "activity_type": "created",
            }
            for i in range(10)
        ]

        with patch("memex.config.get_kb_root") as mock_get_root, \
             patch("memex.context.get_kb_context") as mock_get_context, \
             patch("memex.cli._get_recent_entries_for_status") as mock_get_entries:

            mock_get_root.return_value = Path("/tmp/kb")
            mock_get_context.return_value = None
            mock_get_entries.return_value = entries

            result = runner.invoke(cli, [])

            # Count how many entry paths appear
            entry_count = sum(1 for i in range(10) if f"entry{i}.md" in result.output)
            assert entry_count <= 5
            assert result.exit_code == 0


# ─────────────────────────────────────────────────────────────────────────────
# Error Conditions
# ─────────────────────────────────────────────────────────────────────────────


class TestStatusErrorHandling:
    """Test status output when things go wrong."""

    @patch("memex.config.get_kb_root")
    @patch("memex.context.get_kb_context")
    def test_status_when_kb_not_configured(
        self, mock_get_context, mock_get_root, runner
    ):
        """Status shows helpful message when KB is not configured.

        Category: Error condition
        Purpose: Verify graceful handling of missing KB config
        """
        from memex.config import ConfigurationError

        mock_get_root.side_effect = ConfigurationError("KB not configured")
        mock_get_context.return_value = None

        result = runner.invoke(cli, [])

        assert "KB Root: NOT CONFIGURED" in result.output
        assert "Set MEMEX_KB_ROOT and MEMEX_INDEX_ROOT" in result.output
        assert "mx --help" in result.output
        assert result.exit_code == 0

    @patch("memex.config.get_kb_root")
    @patch("memex.context.get_kb_context")
    @patch("memex.context.detect_project_context")
    @patch("memex.cli._get_recent_entries_for_status")
    def test_status_resilient_to_entry_fetch_errors(
        self, mock_get_entries, mock_detect, mock_get_context, mock_get_root, runner, mock_kb_root
    ):
        """Status continues gracefully even if fetching entries fails.

        Category: Error condition
        Purpose: Verify resilience to data fetch failures
        """
        mock_get_root.return_value = mock_kb_root
        mock_get_context.return_value = None
        mock_detect.return_value = None
        # Simulate error in getting entries - returns empty list (fail silently)
        mock_get_entries.return_value = []

        result = runner.invoke(cli, [])

        # Should still show status with KB root
        assert "Memex Knowledge Base" in result.output
        assert f"KB Root: {mock_kb_root}" in result.output
        # Should still show commands
        assert "Commands" in result.output
        assert result.exit_code == 0


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests for Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


class TestGetRecentEntriesForStatus:
    """Test _get_recent_entries_for_status() helper function."""

    @patch("memex.cli.run_async")
    def test_gets_project_specific_entries_when_available(
        self, mock_run_async, mock_kb_root, mock_recent_entries
    ):
        """Gets project-specific entries when project name provided.

        Category: Happy path
        Purpose: Verify project filtering works
        """
        mock_run_async.return_value = mock_recent_entries

        result = _get_recent_entries_for_status(mock_kb_root, "myproject", limit=5)

        assert result == mock_recent_entries
        assert mock_run_async.called

    @patch("memex.cli.run_async")
    def test_falls_back_to_all_entries_if_no_project_entries(
        self, mock_run_async, mock_kb_root, mock_recent_entries
    ):
        """Falls back to all entries if project has no recent entries.

        Category: Edge case
        Purpose: Verify fallback behavior
        """
        # First call (project-specific) returns empty, second call (all) returns entries
        mock_run_async.side_effect = [[], mock_recent_entries]

        result = _get_recent_entries_for_status(mock_kb_root, "myproject", limit=5)

        assert result == mock_recent_entries
        assert mock_run_async.call_count == 2

    @patch("memex.cli.run_async")
    def test_returns_empty_list_on_error(self, mock_run_async, mock_kb_root):
        """Returns empty list if entry fetching fails (fail silently).

        Category: Error condition
        Purpose: Verify resilient error handling
        """
        mock_run_async.side_effect = Exception("Database error")

        result = _get_recent_entries_for_status(mock_kb_root, "myproject")

        assert result == []


class TestOutputStatus:
    """Test _output_status() helper function."""

    def test_output_includes_all_sections(self, mock_kb_root, mock_recent_entries):
        """Output includes header, context, entries, and commands sections.

        Category: Happy path
        Purpose: Verify complete output structure
        """
        from io import StringIO
        from unittest.mock import patch

        output = StringIO()

        with patch("click.echo", side_effect=lambda x: output.write(x + "\n")):
            _output_status(
                kb_root=mock_kb_root,
                context=None,
                detected=None,
                entries=mock_recent_entries,
                project_name=None,
            )

        result = output.getvalue()

        assert "Memex Knowledge Base" in result
        assert "=" * 40 in result
        assert f"KB Root: {mock_kb_root}" in result
        assert "Recent Entries" in result
        assert "Commands" in result

    def test_output_without_kb_root(self):
        """Output shows configuration instructions when KB not configured.

        Category: Error condition
        Purpose: Verify helpful error messaging
        """
        from io import StringIO
        from unittest.mock import patch

        output = StringIO()

        with patch("click.echo", side_effect=lambda x: output.write(x + "\n")):
            _output_status(
                kb_root=None,
                context=None,
                detected=None,
                entries=[],
                project_name=None,
            )

        result = output.getvalue()

        assert "KB Root: NOT CONFIGURED" in result
        assert "Set MEMEX_KB_ROOT" in result
        assert "mx --help" in result

    def test_output_shows_project_in_header(self, mock_kb_root, mock_recent_entries):
        """Output includes project name in recent entries header when applicable.

        Category: Happy path
        Purpose: Verify project context is shown
        """
        from io import StringIO
        from unittest.mock import patch

        output = StringIO()

        with patch("click.echo", side_effect=lambda x: output.write(x + "\n")):
            _output_status(
                kb_root=mock_kb_root,
                context=None,
                detected=None,
                entries=mock_recent_entries,
                project_name="myproject",
            )

        result = output.getvalue()

        assert "Recent Entries (myproject)" in result


# ─────────────────────────────────────────────────────────────────────────────
# Boundary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestStatusBoundaryConditions:
    """Test boundary conditions for status output."""

    def test_status_with_exactly_five_entries(self, runner, tmp_path):
        """Status displays exactly 5 entries correctly.

        Category: Boundary value
        Purpose: Verify limit boundary behavior
        """
        kb_root = tmp_path / "kb"
        kb_root.mkdir()  # Ensure KB directory exists

        entries = [
            {
                "path": f"entry{i}.md",
                "title": f"Entry {i}",
                "activity_date": f"2026-01-0{i+1}T10:30:00",
                "activity_type": "created",
            }
            for i in range(5)
        ]

        with patch("memex.config.get_kb_root") as mock_get_root, \
             patch("memex.context.get_kb_context") as mock_get_context, \
             patch("memex.context.detect_project_context") as mock_detect, \
             patch("memex.cli._get_recent_entries_for_status") as mock_get_entries:

            mock_get_root.return_value = kb_root
            mock_get_context.return_value = None
            mock_detect.return_value = None
            mock_get_entries.return_value = entries

            result = runner.invoke(cli, [])

            for i in range(5):
                assert f"entry{i}.md" in result.output
            assert result.exit_code == 0

    def test_status_with_title_exactly_40_chars(self, runner, tmp_path):
        """Status handles title that is exactly at truncation boundary.

        Category: Boundary value
        Purpose: Verify title truncation boundary
        """
        kb_root = tmp_path / "kb"
        kb_root.mkdir()

        title_40_chars = "A" * 40
        entries = [
            {
                "path": "test.md",
                "title": title_40_chars,
                "activity_date": "2026-01-05T10:30:00",
                "activity_type": "created",
            }
        ]

        with patch("memex.config.get_kb_root") as mock_get_root, \
             patch("memex.context.get_kb_context") as mock_get_context, \
             patch("memex.context.detect_project_context") as mock_detect, \
             patch("memex.cli._get_recent_entries_for_status") as mock_get_entries:

            mock_get_root.return_value = kb_root
            mock_get_context.return_value = None
            mock_detect.return_value = None
            mock_get_entries.return_value = entries

            result = runner.invoke(cli, [])

            assert title_40_chars in result.output
            assert result.exit_code == 0

    def test_status_with_path_exactly_35_chars(self, runner, tmp_path):
        """Status handles path at truncation boundary (35 chars = no truncation, 36+ = truncation).

        Category: Boundary value
        Purpose: Verify path truncation boundary
        """
        kb_root = tmp_path / "kb"
        kb_root.mkdir()

        # Test exactly 35 characters - should NOT be truncated
        # 1 (p) + 31 (x's) + 3 (.md) = 35
        path_35_chars = "p" + ("x" * 31) + ".md"
        # Test exactly 36 characters - should be truncated
        # 1 (p) + 32 (x's) + 3 (.md) = 36
        path_36_chars = "p" + ("x" * 32) + ".md"

        entries = [
            {
                "path": path_35_chars,
                "title": "Test 35",
                "activity_date": "2026-01-05T10:30:00",
                "activity_type": "created",
            },
            {
                "path": path_36_chars,
                "title": "Test 36",
                "activity_date": "2026-01-04T10:30:00",
                "activity_type": "created",
            }
        ]

        with patch("memex.config.get_kb_root") as mock_get_root, \
             patch("memex.context.get_kb_context") as mock_get_context, \
             patch("memex.context.detect_project_context") as mock_detect, \
             patch("memex.cli._get_recent_entries_for_status") as mock_get_entries:

            mock_get_root.return_value = kb_root
            mock_get_context.return_value = None
            mock_detect.return_value = None
            mock_get_entries.return_value = entries

            result = runner.invoke(cli, [])

            # 35 chars should NOT be truncated
            assert path_35_chars in result.output
            # 36 chars SHOULD be truncated (starts with ...)
            assert "..." in result.output
            # The truncated version should show the last 32 characters
            assert path_36_chars[-32:] in result.output
            assert result.exit_code == 0
