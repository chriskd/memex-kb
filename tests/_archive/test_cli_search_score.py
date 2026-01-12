"""Tests for CLI search score features (--min-score, confidence labels)."""

from click.testing import CliRunner

from memex.cli import cli, _score_confidence, _score_confidence_short


class TestScoreConfidenceHelpers:
    """Tests for score confidence helper functions."""

    def test_score_confidence_high(self):
        """Scores >= 0.7 return 'high'."""
        assert _score_confidence(0.7) == "high"
        assert _score_confidence(0.85) == "high"
        assert _score_confidence(1.0) == "high"

    def test_score_confidence_moderate(self):
        """Scores 0.4-0.7 return 'moderate'."""
        assert _score_confidence(0.4) == "moderate"
        assert _score_confidence(0.55) == "moderate"
        assert _score_confidence(0.69) == "moderate"

    def test_score_confidence_weak(self):
        """Scores < 0.4 return 'weak'."""
        assert _score_confidence(0.0) == "weak"
        assert _score_confidence(0.2) == "weak"
        assert _score_confidence(0.39) == "weak"

    def test_score_confidence_short_high(self):
        """Short confidence for high scores."""
        assert _score_confidence_short(0.7) == "high"
        assert _score_confidence_short(1.0) == "high"

    def test_score_confidence_short_moderate(self):
        """Short confidence for moderate scores is 'mod'."""
        assert _score_confidence_short(0.5) == "mod"
        assert _score_confidence_short(0.69) == "mod"

    def test_score_confidence_short_weak(self):
        """Short confidence for weak scores."""
        assert _score_confidence_short(0.1) == "weak"
        assert _score_confidence_short(0.39) == "weak"


class TestMinScoreValidation:
    """Tests for --min-score option validation."""

    def test_min_score_below_zero_rejected(self):
        """--min-score below 0 should be rejected."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "test", "--min-score", "-0.1"])
        assert result.exit_code != 0
        assert "not in the range" in result.output or "Invalid" in result.output

    def test_min_score_above_one_rejected(self):
        """--min-score above 1.0 should be rejected."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "test", "--min-score", "1.5"])
        assert result.exit_code != 0
        assert "not in the range" in result.output or "Invalid" in result.output

    def test_min_score_valid_range_accepted(self):
        """--min-score in 0.0-1.0 range should be accepted (validation passes)."""
        runner = CliRunner()
        # We just check that the validation passes - command may fail for other reasons
        result = runner.invoke(cli, ["search", "test", "--min-score", "0.5"])
        assert "not in the range" not in result.output

    def test_min_score_zero_accepted(self):
        """--min-score 0.0 should be accepted."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "test", "--min-score", "0.0"])
        assert "not in the range" not in result.output

    def test_min_score_one_accepted(self):
        """--min-score 1.0 should be accepted."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "test", "--min-score", "1.0"])
        assert "not in the range" not in result.output


class TestSearchHelpText:
    """Tests for search command help text."""

    def test_search_help_includes_score_semantics(self):
        """Search help should document score semantics."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "--help"])
        assert result.exit_code == 0
        # Check for score documentation
        assert "0.0-1.0" in result.output
        assert "High confidence" in result.output or ">= 0.7" in result.output
        assert "Moderate" in result.output or "0.4-0.7" in result.output
        assert "Weak" in result.output or "< 0.4" in result.output

    def test_search_help_includes_min_score_option(self):
        """Search help should include --min-score option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["search", "--help"])
        assert result.exit_code == 0
        assert "--min-score" in result.output
