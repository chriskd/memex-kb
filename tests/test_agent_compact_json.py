import json

from click.testing import CliRunner

from conftest import create_entry
from memex.cli import cli


def test_prime_json_compact_omits_content_and_is_single_line(
    runner: CliRunner, multi_kb: dict
) -> None:
    create_entry(multi_kb["project_kb"], "general/project-guide.md", "Project Guide", "Hello", ["x"])

    result = runner.invoke(cli, ["prime", "--json", "--compact"])
    assert result.exit_code == 0, result.output

    out = result.output.strip()
    assert "\n" not in out
    data = json.loads(out)
    assert "content" not in data
    assert isinstance(data.get("entries"), list)


def test_session_context_json_compact_omits_content_and_is_single_line(
    runner: CliRunner, multi_kb: dict
) -> None:
    create_entry(multi_kb["project_kb"], "general/project-guide.md", "Project Guide", "Hello", ["x"])

    result = runner.invoke(cli, ["session-context", "--json", "--compact"])
    assert result.exit_code == 0, result.output

    out = result.output.strip()
    assert "\n" not in out
    data = json.loads(out)
    assert "content" not in data
    assert data["project"]


def test_session_context_json_includes_content_when_not_compact(
    runner: CliRunner, multi_kb: dict
) -> None:
    create_entry(multi_kb["project_kb"], "general/project-guide.md", "Project Guide", "Hello", ["x"])

    result = runner.invoke(cli, ["session-context", "--json"])
    assert result.exit_code == 0, result.output

    data = json.loads(result.output)
    assert "content" in data


def test_compact_max_bytes_truncates_entries(runner: CliRunner, multi_kb: dict) -> None:
    for i in range(10):
        create_entry(
            multi_kb["project_kb"],
            f"general/project-guide-{i}.md",
            f"Project Guide {i} " + ("x" * 50),
            "Hello",
            ["x"],
        )

    baseline = runner.invoke(cli, ["prime", "--json", "--compact", "--max-entries", "10"])
    assert baseline.exit_code == 0, baseline.output
    baseline_bytes = len(baseline.output.strip().encode("utf-8"))
    assert baseline_bytes > 0

    max_bytes = max(150, baseline_bytes - 50)
    bounded = runner.invoke(
        cli,
        [
            "prime",
            "--json",
            "--compact",
            "--max-entries",
            "10",
            "--max-bytes",
            str(max_bytes),
        ],
    )
    assert bounded.exit_code == 0, bounded.output

    out = bounded.output.strip()
    assert len(out.encode("utf-8")) <= max_bytes
    data = json.loads(out)
    assert "truncated" in data
    assert isinstance(data.get("entries"), list)
    assert len(data["entries"]) < 10

