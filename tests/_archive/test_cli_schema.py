"""Tests for mx schema command."""

import json
import subprocess
import sys

import pytest


def run_mx(*args: str) -> tuple[int, str, str]:
    """Run mx CLI command and return (exit_code, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-m", "memex.cli", *args],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


class TestSchemaCommand:
    """Tests for the schema command output."""

    def test_schema_outputs_valid_json(self):
        """Schema command outputs valid JSON."""
        exit_code, stdout, stderr = run_mx("schema")

        assert exit_code == 0
        assert stderr == ""

        # Should be valid JSON
        schema = json.loads(stdout)
        assert isinstance(schema, dict)

    def test_schema_has_required_top_level_keys(self):
        """Schema contains all required top-level keys."""
        exit_code, stdout, _ = run_mx("schema")

        schema = json.loads(stdout)

        assert "version" in schema
        assert "description" in schema
        assert "commands" in schema
        assert "global_options" in schema
        assert "workflows" in schema

    def test_schema_version_format(self):
        """Schema version follows semantic versioning."""
        exit_code, stdout, _ = run_mx("schema")

        schema = json.loads(stdout)

        # Version should be in semver format (e.g., "0.1.0")
        version = schema["version"]
        parts = version.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)

    def test_schema_contains_core_commands(self):
        """Schema contains all core commands."""
        exit_code, stdout, _ = run_mx("schema")

        schema = json.loads(stdout)
        commands = schema["commands"]

        core_commands = [
            "search", "get", "add", "append", "update", "patch",
            "delete", "list", "tree", "tags", "health", "hubs",
            "suggest-links", "whats-new", "prime", "reindex",
            "context", "schema"
        ]

        for cmd in core_commands:
            assert cmd in commands, f"Missing command: {cmd}"

    def test_command_has_required_fields(self):
        """Each command has required fields."""
        exit_code, stdout, _ = run_mx("schema")

        schema = json.loads(stdout)

        required_fields = ["description", "aliases", "options", "related", "common_mistakes", "examples"]

        for cmd_name, cmd_data in schema["commands"].items():
            for field in required_fields:
                assert field in cmd_data, f"Command '{cmd_name}' missing field: {field}"

    def test_patch_command_has_common_mistakes(self):
        """Patch command includes common mistakes for agent guidance."""
        exit_code, stdout, _ = run_mx("schema")

        schema = json.loads(stdout)
        patch = schema["commands"]["patch"]

        assert len(patch["common_mistakes"]) > 0
        # Should warn about --find without --replace
        assert "--find without --replace" in patch["common_mistakes"]

    def test_patch_command_has_exit_codes(self):
        """Patch command documents exit codes."""
        exit_code, stdout, _ = run_mx("schema")

        schema = json.loads(stdout)
        patch = schema["commands"]["patch"]

        assert "exit_codes" in patch
        assert "0" in patch["exit_codes"]  # Success
        assert "1" in patch["exit_codes"]  # Text not found

    def test_command_option_with_command_flag(self):
        """--command flag shows schema for specific command only."""
        exit_code, stdout, stderr = run_mx("schema", "--command=patch")

        assert exit_code == 0
        assert stderr == ""

        result = json.loads(stdout)

        # Should have command name at top level
        assert result["command"] == "patch"
        assert "description" in result
        assert "options" in result
        assert "common_mistakes" in result

        # Should NOT have other commands
        assert "commands" not in result

    def test_command_flag_short_form(self):
        """Short form -c works for command filter."""
        exit_code, stdout, _ = run_mx("schema", "-c", "search")

        assert exit_code == 0

        result = json.loads(stdout)
        assert result["command"] == "search"

    def test_command_flag_unknown_command(self):
        """Unknown command returns error."""
        exit_code, stdout, stderr = run_mx("schema", "--command=nonexistent")

        assert exit_code == 1
        assert "Unknown command" in stderr
        assert "nonexistent" in stderr

    def test_compact_flag(self):
        """--compact flag produces minimal output."""
        exit_code, stdout, _ = run_mx("schema", "--compact")

        assert exit_code == 0

        result = json.loads(stdout)

        assert "version" in result
        assert "commands" in result

        # Commands should have simplified structure
        for cmd_name, cmd_data in result["commands"].items():
            assert "description" in cmd_data
            assert "options" in cmd_data
            # Options should be just names, not full objects
            if cmd_data["options"]:
                assert isinstance(cmd_data["options"][0], str)

        # Should NOT have workflows in compact mode
        assert "workflows" not in result

    def test_search_command_options_documented(self):
        """Search command has all options documented."""
        exit_code, stdout, _ = run_mx("schema")

        schema = json.loads(stdout)
        search = schema["commands"]["search"]

        option_names = [opt["name"] for opt in search["options"]]

        expected_options = [
            "--tags", "--mode", "--limit", "--min-score",
            "--content", "--strict", "--terse", "--full-titles", "--json"
        ]

        for opt in expected_options:
            assert opt in option_names, f"Missing option: {opt}"

    def test_related_commands_are_valid(self):
        """Related commands reference valid command names."""
        exit_code, stdout, _ = run_mx("schema")

        schema = json.loads(stdout)
        all_commands = set(schema["commands"].keys())

        for cmd_name, cmd_data in schema["commands"].items():
            for related in cmd_data["related"]:
                assert related in all_commands, f"Command '{cmd_name}' references unknown related command: {related}"

    def test_workflows_have_required_fields(self):
        """Workflows have description and steps."""
        exit_code, stdout, _ = run_mx("schema")

        schema = json.loads(stdout)

        for workflow_name, workflow_data in schema["workflows"].items():
            assert "description" in workflow_data, f"Workflow '{workflow_name}' missing description"
            assert "steps" in workflow_data, f"Workflow '{workflow_name}' missing steps"
            assert len(workflow_data["steps"]) > 0, f"Workflow '{workflow_name}' has no steps"

    def test_global_options_documented(self):
        """Global options are documented."""
        exit_code, stdout, _ = run_mx("schema")

        schema = json.loads(stdout)

        assert len(schema["global_options"]) > 0

        option_names = [opt["name"] for opt in schema["global_options"]]
        assert "--json-errors" in option_names
        assert "--help" in option_names

    def test_schema_help_text(self):
        """Schema command has proper help text."""
        exit_code, stdout, stderr = run_mx("schema", "--help")

        assert exit_code == 0
        assert "agent-friendly metadata" in stdout.lower() or "introspection" in stdout.lower()
        assert "--command" in stdout
        assert "--compact" in stdout


class TestSchemaAgentUsability:
    """Tests focused on agent usability of the schema."""

    def test_common_mistakes_provide_guidance(self):
        """Common mistakes provide actionable guidance."""
        exit_code, stdout, _ = run_mx("schema")

        schema = json.loads(stdout)

        # Check append command's common mistake about path vs title
        append = schema["commands"]["append"]
        mistakes = append["common_mistakes"]

        assert "using path instead of title" in mistakes
        guidance = mistakes["using path instead of title"]
        assert "mx append" in guidance  # Should suggest correct usage

    def test_examples_are_runnable_commands(self):
        """Examples start with 'mx' and look like valid commands."""
        exit_code, stdout, _ = run_mx("schema")

        schema = json.loads(stdout)

        for cmd_name, cmd_data in schema["commands"].items():
            for example in cmd_data["examples"]:
                assert example.startswith("mx "), f"Example doesn't start with 'mx': {example}"
                # Should contain the command name
                assert cmd_name in example or cmd_name.replace("-", " ") in example, \
                    f"Example doesn't contain command '{cmd_name}': {example}"

    def test_option_types_are_documented(self):
        """Options have type information for validation."""
        exit_code, stdout, _ = run_mx("schema")

        schema = json.loads(stdout)

        # Check search command options have types
        search = schema["commands"]["search"]
        for opt in search["options"]:
            assert "type" in opt, f"Option {opt['name']} missing type"
            assert opt["type"] in ["string", "integer", "float", "flag", "choice", "path"], \
                f"Unknown type for {opt['name']}: {opt['type']}"

    def test_choice_options_list_valid_choices(self):
        """Choice-type options list their valid values."""
        exit_code, stdout, _ = run_mx("schema")

        schema = json.loads(stdout)

        # Search --mode is a choice option
        search = schema["commands"]["search"]
        mode_opt = next(opt for opt in search["options"] if opt["name"] == "--mode")

        assert mode_opt["type"] == "choice"
        assert "choices" in mode_opt
        assert "hybrid" in mode_opt["choices"]
        assert "keyword" in mode_opt["choices"]
        assert "semantic" in mode_opt["choices"]

    def test_required_options_marked(self):
        """Required options are marked as such."""
        exit_code, stdout, _ = run_mx("schema")

        schema = json.loads(stdout)

        # add command has required --title and --tags
        add = schema["commands"]["add"]
        title_opt = next(opt for opt in add["options"] if opt["name"] == "--title")
        tags_opt = next(opt for opt in add["options"] if opt["name"] == "--tags")

        assert title_opt.get("required") is True
        assert tags_opt.get("required") is True

    def test_default_values_documented(self):
        """Options with defaults have them documented."""
        exit_code, stdout, _ = run_mx("schema")

        schema = json.loads(stdout)

        # search --limit has default of 10
        search = schema["commands"]["search"]
        limit_opt = next(opt for opt in search["options"] if opt["name"] == "--limit")

        assert "default" in limit_opt
        assert limit_opt["default"] == 10
