"""Intent detection for CLI error messages.

Detects when users pick the wrong command but express clear intent through flags,
and suggests the correct command.
"""

from dataclasses import dataclass


@dataclass
class IntentSuggestion:
    """A suggested alternative command based on detected intent."""
    description: str
    command: str


@dataclass
class IntentMismatch:
    """Detected intent mismatch with suggestions."""
    message: str
    suggestions: list[IntentSuggestion]

    def format_error(self) -> str:
        """Format the error message with suggestions."""
        lines = [f"Error: {self.message}", "", "Did you mean:"]
        for suggestion in self.suggestions:
            lines.append(f"  - {suggestion.description}: {suggestion.command}")
        return "\n".join(lines)


def detect_patch_intent_mismatch(
    path: str,
    find_text: str | None,
    replace_text: str | None,
    content: str | None = None,
    append: str | None = None,
) -> IntentMismatch | None:
    """Detect intent mismatch for the patch command.

    Returns IntentMismatch if the flags suggest a different command,
    or None if the flags are appropriate for patch.

    Common mismatches:
    - `patch` with `--content` but no `--find` -> suggest append/update
    - `patch` with `--append` -> suggest append command
    """
    # User tried to use --append flag (which doesn't exist on patch)
    if append is not None:
        return IntentMismatch(
            message="No such option: --append",
            suggestions=[
                IntentSuggestion(
                    description="For appending content",
                    command="mx append 'Title' --content='...'",
                ),
                IntentSuggestion(
                    description="For find-replace",
                    command=f"mx patch {path} --find 'x' --replace 'y'",
                ),
            ],
        )

    # User provided --content but no --find (suggests they want append/update, not patch)
    if content is not None and find_text is None:
        return IntentMismatch(
            message="--find is required for find-replace operations.",
            suggestions=[
                IntentSuggestion(
                    description="To append content",
                    command="mx append 'Title' --content='...'",
                ),
                IntentSuggestion(
                    description="To replace text",
                    command=f"mx patch {path} --find 'x' --replace 'y'",
                ),
                IntentSuggestion(
                    description="To overwrite entry",
                    command=f"mx update {path} --content='...'",
                ),
            ],
        )

    return None


def detect_update_intent_mismatch(
    path: str,
    find_text: str | None = None,
    replace_text: str | None = None,
) -> IntentMismatch | None:
    """Detect intent mismatch for the update command.

    Returns IntentMismatch if the flags suggest a different command,
    or None if the flags are appropriate for update.

    Common mismatches:
    - `update` with `--find` -> suggest patch
    """
    # User tried to use --find flag (which doesn't exist on update)
    if find_text is not None:
        return IntentMismatch(
            message="No such option: --find. Use 'mx patch' for find-replace operations.",
            suggestions=[
                IntentSuggestion(
                    description="For find-replace",
                    command=f"mx patch {path} --find 'x' --replace 'y'",
                ),
                IntentSuggestion(
                    description="To overwrite content",
                    command=f"mx update {path} --content='...'",
                ),
            ],
        )

    # User tried to use --replace without --find context (might mean they want patch)
    if replace_text is not None:
        return IntentMismatch(
            message="No such option: --replace. Use 'mx patch' for find-replace operations.",
            suggestions=[
                IntentSuggestion(
                    description="For find-replace",
                    command=f"mx patch {path} --find 'x' --replace 'y'",
                ),
                IntentSuggestion(
                    description="To overwrite content",
                    command=f"mx update {path} --content='...'",
                ),
            ],
        )

    return None
