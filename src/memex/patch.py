"""Surgical find-replace patch operations for KB entries.

This module provides the core logic for patching KB entries with exact
string replacement, mirroring Claude Code's Edit tool interface.
"""

from __future__ import annotations

import bisect
import difflib
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path


class PatchExitCode(IntEnum):
    """Exit codes for patch operations."""

    SUCCESS = 0
    NOT_FOUND = 1  # old_string not found in content
    AMBIGUOUS = 2  # multiple matches without --replace-all
    FILE_ERROR = 3  # file not found, permission denied, encoding error


@dataclass
class MatchContext:
    """Context around a match for error reporting."""

    match_number: int  # 1-indexed match number
    start_pos: int  # byte offset in content
    line_number: int  # 1-indexed line number
    context_before: str  # ~50 chars before match
    match_text: str  # the matched text
    context_after: str  # ~50 chars after match

    def format_preview(self) -> str:
        """Format as human-readable preview."""
        before = self.context_before.replace("\n", "\\n")
        after = self.context_after.replace("\n", "\\n")
        match = self.match_text.replace("\n", "\\n")
        # Truncate long contexts
        if len(before) > 30:
            before = "..." + before[-30:]
        if len(after) > 30:
            after = after[:30] + "..."
        return f"Match {self.match_number} (line {self.line_number}): ...{before}[{match}]{after}..."


@dataclass
class PatchResult:
    """Result of a patch operation."""

    success: bool
    exit_code: PatchExitCode
    message: str
    matches_found: int = 0
    replacements_made: int = 0
    match_contexts: list[MatchContext] = field(default_factory=list)
    new_content: str | None = None  # For dry-run preview or successful patch
    diff: str | None = None  # Unified diff for dry-run

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        result = {
            "success": self.success,
            "exit_code": int(self.exit_code),
            "message": self.message,
            "matches_found": self.matches_found,
            "replacements_made": self.replacements_made,
        }
        if self.match_contexts:
            result["match_contexts"] = [
                {
                    "match_number": mc.match_number,
                    "line_number": mc.line_number,
                    "preview": mc.format_preview(),
                }
                for mc in self.match_contexts
            ]
        if self.diff:
            result["diff"] = self.diff
        return result


def find_matches(content: str, old_string: str) -> list[MatchContext]:
    """Find all occurrences of old_string in content with context.

    Args:
        content: The full file content (body only, no frontmatter).
        old_string: The exact string to find.

    Returns:
        List of MatchContext for each occurrence.
    """
    if not old_string:
        return []

    matches = []
    lines = content.split("\n")

    # Build line offset map for line number lookup
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line) + 1)

    start = 0
    match_num = 0
    while True:
        pos = content.find(old_string, start)
        if pos == -1:
            break

        match_num += 1

        # Find line number (1-indexed)
        line_num = bisect.bisect_right(line_offsets, pos)

        # Extract context (~50 chars before/after)
        ctx_start = max(0, pos - 50)
        ctx_end = min(len(content), pos + len(old_string) + 50)

        matches.append(
            MatchContext(
                match_number=match_num,
                start_pos=pos,
                line_number=line_num,
                context_before=content[ctx_start:pos],
                match_text=old_string,
                context_after=content[pos + len(old_string) : ctx_end],
            )
        )

        # Move past this match (non-overlapping)
        start = pos + len(old_string)

    return matches


def apply_patch(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> PatchResult:
    """Apply a find-replace patch to content.

    This is the pure logic function - no file I/O.

    Args:
        content: The file content to patch.
        old_string: Exact text to find and replace.
        new_string: Replacement text.
        replace_all: If True, replace all occurrences.
                     If False, fail on multiple matches.

    Returns:
        PatchResult with success status, message, and new content if successful.
    """
    if not old_string:
        return PatchResult(
            success=False,
            exit_code=PatchExitCode.NOT_FOUND,
            message="Empty search string",
            matches_found=0,
        )

    matches = find_matches(content, old_string)

    if len(matches) == 0:
        preview = old_string[:50] + ("..." if len(old_string) > 50 else "")
        return PatchResult(
            success=False,
            exit_code=PatchExitCode.NOT_FOUND,
            message=f"Text not found: {preview}",
            matches_found=0,
        )

    if len(matches) > 1 and not replace_all:
        return PatchResult(
            success=False,
            exit_code=PatchExitCode.AMBIGUOUS,
            message=f"Found {len(matches)} matches. Use --replace-all or provide more context.",
            matches_found=len(matches),
            match_contexts=matches[:3],  # First 3 for preview
        )

    # Perform replacement
    if replace_all:
        new_content = content.replace(old_string, new_string)
        replacement_count = len(matches)
    else:
        new_content = content.replace(old_string, new_string, 1)
        replacement_count = 1

    return PatchResult(
        success=True,
        exit_code=PatchExitCode.SUCCESS,
        message=f"Replaced {replacement_count} occurrence(s)",
        matches_found=len(matches),
        replacements_made=replacement_count,
        new_content=new_content,
    )


def generate_diff(
    original: str,
    patched: str,
    filename: str = "entry.md",
) -> str:
    """Generate unified diff between original and patched content.

    Args:
        original: Original content.
        patched: Patched content.
        filename: Filename for diff header.

    Returns:
        Unified diff string.
    """
    original_lines = original.splitlines(keepends=True)
    patched_lines = patched.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        patched_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm="",
    )
    return "".join(diff)


def read_file_safely(path: Path) -> tuple[str, str, PatchResult | None]:
    """Read file with encoding validation and BOM handling.

    Separates frontmatter from body content.

    Args:
        path: Path to the file.

    Returns:
        Tuple of (frontmatter, body_content, error_result).
        If error_result is not None, frontmatter and body will be empty strings.
    """
    try:
        raw_bytes = path.read_bytes()
    except PermissionError:
        return (
            "",
            "",
            PatchResult(
                success=False,
                exit_code=PatchExitCode.FILE_ERROR,
                message=f"Permission denied: {path}",
            ),
        )
    except FileNotFoundError:
        return (
            "",
            "",
            PatchResult(
                success=False,
                exit_code=PatchExitCode.FILE_ERROR,
                message=f"File not found: {path}",
            ),
        )

    # Handle UTF-8 BOM
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        raw_bytes = raw_bytes[3:]

    # Decode as UTF-8
    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        return (
            "",
            "",
            PatchResult(
                success=False,
                exit_code=PatchExitCode.FILE_ERROR,
                message=f"File is not valid UTF-8: {e}",
            ),
        )

    # Split frontmatter and body
    # Frontmatter is between first --- and second ---
    if content.startswith("---"):
        # Find the closing ---
        end_idx = content.find("\n---", 3)
        if end_idx != -1:
            # Find the end of the --- line
            newline_after = content.find("\n", end_idx + 4)
            if newline_after == -1:
                # Frontmatter only, no body
                frontmatter = content
                body = ""
            else:
                frontmatter = content[: newline_after + 1]
                body = content[newline_after + 1 :]
            return frontmatter, body, None

    # No frontmatter - treat entire content as body
    return "", content, None


def write_file_atomically(
    path: Path,
    frontmatter: str,
    content: str,
    backup: bool = False,
) -> PatchResult | None:
    """Write file atomically (temp file + rename) with optional backup.

    Args:
        path: Target file path.
        frontmatter: YAML frontmatter including --- delimiters.
        content: Body content.
        backup: If True, create .bak backup before overwriting.

    Returns:
        PatchResult with error if write failed, None on success.
    """
    full_content = frontmatter + content

    # Create backup if requested
    if backup and path.exists():
        backup_path = path.with_suffix(path.suffix + ".bak")
        try:
            shutil.copy2(path, backup_path)
        except Exception as e:
            return PatchResult(
                success=False,
                exit_code=PatchExitCode.FILE_ERROR,
                message=f"Failed to create backup: {e}",
            )

    # Write to temp file in same directory (for atomic rename)
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=".patch_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(full_content)

            # Preserve original file permissions
            if path.exists():
                shutil.copymode(path, temp_path)

            # Atomic rename
            os.replace(temp_path, path)
            temp_path = None  # Mark as successfully moved
        except Exception:
            raise
    except Exception as e:
        # Clean up temp file on error
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        return PatchResult(
            success=False,
            exit_code=PatchExitCode.FILE_ERROR,
            message=f"Failed to write file: {e}",
        )

    return None  # Success
