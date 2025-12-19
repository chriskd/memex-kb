"""Markdown parsing with YAML frontmatter support."""

import re
from pathlib import Path

import frontmatter
from pydantic import ValidationError

from ..models import DocumentChunk, EntryMetadata


class ParseError(Exception):
    """Raised when markdown parsing fails."""

    def __init__(self, path: Path, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


def parse_entry(path: Path) -> tuple[EntryMetadata, str, list[DocumentChunk]]:
    """Parse a markdown file with YAML frontmatter.

    Args:
        path: Path to the markdown file.

    Returns:
        Tuple of (metadata, raw_content, chunks).

    Raises:
        ParseError: If the file cannot be parsed or has invalid frontmatter.
    """
    if not path.exists():
        raise ParseError(path, "File does not exist")

    if not path.is_file():
        raise ParseError(path, "Path is not a file")

    try:
        post = frontmatter.load(path)
    except Exception as e:
        raise ParseError(path, f"Failed to parse frontmatter: {e}") from e

    if not post.metadata:
        raise ParseError(path, "Missing frontmatter (YAML block required at start of file)")

    try:
        metadata = EntryMetadata.model_validate(post.metadata)
    except ValidationError as e:
        errors = []
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]
            errors.append(f"  - {loc}: {msg}")
        raise ParseError(path, "Invalid frontmatter:\n" + "\n".join(errors)) from e

    content = post.content
    path_str = str(path)

    chunks = _chunk_by_h2(path_str, content, metadata)

    return metadata, content, chunks


def _chunk_by_h2(path: str, content: str, metadata: EntryMetadata) -> list[DocumentChunk]:
    """Split content into chunks by H2 headers.

    Args:
        path: File path for the chunks.
        content: Markdown content to chunk.
        metadata: Entry metadata to attach to chunks.

    Returns:
        List of DocumentChunk objects.
    """
    # Pattern matches H2 headers (## Title) at the start of a line
    h2_pattern = re.compile(r"^## (.+)$", re.MULTILINE)

    chunks: list[DocumentChunk] = []
    matches = list(h2_pattern.finditer(content))

    if not matches:
        # No H2 headers - entire content is one chunk
        stripped = content.strip()
        if stripped:
            chunks.append(
                DocumentChunk(
                    path=path,
                    section=None,
                    content=stripped,
                    metadata=metadata,
                )
            )
        return chunks

    # Handle intro section (content before first H2)
    intro_end = matches[0].start()
    intro_content = content[:intro_end].strip()
    if intro_content:
        chunks.append(
            DocumentChunk(
                path=path,
                section=None,
                content=intro_content,
                metadata=metadata,
            )
        )

    # Handle each H2 section
    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        section_start = match.end()

        # Section ends at next H2 or end of content
        if i + 1 < len(matches):
            section_end = matches[i + 1].start()
        else:
            section_end = len(content)

        section_content = content[section_start:section_end].strip()

        # Only create chunk if there's actual content
        if section_content:
            chunks.append(
                DocumentChunk(
                    path=path,
                    section=section_name,
                    content=section_content,
                    metadata=metadata,
                )
            )

    return chunks
