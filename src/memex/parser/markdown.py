"""Markdown parsing with YAML frontmatter support."""

import re
from pathlib import Path

import frontmatter
import tiktoken
from pydantic import ValidationError

from ..config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS
from ..models import DocumentChunk, EntryMetadata

# Cached encoder for token counting (cl100k_base is Claude/GPT-4 compatible)
_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def _get_token_count(text: str) -> int:
    """Count tokens using cl100k_base encoding.

    Args:
        text: Text to count tokens for.

    Returns:
        Number of tokens in the text.
    """
    return len(_get_encoder().encode(text))


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
        post = frontmatter.load(str(path))
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


def _split_by_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, int]]:
    stripped = text.strip()
    if not stripped:
        return []

    if max_tokens <= 0:
        return [(stripped, _get_token_count(stripped))]

    encoder = _get_encoder()
    tokens = encoder.encode(stripped)
    if len(tokens) <= max_tokens:
        return [(stripped, len(tokens))]

    overlap = max(0, overlap_tokens)
    if max_tokens > 1:
        overlap = min(overlap, max_tokens - 1)
    else:
        overlap = 0

    step = max_tokens - overlap if max_tokens > 0 else max_tokens
    if step <= 0:
        step = max_tokens

    chunks: list[tuple[str, int]] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens).strip()
        if chunk_text:
            chunks.append((chunk_text, len(chunk_tokens)))
        if end >= len(tokens):
            break
        start += step
    return chunks


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
    chunk_index = 0

    def add_chunks(section: str | None, text: str) -> None:
        nonlocal chunk_index
        for chunk_text, token_count in _split_by_tokens(
            text,
            CHUNK_MAX_TOKENS,
            CHUNK_OVERLAP_TOKENS,
        ):
            chunks.append(
                DocumentChunk(
                    path=path,
                    section=section,
                    content=chunk_text,
                    metadata=metadata,
                    token_count=token_count,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
    matches = list(h2_pattern.finditer(content))

    if not matches:
        # No H2 headers - entire content is one chunk (with token splitting if needed)
        add_chunks(None, content)
        return chunks

    # Handle intro section (content before first H2)
    intro_end = matches[0].start()
    intro_content = content[:intro_end].strip()
    if intro_content:
        add_chunks(None, intro_content)

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
            add_chunks(section_name, section_content)

    return chunks
