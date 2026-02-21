"""
Text chunking for RAG ingestion.

Implements recursive character splitting with configurable chunk size and overlap.
Preserves paragraph boundaries and tracks metadata per chunk.
"""
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Default separators for recursive splitting (in order of preference)
DEFAULT_SEPARATORS = [
    "\n\n\n",  # Multiple paragraph breaks
    "\n\n",    # Paragraph breaks
    "\n",      # Line breaks
    ". ",      # Sentence endings
    "? ",      # Question endings
    "! ",      # Exclamation endings
    "; ",      # Semicolon breaks
    ", ",      # Comma breaks
    " ",       # Word breaks
    "",        # Character level (last resort)
]


@dataclass
class Chunk:
    """A single text chunk with position metadata."""
    text: str
    chunk_index: int
    char_start: int
    char_end: int
    token_estimate: int


@dataclass
class ChunkMetadata:
    """Full metadata for a chunk in the index."""
    chunk_id: str
    document_id: str
    chunk_text: str
    chunk_index: int
    page_number: Optional[int]
    section_title: Optional[str]
    source_url: Optional[str]
    doc_title: str
    doc_type: str


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Rough estimate: ~4 characters per token for English text.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // 4


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: Optional[list[str]] = None,
) -> list[Chunk]:
    """
    Split text into chunks using recursive character splitting.

    Tries to split on larger semantic boundaries first (paragraphs, sentences),
    falling back to smaller boundaries if necessary.

    Args:
        text: Text to split
        chunk_size: Target tokens per chunk (~4 chars per token)
        chunk_overlap: Overlap tokens between consecutive chunks
        separators: List of separators to try, in order of preference

    Returns:
        List of Chunk objects with position metadata
    """
    if not text:
        return []

    if separators is None:
        separators = DEFAULT_SEPARATORS.copy()

    # Convert token counts to character counts (rough estimate)
    max_chars = chunk_size * 4
    overlap_chars = chunk_overlap * 4

    # If text is already small enough, return as single chunk
    if len(text) <= max_chars:
        return [Chunk(
            text=text.strip(),
            chunk_index=0,
            char_start=0,
            char_end=len(text),
            token_estimate=estimate_tokens(text),
        )]

    chunks = []
    current_pos = 0

    while current_pos < len(text):
        # Find the end position for this chunk
        chunk_end = min(current_pos + max_chars, len(text))

        # If we're not at the end, try to find a good break point
        if chunk_end < len(text):
            best_break = None

            # Try each separator in order of preference
            for separator in separators:
                if not separator:
                    # Last resort: just break at max_chars
                    best_break = chunk_end
                    break

                # Look for separator within the chunk
                search_start = current_pos
                search_end = chunk_end

                # Find the last occurrence of separator in the range
                chunk_text = text[search_start:search_end]
                last_sep = chunk_text.rfind(separator)

                if last_sep != -1:
                    # Found a separator - break after it
                    best_break = search_start + last_sep + len(separator)
                    break

            chunk_end = best_break if best_break else chunk_end

        # Extract chunk text
        chunk_text = text[current_pos:chunk_end].strip()

        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                chunk_index=len(chunks),
                char_start=current_pos,
                char_end=chunk_end,
                token_estimate=estimate_tokens(chunk_text),
            ))

        # Move to next position with overlap
        # The overlap should start from content, not whitespace
        next_pos = chunk_end - overlap_chars
        if next_pos <= current_pos:
            next_pos = chunk_end  # Avoid infinite loop

        # Try to start on a word boundary for the overlap
        while next_pos < len(text) and text[next_pos].isspace():
            next_pos += 1

        current_pos = next_pos

    return chunks


def get_page_number(
    char_position: int,
    page_boundaries: list[int],
) -> Optional[int]:
    """
    Get the page number for a character position.

    Args:
        char_position: Character position in the full text
        page_boundaries: List of character positions where pages start

    Returns:
        1-indexed page number, or None if cannot be determined
    """
    if not page_boundaries:
        return None

    for i, boundary in enumerate(page_boundaries):
        if char_position < boundary:
            return i  # 1-indexed

    return len(page_boundaries)  # Last page


def get_section_title(
    char_position: int,
    section_titles: dict[int, str],
) -> Optional[str]:
    """
    Get the section title for a character position.

    Args:
        char_position: Character position in the full text
        section_titles: Dict mapping positions to section titles

    Returns:
        Section title, or None if no section found
    """
    if not section_titles:
        return None

    # Find the most recent section title before this position
    current_section = None
    sorted_positions = sorted(section_titles.keys())

    for pos in sorted_positions:
        if pos <= char_position:
            current_section = section_titles[pos]
        else:
            break

    return current_section


def chunk_document(
    text: str,
    document_id: str,
    doc_title: str,
    doc_type: str,
    page_boundaries: Optional[list[int]] = None,
    section_titles: Optional[dict[int, str]] = None,
    source_url: Optional[str] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[ChunkMetadata]:
    """
    Chunk a document and create metadata for each chunk.

    Args:
        text: Full document text
        document_id: Unique document identifier
        doc_title: Document title
        doc_type: Document type ("paper" or "trial")
        page_boundaries: Character positions where pages start
        section_titles: Dict mapping positions to section titles
        source_url: Source URL for the document
        chunk_size: Target tokens per chunk
        chunk_overlap: Overlap tokens between chunks

    Returns:
        List of ChunkMetadata objects ready for embedding
    """
    # Split text into chunks
    raw_chunks = split_text(text, chunk_size, chunk_overlap)

    metadata_list = []

    for chunk in raw_chunks:
        # Determine page number
        page_number = get_page_number(
            chunk.char_start,
            page_boundaries or [],
        )

        # Determine section title
        section_title = get_section_title(
            chunk.char_start,
            section_titles or {},
        )

        # Create unique chunk ID
        chunk_id = f"{document_id}_chunk_{chunk.chunk_index}"

        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            document_id=document_id,
            chunk_text=chunk.text,
            chunk_index=chunk.chunk_index,
            page_number=page_number,
            section_title=section_title,
            source_url=source_url,
            doc_title=doc_title,
            doc_type=doc_type,
        )

        metadata_list.append(metadata)

    logger.info(f"Created {len(metadata_list)} chunks for document {document_id}")

    return metadata_list
