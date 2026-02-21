"""
Text chunking for RAG ingestion.

Implements recursive character splitting with configurable chunk size and overlap.
Preserves paragraph boundaries and tracks metadata per chunk.
"""
import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    # Core identification
    chunk_id: str
    document_id: str
    chunk_text: str
    chunk_index: int
    page_number: Optional[int]
    section_title: Optional[str]
    source_url: Optional[str]
    doc_title: str
    doc_type: str

    # Lineage fields
    ingestion_timestamp: str = ""  # ISO 8601
    parser_name: str = ""  # "PDFParser", "ClinicalTrialParser", etc.
    parser_version: str = "1.0.0"
    document_hash: str = ""  # SHA256 of original document text
    extraction_confidence: float = 1.0  # 0.0-1.0

    # Quality metrics
    char_count: int = 0
    word_count: int = 0
    avg_word_length: float = 0.0
    whitespace_ratio: float = 0.0
    has_ocr_artifacts: bool = False


@dataclass
class LineageInfo:
    """Lineage information for tracking document provenance."""
    parser_name: str
    parser_version: str
    document_hash: str
    ingestion_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def compute_document_hash(text: str) -> str:
    """
    Compute SHA256 hash of document text for lineage tracking.

    Args:
        text: Document text to hash

    Returns:
        SHA256 hash as hex string
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_chunk_quality_metrics(text: str) -> dict:
    """
    Compute quality metrics for a text chunk.

    Args:
        text: Chunk text to analyze

    Returns:
        Dictionary with quality metrics
    """
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0
    whitespace_count = sum(1 for c in text if c.isspace())
    whitespace_ratio = whitespace_count / char_count if char_count > 0 else 0.0

    # Detect common OCR artifacts
    ocr_artifact_patterns = [
        r"[^\x00-\x7F]{3,}",  # Long sequences of non-ASCII
        r"\b[A-Z]{10,}\b",  # Long all-caps words (likely OCR errors)
        r"(?<!\d)[0oO]{3,}(?!\d)",  # Repeated 0/o/O (common OCR confusion)
        r"[|!l1I]{5,}",  # Confused vertical characters
    ]
    has_ocr_artifacts = any(
        re.search(pattern, text) for pattern in ocr_artifact_patterns
    )

    return {
        "char_count": char_count,
        "word_count": word_count,
        "avg_word_length": avg_word_length,
        "whitespace_ratio": whitespace_ratio,
        "has_ocr_artifacts": has_ocr_artifacts,
    }


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
    lineage: Optional[LineageInfo] = None,
    extraction_confidence: float = 1.0,
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
        lineage: Optional lineage information for provenance tracking
        extraction_confidence: Confidence score for the extraction (0.0-1.0)

    Returns:
        List of ChunkMetadata objects ready for embedding
    """
    # Split text into chunks
    raw_chunks = split_text(text, chunk_size, chunk_overlap)

    # Compute document hash once
    document_hash = compute_document_hash(text)

    # Get current timestamp
    ingestion_timestamp = datetime.now(timezone.utc).isoformat()

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

        # Compute quality metrics for this chunk
        quality_metrics = compute_chunk_quality_metrics(chunk.text)

        metadata = ChunkMetadata(
            # Core fields
            chunk_id=chunk_id,
            document_id=document_id,
            chunk_text=chunk.text,
            chunk_index=chunk.chunk_index,
            page_number=page_number,
            section_title=section_title,
            source_url=source_url,
            doc_title=doc_title,
            doc_type=doc_type,
            # Lineage fields
            ingestion_timestamp=lineage.ingestion_timestamp if lineage else ingestion_timestamp,
            parser_name=lineage.parser_name if lineage else "",
            parser_version=lineage.parser_version if lineage else "1.0.0",
            document_hash=lineage.document_hash if lineage else document_hash,
            extraction_confidence=extraction_confidence,
            # Quality metrics
            char_count=quality_metrics["char_count"],
            word_count=quality_metrics["word_count"],
            avg_word_length=quality_metrics["avg_word_length"],
            whitespace_ratio=quality_metrics["whitespace_ratio"],
            has_ocr_artifacts=quality_metrics["has_ocr_artifacts"],
        )

        metadata_list.append(metadata)

    logger.info(f"Created {len(metadata_list)} chunks for document {document_id}")

    return metadata_list
