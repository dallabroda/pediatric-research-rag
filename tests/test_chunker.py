"""Tests for the chunker module."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lambdas.ingest.chunker import (
    Chunk,
    ChunkMetadata,
    chunk_document,
    estimate_tokens,
    get_page_number,
    get_section_title,
    split_text,
)


class TestEstimateTokens:
    """Tests for token estimation."""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_string(self):
        # "Hello" = 5 chars -> ~1 token
        result = estimate_tokens("Hello")
        assert result == 1

    def test_longer_string(self):
        # 100 chars -> ~25 tokens
        text = "a" * 100
        result = estimate_tokens(text)
        assert result == 25


class TestSplitText:
    """Tests for text splitting."""

    def test_empty_text(self):
        result = split_text("")
        assert result == []

    def test_short_text_single_chunk(self):
        text = "This is a short text."
        result = split_text(text, chunk_size=500)
        assert len(result) == 1
        assert result[0].text == text

    def test_splits_on_paragraphs(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = split_text(text, chunk_size=10, chunk_overlap=2)
        assert len(result) >= 2
        # First chunk should contain first paragraph
        assert "First" in result[0].text

    def test_chunk_indices(self):
        text = "Word " * 1000  # Long text
        result = split_text(text, chunk_size=50, chunk_overlap=10)
        for i, chunk in enumerate(result):
            assert chunk.chunk_index == i

    def test_char_positions(self):
        text = "First paragraph.\n\nSecond paragraph."
        result = split_text(text, chunk_size=100)
        for chunk in result:
            assert chunk.char_start >= 0
            assert chunk.char_end <= len(text)
            assert chunk.char_end > chunk.char_start


class TestGetPageNumber:
    """Tests for page number determination."""

    def test_empty_boundaries(self):
        result = get_page_number(100, [])
        assert result is None

    def test_first_page(self):
        boundaries = [0, 1000, 2000]
        result = get_page_number(500, boundaries)
        assert result == 1

    def test_second_page(self):
        boundaries = [0, 1000, 2000]
        result = get_page_number(1500, boundaries)
        assert result == 2

    def test_last_page(self):
        boundaries = [0, 1000, 2000]
        result = get_page_number(2500, boundaries)
        assert result == 3


class TestGetSectionTitle:
    """Tests for section title determination."""

    def test_empty_sections(self):
        result = get_section_title(100, {})
        assert result is None

    def test_finds_section(self):
        sections = {0: "Introduction", 500: "Methods", 1000: "Results"}
        result = get_section_title(750, sections)
        assert result == "Methods"

    def test_first_section(self):
        sections = {0: "Abstract", 100: "Introduction"}
        result = get_section_title(50, sections)
        assert result == "Abstract"


class TestChunkDocument:
    """Tests for full document chunking."""

    def test_basic_chunking(self):
        text = "This is test content. " * 50
        result = chunk_document(
            text=text,
            document_id="test_doc",
            doc_title="Test Document",
            doc_type="paper",
            chunk_size=100,
            chunk_overlap=10,
        )

        assert len(result) >= 1
        assert all(isinstance(c, ChunkMetadata) for c in result)
        assert all(c.document_id == "test_doc" for c in result)
        assert all(c.doc_type == "paper" for c in result)

    def test_chunk_ids_unique(self):
        text = "Content. " * 200
        result = chunk_document(
            text=text,
            document_id="test",
            doc_title="Test",
            doc_type="paper",
        )

        chunk_ids = [c.chunk_id for c in result]
        assert len(chunk_ids) == len(set(chunk_ids))  # All unique

    def test_preserves_metadata(self):
        text = "Introduction\n\nThis is the intro.\n\nMethods\n\nThis is methods."
        result = chunk_document(
            text=text,
            document_id="PMC123",
            doc_title="Research Paper",
            doc_type="paper",
            source_url="https://example.com",
            page_boundaries=[0, 100],
            section_titles={0: "Introduction", 30: "Methods"},
        )

        assert len(result) >= 1
        first = result[0]
        assert first.document_id == "PMC123"
        assert first.doc_title == "Research Paper"
        assert first.source_url == "https://example.com"
