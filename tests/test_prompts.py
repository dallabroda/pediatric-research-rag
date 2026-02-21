"""Tests for the prompts module."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lambdas.query.prompts import (
    SYSTEM_PROMPT,
    ContextChunk,
    build_context_prompt,
    build_no_context_response,
    build_user_prompt,
    create_messages,
    format_context_chunk,
)


class TestSystemPrompt:
    """Tests for system prompt."""

    def test_system_prompt_exists(self):
        assert len(SYSTEM_PROMPT) > 0

    def test_mentions_citations(self):
        assert "cite" in SYSTEM_PROMPT.lower()

    def test_mentions_st_jude(self):
        assert "St. Jude" in SYSTEM_PROMPT


class TestFormatContextChunk:
    """Tests for context chunk formatting."""

    def test_formats_basic_chunk(self):
        chunk = ContextChunk(
            text="This is the chunk text.",
            doc_title="Test Document",
            doc_type="paper",
        )

        result = format_context_chunk(chunk, 1)

        assert "[Context 1]" in result
        assert "Test Document" in result
        assert "paper" in result
        assert "This is the chunk text." in result

    def test_includes_optional_fields(self):
        chunk = ContextChunk(
            text="Chunk text",
            doc_title="Document",
            doc_type="trial",
            source_url="https://example.com",
            section_title="Methods",
            page_number=5,
        )

        result = format_context_chunk(chunk, 2)

        assert "Methods" in result
        assert "Page: 5" in result
        assert "https://example.com" in result


class TestBuildContextPrompt:
    """Tests for context prompt building."""

    def test_empty_chunks(self):
        result = build_context_prompt([])
        assert "No relevant context" in result

    def test_multiple_chunks(self):
        chunks = [
            ContextChunk(text="First chunk", doc_title="Doc 1", doc_type="paper"),
            ContextChunk(text="Second chunk", doc_title="Doc 2", doc_type="trial"),
        ]

        result = build_context_prompt(chunks)

        assert "[Context 1]" in result
        assert "[Context 2]" in result
        assert "First chunk" in result
        assert "Second chunk" in result


class TestBuildUserPrompt:
    """Tests for user prompt building."""

    def test_includes_question(self):
        chunks = [
            ContextChunk(text="Context", doc_title="Doc", doc_type="paper"),
        ]

        result = build_user_prompt("What is ALL?", chunks)

        assert "What is ALL?" in result
        assert "QUESTION:" in result

    def test_includes_context(self):
        chunks = [
            ContextChunk(text="Relevant info", doc_title="Paper", doc_type="paper"),
        ]

        result = build_user_prompt("Question?", chunks)

        assert "Relevant info" in result


class TestBuildNoContextResponse:
    """Tests for no-context response."""

    def test_includes_question(self):
        result = build_no_context_response("Test question?")
        assert "Test question?" in result

    def test_explains_no_results(self):
        result = build_no_context_response("Question")
        assert "couldn't find" in result.lower() or "no" in result.lower()


class TestCreateMessages:
    """Tests for message creation."""

    def test_creates_user_message(self):
        chunks = [
            ContextChunk(text="Info", doc_title="Doc", doc_type="paper"),
        ]

        result = create_messages("Question?", chunks)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Question?" in result[0]["content"]
