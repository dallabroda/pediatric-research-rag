"""
Prompt templates for the RAG pipeline.

Contains system prompts and formatting functions for Claude.
"""
from dataclasses import dataclass
from typing import Optional


SYSTEM_PROMPT = """You are a research assistant specializing in pediatric cancer research from St. Jude Children's Research Hospital. Your role is to answer questions based ONLY on the provided research documents.

Guidelines:
1. Only use information from the provided context chunks
2. Always cite your sources using the format [Source: Document Title]
3. If the context doesn't contain enough information, say so clearly
4. Use precise medical terminology but explain complex terms when needed
5. For clinical trial information, note the trial status and NCT ID
6. Never make up or hallucinate information not in the provided context

Response format:
- Provide a clear, concise answer
- Include relevant citations inline
- If multiple sources agree, synthesize the information
- If sources conflict, note the discrepancy"""


@dataclass
class ContextChunk:
    """A chunk of context for the prompt."""
    text: str
    doc_title: str
    doc_type: str
    source_url: Optional[str] = None
    section_title: Optional[str] = None
    page_number: Optional[int] = None


def format_context_chunk(chunk: ContextChunk, index: int) -> str:
    """
    Format a single context chunk for the prompt.

    Args:
        chunk: Context chunk
        index: Chunk index (1-based)

    Returns:
        Formatted chunk string
    """
    lines = [f"[Context {index}]"]
    lines.append(f"Document: {chunk.doc_title}")
    lines.append(f"Type: {chunk.doc_type}")

    if chunk.section_title:
        lines.append(f"Section: {chunk.section_title}")
    if chunk.page_number:
        lines.append(f"Page: {chunk.page_number}")
    if chunk.source_url:
        lines.append(f"URL: {chunk.source_url}")

    lines.append("")
    lines.append(chunk.text)
    lines.append("")

    return "\n".join(lines)


def build_context_prompt(chunks: list[ContextChunk]) -> str:
    """
    Build the context section of the prompt.

    Args:
        chunks: List of context chunks

    Returns:
        Formatted context string
    """
    if not chunks:
        return "No relevant context found."

    sections = ["RELEVANT RESEARCH CONTEXT:", "=" * 40, ""]

    for i, chunk in enumerate(chunks, 1):
        sections.append(format_context_chunk(chunk, i))
        sections.append("-" * 40)

    return "\n".join(sections)


def build_user_prompt(question: str, context_chunks: list[ContextChunk]) -> str:
    """
    Build the complete user prompt with context and question.

    Args:
        question: User's question
        context_chunks: Retrieved context chunks

    Returns:
        Complete user prompt
    """
    context = build_context_prompt(context_chunks)

    prompt = f"""{context}

QUESTION:
{question}

Please provide a comprehensive answer based on the research context above. Remember to cite your sources."""

    return prompt


def build_no_context_response(question: str) -> str:
    """
    Build response when no relevant context is found.

    Args:
        question: User's question

    Returns:
        Response explaining lack of context
    """
    return f"""I couldn't find relevant information in the research database to answer your question: "{question}"

This could mean:
1. The topic isn't covered in the current document collection
2. The question needs to be rephrased for better matching
3. The specific information isn't available in open-access pediatric cancer research

Please try rephrasing your question or asking about a related topic in pediatric cancer research from St. Jude Children's Research Hospital."""


def create_messages(question: str, context_chunks: list[ContextChunk]) -> list[dict]:
    """
    Create the messages array for Claude API.

    Args:
        question: User's question
        context_chunks: Retrieved context chunks

    Returns:
        Messages list for Claude API
    """
    user_prompt = build_user_prompt(question, context_chunks)

    return [
        {"role": "user", "content": user_prompt}
    ]


# MCP-specific prompts

MCP_RESEARCH_QA_PROMPT = """You are a research assistant for pediatric cancer research. Answer questions using the provided research context.

Instructions:
1. Use ONLY the provided context to answer
2. Cite sources using [Source: Title] format
3. Note clinical trial NCT IDs and status when relevant
4. If context is insufficient, state this clearly
5. Be precise with medical terminology

Context will be provided in the user message."""


def get_mcp_prompt(prompt_name: str) -> dict:
    """
    Get an MCP prompt definition.

    Args:
        prompt_name: Name of the prompt

    Returns:
        Prompt definition dict
    """
    prompts = {
        "research_qa": {
            "name": "research_qa",
            "description": "Structured Q&A prompt for pediatric cancer research with citation requirements",
            "arguments": [
                {
                    "name": "question",
                    "description": "The research question to answer",
                    "required": True,
                }
            ],
        }
    }

    return prompts.get(prompt_name)
