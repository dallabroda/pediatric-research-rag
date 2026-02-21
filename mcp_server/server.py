#!/usr/bin/env python3
"""
FastMCP server for pediatric research RAG.

Exposes tools, resources, and prompts for Claude Desktop integration.

Usage:
    python mcp_server/server.py

For Claude Desktop, add to your config:
{
    "mcpServers": {
        "pediatric-research": {
            "command": "python",
            "args": ["path/to/mcp_server/server.py"]
        }
    }
}
"""
import logging
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from fastmcp import FastMCP

from mcp_server.tools import (
    ask_research_question,
    get_document,
    list_clinical_trials,
    list_documents,
    search_research,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("pediatric-research-rag")


# ============== TOOLS ==============

@mcp.tool()
def search_pediatric_research(query: str, top_k: int = 5) -> str:
    """
    Search the St. Jude pediatric cancer research database.

    Use this to find relevant research papers and clinical trials
    related to pediatric cancer topics.

    Args:
        query: Search query (e.g., "acute lymphoblastic leukemia treatment")
        top_k: Number of results to return (default: 5)

    Returns:
        JSON string with matching document chunks
    """
    try:
        results = search_research(query, top_k)
        return format_search_results(results)
    except Exception as e:
        logger.exception("Search failed")
        return f"Error searching: {str(e)}"


@mcp.tool()
def ask_pediatric_research_question(question: str) -> str:
    """
    Ask a question about pediatric cancer research and get a cited answer.

    The answer is generated using Claude and cites sources from
    St. Jude research papers and clinical trials.

    Args:
        question: Research question (e.g., "What are the survival rates for pediatric ALL?")

    Returns:
        Answer with citations and sources
    """
    try:
        result = ask_research_question(question)
        return format_answer(result)
    except Exception as e:
        logger.exception("Question answering failed")
        return f"Error answering question: {str(e)}"


@mcp.tool()
def get_clinical_trials(status: Optional[str] = None) -> str:
    """
    List St. Jude clinical trials in the database.

    Args:
        status: Optional filter by status (not yet implemented)

    Returns:
        JSON string with list of clinical trials
    """
    try:
        trials = list_clinical_trials(status)
        return format_trials(trials)
    except Exception as e:
        logger.exception("Failed to list trials")
        return f"Error listing trials: {str(e)}"


@mcp.tool()
def get_research_document(document_id: str) -> str:
    """
    Get details for a specific research document.

    Args:
        document_id: Document ID (PMC ID like "PMC1234567" or NCT ID like "NCT01234567")

    Returns:
        Document details including full text
    """
    try:
        doc = get_document(document_id)
        if doc is None:
            return f"Document not found: {document_id}"
        return format_document(doc)
    except Exception as e:
        logger.exception("Failed to get document")
        return f"Error getting document: {str(e)}"


# ============== RESOURCES ==============

@mcp.resource("research://documents")
def get_all_documents() -> str:
    """
    List all research documents in the database.

    Returns a list of all papers and clinical trials that have been indexed.
    """
    try:
        docs = list_documents()
        lines = ["# Research Documents\n"]

        papers = [d for d in docs if d.get("doc_type") == "paper"]
        trials = [d for d in docs if d.get("doc_type") == "trial"]

        if papers:
            lines.append(f"## Research Papers ({len(papers)})\n")
            for doc in papers:
                lines.append(f"- **{doc['document_id']}**: {doc['title']}")
                if doc.get("source_url"):
                    lines.append(f"  - URL: {doc['source_url']}")
            lines.append("")

        if trials:
            lines.append(f"## Clinical Trials ({len(trials)})\n")
            for doc in trials:
                lines.append(f"- **{doc['document_id']}**: {doc['title']}")
                if doc.get("source_url"):
                    lines.append(f"  - URL: {doc['source_url']}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error listing documents: {str(e)}"


@mcp.resource("research://trials")
def get_trials_resource() -> str:
    """
    List all clinical trials from St. Jude.

    Returns a formatted list of all clinical trials in the database.
    """
    try:
        trials = list_clinical_trials()

        lines = ["# St. Jude Clinical Trials\n"]
        lines.append(f"Total trials: {len(trials)}\n")

        for trial in trials:
            lines.append(f"## {trial['nct_id']}")
            lines.append(f"**{trial['title']}**")
            if trial.get("source_url"):
                lines.append(f"URL: {trial['source_url']}")
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"Error listing trials: {str(e)}"


# ============== PROMPTS ==============

@mcp.prompt()
def research_qa(question: str) -> str:
    """
    Structured prompt for research Q&A with citation requirements.

    This prompt guides the model to answer questions using
    the pediatric research database and cite sources properly.

    Args:
        question: The research question to answer
    """
    return f"""You are a research assistant specializing in pediatric cancer research from St. Jude Children's Research Hospital.

Please answer the following question using the search_pediatric_research and ask_pediatric_research_question tools.

Guidelines:
1. First search for relevant context using search_pediatric_research
2. If needed, use ask_pediatric_research_question for a synthesized answer
3. Always cite sources using [Source: Document Title] format
4. Note clinical trial NCT IDs and status when relevant
5. If information is insufficient, clearly state this

Question: {question}

Please provide a comprehensive, well-cited answer."""


# ============== FORMATTING HELPERS ==============

def format_search_results(results: list[dict]) -> str:
    """Format search results for display."""
    if not results:
        return "No results found."

    lines = [f"Found {len(results)} results:\n"]

    for i, r in enumerate(results, 1):
        lines.append(f"### Result {i}: {r['title']}")
        lines.append(f"- **Document ID**: {r['document_id']}")
        lines.append(f"- **Type**: {r['doc_type']}")
        lines.append(f"- **Score**: {r['score']}")
        if r.get("section"):
            lines.append(f"- **Section**: {r['section']}")
        if r.get("source_url"):
            lines.append(f"- **URL**: {r['source_url']}")
        lines.append(f"\n**Text:**\n{r['text'][:500]}...")
        lines.append("")

    return "\n".join(lines)


def format_answer(result: dict) -> str:
    """Format Q&A answer for display."""
    lines = [
        f"## Question\n{result['question']}\n",
        f"## Answer\n{result['answer']}\n",
        "## Sources",
    ]

    for source in result.get("sources", []):
        lines.append(f"- **{source['title']}** ({source['doc_type']}, score: {source['score']})")
        if source.get("source_url"):
            lines.append(f"  - URL: {source['source_url']}")

    return "\n".join(lines)


def format_trials(trials: list[dict]) -> str:
    """Format trials list for display."""
    if not trials:
        return "No clinical trials found."

    lines = [f"Found {len(trials)} clinical trials:\n"]

    for trial in trials:
        lines.append(f"- **{trial['nct_id']}**: {trial['title']}")
        if trial.get("source_url"):
            lines.append(f"  - URL: {trial['source_url']}")

    return "\n".join(lines)


def format_document(doc: dict) -> str:
    """Format document details for display."""
    lines = [
        f"# {doc['title']}",
        f"**Document ID**: {doc['document_id']}",
        f"**Type**: {doc['doc_type']}",
    ]

    if doc.get("source_url"):
        lines.append(f"**URL**: {doc['source_url']}")

    if doc.get("sections"):
        lines.append(f"\n**Sections**: {', '.join(doc['sections'])}")

    lines.append(f"\n**Content** ({doc['chunks_count']} chunks):\n")
    lines.append(doc.get("full_text", "No text available")[:5000])

    if len(doc.get("full_text", "")) > 5000:
        lines.append("\n... (truncated)")

    return "\n".join(lines)


# ============== MAIN ==============

if __name__ == "__main__":
    logger.info("Starting Pediatric Research RAG MCP Server")
    mcp.run()
