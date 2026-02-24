"""
MCP Tool implementations for pediatric research RAG.

Provides tools, resources, and prompts for Claude Desktop integration.
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import boto3
from dotenv import load_dotenv

from lambdas.query.prompts import (
    SYSTEM_PROMPT,
    ContextChunk,
    build_user_prompt,
    build_no_context_response,
)
from lambdas.query.retriever import FAISSRetriever

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
S3_BUCKET = os.environ.get("S3_BUCKET", "pediatric-research-rag")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "amazon.nova-pro-v1:0")
LOCAL_INDEX_DIR = os.environ.get("LOCAL_INDEX_DIR", "data/index")
TOP_K = int(os.environ.get("TOP_K", "5"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))

# Cache
_retriever = None
_bedrock_client = None


def get_retriever() -> FAISSRetriever:
    """Get or create cached retriever."""
    global _retriever

    if _retriever is None:
        try:
            _retriever = FAISSRetriever.from_local(LOCAL_INDEX_DIR)
            logger.info("Loaded retriever from local index")
        except FileNotFoundError:
            _retriever = FAISSRetriever.from_s3(S3_BUCKET)
            logger.info("Loaded retriever from S3")

    return _retriever


def get_bedrock_client():
    """Get or create cached Bedrock client."""
    global _bedrock_client

    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )

    return _bedrock_client


def search_research(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Search the pediatric research database.

    Args:
        query: Search query
        top_k: Number of results to return

    Returns:
        List of matching document chunks with metadata
    """
    retriever = get_retriever()
    results = retriever.search(query, top_k=top_k)

    return [
        {
            "document_id": r.document_id,
            "title": r.doc_title,
            "doc_type": r.doc_type,
            "text": r.chunk_text,
            "score": round(r.score, 3),
            "source_url": r.source_url,
            "section": r.section_title,
            "page": r.page_number,
        }
        for r in results
    ]


def ask_research_question(
    question: str,
    top_k: int = 5,
) -> dict:
    """
    Ask a question and get an AI-generated answer with citations.

    Args:
        question: Research question
        top_k: Number of context chunks to use

    Returns:
        Answer with sources
    """
    # Search for context
    retriever = get_retriever()
    results = retriever.search(question, top_k=top_k)

    if not results:
        return {
            "question": question,
            "answer": build_no_context_response(question),
            "sources": [],
        }

    # Build context chunks
    context_chunks = [
        ContextChunk(
            text=r.chunk_text,
            doc_title=r.doc_title,
            doc_type=r.doc_type,
            source_url=r.source_url,
            section_title=r.section_title,
            page_number=r.page_number,
        )
        for r in results
    ]

    # Build prompt
    user_prompt = build_user_prompt(question, context_chunks)

    # Call Claude via Bedrock
    bedrock = get_bedrock_client()

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_prompt}],
    })

    response = bedrock.invoke_model(
        modelId=LLM_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())
    answer = response_body["content"][0]["text"]

    sources = [
        {
            "title": r.doc_title,
            "doc_type": r.doc_type,
            "source_url": r.source_url,
            "score": round(r.score, 3),
        }
        for r in results
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }


def list_clinical_trials(status: Optional[str] = None) -> list[dict]:
    """
    List clinical trials in the database.

    Args:
        status: Optional filter by trial status

    Returns:
        List of clinical trials
    """
    retriever = get_retriever()

    trials = []
    seen = set()

    for chunk in retriever.metadata:
        if chunk.get("doc_type") != "trial":
            continue

        doc_id = chunk.get("document_id")
        if doc_id in seen:
            continue
        seen.add(doc_id)

        trials.append({
            "nct_id": doc_id,
            "title": chunk.get("doc_title", ""),
            "source_url": chunk.get("source_url"),
        })

    return trials


def get_document(document_id: str) -> dict | None:
    """
    Get full details for a specific document.

    Args:
        document_id: Document ID (PMC ID or NCT ID)

    Returns:
        Document details or None if not found
    """
    retriever = get_retriever()
    chunks = retriever.get_document_chunks(document_id)

    if not chunks:
        return None

    first_chunk = chunks[0]

    # Reconstruct text from chunks
    full_text = "\n\n".join(c.get("chunk_text", "") for c in chunks)

    # Get unique sections
    sections = []
    for c in chunks:
        section = c.get("section_title")
        if section and section not in sections:
            sections.append(section)

    return {
        "document_id": document_id,
        "title": first_chunk.get("doc_title", ""),
        "doc_type": first_chunk.get("doc_type", ""),
        "source_url": first_chunk.get("source_url"),
        "sections": sections,
        "chunks_count": len(chunks),
        "full_text": full_text,
    }


def list_documents() -> list[dict]:
    """
    List all documents in the database.

    Returns:
        List of document summaries
    """
    retriever = get_retriever()

    documents = []
    seen = set()

    for chunk in retriever.metadata:
        doc_id = chunk.get("document_id")
        if doc_id in seen:
            continue
        seen.add(doc_id)

        documents.append({
            "document_id": doc_id,
            "title": chunk.get("doc_title", ""),
            "doc_type": chunk.get("doc_type", ""),
            "source_url": chunk.get("source_url"),
        })

    return documents
