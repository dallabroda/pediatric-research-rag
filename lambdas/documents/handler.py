"""
Lambda handler for documents API.

Provides endpoints to list documents and get document details.
"""
import json
import logging
import os
from typing import Any

from lambdas.query.retriever import FAISSRetriever

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
S3_BUCKET = os.environ.get("S3_BUCKET", "pediatric-research-rag")

# Cache for retriever
_retriever_cache = None


def get_retriever() -> FAISSRetriever:
    """Get or create cached retriever."""
    global _retriever_cache

    if _retriever_cache is None:
        try:
            _retriever_cache = FAISSRetriever.from_local()
        except FileNotFoundError:
            _retriever_cache = FAISSRetriever.from_s3(S3_BUCKET)

    return _retriever_cache


def list_documents() -> list[dict]:
    """
    List all documents in the index.

    Returns:
        List of document summaries
    """
    retriever = get_retriever()
    doc_ids = retriever.get_document_ids()

    documents = []

    for doc_id in doc_ids:
        chunks = retriever.get_document_chunks(doc_id)
        if not chunks:
            continue

        first_chunk = chunks[0]
        documents.append({
            "document_id": doc_id,
            "title": first_chunk.get("doc_title", doc_id),
            "doc_type": first_chunk.get("doc_type", "unknown"),
            "source_url": first_chunk.get("source_url"),
            "chunks_count": len(chunks),
        })

    return documents


def get_document(document_id: str) -> dict | None:
    """
    Get details for a specific document.

    Args:
        document_id: Document ID

    Returns:
        Document details or None if not found
    """
    retriever = get_retriever()
    chunks = retriever.get_document_chunks(document_id)

    if not chunks:
        return None

    first_chunk = chunks[0]

    # Get unique sections
    sections = []
    seen_sections = set()
    for chunk in chunks:
        section = chunk.get("section_title")
        if section and section not in seen_sections:
            sections.append(section)
            seen_sections.add(section)

    # Reconstruct full text (approximately)
    full_text = "\n\n".join(chunk.get("chunk_text", "") for chunk in chunks)

    return {
        "document_id": document_id,
        "title": first_chunk.get("doc_title", document_id),
        "doc_type": first_chunk.get("doc_type", "unknown"),
        "source_url": first_chunk.get("source_url"),
        "chunks_count": len(chunks),
        "sections": sections,
        "text_preview": full_text[:2000] + "..." if len(full_text) > 2000 else full_text,
        "chunks": [
            {
                "chunk_id": c.get("chunk_id"),
                "chunk_index": c.get("chunk_index"),
                "section_title": c.get("section_title"),
                "page_number": c.get("page_number"),
                "text_preview": c.get("chunk_text", "")[:200] + "...",
            }
            for c in chunks
        ],
    }


def handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for documents API.

    Routes:
        GET /documents - List all documents
        GET /documents/{id} - Get document details

    Args:
        event: API Gateway event
        context: Lambda context

    Returns:
        Response dict with status code and body
    """
    try:
        # Get HTTP method and path
        http_method = event.get("httpMethod", event.get("requestContext", {}).get("http", {}).get("method", "GET"))
        path = event.get("path", event.get("rawPath", "/documents"))
        path_params = event.get("pathParameters", {}) or {}

        # CORS headers
        headers = {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }

        # Handle OPTIONS for CORS
        if http_method == "OPTIONS":
            return {
                "statusCode": 200,
                "headers": headers,
                "body": "",
            }

        # Route request
        document_id = path_params.get("id") or path_params.get("document_id")

        if document_id:
            # GET /documents/{id}
            document = get_document(document_id)

            if document is None:
                return {
                    "statusCode": 404,
                    "headers": headers,
                    "body": json.dumps({"error": f"Document not found: {document_id}"}),
                }

            return {
                "statusCode": 200,
                "headers": headers,
                "body": json.dumps(document),
            }

        else:
            # GET /documents
            documents = list_documents()

            return {
                "statusCode": 200,
                "headers": headers,
                "body": json.dumps({
                    "documents": documents,
                    "count": len(documents),
                }),
            }

    except Exception as e:
        logger.exception("Error handling documents request")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"error": str(e)}),
        }
