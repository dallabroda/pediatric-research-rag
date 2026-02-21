"""
Lambda handler for lineage API endpoint.

Provides provenance tracking for chunks in the RAG system.
Returns lineage information including source, parsing, and embedding metadata.
"""
import json
import logging
import os
from typing import Any, Optional

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
S3_BUCKET = os.environ.get("S3_BUCKET", "pediatric-research-rag")
INDEX_PREFIX = os.environ.get("INDEX_PREFIX", "processed/index/")
EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")

# Chunking configuration (from settings)
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))


def get_s3_client():
    """Get boto3 S3 client."""
    return boto3.client("s3")


def load_metadata(bucket: str) -> list[dict]:
    """
    Load chunk metadata from S3.

    Args:
        bucket: S3 bucket name

    Returns:
        List of chunk metadata dictionaries
    """
    s3 = get_s3_client()
    metadata_key = f"{INDEX_PREFIX}faiss_metadata.json"

    try:
        response = s3.get_object(Bucket=bucket, Key=metadata_key)
        return json.loads(response["Body"].read())
    except s3.exceptions.NoSuchKey:
        logger.error(f"Metadata file not found: s3://{bucket}/{metadata_key}")
        return []
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        return []


def find_chunk_by_id(metadata: list[dict], chunk_id: str) -> Optional[dict]:
    """
    Find a chunk by its ID.

    Args:
        metadata: List of all chunk metadata
        chunk_id: Chunk ID to find

    Returns:
        Chunk metadata dict or None if not found
    """
    for chunk in metadata:
        if chunk.get("chunk_id") == chunk_id:
            return chunk
    return None


def find_chunks_by_document(metadata: list[dict], document_id: str) -> list[dict]:
    """
    Find all chunks belonging to a document.

    Args:
        metadata: List of all chunk metadata
        document_id: Document ID to filter by

    Returns:
        List of chunk metadata for the document
    """
    return [
        chunk for chunk in metadata
        if chunk.get("document_id") == document_id
    ]


def build_lineage_response(chunk: dict) -> dict:
    """
    Build the lineage response for a chunk.

    Args:
        chunk: Chunk metadata dictionary

    Returns:
        Formatted lineage response
    """
    return {
        "chunk_id": chunk.get("chunk_id", ""),
        "document_id": chunk.get("document_id", ""),
        "lineage": {
            "source_url": chunk.get("source_url"),
            "document_hash": chunk.get("document_hash", ""),
            "ingestion_timestamp": chunk.get("ingestion_timestamp", ""),
            "parser": f"{chunk.get('parser_name', 'unknown')}:v{chunk.get('parser_version', '1.0.0')}",
            "chunking_method": "recursive_char",
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_model": EMBEDDING_MODEL_ID,
            "chunk_index": chunk.get("chunk_index", 0),
            "section_title": chunk.get("section_title"),
            "page_number": chunk.get("page_number"),
        },
        "quality": {
            "extraction_confidence": chunk.get("extraction_confidence", 1.0),
            "char_count": chunk.get("char_count", len(chunk.get("chunk_text", ""))),
            "word_count": chunk.get("word_count", 0),
            "avg_word_length": chunk.get("avg_word_length", 0.0),
            "whitespace_ratio": chunk.get("whitespace_ratio", 0.0),
            "has_ocr_artifacts": chunk.get("has_ocr_artifacts", False),
        },
        "document": {
            "title": chunk.get("doc_title", ""),
            "type": chunk.get("doc_type", ""),
        },
    }


def handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for lineage endpoint.

    Supports:
    - GET /lineage/{chunk_id} - Get lineage for specific chunk
    - GET /lineage?document_id={id} - Get lineage for all chunks in a document

    Args:
        event: API Gateway event
        context: Lambda context

    Returns:
        Response dict with status code and body
    """
    try:
        # Parse request
        http_method = event.get("httpMethod", "GET")
        path_params = event.get("pathParameters") or {}
        query_params = event.get("queryStringParameters") or {}

        chunk_id = path_params.get("id")
        document_id = query_params.get("document_id")

        if http_method != "GET":
            return {
                "statusCode": 405,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                },
                "body": json.dumps({"error": "Method not allowed"}),
            }

        # Load metadata
        metadata = load_metadata(S3_BUCKET)

        if not metadata:
            return {
                "statusCode": 404,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                },
                "body": json.dumps({"error": "No metadata found. Index may not be built."}),
            }

        # Get specific chunk by ID
        if chunk_id:
            chunk = find_chunk_by_id(metadata, chunk_id)
            if not chunk:
                return {
                    "statusCode": 404,
                    "headers": {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*",
                    },
                    "body": json.dumps({"error": f"Chunk not found: {chunk_id}"}),
                }

            response = build_lineage_response(chunk)

        # Get all chunks for a document
        elif document_id:
            chunks = find_chunks_by_document(metadata, document_id)
            if not chunks:
                return {
                    "statusCode": 404,
                    "headers": {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*",
                    },
                    "body": json.dumps({"error": f"Document not found: {document_id}"}),
                }

            response = {
                "document_id": document_id,
                "chunk_count": len(chunks),
                "chunks": [build_lineage_response(c) for c in chunks],
            }

        # List all documents with summary
        else:
            # Group by document
            documents = {}
            for chunk in metadata:
                doc_id = chunk.get("document_id", "unknown")
                if doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "title": chunk.get("doc_title", ""),
                        "type": chunk.get("doc_type", ""),
                        "chunk_count": 0,
                        "source_url": chunk.get("source_url"),
                        "ingestion_timestamp": chunk.get("ingestion_timestamp", ""),
                        "extraction_confidence": chunk.get("extraction_confidence", 1.0),
                    }
                documents[doc_id]["chunk_count"] += 1

            response = {
                "total_documents": len(documents),
                "total_chunks": len(metadata),
                "documents": list(documents.values()),
            }

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(response),
        }

    except Exception as e:
        logger.exception("Error processing lineage request")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"error": str(e)}),
        }
