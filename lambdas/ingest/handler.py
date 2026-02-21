"""
Lambda handler for document ingestion.

Triggered by S3 upload, parses and chunks documents, stores chunks for embedding.
"""
import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import boto3

from .chunker import chunk_document
from .parsers import parse_document

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
S3_BUCKET = os.environ.get("S3_BUCKET", "pediatric-research-rag")
CHUNKS_PREFIX = os.environ.get("CHUNKS_PREFIX", "processed/chunks/")


def get_s3_client():
    """Get boto3 S3 client."""
    return boto3.client("s3")


def download_from_s3(bucket: str, key: str, local_path: Path) -> None:
    """Download a file from S3 to local path."""
    s3 = get_s3_client()
    s3.download_file(bucket, key, str(local_path))
    logger.info(f"Downloaded s3://{bucket}/{key} to {local_path}")


def upload_chunks_to_s3(
    chunks: list[dict],
    document_id: str,
    bucket: str,
) -> str:
    """
    Upload chunk metadata to S3.

    Args:
        chunks: List of chunk metadata dictionaries
        document_id: Document identifier
        bucket: S3 bucket name

    Returns:
        S3 key where chunks were stored
    """
    s3 = get_s3_client()

    chunks_key = f"{CHUNKS_PREFIX}{document_id}_chunks.json"

    s3.put_object(
        Bucket=bucket,
        Key=chunks_key,
        Body=json.dumps(chunks, ensure_ascii=False, indent=2),
        ContentType="application/json",
    )

    logger.info(f"Uploaded {len(chunks)} chunks to s3://{bucket}/{chunks_key}")
    return chunks_key


def handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for document ingestion.

    Triggered by S3 upload events. Downloads the document, parses it,
    chunks the text, and stores chunk metadata in S3.

    Args:
        event: S3 event containing bucket and key information
        context: Lambda context

    Returns:
        Response dict with status code and body
    """
    try:
        # Extract S3 event information
        records = event.get("Records", [])
        if not records:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No records in event"}),
            }

        results = []

        for record in records:
            s3_info = record.get("s3", {})
            bucket = s3_info.get("bucket", {}).get("name")
            key = s3_info.get("object", {}).get("key")

            if not bucket or not key:
                logger.warning(f"Missing bucket or key in record: {record}")
                continue

            logger.info(f"Processing s3://{bucket}/{key}")

            # Skip non-document files
            if not key.lower().endswith((".pdf", ".json")):
                logger.info(f"Skipping non-document file: {key}")
                continue

            # Skip metadata JSON files
            if "_metadata.json" in key or "_chunks.json" in key:
                logger.info(f"Skipping metadata file: {key}")
                continue

            # Download file to temp location
            suffix = Path(key).suffix
            with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = Path(tmp.name)

            try:
                download_from_s3(bucket, key, tmp_path)

                # Also download metadata sidecar if it exists
                metadata_key = key.rsplit(".", 1)[0] + "_metadata.json"
                metadata_path = tmp_path.parent / f"{tmp_path.stem}_metadata.json"
                try:
                    download_from_s3(bucket, metadata_key, metadata_path)
                except Exception:
                    logger.info(f"No metadata sidecar found for {key}")

                # Parse document
                parsed = parse_document(tmp_path)

                # Get source URL from metadata
                source_url = parsed.metadata.get("source_url")

                # Chunk document
                chunks = chunk_document(
                    text=parsed.text,
                    document_id=parsed.document_id,
                    doc_title=parsed.title,
                    doc_type=parsed.doc_type,
                    page_boundaries=parsed.page_boundaries,
                    section_titles=parsed.section_titles,
                    source_url=source_url,
                )

                # Convert to dicts for JSON serialization
                chunk_dicts = []
                for chunk in chunks:
                    chunk_dicts.append({
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "chunk_text": chunk.chunk_text,
                        "chunk_index": chunk.chunk_index,
                        "page_number": chunk.page_number,
                        "section_title": chunk.section_title,
                        "source_url": chunk.source_url,
                        "doc_title": chunk.doc_title,
                        "doc_type": chunk.doc_type,
                    })

                # Upload chunks to S3
                chunks_key = upload_chunks_to_s3(chunk_dicts, parsed.document_id, bucket)

                results.append({
                    "document_id": parsed.document_id,
                    "title": parsed.title,
                    "chunks_count": len(chunks),
                    "chunks_key": chunks_key,
                })

            finally:
                # Cleanup temp files
                if tmp_path.exists():
                    tmp_path.unlink()
                if metadata_path.exists():
                    metadata_path.unlink()

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Documents processed successfully",
                "results": results,
            }),
        }

    except Exception as e:
        logger.exception("Error processing documents")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }
