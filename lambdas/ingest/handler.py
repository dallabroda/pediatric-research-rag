"""
Lambda handler for document ingestion.

Triggered by S3 upload, parses and chunks documents, stores chunks for embedding.
Includes hash-based deduplication to skip already-indexed documents.
"""
import hashlib
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
INDEX_PREFIX = os.environ.get("INDEX_PREFIX", "processed/index/")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


def get_s3_client():
    """Get boto3 S3 client."""
    return boto3.client("s3", region_name=AWS_REGION)


def compute_document_hash(file_path: Path) -> str:
    """
    Compute SHA-256 hash of document content.

    Args:
        file_path: Path to the document file

    Returns:
        Hex string of the SHA-256 hash
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def check_duplicate(document_hash: str, bucket: str) -> tuple[bool, str | None]:
    """
    Check if document hash already exists in the index metadata.

    Args:
        document_hash: SHA-256 hash of the document
        bucket: S3 bucket name

    Returns:
        Tuple of (is_duplicate, existing_document_id or None)
    """
    s3 = get_s3_client()

    try:
        # Load existing metadata
        metadata_key = f"{INDEX_PREFIX}faiss_metadata.json"
        response = s3.get_object(Bucket=bucket, Key=metadata_key)
        metadata = json.loads(response["Body"].read())

        # Check for matching hash
        for chunk in metadata:
            if chunk.get("document_hash") == document_hash:
                return True, chunk.get("document_id")

        return False, None

    except s3.exceptions.NoSuchKey:
        # No index yet, definitely not a duplicate
        logger.info("No existing index found, document is new")
        return False, None
    except Exception as e:
        logger.warning(f"Could not check for duplicates: {e}")
        return False, None


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

                # Check for duplicate before processing
                doc_hash = compute_document_hash(tmp_path)
                is_dup, existing_doc_id = check_duplicate(doc_hash, bucket)

                if is_dup:
                    logger.info(f"Skipping duplicate document: {key} (matches {existing_doc_id})")
                    results.append({
                        "document_id": existing_doc_id,
                        "status": "skipped",
                        "reason": "duplicate",
                        "hash": doc_hash[:16] + "...",
                    })
                    continue

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
                        "document_hash": doc_hash,  # Store hash for deduplication
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
