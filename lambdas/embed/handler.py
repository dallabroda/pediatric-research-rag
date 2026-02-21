"""
Lambda handler for embedding generation.

Takes chunked documents and creates embeddings using Bedrock Titan,
then builds/updates the FAISS index.
"""
import json
import logging
import os
import time
from io import BytesIO
from tempfile import TemporaryDirectory
from typing import Any

import boto3
import faiss
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
S3_BUCKET = os.environ.get("S3_BUCKET", "pediatric-research-rag")
EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
EMBEDDING_DIMENSION = 1024
INDEX_PREFIX = os.environ.get("INDEX_PREFIX", "processed/index/")
FAISS_INDEX_FILE = "faiss_index.bin"
FAISS_METADATA_FILE = "faiss_metadata.json"


def get_bedrock_client():
    """Get boto3 Bedrock Runtime client."""
    return boto3.client("bedrock-runtime")


def get_s3_client():
    """Get boto3 S3 client."""
    return boto3.client("s3")


def get_embedding(text: str, bedrock_client=None) -> list[float]:
    """
    Get embedding for text using Bedrock Titan Embeddings V2.

    Args:
        text: Text to embed
        bedrock_client: Optional pre-created Bedrock client

    Returns:
        1024-dimensional embedding vector
    """
    if bedrock_client is None:
        bedrock_client = get_bedrock_client()

    # Titan V2 accepts text input
    body = json.dumps({
        "inputText": text,
        "dimensions": EMBEDDING_DIMENSION,
        "normalize": True,  # L2 normalize for cosine similarity
    })

    # Retry with exponential backoff for throttling
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = bedrock_client.invoke_model(
                modelId=EMBEDDING_MODEL_ID,
                body=body,
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())
            return response_body["embedding"]

        except bedrock_client.exceptions.ThrottlingException:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Throttled, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise


def embed_chunks(chunks: list[dict], bedrock_client=None) -> tuple[np.ndarray, list[dict]]:
    """
    Generate embeddings for a list of chunks.

    Args:
        chunks: List of chunk metadata dictionaries
        bedrock_client: Optional pre-created Bedrock client

    Returns:
        Tuple of (embeddings array, chunk metadata list)
    """
    if bedrock_client is None:
        bedrock_client = get_bedrock_client()

    embeddings = []
    valid_chunks = []

    for i, chunk in enumerate(chunks):
        text = chunk.get("chunk_text", "")
        if not text.strip():
            logger.warning(f"Empty chunk {i}, skipping")
            continue

        try:
            embedding = get_embedding(text, bedrock_client)
            embeddings.append(embedding)
            valid_chunks.append(chunk)

            if (i + 1) % 10 == 0:
                logger.info(f"Embedded {i + 1}/{len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Failed to embed chunk {i}: {e}")
            continue

    embeddings_array = np.array(embeddings, dtype=np.float32)
    logger.info(f"Generated {len(embeddings)} embeddings, shape: {embeddings_array.shape}")

    return embeddings_array, valid_chunks


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS index from embeddings.

    Uses IndexFlatIP (inner product) with L2-normalized vectors
    for cosine similarity search.

    Args:
        embeddings: 2D array of embeddings (N x dim)

    Returns:
        FAISS index ready for search
    """
    dimension = embeddings.shape[1]

    # Use inner product index - with L2 normalized vectors this equals cosine similarity
    index = faiss.IndexFlatIP(dimension)

    # Ensure embeddings are L2 normalized
    faiss.normalize_L2(embeddings)

    index.add(embeddings)

    logger.info(f"Built FAISS index with {index.ntotal} vectors")
    return index


def save_index_to_s3(
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    bucket: str,
) -> dict:
    """
    Save FAISS index and metadata to S3.

    Args:
        index: FAISS index to save
        metadata: List of chunk metadata
        bucket: S3 bucket name

    Returns:
        Dict with S3 keys for index and metadata files
    """
    s3 = get_s3_client()

    # Save FAISS index to bytes
    index_bytes = BytesIO()
    faiss.write_index(index, faiss.PyCallbackIOWriter(index_bytes.write))
    index_bytes.seek(0)

    index_key = f"{INDEX_PREFIX}{FAISS_INDEX_FILE}"
    s3.put_object(
        Bucket=bucket,
        Key=index_key,
        Body=index_bytes.getvalue(),
        ContentType="application/octet-stream",
    )
    logger.info(f"Saved FAISS index to s3://{bucket}/{index_key}")

    # Save metadata as JSON
    metadata_key = f"{INDEX_PREFIX}{FAISS_METADATA_FILE}"
    s3.put_object(
        Bucket=bucket,
        Key=metadata_key,
        Body=json.dumps(metadata, ensure_ascii=False, indent=2),
        ContentType="application/json",
    )
    logger.info(f"Saved metadata to s3://{bucket}/{metadata_key}")

    return {
        "index_key": index_key,
        "metadata_key": metadata_key,
    }


def load_existing_index(bucket: str) -> tuple[faiss.IndexFlatIP | None, list[dict]]:
    """
    Load existing FAISS index and metadata from S3.

    Args:
        bucket: S3 bucket name

    Returns:
        Tuple of (index, metadata) or (None, []) if not found
    """
    s3 = get_s3_client()

    try:
        # Load FAISS index
        index_key = f"{INDEX_PREFIX}{FAISS_INDEX_FILE}"
        response = s3.get_object(Bucket=bucket, Key=index_key)
        index_bytes = response["Body"].read()

        # Write to temp file for FAISS to read
        with TemporaryDirectory() as tmpdir:
            index_path = f"{tmpdir}/index.bin"
            with open(index_path, "wb") as f:
                f.write(index_bytes)
            index = faiss.read_index(index_path)

        # Load metadata
        metadata_key = f"{INDEX_PREFIX}{FAISS_METADATA_FILE}"
        response = s3.get_object(Bucket=bucket, Key=metadata_key)
        metadata = json.loads(response["Body"].read())

        logger.info(f"Loaded existing index with {index.ntotal} vectors")
        return index, metadata

    except s3.exceptions.NoSuchKey:
        logger.info("No existing index found")
        return None, []
    except Exception as e:
        logger.warning(f"Failed to load existing index: {e}")
        return None, []


def handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for embedding generation.

    Processes chunk files from S3 and updates the FAISS index.

    Args:
        event: Event containing chunk file locations
        context: Lambda context

    Returns:
        Response dict with status code and body
    """
    try:
        # Get bucket and chunks to process
        bucket = event.get("bucket", S3_BUCKET)
        chunk_keys = event.get("chunk_keys", [])

        if not chunk_keys:
            # If no specific keys provided, this might be triggered after ingest
            # Check for new chunk files in the chunks prefix
            s3 = get_s3_client()
            response = s3.list_objects_v2(
                Bucket=bucket,
                Prefix="processed/chunks/",
            )
            chunk_keys = [
                obj["Key"] for obj in response.get("Contents", [])
                if obj["Key"].endswith("_chunks.json")
            ]

        if not chunk_keys:
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "No chunks to process"}),
            }

        # Load all chunks
        s3 = get_s3_client()
        all_chunks = []

        for key in chunk_keys:
            try:
                response = s3.get_object(Bucket=bucket, Key=key)
                chunks = json.loads(response["Body"].read())
                all_chunks.extend(chunks)
                logger.info(f"Loaded {len(chunks)} chunks from {key}")
            except Exception as e:
                logger.error(f"Failed to load {key}: {e}")
                continue

        if not all_chunks:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No valid chunks found"}),
            }

        # Generate embeddings
        bedrock = get_bedrock_client()
        embeddings, valid_chunks = embed_chunks(all_chunks, bedrock)

        if len(valid_chunks) == 0:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Failed to embed any chunks"}),
            }

        # Build new index (we rebuild from scratch for simplicity at demo scale)
        index = build_faiss_index(embeddings)

        # Save to S3
        result = save_index_to_s3(index, valid_chunks, bucket)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Index built successfully",
                "vectors_count": index.ntotal,
                "index_key": result["index_key"],
                "metadata_key": result["metadata_key"],
            }),
        }

    except Exception as e:
        logger.exception("Error building index")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }
