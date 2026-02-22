"""
Lambda handler for health check endpoint.

Returns system health status including FAISS index, Bedrock, and S3 status.
"""
import json
import logging
import os
from datetime import datetime
from typing import Any

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
S3_BUCKET = os.environ.get("S3_BUCKET", "pediatric-research-rag")
INDEX_PREFIX = os.environ.get("INDEX_PREFIX", "processed/index/")
EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


def get_s3_client():
    """Get boto3 S3 client."""
    return boto3.client("s3", region_name=AWS_REGION)


def check_faiss_index(bucket: str) -> dict:
    """
    Check FAISS index status.

    Returns:
        Dict with status, vector count, document count, and last updated time
    """
    s3 = get_s3_client()

    try:
        # Check index file exists
        index_key = f"{INDEX_PREFIX}faiss_index.bin"
        index_response = s3.head_object(Bucket=bucket, Key=index_key)
        last_modified = index_response["LastModified"].isoformat()

        # Load metadata to get stats
        metadata_key = f"{INDEX_PREFIX}faiss_metadata.json"
        metadata_response = s3.get_object(Bucket=bucket, Key=metadata_key)
        metadata = json.loads(metadata_response["Body"].read())

        # Count unique documents
        doc_ids = set()
        for chunk in metadata:
            doc_ids.add(chunk.get("document_id", ""))

        return {
            "status": "ok",
            "vectors": len(metadata),
            "documents": len(doc_ids),
            "last_updated": last_modified,
        }

    except s3.exceptions.NoSuchKey:
        return {
            "status": "not_found",
            "vectors": 0,
            "documents": 0,
            "last_updated": None,
            "message": "Index not yet created",
        }
    except Exception as e:
        logger.error(f"Error checking FAISS index: {e}")
        return {
            "status": "error",
            "message": str(e),
        }


def check_bedrock() -> dict:
    """
    Check Bedrock model availability.

    Returns:
        Dict with status and model info
    """
    try:
        bedrock = boto3.client("bedrock", region_name=AWS_REGION)

        # Check if model is available
        response = bedrock.get_foundation_model(modelIdentifier=LLM_MODEL_ID)

        return {
            "status": "ok",
            "model": LLM_MODEL_ID,
            "model_name": response.get("modelDetails", {}).get("modelName", "Unknown"),
        }

    except bedrock.exceptions.ResourceNotFoundException:
        return {
            "status": "unavailable",
            "model": LLM_MODEL_ID,
            "message": "Model not found or not enabled",
        }
    except Exception as e:
        # Bedrock describe may not be available, but runtime might work
        logger.warning(f"Could not check Bedrock model: {e}")
        return {
            "status": "unknown",
            "model": LLM_MODEL_ID,
            "message": "Could not verify model status (may still work at runtime)",
        }


def check_s3_bucket(bucket: str) -> dict:
    """
    Check S3 bucket accessibility.

    Returns:
        Dict with status and bucket info
    """
    s3 = get_s3_client()

    try:
        s3.head_bucket(Bucket=bucket)

        # Count objects in raw/ and processed/
        raw_count = 0
        processed_count = 0

        raw_response = s3.list_objects_v2(Bucket=bucket, Prefix="raw/", MaxKeys=1000)
        raw_count = raw_response.get("KeyCount", 0)

        processed_response = s3.list_objects_v2(
            Bucket=bucket, Prefix="processed/chunks/", MaxKeys=1000
        )
        processed_count = processed_response.get("KeyCount", 0)

        return {
            "status": "ok",
            "bucket": bucket,
            "raw_objects": raw_count,
            "processed_objects": processed_count,
        }

    except Exception as e:
        logger.error(f"Error checking S3 bucket: {e}")
        return {
            "status": "error",
            "bucket": bucket,
            "message": str(e),
        }


def handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for health check endpoint.

    Returns system health status.

    Args:
        event: API Gateway event
        context: Lambda context

    Returns:
        Response dict with status code and body
    """
    try:
        # Run all health checks
        faiss_status = check_faiss_index(S3_BUCKET)
        bedrock_status = check_bedrock()
        s3_status = check_s3_bucket(S3_BUCKET)

        # Determine overall health
        all_ok = (
            faiss_status.get("status") == "ok"
            and s3_status.get("status") == "ok"
            and bedrock_status.get("status") in ("ok", "unknown")
        )

        overall_status = "healthy" if all_ok else "degraded"

        # Build response
        response_body = {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "components": {
                "faiss_index": faiss_status,
                "bedrock": bedrock_status,
                "s3": s3_status,
            },
        }

        # Add last refresh info if available
        if faiss_status.get("last_updated"):
            response_body["last_index_update"] = faiss_status["last_updated"]

        status_code = 200 if overall_status == "healthy" else 503

        return {
            "statusCode": status_code,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(response_body, indent=2),
        }

    except Exception as e:
        logger.exception("Error in health check")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({
                "status": "error",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "error": str(e),
            }),
        }
