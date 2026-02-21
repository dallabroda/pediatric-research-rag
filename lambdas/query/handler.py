"""
Lambda handler for RAG query pipeline.

Receives questions via API Gateway, retrieves relevant context,
and generates answers using Claude via Bedrock.
"""
import json
import logging
import os
import time
from typing import Any

import boto3

from .prompts import (
    SYSTEM_PROMPT,
    ContextChunk,
    build_no_context_response,
    create_messages,
)
from .retriever import FAISSRetriever

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
S3_BUCKET = os.environ.get("S3_BUCKET", "pediatric-research-rag")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
TOP_K = int(os.environ.get("TOP_K", "5"))

# Cache for retriever (reuse across warm Lambda invocations)
_retriever_cache = None


def get_retriever() -> FAISSRetriever:
    """Get or create cached retriever."""
    global _retriever_cache

    if _retriever_cache is None:
        # Try local first (for local development), then S3
        try:
            _retriever_cache = FAISSRetriever.from_local()
            logger.info("Loaded retriever from local index")
        except FileNotFoundError:
            _retriever_cache = FAISSRetriever.from_s3(S3_BUCKET)
            logger.info("Loaded retriever from S3")

    return _retriever_cache


def get_bedrock_client():
    """Get boto3 Bedrock Runtime client."""
    return boto3.client("bedrock-runtime")


def generate_answer(
    question: str,
    context_chunks: list[ContextChunk],
    bedrock_client=None,
) -> str:
    """
    Generate an answer using Claude.

    Args:
        question: User's question
        context_chunks: Retrieved context
        bedrock_client: Optional Bedrock client

    Returns:
        Generated answer text
    """
    if bedrock_client is None:
        bedrock_client = get_bedrock_client()

    # Build messages
    messages = create_messages(question, context_chunks)

    # Call Claude via Bedrock
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": messages,
    })

    # Retry with exponential backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = bedrock_client.invoke_model(
                modelId=LLM_MODEL_ID,
                body=body,
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())
            answer = response_body["content"][0]["text"]

            return answer

        except Exception as e:
            if "ThrottlingException" in str(type(e).__name__):
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Throttled, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
            else:
                raise


def handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for query endpoint.

    Expects POST request with JSON body: {"question": "..."}

    Args:
        event: API Gateway event
        context: Lambda context

    Returns:
        Response dict with status code and body
    """
    try:
        # Parse request
        if event.get("body"):
            if isinstance(event["body"], str):
                body = json.loads(event["body"])
            else:
                body = event["body"]
        else:
            body = event  # Direct invocation

        question = body.get("question", "").strip()

        if not question:
            return {
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                },
                "body": json.dumps({"error": "Question is required"}),
            }

        logger.info(f"Processing question: {question[:100]}...")

        # Get retriever and search
        retriever = get_retriever()
        results = retriever.search(question, top_k=TOP_K)

        # Convert to context chunks
        context_chunks = []
        sources = []

        for result in results:
            context_chunks.append(ContextChunk(
                text=result.chunk_text,
                doc_title=result.doc_title,
                doc_type=result.doc_type,
                source_url=result.source_url,
                section_title=result.section_title,
                page_number=result.page_number,
            ))

            sources.append({
                "document_id": result.document_id,
                "doc_title": result.doc_title,
                "doc_type": result.doc_type,
                "score": result.score,
                "source_url": result.source_url,
                "section": result.section_title,
                "page": result.page_number,
            })

        # Generate answer
        if context_chunks:
            bedrock = get_bedrock_client()
            answer = generate_answer(question, context_chunks, bedrock)
        else:
            answer = build_no_context_response(question)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({
                "question": question,
                "answer": answer,
                "sources": sources,
                "chunks_retrieved": len(context_chunks),
            }),
        }

    except Exception as e:
        logger.exception("Error processing query")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"error": str(e)}),
        }
