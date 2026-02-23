"""
Lambda handler for RAG query pipeline.

Receives questions via API Gateway, retrieves relevant context,
and generates answers using Claude via Bedrock.
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import boto3

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.retry import retry_with_backoff

from .confidence import (
    ConfidenceScore,
    RetrievalResult as ConfidenceRetrievalResult,
    analyze_sources,
    compute_answer_confidence,
)
from .prompts import (
    CROSS_DOCUMENT_SYSTEM_PROMPT,
    LOW_CONFIDENCE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    ContextChunk,
    build_low_confidence_prompt,
    build_no_context_response,
    build_synthesis_prompt,
    create_messages,
)
from .retriever import FAISSRetriever
from .analytics import (
    QueryMetrics,
    QueryTimer,
    generate_query_id,
    log_query,
    publish_query_metrics,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
S3_BUCKET = os.environ.get("S3_BUCKET", "pediatric-research-rag")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
TOP_K = int(os.environ.get("TOP_K", "5"))
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

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
    # Read region at runtime for Streamlit Cloud compatibility
    region = os.environ.get("AWS_REGION", "us-east-1")
    return boto3.client("bedrock-runtime", region_name=region)


def _is_claude_model(model_id: str) -> bool:
    """Check if model is Claude/Anthropic."""
    return "anthropic" in model_id.lower() or "claude" in model_id.lower()


@retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0)
def generate_answer(
    question: str,
    context_chunks: list[ContextChunk],
    bedrock_client=None,
    system_prompt: str = None,
) -> str:
    """
    Generate an answer using Claude or Amazon Nova.

    Args:
        question: User's question
        context_chunks: Retrieved context
        bedrock_client: Optional Bedrock client
        system_prompt: Optional override for system prompt

    Returns:
        Generated answer text
    """
    if bedrock_client is None:
        bedrock_client = get_bedrock_client()

    # Build messages
    messages = create_messages(question, context_chunks)
    effective_system_prompt = system_prompt or SYSTEM_PROMPT

    # Build request body based on model type
    if _is_claude_model(LLM_MODEL_ID):
        # Claude/Anthropic format
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_TOKENS,
            "system": effective_system_prompt,
            "messages": messages,
        })
    else:
        # Amazon Nova format
        user_content = messages[0]["content"] if messages else ""
        body = json.dumps({
            "system": [{"text": effective_system_prompt}],
            "messages": [{"role": "user", "content": [{"text": user_content}]}],
            "inferenceConfig": {"maxTokens": MAX_TOKENS},
        })

    response = bedrock_client.invoke_model(
        modelId=LLM_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())

    # Parse response based on model type
    if _is_claude_model(LLM_MODEL_ID):
        answer = response_body["content"][0]["text"]
    else:
        # Amazon Nova format
        answer = response_body["output"]["message"]["content"][0]["text"]

    return answer


def _invoke_llm(
    user_content: str,
    system_prompt: str,
    bedrock_client,
) -> str:
    """Helper to invoke LLM with correct format for model type."""
    if _is_claude_model(LLM_MODEL_ID):
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_TOKENS,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_content}],
        })
    else:
        body = json.dumps({
            "system": [{"text": system_prompt}],
            "messages": [{"role": "user", "content": [{"text": user_content}]}],
            "inferenceConfig": {"maxTokens": MAX_TOKENS},
        })

    response = bedrock_client.invoke_model(
        modelId=LLM_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())

    if _is_claude_model(LLM_MODEL_ID):
        return response_body["content"][0]["text"]
    else:
        return response_body["output"]["message"]["content"][0]["text"]


@retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0)
def generate_synthesis_answer(
    question: str,
    prompt: str,
    bedrock_client=None,
) -> str:
    """
    Generate a synthesis answer for cross-document analysis.

    Args:
        question: User's question
        prompt: Pre-built synthesis prompt
        bedrock_client: Optional Bedrock client

    Returns:
        Generated synthesis answer
    """
    if bedrock_client is None:
        bedrock_client = get_bedrock_client()

    return _invoke_llm(prompt, CROSS_DOCUMENT_SYSTEM_PROMPT, bedrock_client)


@retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0)
def generate_low_confidence_answer(
    question: str,
    prompt: str,
    bedrock_client=None,
) -> str:
    """
    Generate an answer when confidence is low.

    Args:
        question: User's question
        prompt: Pre-built low-confidence prompt
        bedrock_client: Optional Bedrock client

    Returns:
        Generated answer acknowledging limitations
    """
    if bedrock_client is None:
        bedrock_client = get_bedrock_client()

    return _invoke_llm(prompt, LOW_CONFIDENCE_SYSTEM_PROMPT, bedrock_client)


def build_graceful_degradation_response(
    question: str,
    closest_findings: list[dict],
) -> dict:
    """
    Build a graceful degradation response when no good context is found.

    Args:
        question: User's question
        closest_findings: Any marginally relevant findings

    Returns:
        Response dict with helpful suggestions
    """
    suggested_queries = [
        "Try asking about specific treatment protocols",
        "Try narrowing to a specific cancer type (e.g., leukemia, neuroblastoma)",
        "Try asking about clinical trial outcomes or eligibility",
    ]

    return {
        "answer": (
            "I don't have enough information in the research database to answer this "
            "question confidently. The available documents may not cover this specific topic."
        ),
        "confidence": {
            "level": "low",
            "score": 0.0,
            "reasoning": "No relevant sources found in the research database",
        },
        "closest_findings": closest_findings,
        "suggested_queries": suggested_queries,
    }


def handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for query endpoint.

    Expects POST request with JSON body: {"question": "..."}

    Supports optional parameters:
    - synthesis: bool - Enable cross-document synthesis mode
    - include_confidence: bool - Include confidence scoring (default: true)

    Args:
        event: API Gateway event
        context: Lambda context

    Returns:
        Response dict with status code and body
    """
    query_id = generate_query_id()
    timer = QueryTimer()

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
        enable_synthesis = body.get("synthesis", False)
        include_confidence = body.get("include_confidence", True)

        if not question:
            return {
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                },
                "body": json.dumps({"error": "Question is required"}),
            }

        logger.info(f"Processing query {query_id}: {question[:100]}...")

        # Start timing the query
        timer.__enter__()

        # Get retriever and search
        retriever = get_retriever()
        results = retriever.search(question, top_k=TOP_K)

        # Convert to context chunks and prepare for confidence scoring
        context_chunks = []
        sources = []
        confidence_results = []

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

            # Build confidence-compatible result
            confidence_results.append(ConfidenceRetrievalResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                chunk_text=result.chunk_text,
                doc_title=result.doc_title,
                doc_type=result.doc_type,
                score=result.score,
                page_number=result.page_number,
                section_title=result.section_title,
                source_url=result.source_url,
            ))

        # Compute confidence
        confidence = compute_answer_confidence(question, confidence_results)

        # Analyze sources for related findings
        source_analysis = analyze_sources(confidence_results) if confidence_results else {}

        # Get Bedrock client
        bedrock = get_bedrock_client()

        # Generate answer based on confidence level
        if not context_chunks:
            # No results at all
            response_data = build_graceful_degradation_response(question, [])
            answer = response_data["answer"]
        elif confidence.level == "low":
            # Low confidence - use graceful degradation
            prompt = build_low_confidence_prompt(question, context_chunks)
            answer = generate_low_confidence_answer(question, prompt, bedrock)
        elif enable_synthesis and len(set(r.document_id for r in confidence_results)) >= 2:
            # Multiple unique documents - use synthesis mode
            from .prompts import ContextChunkWithDate
            synthesis_chunks = [
                ContextChunkWithDate(
                    text=r.chunk_text,
                    doc_title=r.doc_title,
                    doc_type=r.doc_type,
                    source_url=r.source_url,
                    section_title=r.section_title,
                    page_number=r.page_number,
                    similarity_score=r.score,
                )
                for r in confidence_results
            ]
            prompt = build_synthesis_prompt(question, synthesis_chunks)
            answer = generate_synthesis_answer(question, prompt, bedrock)
        else:
            # Standard answer generation
            answer = generate_answer(question, context_chunks, bedrock)

        # Build response
        response_body = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks_retrieved": len(context_chunks),
        }

        # Add confidence if requested
        if include_confidence:
            response_body["confidence"] = {
                "level": confidence.level,
                "score": round(confidence.score, 3),
                "reasoning": confidence.reasoning,
            }

        # Add related findings if any
        if source_analysis.get("related_findings"):
            response_body["related_findings"] = source_analysis["related_findings"]

        # Add suggested queries for low confidence
        if confidence.level == "low":
            response_body["suggested_queries"] = [
                "Try asking about specific treatment protocols",
                "Try narrowing to a specific cancer type",
                "Try asking about clinical trial outcomes",
            ]

        # Stop timer and add query_id to response
        timer.__exit__(None, None, None)
        response_body["query_id"] = query_id

        # Log and publish metrics (async, non-blocking)
        metrics = QueryMetrics(
            query_id=query_id,
            question=question,
            chunks_retrieved=len(context_chunks),
            chunk_ids=[r.chunk_id for r in confidence_results],
            chunk_scores=[r.score for r in confidence_results],
            confidence_level=confidence.level,
            confidence_score=confidence.score,
            response_length=len(answer),
            latency_ms=timer.elapsed_ms,
            model_id=LLM_MODEL_ID,
            success=True,
        )
        log_query(metrics)
        publish_query_metrics(metrics)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(response_body),
        }

    except Exception as e:
        logger.exception("Error processing query")

        # Log failed query metrics
        try:
            metrics = QueryMetrics(
                query_id=query_id,
                question=body.get("question", "") if "body" in dir() else "",
                chunks_retrieved=0,
                chunk_ids=[],
                chunk_scores=[],
                confidence_level="error",
                confidence_score=0.0,
                response_length=0,
                latency_ms=timer.elapsed_ms if timer.end_time else 0,
                model_id=LLM_MODEL_ID,
                success=False,
                error_message=str(e),
            )
            log_query(metrics)
            publish_query_metrics(metrics)
        except Exception:
            pass  # Don't fail on metrics logging failure

        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"error": str(e), "query_id": query_id}),
        }
