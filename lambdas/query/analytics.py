"""
Query analytics and logging module.

Logs queries to DynamoDB for analytics and debugging.
Publishes metrics to CloudWatch for monitoring.
"""
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import boto3

logger = logging.getLogger(__name__)

# Configuration
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
QUERY_LOG_TABLE = os.environ.get("QUERY_LOG_TABLE", "pediatric-rag-query-logs")
METRICS_NAMESPACE = "PediatricRAG"
ENABLE_QUERY_LOGGING = os.environ.get("ENABLE_QUERY_LOGGING", "true").lower() == "true"


@dataclass
class QueryMetrics:
    """Metrics collected for a single query."""
    query_id: str
    question: str
    chunks_retrieved: int
    chunk_ids: list[str]
    chunk_scores: list[float]
    confidence_level: str
    confidence_score: float
    response_length: int
    latency_ms: int
    model_id: str
    success: bool
    error_message: Optional[str] = None


def get_dynamodb_resource():
    """Get boto3 DynamoDB resource."""
    return boto3.resource("dynamodb", region_name=AWS_REGION)


def get_cloudwatch_client():
    """Get boto3 CloudWatch client."""
    return boto3.client("cloudwatch", region_name=AWS_REGION)


def log_query(metrics: QueryMetrics) -> bool:
    """
    Log query metrics to DynamoDB.

    Args:
        metrics: QueryMetrics dataclass with query data

    Returns:
        True if logged successfully, False otherwise
    """
    if not ENABLE_QUERY_LOGGING:
        return True

    try:
        dynamodb = get_dynamodb_resource()
        table = dynamodb.Table(QUERY_LOG_TABLE)

        item = {
            "query_id": metrics.query_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "question": metrics.question[:500],  # Truncate long questions
            "chunks_retrieved": metrics.chunks_retrieved,
            "chunk_ids": metrics.chunk_ids,
            "chunk_scores": [str(s) for s in metrics.chunk_scores],  # DynamoDB doesn't like floats
            "confidence_level": metrics.confidence_level,
            "confidence_score": str(metrics.confidence_score),
            "response_length": metrics.response_length,
            "latency_ms": metrics.latency_ms,
            "model_id": metrics.model_id,
            "success": metrics.success,
        }

        if metrics.error_message:
            item["error_message"] = metrics.error_message[:500]

        table.put_item(Item=item)
        logger.info(f"Logged query {metrics.query_id} to DynamoDB")
        return True

    except Exception as e:
        logger.warning(f"Failed to log query to DynamoDB: {e}")
        return False


def publish_query_metrics(metrics: QueryMetrics) -> bool:
    """
    Publish query metrics to CloudWatch.

    Args:
        metrics: QueryMetrics dataclass with query data

    Returns:
        True if published successfully, False otherwise
    """
    try:
        cloudwatch = get_cloudwatch_client()

        metric_data = [
            {
                "MetricName": "QueryLatency",
                "Value": metrics.latency_ms,
                "Unit": "Milliseconds",
                "Dimensions": [
                    {"Name": "ConfidenceLevel", "Value": metrics.confidence_level},
                ],
            },
            {
                "MetricName": "ChunksRetrieved",
                "Value": metrics.chunks_retrieved,
                "Unit": "Count",
            },
            {
                "MetricName": "ConfidenceScore",
                "Value": metrics.confidence_score,
                "Unit": "None",
            },
            {
                "MetricName": "QuerySuccess",
                "Value": 1 if metrics.success else 0,
                "Unit": "Count",
            },
            {
                "MetricName": "ResponseLength",
                "Value": metrics.response_length,
                "Unit": "Count",
            },
        ]

        # Add error metric if query failed
        if not metrics.success:
            metric_data.append({
                "MetricName": "QueryErrors",
                "Value": 1,
                "Unit": "Count",
            })

        cloudwatch.put_metric_data(
            Namespace=METRICS_NAMESPACE,
            MetricData=metric_data,
        )
        logger.info(f"Published metrics for query {metrics.query_id}")
        return True

    except Exception as e:
        logger.warning(f"Failed to publish CloudWatch metrics: {e}")
        return False


class QueryTimer:
    """Context manager for timing queries."""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        return False

    @property
    def elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds."""
        if self.start_time is None or self.end_time is None:
            return 0
        return int((self.end_time - self.start_time) * 1000)


def generate_query_id() -> str:
    """Generate a unique query ID."""
    return str(uuid.uuid4())
