"""Tests for the query analytics module."""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from lambdas.query.analytics import (
    QueryMetrics,
    QueryTimer,
    generate_query_id,
    log_query,
    publish_query_metrics,
)


class TestQueryTimer:
    """Tests for QueryTimer context manager."""

    def test_timer_records_elapsed_time(self):
        """Test that timer records elapsed time."""
        import time

        timer = QueryTimer()
        timer.__enter__()
        time.sleep(0.1)  # 100ms
        timer.__exit__(None, None, None)

        # Should be at least 100ms
        assert timer.elapsed_ms >= 100
        assert timer.elapsed_ms < 500  # But not too long

    def test_timer_initially_zero(self):
        """Test that timer starts at zero."""
        timer = QueryTimer()
        assert timer.elapsed_ms == 0


class TestGenerateQueryId:
    """Tests for query ID generation."""

    def test_generates_uuid(self):
        """Test that query ID is a valid UUID."""
        query_id = generate_query_id()
        assert len(query_id) == 36  # UUID format
        assert query_id.count("-") == 4

    def test_unique_ids(self):
        """Test that IDs are unique."""
        ids = [generate_query_id() for _ in range(100)]
        assert len(ids) == len(set(ids))


class TestQueryMetrics:
    """Tests for QueryMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating QueryMetrics."""
        metrics = QueryMetrics(
            query_id="test-123",
            question="What is cancer?",
            chunks_retrieved=5,
            chunk_ids=["c1", "c2", "c3", "c4", "c5"],
            chunk_scores=[0.9, 0.85, 0.8, 0.75, 0.7],
            confidence_level="high",
            confidence_score=0.85,
            response_length=500,
            latency_ms=1500,
            model_id="claude-haiku",
            success=True,
        )

        assert metrics.query_id == "test-123"
        assert metrics.chunks_retrieved == 5
        assert metrics.success is True
        assert metrics.error_message is None


class TestLogQuery:
    """Tests for DynamoDB logging."""

    @patch("lambdas.query.analytics.ENABLE_QUERY_LOGGING", True)
    @patch("lambdas.query.analytics.get_dynamodb_resource")
    def test_log_query_success(self, mock_get_resource):
        """Test successful query logging."""
        mock_table = MagicMock()
        mock_resource = MagicMock()
        mock_resource.Table.return_value = mock_table
        mock_get_resource.return_value = mock_resource

        metrics = QueryMetrics(
            query_id="test-123",
            question="Test question",
            chunks_retrieved=3,
            chunk_ids=["c1", "c2", "c3"],
            chunk_scores=[0.9, 0.8, 0.7],
            confidence_level="high",
            confidence_score=0.85,
            response_length=100,
            latency_ms=1000,
            model_id="claude",
            success=True,
        )

        result = log_query(metrics)

        assert result is True
        mock_table.put_item.assert_called_once()

    @patch("lambdas.query.analytics.ENABLE_QUERY_LOGGING", False)
    def test_log_query_disabled(self):
        """Test logging when disabled."""
        metrics = QueryMetrics(
            query_id="test-123",
            question="Test",
            chunks_retrieved=0,
            chunk_ids=[],
            chunk_scores=[],
            confidence_level="low",
            confidence_score=0.0,
            response_length=0,
            latency_ms=0,
            model_id="test",
            success=False,
        )

        result = log_query(metrics)
        assert result is True  # Returns True when disabled


class TestPublishQueryMetrics:
    """Tests for CloudWatch metrics publishing."""

    @patch("lambdas.query.analytics.get_cloudwatch_client")
    def test_publish_success_metrics(self, mock_get_client):
        """Test publishing successful query metrics."""
        mock_cloudwatch = MagicMock()
        mock_get_client.return_value = mock_cloudwatch

        metrics = QueryMetrics(
            query_id="test-123",
            question="Test",
            chunks_retrieved=5,
            chunk_ids=["c1"],
            chunk_scores=[0.9],
            confidence_level="high",
            confidence_score=0.85,
            response_length=500,
            latency_ms=1500,
            model_id="claude",
            success=True,
        )

        result = publish_query_metrics(metrics)

        assert result is True
        mock_cloudwatch.put_metric_data.assert_called_once()

        # Verify the metrics sent
        call_args = mock_cloudwatch.put_metric_data.call_args
        assert call_args[1]["Namespace"] == "PediatricRAG"
        metric_names = [m["MetricName"] for m in call_args[1]["MetricData"]]
        assert "QueryLatency" in metric_names
        assert "ConfidenceScore" in metric_names
        assert "QuerySuccess" in metric_names
