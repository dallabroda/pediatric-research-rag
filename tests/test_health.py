"""Tests for the health check handler."""
import json
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from lambdas.health.handler import (
    check_faiss_index,
    check_s3_bucket,
    handler,
)


class TestCheckFaissIndex:
    """Tests for FAISS index health check."""

    @patch("lambdas.health.handler.get_s3_client")
    def test_index_found(self, mock_get_client):
        """Test when index exists and is healthy."""
        mock_s3 = MagicMock()
        mock_get_client.return_value = mock_s3

        # Mock head_object for index file
        from datetime import datetime
        mock_s3.head_object.return_value = {
            "LastModified": datetime(2024, 2, 21, 12, 0, 0)
        }

        # Mock get_object for metadata
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps([
            {"document_id": "doc1", "chunk_id": "c1"},
            {"document_id": "doc1", "chunk_id": "c2"},
            {"document_id": "doc2", "chunk_id": "c3"},
        ]).encode()
        mock_s3.get_object.return_value = {"Body": mock_body}

        result = check_faiss_index("test-bucket")

        assert result["status"] == "ok"
        assert result["vectors"] == 3
        assert result["documents"] == 2
        assert result["last_updated"] is not None

    @patch("lambdas.health.handler.get_s3_client")
    def test_index_not_found(self, mock_get_client):
        """Test when index doesn't exist."""
        mock_s3 = MagicMock()
        mock_get_client.return_value = mock_s3

        # Simulate NoSuchKey error
        mock_s3.exceptions.NoSuchKey = Exception
        mock_s3.head_object.side_effect = mock_s3.exceptions.NoSuchKey()

        result = check_faiss_index("test-bucket")

        assert result["status"] == "not_found"
        assert result["vectors"] == 0


class TestCheckS3Bucket:
    """Tests for S3 bucket health check."""

    @patch("lambdas.health.handler.get_s3_client")
    def test_bucket_accessible(self, mock_get_client):
        """Test when bucket is accessible."""
        mock_s3 = MagicMock()
        mock_get_client.return_value = mock_s3

        mock_s3.head_bucket.return_value = {}
        mock_s3.list_objects_v2.return_value = {"KeyCount": 5}

        result = check_s3_bucket("test-bucket")

        assert result["status"] == "ok"
        assert result["bucket"] == "test-bucket"

    @patch("lambdas.health.handler.get_s3_client")
    def test_bucket_error(self, mock_get_client):
        """Test when bucket access fails."""
        mock_s3 = MagicMock()
        mock_get_client.return_value = mock_s3
        mock_s3.head_bucket.side_effect = Exception("Access denied")

        result = check_s3_bucket("test-bucket")

        assert result["status"] == "error"
        assert "Access denied" in result["message"]


class TestHandler:
    """Tests for the health handler."""

    @patch("lambdas.health.handler.check_faiss_index")
    @patch("lambdas.health.handler.check_bedrock")
    @patch("lambdas.health.handler.check_s3_bucket")
    def test_healthy_response(self, mock_s3, mock_bedrock, mock_faiss):
        """Test healthy system returns 200."""
        mock_faiss.return_value = {"status": "ok", "vectors": 100, "documents": 10}
        mock_bedrock.return_value = {"status": "ok", "model": "test"}
        mock_s3.return_value = {"status": "ok", "bucket": "test"}

        response = handler({}, None)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["status"] == "healthy"
        assert "components" in body

    @patch("lambdas.health.handler.check_faiss_index")
    @patch("lambdas.health.handler.check_bedrock")
    @patch("lambdas.health.handler.check_s3_bucket")
    def test_degraded_response(self, mock_s3, mock_bedrock, mock_faiss):
        """Test degraded system returns 503."""
        mock_faiss.return_value = {"status": "error", "message": "Not found"}
        mock_bedrock.return_value = {"status": "ok", "model": "test"}
        mock_s3.return_value = {"status": "ok", "bucket": "test"}

        response = handler({}, None)

        assert response["statusCode"] == 503
        body = json.loads(response["body"])
        assert body["status"] == "degraded"
