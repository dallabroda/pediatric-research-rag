"""Tests for document deduplication in the ingest handler."""
import hashlib
import json
import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from lambdas.ingest.handler import (
    check_duplicate,
    compute_document_hash,
)


class TestComputeDocumentHash:
    """Tests for document hash computation."""

    def test_compute_hash_for_text_file(self):
        """Test computing hash for a text file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is test content for hashing.")
            temp_path = Path(f.name)

        try:
            result = compute_document_hash(temp_path)

            # Verify it's a valid SHA256 hash (64 hex chars)
            assert len(result) == 64
            assert all(c in "0123456789abcdef" for c in result)
        finally:
            temp_path.unlink()

    def test_same_content_same_hash(self):
        """Test that identical content produces same hash."""
        content = "Identical content for testing."

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f1:
            f1.write(content)
            path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f2:
            f2.write(content)
            path2 = Path(f2.name)

        try:
            hash1 = compute_document_hash(path1)
            hash2 = compute_document_hash(path2)
            assert hash1 == hash2
        finally:
            path1.unlink()
            path2.unlink()

    def test_different_content_different_hash(self):
        """Test that different content produces different hash."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f1:
            f1.write("Content A")
            path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f2:
            f2.write("Content B")
            path2 = Path(f2.name)

        try:
            hash1 = compute_document_hash(path1)
            hash2 = compute_document_hash(path2)
            assert hash1 != hash2
        finally:
            path1.unlink()
            path2.unlink()


class TestCheckDuplicate:
    """Tests for duplicate checking."""

    @patch("lambdas.ingest.handler.get_s3_client")
    def test_not_duplicate_when_no_index(self, mock_get_client):
        """Test that new documents are not duplicates when no index exists."""
        mock_s3 = MagicMock()
        mock_get_client.return_value = mock_s3

        # Simulate NoSuchKey error
        mock_s3.exceptions.NoSuchKey = Exception
        mock_s3.get_object.side_effect = mock_s3.exceptions.NoSuchKey()

        is_dup, doc_id = check_duplicate("abc123hash", "test-bucket")

        assert is_dup is False
        assert doc_id is None

    @patch("lambdas.ingest.handler.get_s3_client")
    def test_finds_duplicate(self, mock_get_client):
        """Test that existing documents are detected as duplicates."""
        mock_s3 = MagicMock()
        mock_get_client.return_value = mock_s3

        # Mock existing metadata with matching hash
        existing_metadata = [
            {"document_id": "doc1", "chunk_id": "c1", "document_hash": "abc123hash"},
            {"document_id": "doc2", "chunk_id": "c2", "document_hash": "def456hash"},
        ]
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(existing_metadata).encode()
        mock_s3.get_object.return_value = {"Body": mock_body}

        is_dup, doc_id = check_duplicate("abc123hash", "test-bucket")

        assert is_dup is True
        assert doc_id == "doc1"

    @patch("lambdas.ingest.handler.get_s3_client")
    def test_not_duplicate_new_hash(self, mock_get_client):
        """Test that new hashes are not detected as duplicates."""
        mock_s3 = MagicMock()
        mock_get_client.return_value = mock_s3

        # Mock existing metadata without matching hash
        existing_metadata = [
            {"document_id": "doc1", "chunk_id": "c1", "document_hash": "existinghash1"},
            {"document_id": "doc2", "chunk_id": "c2", "document_hash": "existinghash2"},
        ]
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(existing_metadata).encode()
        mock_s3.get_object.return_value = {"Body": mock_body}

        is_dup, doc_id = check_duplicate("brandnewhash", "test-bucket")

        assert is_dup is False
        assert doc_id is None
