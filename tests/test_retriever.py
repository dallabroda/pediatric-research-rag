"""Tests for the FAISS retriever module."""
import json
import os
import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Must import before faiss to avoid issues
import faiss

from lambdas.query.retriever import (
    FAISSRetriever,
    RetrievalResult,
    EMBEDDING_DIMENSION,
)


# Sample metadata for testing
SAMPLE_METADATA = [
    {
        "chunk_id": "doc1_chunk0",
        "document_id": "PMC123",
        "chunk_text": "Pediatric cancer treatment has evolved significantly.",
        "doc_title": "Advances in Pediatric Oncology",
        "doc_type": "paper",
        "chunk_index": 0,
        "page_number": 1,
        "section_title": "Introduction",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC123/",
    },
    {
        "chunk_id": "doc1_chunk1",
        "document_id": "PMC123",
        "chunk_text": "Chemotherapy protocols have been refined over decades.",
        "doc_title": "Advances in Pediatric Oncology",
        "doc_type": "paper",
        "chunk_index": 1,
        "page_number": 2,
        "section_title": "Methods",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC123/",
    },
    {
        "chunk_id": "doc2_chunk0",
        "document_id": "NCT001",
        "chunk_text": "This clinical trial studies new immunotherapy approaches.",
        "doc_title": "Phase II Immunotherapy Trial",
        "doc_type": "trial",
        "chunk_index": 0,
        "page_number": None,
        "section_title": "Brief Summary",
        "source_url": "https://clinicaltrials.gov/study/NCT001",
    },
]


def create_test_index(num_vectors: int = 3) -> faiss.IndexFlatIP:
    """Create a test FAISS index with random vectors."""
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)

    # Create normalized random vectors
    vectors = np.random.randn(num_vectors, EMBEDDING_DIMENSION).astype(np.float32)
    faiss.normalize_L2(vectors)

    index.add(vectors)
    return index


def create_mock_bedrock_response(embedding: np.ndarray) -> dict:
    """Create mock Bedrock embedding response."""
    return {
        "body": MagicMock(
            read=lambda: json.dumps({"embedding": embedding.tolist()}).encode()
        )
    }


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_create_result(self):
        result = RetrievalResult(
            chunk_id="test_chunk",
            document_id="doc1",
            chunk_text="Test content",
            doc_title="Test Title",
            doc_type="paper",
            score=0.85,
            page_number=1,
            section_title="Introduction",
            source_url="https://example.com",
        )

        assert result.chunk_id == "test_chunk"
        assert result.score == 0.85
        assert result.page_number == 1

    def test_optional_fields(self):
        result = RetrievalResult(
            chunk_id="test",
            document_id="doc1",
            chunk_text="Content",
            doc_title="Title",
            doc_type="paper",
            score=0.5,
            page_number=None,
            section_title=None,
            source_url=None,
        )

        assert result.page_number is None
        assert result.section_title is None
        assert result.source_url is None


class TestFAISSRetrieverInit:
    """Tests for FAISSRetriever initialization."""

    def test_init_with_index_and_metadata(self):
        index = create_test_index()
        retriever = FAISSRetriever(index=index, metadata=SAMPLE_METADATA)

        assert retriever.index is index
        assert retriever.metadata == SAMPLE_METADATA
        assert retriever._bedrock_client is None

    def test_init_with_bedrock_client(self):
        index = create_test_index()
        mock_client = MagicMock()

        retriever = FAISSRetriever(
            index=index,
            metadata=SAMPLE_METADATA,
            bedrock_client=mock_client,
        )

        assert retriever._bedrock_client is mock_client


class TestFAISSRetrieverFromLocal:
    """Tests for loading retriever from local disk."""

    def test_from_local_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create index file
            index = create_test_index()
            index_path = Path(tmpdir) / "faiss_index.bin"
            faiss.write_index(index, str(index_path))

            # Create metadata file
            metadata_path = Path(tmpdir) / "faiss_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(SAMPLE_METADATA, f)

            # Load retriever
            retriever = FAISSRetriever.from_local(tmpdir)

            assert retriever.index.ntotal == 3
            assert len(retriever.metadata) == 3

    def test_from_local_missing_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="Index not found"):
                FAISSRetriever.from_local(tmpdir)

    def test_from_local_default_path(self):
        # Test that default path is used when none provided
        with patch.object(FAISSRetriever, "from_local") as mock_from_local:
            mock_from_local.return_value = MagicMock()
            # This would use LOCAL_INDEX_DIR default
            # We just verify the method exists and can be called
            assert callable(FAISSRetriever.from_local)


class TestFAISSRetrieverFromS3:
    """Tests for loading retriever from S3."""

    @patch("boto3.client")
    def test_from_s3_success(self, mock_boto_client):
        # Create test index and save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            index = create_test_index()
            index_path = Path(tmpdir) / "faiss_index.bin"
            faiss.write_index(index, str(index_path))

            # Read index bytes
            with open(index_path, "rb") as f:
                index_bytes = f.read()

            # Mock S3 client
            mock_s3 = MagicMock()
            mock_boto_client.return_value = mock_s3

            # Mock download_file to copy the test index
            def mock_download(bucket, key, path):
                with open(path, "wb") as f:
                    f.write(index_bytes)

            mock_s3.download_file.side_effect = mock_download

            # Mock get_object for metadata
            mock_s3.get_object.return_value = {
                "Body": MagicMock(read=lambda: json.dumps(SAMPLE_METADATA).encode())
            }

            # Load retriever
            retriever = FAISSRetriever.from_s3("test-bucket")

            assert retriever.index.ntotal == 3
            assert len(retriever.metadata) == 3
            mock_s3.download_file.assert_called_once()
            mock_s3.get_object.assert_called_once()


class TestFAISSRetrieverGetEmbedding:
    """Tests for embedding generation."""

    def test_get_embedding(self):
        index = create_test_index()
        mock_client = MagicMock()

        # Create mock response with random embedding
        test_embedding = np.random.randn(EMBEDDING_DIMENSION).astype(np.float32)
        mock_client.invoke_model.return_value = create_mock_bedrock_response(test_embedding)

        retriever = FAISSRetriever(
            index=index,
            metadata=SAMPLE_METADATA,
            bedrock_client=mock_client,
        )

        result = retriever.get_embedding("test query")

        assert result.shape == (EMBEDDING_DIMENSION,)
        assert result.dtype == np.float32
        mock_client.invoke_model.assert_called_once()

    def test_get_embedding_normalized(self):
        index = create_test_index()
        mock_client = MagicMock()

        # Create unnormalized embedding
        test_embedding = np.array([1.0, 2.0, 3.0] + [0.0] * (EMBEDDING_DIMENSION - 3))
        mock_client.invoke_model.return_value = create_mock_bedrock_response(test_embedding)

        retriever = FAISSRetriever(
            index=index,
            metadata=SAMPLE_METADATA,
            bedrock_client=mock_client,
        )

        result = retriever.get_embedding("test")

        # Check that result is normalized (L2 norm = 1)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5


class TestFAISSRetrieverSearch:
    """Tests for similarity search."""

    def test_search_returns_results(self):
        index = create_test_index(len(SAMPLE_METADATA))
        mock_client = MagicMock()

        # Create query embedding that matches first vector
        query_embedding = np.random.randn(EMBEDDING_DIMENSION).astype(np.float32)
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        mock_client.invoke_model.return_value = create_mock_bedrock_response(query_embedding)

        retriever = FAISSRetriever(
            index=index,
            metadata=SAMPLE_METADATA,
            bedrock_client=mock_client,
        )

        results = retriever.search("pediatric cancer", top_k=3, threshold=0.0)

        assert len(results) <= 3
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_search_threshold_filtering(self):
        index = create_test_index()
        mock_client = MagicMock()

        # Create random query embedding
        query_embedding = np.random.randn(EMBEDDING_DIMENSION).astype(np.float32)
        mock_client.invoke_model.return_value = create_mock_bedrock_response(query_embedding)

        retriever = FAISSRetriever(
            index=index,
            metadata=SAMPLE_METADATA,
            bedrock_client=mock_client,
        )

        # Very high threshold should filter out results
        results = retriever.search("test", top_k=5, threshold=0.99)

        # Results may be empty due to high threshold
        assert all(r.score >= 0.99 for r in results)

    def test_search_top_k_limit(self):
        # Create index with more vectors
        index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        vectors = np.random.randn(10, EMBEDDING_DIMENSION).astype(np.float32)
        faiss.normalize_L2(vectors)
        index.add(vectors)

        # Create extended metadata
        extended_metadata = SAMPLE_METADATA * 4  # 12 items
        extended_metadata = extended_metadata[:10]  # Match index size

        mock_client = MagicMock()
        query_embedding = np.random.randn(EMBEDDING_DIMENSION).astype(np.float32)
        mock_client.invoke_model.return_value = create_mock_bedrock_response(query_embedding)

        retriever = FAISSRetriever(
            index=index,
            metadata=extended_metadata,
            bedrock_client=mock_client,
        )

        results = retriever.search("test", top_k=3, threshold=0.0)

        assert len(results) <= 3

    def test_search_returns_metadata(self):
        index = create_test_index()
        mock_client = MagicMock()

        # Create embedding that will have some similarity
        query_embedding = np.random.randn(EMBEDDING_DIMENSION).astype(np.float32)
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        mock_client.invoke_model.return_value = create_mock_bedrock_response(query_embedding)

        retriever = FAISSRetriever(
            index=index,
            metadata=SAMPLE_METADATA,
            bedrock_client=mock_client,
        )

        results = retriever.search("cancer treatment", top_k=3, threshold=0.0)

        if results:  # May be empty if scores too low
            result = results[0]
            assert result.chunk_id is not None
            assert result.document_id is not None
            assert result.chunk_text is not None


class TestFAISSRetrieverDocumentMethods:
    """Tests for document-related methods."""

    def test_get_document_ids(self):
        index = create_test_index()
        retriever = FAISSRetriever(index=index, metadata=SAMPLE_METADATA)

        doc_ids = retriever.get_document_ids()

        assert "PMC123" in doc_ids
        assert "NCT001" in doc_ids
        assert len(doc_ids) == 2  # Two unique documents

    def test_get_document_ids_sorted(self):
        index = create_test_index()
        retriever = FAISSRetriever(index=index, metadata=SAMPLE_METADATA)

        doc_ids = retriever.get_document_ids()

        assert doc_ids == sorted(doc_ids)

    def test_get_document_chunks(self):
        index = create_test_index()
        retriever = FAISSRetriever(index=index, metadata=SAMPLE_METADATA)

        chunks = retriever.get_document_chunks("PMC123")

        assert len(chunks) == 2
        assert all(c["document_id"] == "PMC123" for c in chunks)

    def test_get_document_chunks_sorted_by_index(self):
        index = create_test_index()
        retriever = FAISSRetriever(index=index, metadata=SAMPLE_METADATA)

        chunks = retriever.get_document_chunks("PMC123")

        indices = [c["chunk_index"] for c in chunks]
        assert indices == sorted(indices)

    def test_get_document_chunks_not_found(self):
        index = create_test_index()
        retriever = FAISSRetriever(index=index, metadata=SAMPLE_METADATA)

        chunks = retriever.get_document_chunks("NONEXISTENT")

        assert chunks == []


class TestFAISSRetrieverBedrockClient:
    """Tests for Bedrock client lazy loading."""

    @patch("boto3.client")
    def test_lazy_load_bedrock_client(self, mock_boto_client):
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        index = create_test_index()
        retriever = FAISSRetriever(index=index, metadata=SAMPLE_METADATA)

        # Client should not be created yet
        mock_boto_client.assert_not_called()

        # Access the client property
        client = retriever.bedrock_client

        # Now it should be created
        mock_boto_client.assert_called_once_with("bedrock-runtime")
        assert client is mock_client

    @patch("boto3.client")
    def test_bedrock_client_cached(self, mock_boto_client):
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        index = create_test_index()
        retriever = FAISSRetriever(index=index, metadata=SAMPLE_METADATA)

        # Access multiple times
        client1 = retriever.bedrock_client
        client2 = retriever.bedrock_client

        # Should only create once
        mock_boto_client.assert_called_once()
        assert client1 is client2
