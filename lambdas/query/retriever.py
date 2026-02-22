"""
FAISS index retriever for semantic search.

Loads index from S3 or local disk, performs similarity search,
and returns relevant document chunks.
"""
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import boto3
import faiss
import numpy as np

logger = logging.getLogger(__name__)

# Configuration
S3_BUCKET = os.environ.get("S3_BUCKET", "pediatric-research-rag")
INDEX_PREFIX = os.environ.get("INDEX_PREFIX", "processed/index/")
LOCAL_INDEX_DIR = os.environ.get("LOCAL_INDEX_DIR", "data/index")
EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
EMBEDDING_DIMENSION = 1024
TOP_K = int(os.environ.get("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.3"))
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


@dataclass
class RetrievalResult:
    """Result from similarity search."""
    chunk_id: str
    document_id: str
    chunk_text: str
    doc_title: str
    doc_type: str
    score: float
    page_number: Optional[int]
    section_title: Optional[str]
    source_url: Optional[str]


class FAISSRetriever:
    """
    Retriever that loads FAISS index and performs similarity search.

    Can load from local disk or S3.
    """

    def __init__(
        self,
        index: faiss.IndexFlatIP,
        metadata: list[dict],
        bedrock_client=None,
    ):
        """
        Initialize retriever with loaded index.

        Args:
            index: FAISS index
            metadata: List of chunk metadata dicts
            bedrock_client: Optional Bedrock client for embeddings
        """
        self.index = index
        self.metadata = metadata
        self._bedrock_client = bedrock_client

    @property
    def bedrock_client(self):
        """Lazy-load Bedrock client."""
        if self._bedrock_client is None:
            self._bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        return self._bedrock_client

    @classmethod
    def from_local(cls, index_dir: str = None) -> "FAISSRetriever":
        """
        Load retriever from local disk.

        Args:
            index_dir: Directory containing index files

        Returns:
            FAISSRetriever instance
        """
        if index_dir is None:
            index_dir = LOCAL_INDEX_DIR

        index_path = Path(index_dir)

        # Load FAISS index
        index_file = index_path / "faiss_index.bin"
        if not index_file.exists():
            raise FileNotFoundError(f"Index not found: {index_file}")

        index = faiss.read_index(str(index_file))
        logger.info(f"Loaded index with {index.ntotal} vectors from {index_file}")

        # Load metadata
        metadata_file = index_path / "faiss_metadata.json"
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return cls(index=index, metadata=metadata)

    @classmethod
    def from_s3(cls, bucket: str = None) -> "FAISSRetriever":
        """
        Load retriever from S3.

        Args:
            bucket: S3 bucket name

        Returns:
            FAISSRetriever instance
        """
        if bucket is None:
            bucket = S3_BUCKET

        s3 = boto3.client("s3", region_name=AWS_REGION)

        with TemporaryDirectory() as tmpdir:
            # Download index
            index_key = f"{INDEX_PREFIX}faiss_index.bin"
            index_path = f"{tmpdir}/faiss_index.bin"

            s3.download_file(bucket, index_key, index_path)
            index = faiss.read_index(index_path)
            logger.info(f"Loaded index with {index.ntotal} vectors from s3://{bucket}/{index_key}")

        # Download metadata
        metadata_key = f"{INDEX_PREFIX}faiss_metadata.json"
        response = s3.get_object(Bucket=bucket, Key=metadata_key)
        metadata = json.loads(response["Body"].read())

        return cls(index=index, metadata=metadata)

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for query text.

        Args:
            text: Query text

        Returns:
            Embedding as numpy array
        """
        body = json.dumps({
            "inputText": text,
            "dimensions": EMBEDDING_DIMENSION,
            "normalize": True,
        })

        response = self.bedrock_client.invoke_model(
            modelId=EMBEDDING_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )

        response_body = json.loads(response["body"].read())
        embedding = np.array(response_body["embedding"], dtype=np.float32)

        # Ensure normalized
        faiss.normalize_L2(embedding.reshape(1, -1))

        return embedding

    def search(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None,
    ) -> list[RetrievalResult]:
        """
        Search for relevant chunks.

        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity score

        Returns:
            List of RetrievalResult objects, sorted by score descending
        """
        if top_k is None:
            top_k = TOP_K
        if threshold is None:
            threshold = SIMILARITY_THRESHOLD

        # Get query embedding
        query_embedding = self.get_embedding(query)

        # Search index
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1),
            top_k,
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # No more results
                break
            if score < threshold:
                continue

            chunk_meta = self.metadata[idx]

            results.append(RetrievalResult(
                chunk_id=chunk_meta.get("chunk_id", ""),
                document_id=chunk_meta.get("document_id", ""),
                chunk_text=chunk_meta.get("chunk_text", ""),
                doc_title=chunk_meta.get("doc_title", ""),
                doc_type=chunk_meta.get("doc_type", ""),
                score=float(score),
                page_number=chunk_meta.get("page_number"),
                section_title=chunk_meta.get("section_title"),
                source_url=chunk_meta.get("source_url"),
            ))

        logger.info(f"Found {len(results)} results for query: {query[:50]}...")
        return results

    def get_document_ids(self) -> list[str]:
        """Get list of unique document IDs in the index."""
        doc_ids = set()
        for chunk in self.metadata:
            doc_ids.add(chunk.get("document_id", ""))
        return sorted(doc_ids)

    def get_document_chunks(self, document_id: str) -> list[dict]:
        """Get all chunks for a specific document."""
        chunks = []
        for chunk in self.metadata:
            if chunk.get("document_id") == document_id:
                chunks.append(chunk)
        return sorted(chunks, key=lambda x: x.get("chunk_index", 0))
