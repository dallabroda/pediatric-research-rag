#!/usr/bin/env python3
"""
Build FAISS index from sample documents.

This script:
1. Parses all documents from local directory or S3
2. Chunks the text
3. Generates embeddings using Bedrock Titan
4. Builds a FAISS index
5. Saves index locally and optionally to S3
"""
import argparse
import io
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import boto3
import faiss
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from lambdas.ingest.chunker import chunk_document
from lambdas.ingest.parsers import parse_document

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
EMBEDDING_DIMENSION = 1024
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))


def get_s3_client():
    """Get boto3 S3 client."""
    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )


def find_documents_in_s3(s3_bucket: str) -> list[dict]:
    """
    Find all parseable documents in S3 bucket.

    Args:
        s3_bucket: S3 bucket name

    Returns:
        List of dicts with 'key', 'type' (pdf/trial), and 'size'
    """
    s3 = get_s3_client()
    documents = []

    # Find PDFs in raw/papers/
    paginator = s3.get_paginator("list_objects_v2")

    logger.info("Scanning S3 for papers...")
    for page in paginator.paginate(Bucket=s3_bucket, Prefix="raw/papers/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".pdf"):
                documents.append({
                    "key": key,
                    "type": "pdf",
                    "size": obj["Size"],
                })

    logger.info("Scanning S3 for trials...")
    for page in paginator.paginate(Bucket=s3_bucket, Prefix="raw/trials/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Only get .json files (not _text.txt)
            if key.endswith(".json") and "_text.txt" not in key:
                documents.append({
                    "key": key,
                    "type": "trial",
                    "size": obj["Size"],
                })

    return documents


def download_s3_document(s3_bucket: str, key: str, doc_type: str) -> Path:
    """
    Download a document from S3 to a temporary file.

    Args:
        s3_bucket: S3 bucket name
        key: S3 object key
        doc_type: 'pdf' or 'trial'

    Returns:
        Path to temporary file
    """
    s3 = get_s3_client()

    # Create temp file with appropriate extension
    if doc_type == "pdf":
        suffix = ".pdf"
    else:
        suffix = ".json"

    # Extract filename from key
    filename = key.split("/")[-1]

    # Download to temp file
    temp_dir = Path(tempfile.gettempdir()) / "rag_index"
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / filename

    # Only download if not already cached
    if not temp_path.exists():
        response = s3.get_object(Bucket=s3_bucket, Key=key)
        with open(temp_path, "wb") as f:
            f.write(response["Body"].read())

    return temp_path


def get_bedrock_client():
    """Get boto3 Bedrock Runtime client."""
    return boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )


def get_embedding(text: str, bedrock_client) -> list[float]:
    """
    Get embedding for text using Bedrock Titan Embeddings V2.

    Args:
        text: Text to embed
        bedrock_client: Bedrock Runtime client

    Returns:
        1024-dimensional embedding vector
    """
    body = json.dumps({
        "inputText": text,
        "dimensions": EMBEDDING_DIMENSION,
        "normalize": True,
    })

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


def find_documents(data_dir: Path) -> list[Path]:
    """Find all parseable documents in the data directory."""
    documents = []

    # Find PDFs
    for pdf in data_dir.rglob("*.pdf"):
        documents.append(pdf)

    # Find clinical trial JSON files (not metadata files)
    for json_file in data_dir.rglob("*.json"):
        if "_metadata.json" not in json_file.name:
            # Check if it's a clinical trial
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "nct_id" in data:
                    documents.append(json_file)
            except Exception:
                continue

    return documents


def build_index(
    data_dir: str,
    output_dir: str,
    upload_to_s3: bool = False,
    s3_bucket: str = None,
) -> dict:
    """
    Build FAISS index from local documents.

    Args:
        data_dir: Directory containing documents
        output_dir: Directory to save index files
        upload_to_s3: Whether to upload to S3
        s3_bucket: S3 bucket name (required if upload_to_s3 is True)

    Returns:
        Dict with build statistics
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all documents
    documents = find_documents(data_path)
    logger.info(f"Found {len(documents)} documents to process")

    if not documents:
        logger.warning("No documents found! Make sure to run download scripts first.")
        return {"documents": 0, "chunks": 0, "vectors": 0}

    # Initialize Bedrock client
    bedrock = get_bedrock_client()

    all_chunks = []
    all_embeddings = []

    for doc_path in documents:
        try:
            logger.info(f"Processing: {doc_path.name}")

            # Parse document
            parsed = parse_document(doc_path)
            logger.info(f"  Title: {parsed.title[:60]}...")
            logger.info(f"  Text length: {len(parsed.text)} chars")

            # Chunk document
            chunks = chunk_document(
                text=parsed.text,
                document_id=parsed.document_id,
                doc_title=parsed.title,
                doc_type=parsed.doc_type,
                page_boundaries=parsed.page_boundaries,
                section_titles=parsed.section_titles,
                source_url=parsed.metadata.get("source_url"),
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            logger.info(f"  Created {len(chunks)} chunks")

            # Generate embeddings for each chunk
            for i, chunk in enumerate(chunks):
                try:
                    embedding = get_embedding(chunk.chunk_text, bedrock)
                    all_embeddings.append(embedding)

                    # Convert ChunkMetadata to dict
                    chunk_dict = {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "chunk_text": chunk.chunk_text,
                        "chunk_index": chunk.chunk_index,
                        "page_number": chunk.page_number,
                        "section_title": chunk.section_title,
                        "source_url": chunk.source_url,
                        "doc_title": chunk.doc_title,
                        "doc_type": chunk.doc_type,
                    }
                    all_chunks.append(chunk_dict)

                    # Rate limiting - be gentle with Bedrock
                    if (len(all_embeddings)) % 10 == 0:
                        logger.info(f"  Embedded {len(all_embeddings)} chunks total...")
                        time.sleep(0.1)

                except Exception as e:
                    logger.error(f"  Failed to embed chunk {i}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to process {doc_path.name}: {e}")
            continue

    if not all_embeddings:
        logger.error("No embeddings generated!")
        return {"documents": len(documents), "chunks": 0, "vectors": 0}

    # Build FAISS index
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    logger.info(f"Building FAISS index from {len(all_embeddings)} vectors...")

    # L2 normalize for cosine similarity with inner product
    faiss.normalize_L2(embeddings_array)

    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(embeddings_array)

    logger.info(f"Index built with {index.ntotal} vectors")

    # Save locally
    index_path = output_path / "faiss_index.bin"
    metadata_path = output_path / "faiss_metadata.json"

    faiss.write_index(index, str(index_path))
    logger.info(f"Saved index to {index_path}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved metadata to {metadata_path}")

    # Upload to S3 if requested
    if upload_to_s3:
        if not s3_bucket:
            s3_bucket = os.environ.get("S3_BUCKET", "pediatric-research-rag")

        s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))

        # Upload index
        with open(index_path, "rb") as f:
            s3.put_object(
                Bucket=s3_bucket,
                Key="processed/index/faiss_index.bin",
                Body=f.read(),
                ContentType="application/octet-stream",
            )
        logger.info(f"Uploaded index to s3://{s3_bucket}/processed/index/faiss_index.bin")

        # Upload metadata
        with open(metadata_path, "rb") as f:
            s3.put_object(
                Bucket=s3_bucket,
                Key="processed/index/faiss_metadata.json",
                Body=f.read(),
                ContentType="application/json",
            )
        logger.info(f"Uploaded metadata to s3://{s3_bucket}/processed/index/faiss_metadata.json")

    return {
        "documents": len(documents),
        "chunks": len(all_chunks),
        "vectors": index.ntotal,
    }


def build_index_from_s3(
    s3_bucket: str,
    output_dir: str,
    upload_to_s3: bool = False,
) -> dict:
    """
    Build FAISS index from documents in S3.

    Args:
        s3_bucket: S3 bucket name containing raw documents
        output_dir: Local directory to save index files
        upload_to_s3: Whether to upload index back to S3

    Returns:
        Dict with build statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all documents in S3
    logger.info(f"Scanning s3://{s3_bucket} for documents...")
    s3_documents = find_documents_in_s3(s3_bucket)
    logger.info(f"Found {len(s3_documents)} documents in S3")

    if not s3_documents:
        logger.warning("No documents found in S3!")
        return {"documents": 0, "chunks": 0, "vectors": 0}

    # Initialize Bedrock client
    bedrock = get_bedrock_client()

    all_chunks = []
    all_embeddings = []
    processed_docs = 0
    failed_docs = 0

    # Process documents with progress bar
    with tqdm(total=len(s3_documents), desc="Processing documents", unit="doc") as doc_pbar:
        for s3_doc in s3_documents:
            try:
                # Download document from S3
                doc_path = download_s3_document(s3_bucket, s3_doc["key"], s3_doc["type"])
                doc_pbar.set_postfix({"current": doc_path.name[:30]})

                # Parse document
                parsed = parse_document(doc_path)

                # Chunk document
                chunks = chunk_document(
                    text=parsed.text,
                    document_id=parsed.document_id,
                    doc_title=parsed.title,
                    doc_type=parsed.doc_type,
                    page_boundaries=parsed.page_boundaries,
                    section_titles=parsed.section_titles,
                    source_url=parsed.metadata.get("source_url"),
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                )

                # Generate embeddings for each chunk
                for chunk in chunks:
                    try:
                        embedding = get_embedding(chunk.chunk_text, bedrock)
                        all_embeddings.append(embedding)

                        # Convert ChunkMetadata to dict
                        chunk_dict = {
                            "chunk_id": chunk.chunk_id,
                            "document_id": chunk.document_id,
                            "chunk_text": chunk.chunk_text,
                            "chunk_index": chunk.chunk_index,
                            "page_number": chunk.page_number,
                            "section_title": chunk.section_title,
                            "source_url": chunk.source_url,
                            "doc_title": chunk.doc_title,
                            "doc_type": chunk.doc_type,
                        }
                        all_chunks.append(chunk_dict)

                        # Rate limiting
                        if len(all_embeddings) % 50 == 0:
                            time.sleep(0.1)

                    except Exception as e:
                        logger.error(f"Failed to embed chunk: {e}")
                        continue

                processed_docs += 1
                doc_pbar.update(1)

            except Exception as e:
                logger.error(f"Failed to process {s3_doc['key']}: {e}")
                failed_docs += 1
                doc_pbar.update(1)
                continue

    logger.info(f"Processed {processed_docs} documents, {failed_docs} failed")
    logger.info(f"Generated {len(all_embeddings)} embeddings from {len(all_chunks)} chunks")

    if not all_embeddings:
        logger.error("No embeddings generated!")
        return {"documents": processed_docs, "chunks": 0, "vectors": 0}

    # Build FAISS index
    logger.info(f"Building FAISS index from {len(all_embeddings)} vectors...")
    embeddings_array = np.array(all_embeddings, dtype=np.float32)

    # L2 normalize for cosine similarity with inner product
    faiss.normalize_L2(embeddings_array)

    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(embeddings_array)

    logger.info(f"Index built with {index.ntotal} vectors")

    # Save locally
    index_path = output_path / "faiss_index.bin"
    metadata_path = output_path / "faiss_metadata.json"

    faiss.write_index(index, str(index_path))
    logger.info(f"Saved index to {index_path}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved metadata to {metadata_path}")

    # Upload to S3 if requested
    if upload_to_s3:
        s3 = get_s3_client()

        # Upload index
        with open(index_path, "rb") as f:
            s3.put_object(
                Bucket=s3_bucket,
                Key="processed/index/faiss_index.bin",
                Body=f.read(),
                ContentType="application/octet-stream",
            )
        logger.info(f"Uploaded index to s3://{s3_bucket}/processed/index/faiss_index.bin")

        # Upload metadata
        with open(metadata_path, "rb") as f:
            s3.put_object(
                Bucket=s3_bucket,
                Key="processed/index/faiss_metadata.json",
                Body=f.read(),
                ContentType="application/json",
            )
        logger.info(f"Uploaded metadata to s3://{s3_bucket}/processed/index/faiss_metadata.json")

    return {
        "documents": processed_docs,
        "chunks": len(all_chunks),
        "vectors": index.ntotal,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index from documents"
    )
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["local", "s3"],
        default="local",
        help="Where to read documents from: 'local' directory or 's3' bucket",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/sample",
        help="Directory containing documents (local mode only)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/index",
        help="Directory to save index files locally",
    )
    parser.add_argument(
        "--upload-to-s3",
        action="store_true",
        help="Upload index to S3 after building",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=None,
        help="S3 bucket name (default: S3_BUCKET env var or 'pediatric-research-rag')",
    )

    args = parser.parse_args()

    s3_bucket = args.s3_bucket or os.environ.get("S3_BUCKET", "pediatric-research-rag")

    print("\n" + "=" * 60)
    print("Building FAISS Index for Pediatric Research RAG")
    print("=" * 60)
    print(f"  Data source: {args.data_source}")
    if args.data_source == "s3":
        print(f"  S3 bucket: {s3_bucket}")
    else:
        print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Upload to S3: {args.upload_to_s3}")
    print("=" * 60 + "\n")

    if args.data_source == "s3":
        result = build_index_from_s3(
            s3_bucket=s3_bucket,
            output_dir=args.output_dir,
            upload_to_s3=args.upload_to_s3,
        )
    else:
        result = build_index(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            upload_to_s3=args.upload_to_s3,
            s3_bucket=s3_bucket,
        )

    print("\n" + "=" * 60)
    print("Build Complete!")
    print("=" * 60)
    print(f"  Documents processed: {result['documents']}")
    print(f"  Chunks created: {result['chunks']}")
    print(f"  Vectors in index: {result['vectors']}")
    print()


if __name__ == "__main__":
    main()
