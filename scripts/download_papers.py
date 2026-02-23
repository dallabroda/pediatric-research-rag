#!/usr/bin/env python3
"""
Download open-access research papers from PubMed Central (PMC).

Uses NCBI E-utilities API and PMC Open Access Web Service.
Rate limit: 3 requests/second without API key.

Supports direct upload to S3 with --upload-to-s3 flag.
"""
import argparse
import io
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree

import boto3
import requests
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lambdas.shared.retry import retry_with_backoff

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# NCBI E-utilities base URLs
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

# Rate limiting
REQUEST_DELAY = 0.34  # ~3 requests per second


def get_s3_client():
    """Get boto3 S3 client."""
    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )


def get_existing_pmc_ids(s3_bucket: str, prefix: str = "raw/papers/") -> set[str]:
    """
    List PMC IDs already present in S3 bucket.

    Args:
        s3_bucket: S3 bucket name
        prefix: S3 key prefix for papers

    Returns:
        Set of PMC IDs (e.g., {"PMC1234567", "PMC7654321"})
    """
    s3 = get_s3_client()
    pmc_ids = set()

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Extract PMC ID from filename like "raw/papers/PMC1234567.pdf"
            filename = key.split("/")[-1]
            if filename.endswith(".pdf"):
                pmc_id = filename.replace(".pdf", "")
                pmc_ids.add(pmc_id)

    return pmc_ids


@retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=30.0)
def download_pdf_to_memory(url: str) -> Optional[bytes]:
    """
    Download PDF to memory (bytes) instead of disk.

    Args:
        url: URL to download from

    Returns:
        PDF bytes if successful, None otherwise
    """
    time.sleep(REQUEST_DELAY)

    # Handle FTP URLs by converting to HTTPS
    if url.startswith("ftp://"):
        url = url.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")

    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()

    # Read into memory
    content = io.BytesIO()
    for chunk in response.iter_content(chunk_size=8192):
        content.write(chunk)

    return content.getvalue()


def upload_paper_to_s3(
    pmc_id: str,
    pdf_bytes: Optional[bytes],
    metadata: dict,
    s3_bucket: str,
    prefix: str = "raw/papers/",
) -> bool:
    """
    Upload PDF and metadata to S3.

    Args:
        pmc_id: PMC ID (e.g., "PMC1234567")
        pdf_bytes: PDF content as bytes, or None if no PDF
        metadata: Paper metadata dictionary
        s3_bucket: S3 bucket name
        prefix: S3 key prefix

    Returns:
        True if successful, False otherwise
    """
    s3 = get_s3_client()

    try:
        # Upload PDF if available
        if pdf_bytes:
            pdf_key = f"{prefix}{pmc_id}.pdf"
            s3.put_object(
                Bucket=s3_bucket,
                Key=pdf_key,
                Body=pdf_bytes,
                ContentType="application/pdf",
            )

        # Upload metadata
        metadata_key = f"{prefix}{pmc_id}_metadata.json"
        s3.put_object(
            Bucket=s3_bucket,
            Key=metadata_key,
            Body=json.dumps(metadata, indent=2, ensure_ascii=False),
            ContentType="application/json",
        )

        return True

    except Exception as e:
        logger.error(f"Failed to upload {pmc_id} to S3: {e}")
        return False


@dataclass
class PaperMetadata:
    """Metadata for a downloaded paper."""
    pmc_id: str
    pmid: Optional[str]
    title: str
    authors: list[str]
    journal: str
    pub_date: str
    abstract: str
    pdf_path: Optional[str]
    source_url: str


@retry_with_backoff(max_retries=3, base_delay=1.0)
def search_pmc(query: str, max_results: int = 15) -> list[str]:
    """
    Search PubMed Central for articles matching the query.

    Args:
        query: Search query (e.g., "pediatric cancer St. Jude")
        max_results: Maximum number of PMC IDs to return

    Returns:
        List of PMC IDs (e.g., ["PMC1234567", "PMC7654321"])
    """
    params = {
        "db": "pmc",
        "term": f"{query} AND open access[filter]",
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }

    logger.info(f"Searching PMC for: {query}")
    response = requests.get(ESEARCH_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    id_list = data.get("esearchresult", {}).get("idlist", [])

    # Convert PMCID numbers to PMC format
    pmc_ids = [f"PMC{uid}" for uid in id_list]
    logger.info(f"Found {len(pmc_ids)} articles")

    return pmc_ids


@retry_with_backoff(max_retries=3, base_delay=1.0)
def get_article_metadata(pmc_id: str) -> dict:
    """
    Fetch article metadata from PMC using efetch.

    Args:
        pmc_id: PMC ID (e.g., "PMC1234567")

    Returns:
        Dictionary with article metadata
    """
    # Strip "PMC" prefix for API call
    uid = pmc_id.replace("PMC", "")

    params = {
        "db": "pmc",
        "id": uid,
        "retmode": "xml",
    }

    time.sleep(REQUEST_DELAY)
    response = requests.get(EFETCH_URL, params=params, timeout=30)
    response.raise_for_status()

    # Parse XML response
    root = ElementTree.fromstring(response.content)

    metadata = {
        "pmc_id": pmc_id,
        "pmid": None,
        "title": "",
        "authors": [],
        "journal": "",
        "pub_date": "",
        "abstract": "",
    }

    # Extract article metadata from XML
    article = root.find(".//article")
    if article is None:
        return metadata

    # Title
    title_elem = article.find(".//article-title")
    if title_elem is not None:
        metadata["title"] = "".join(title_elem.itertext()).strip()

    # Authors
    authors = []
    for contrib in article.findall(".//contrib[@contrib-type='author']"):
        surname = contrib.find(".//surname")
        given = contrib.find(".//given-names")
        if surname is not None:
            name = surname.text or ""
            if given is not None and given.text:
                name = f"{given.text} {name}"
            authors.append(name.strip())
    metadata["authors"] = authors

    # Journal
    journal_elem = article.find(".//journal-title")
    if journal_elem is not None:
        metadata["journal"] = journal_elem.text or ""

    # Publication date
    pub_date = article.find(".//pub-date[@pub-type='epub']")
    if pub_date is None:
        pub_date = article.find(".//pub-date")
    if pub_date is not None:
        year = pub_date.find("year")
        month = pub_date.find("month")
        day = pub_date.find("day")
        date_parts = []
        if year is not None and year.text:
            date_parts.append(year.text)
        if month is not None and month.text:
            date_parts.append(month.text.zfill(2))
        if day is not None and day.text:
            date_parts.append(day.text.zfill(2))
        metadata["pub_date"] = "-".join(date_parts)

    # Abstract
    abstract_elem = article.find(".//abstract")
    if abstract_elem is not None:
        metadata["abstract"] = "".join(abstract_elem.itertext()).strip()

    # PMID
    for article_id in article.findall(".//article-id"):
        if article_id.get("pub-id-type") == "pmid":
            metadata["pmid"] = article_id.text

    return metadata


@retry_with_backoff(max_retries=3, base_delay=1.0)
def get_pdf_url(pmc_id: str) -> Optional[str]:
    """
    Get the PDF download URL from PMC Open Access Web Service.

    Args:
        pmc_id: PMC ID (e.g., "PMC1234567")

    Returns:
        PDF URL if available, None otherwise
    """
    params = {
        "id": pmc_id,
    }

    time.sleep(REQUEST_DELAY)
    response = requests.get(PMC_OA_URL, params=params, timeout=30)
    response.raise_for_status()

    # Parse XML response
    root = ElementTree.fromstring(response.content)

    # Check for errors
    error = root.find(".//error")
    if error is not None:
        logger.warning(f"No OA PDF available for {pmc_id}: {error.text}")
        return None

    # Find PDF link
    for link in root.findall(".//link"):
        if link.get("format") == "pdf":
            return link.get("href")

    return None


@retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=30.0)
def _download_pdf_with_retry(url: str, output_path: Path) -> None:
    """
    Internal function to download PDF with retry logic.

    Args:
        url: URL to download from
        output_path: Path to save the PDF

    Raises:
        requests.RequestException: On download failure
    """
    time.sleep(REQUEST_DELAY)

    # Handle FTP URLs by converting to HTTPS
    if url.startswith("ftp://"):
        url = url.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")

    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def download_pdf(url: str, output_path: Path) -> bool:
    """
    Download a PDF file from the given URL.

    Args:
        url: URL to download from
        output_path: Path to save the PDF

    Returns:
        True if successful, False otherwise
    """
    try:
        _download_pdf_with_retry(url, output_path)
        logger.info(f"Downloaded: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def download_papers(
    query: str,
    count: int,
    output_dir: str,
) -> list[PaperMetadata]:
    """
    Download papers from PMC matching the query.

    Args:
        query: Search query for PMC
        count: Number of papers to download
        output_dir: Directory to save papers

    Returns:
        List of PaperMetadata for successfully downloaded papers
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Search for articles
    pmc_ids = search_pmc(query, max_results=count * 2)  # Get extra in case some fail

    papers = []
    downloaded = 0

    for pmc_id in pmc_ids:
        if downloaded >= count:
            break

        logger.info(f"Processing {pmc_id}...")

        # Get metadata
        metadata = get_article_metadata(pmc_id)
        if not metadata.get("title"):
            logger.warning(f"No metadata found for {pmc_id}, skipping")
            continue

        # Get PDF URL
        pdf_url = get_pdf_url(pmc_id)
        pdf_path = None

        if pdf_url:
            pdf_file = output_path / f"{pmc_id}.pdf"
            if download_pdf(pdf_url, pdf_file):
                pdf_path = str(pdf_file)
                downloaded += 1
        else:
            logger.warning(f"No PDF available for {pmc_id}, saving metadata only")

        # Create paper metadata
        paper = PaperMetadata(
            pmc_id=metadata["pmc_id"],
            pmid=metadata.get("pmid"),
            title=metadata["title"],
            authors=metadata.get("authors", []),
            journal=metadata.get("journal", ""),
            pub_date=metadata.get("pub_date", ""),
            abstract=metadata.get("abstract", ""),
            pdf_path=pdf_path,
            source_url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/",
        )

        # Save metadata
        metadata_file = output_path / f"{pmc_id}_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(asdict(paper), f, indent=2, ensure_ascii=False)

        papers.append(paper)
        logger.info(f"Saved metadata: {metadata_file}")

    logger.info(f"Downloaded {downloaded} papers out of {count} requested")
    return papers


def download_papers_to_s3(
    query: str,
    count: int,
    s3_bucket: str,
    skip_existing: bool = True,
) -> dict:
    """
    Download papers from PMC and upload directly to S3.

    Args:
        query: Search query for PMC
        count: Number of papers to download
        s3_bucket: S3 bucket name
        skip_existing: Skip papers already in S3

    Returns:
        Dict with download statistics
    """
    # Get existing PMC IDs if skip_existing
    existing_ids = set()
    if skip_existing:
        logger.info(f"Checking existing papers in s3://{s3_bucket}/raw/papers/...")
        existing_ids = get_existing_pmc_ids(s3_bucket)
        logger.info(f"Found {len(existing_ids)} existing papers in S3")

    # Search for more articles than needed (some may fail or already exist)
    search_count = count * 3 if skip_existing else count * 2
    pmc_ids = search_pmc(query, max_results=search_count)

    # Filter out existing
    if skip_existing:
        new_pmc_ids = [pid for pid in pmc_ids if pid not in existing_ids]
        logger.info(f"After filtering existing: {len(new_pmc_ids)} new papers to process")
    else:
        new_pmc_ids = pmc_ids

    stats = {
        "searched": len(pmc_ids),
        "skipped_existing": len(pmc_ids) - len(new_pmc_ids),
        "downloaded": 0,
        "failed": 0,
        "no_pdf": 0,
    }

    # Process papers with progress bar
    downloaded = 0
    with tqdm(total=count, desc="Downloading papers", unit="paper") as pbar:
        for pmc_id in new_pmc_ids:
            if downloaded >= count:
                break

            # Get metadata
            try:
                metadata = get_article_metadata(pmc_id)
                if not metadata.get("title"):
                    logger.warning(f"No metadata found for {pmc_id}, skipping")
                    stats["failed"] += 1
                    continue
            except Exception as e:
                logger.error(f"Failed to get metadata for {pmc_id}: {e}")
                stats["failed"] += 1
                continue

            # Get PDF URL
            try:
                pdf_url = get_pdf_url(pmc_id)
            except Exception as e:
                logger.error(f"Failed to get PDF URL for {pmc_id}: {e}")
                pdf_url = None

            pdf_bytes = None
            if pdf_url:
                try:
                    pdf_bytes = download_pdf_to_memory(pdf_url)
                except Exception as e:
                    logger.error(f"Failed to download PDF for {pmc_id}: {e}")
                    stats["failed"] += 1
                    continue
            else:
                stats["no_pdf"] += 1
                continue  # Skip papers without PDFs

            # Create full metadata
            paper_metadata = {
                "pmc_id": metadata["pmc_id"],
                "pmid": metadata.get("pmid"),
                "title": metadata["title"],
                "authors": metadata.get("authors", []),
                "journal": metadata.get("journal", ""),
                "pub_date": metadata.get("pub_date", ""),
                "abstract": metadata.get("abstract", ""),
                "pdf_path": f"s3://{s3_bucket}/raw/papers/{pmc_id}.pdf" if pdf_bytes else None,
                "source_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/",
            }

            # Upload to S3
            if upload_paper_to_s3(pmc_id, pdf_bytes, paper_metadata, s3_bucket):
                downloaded += 1
                stats["downloaded"] += 1
                pbar.update(1)
                pbar.set_postfix({"last": pmc_id})
            else:
                stats["failed"] += 1

    logger.info(f"Download complete: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download open-access papers from PubMed Central"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="pediatric cancer St. Jude Children's Research Hospital",
        help="Search query for PMC",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=15,
        help="Number of papers to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sample/papers",
        help="Output directory for papers (local mode only)",
    )
    parser.add_argument(
        "--upload-to-s3",
        action="store_true",
        help="Upload directly to S3 instead of local storage",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=None,
        help="S3 bucket name (default: S3_BUCKET env var or 'pediatric-research-rag')",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip papers already in S3 (default: True)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Re-download papers even if they exist in S3",
    )

    args = parser.parse_args()

    if args.upload_to_s3:
        # S3 mode
        s3_bucket = args.s3_bucket or os.environ.get("S3_BUCKET", "pediatric-research-rag")
        print(f"\nDownloading papers to s3://{s3_bucket}/raw/papers/")
        print(f"Skip existing: {args.skip_existing}\n")

        stats = download_papers_to_s3(
            query=args.query,
            count=args.count,
            s3_bucket=s3_bucket,
            skip_existing=args.skip_existing,
        )

        print(f"\nDownload Summary:")
        print(f"  Searched: {stats['searched']}")
        print(f"  Skipped (already in S3): {stats['skipped_existing']}")
        print(f"  Downloaded: {stats['downloaded']}")
        print(f"  No PDF available: {stats['no_pdf']}")
        print(f"  Failed: {stats['failed']}")

    else:
        # Local mode (original behavior)
        papers = download_papers(
            query=args.query,
            count=args.count,
            output_dir=args.output_dir,
        )

        print(f"\nDownloaded {len([p for p in papers if p.pdf_path])} PDFs")
        print(f"Total papers processed: {len(papers)}")

        for paper in papers:
            status = "PDF" if paper.pdf_path else "metadata only"
            print(f"  - {paper.pmc_id}: {paper.title[:60]}... ({status})")


if __name__ == "__main__":
    main()
