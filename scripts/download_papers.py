#!/usr/bin/env python3
"""
Download open-access research papers from PubMed Central (PMC).

Uses NCBI E-utilities API and PMC Open Access Web Service.
Rate limit: 3 requests/second without API key.
"""
import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# NCBI E-utilities base URLs
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

# Rate limiting
REQUEST_DELAY = 0.34  # ~3 requests per second


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
        help="Output directory for papers",
    )

    args = parser.parse_args()

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
