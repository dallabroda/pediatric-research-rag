"""
Lambda handler for scheduled data refresh.

Triggered by EventBridge on a weekly schedule to:
1. Check PMC for new St. Jude publications
2. Check ClinicalTrials.gov for updated trials
3. Download new documents to S3 raw/
4. Trigger ingest pipeline via S3 event
"""
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree

import boto3
import requests

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.retry import retry_with_backoff

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
S3_BUCKET = os.environ.get("S3_BUCKET", "pediatric-research-rag")
RAW_PREFIX = os.environ.get("RAW_PREFIX", "raw/")
MAX_NEW_PAPERS = int(os.environ.get("MAX_NEW_PAPERS", "5"))
MAX_NEW_TRIALS = int(os.environ.get("MAX_NEW_TRIALS", "5"))
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "7"))

# API endpoints
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
CT_API_URL = "https://clinicaltrials.gov/api/v2/studies"

# Search query
PMC_QUERY = "pediatric cancer St. Jude Children's Research Hospital"
CT_SPONSOR = "St. Jude Children's Research Hospital"


def get_s3_client():
    """Get boto3 S3 client."""
    return boto3.client("s3")


def get_existing_documents(bucket: str) -> set[str]:
    """
    Get set of existing document IDs in the bucket.

    Args:
        bucket: S3 bucket name

    Returns:
        Set of document IDs (PMC IDs and NCT IDs)
    """
    s3 = get_s3_client()
    existing = set()

    # List papers
    try:
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=f"{RAW_PREFIX}papers/",
        )
        for obj in response.get("Contents", []):
            key = obj["Key"]
            # Extract PMC ID from filename
            if key.endswith(".pdf"):
                doc_id = Path(key).stem
                existing.add(doc_id)
    except Exception as e:
        logger.warning(f"Error listing papers: {e}")

    # List trials
    try:
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=f"{RAW_PREFIX}trials/",
        )
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json") and not key.endswith("_text.txt"):
                doc_id = Path(key).stem
                existing.add(doc_id)
    except Exception as e:
        logger.warning(f"Error listing trials: {e}")

    return existing


@retry_with_backoff(max_retries=3, base_delay=1.0)
def search_recent_papers(lookback_days: int, max_results: int) -> list[str]:
    """
    Search PMC for recent St. Jude publications.

    Args:
        lookback_days: Number of days to look back
        max_results: Maximum papers to return

    Returns:
        List of PMC IDs
    """
    # Build date range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=lookback_days)
    date_range = f"{start_date.strftime('%Y/%m/%d')}:{end_date.strftime('%Y/%m/%d')}[PDAT]"

    params = {
        "db": "pmc",
        "term": f"({PMC_QUERY}) AND {date_range} AND open access[filter]",
        "retmax": max_results,
        "retmode": "json",
        "sort": "pub_date",
    }

    logger.info(f"Searching PMC for recent papers (last {lookback_days} days)")
    response = requests.get(ESEARCH_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    id_list = data.get("esearchresult", {}).get("idlist", [])

    pmc_ids = [f"PMC{uid}" for uid in id_list]
    logger.info(f"Found {len(pmc_ids)} recent papers")

    return pmc_ids


@retry_with_backoff(max_retries=3, base_delay=1.0)
def search_recent_trials(lookback_days: int, max_results: int) -> list[dict]:
    """
    Search ClinicalTrials.gov for recently updated St. Jude trials.

    Args:
        lookback_days: Number of days to look back
        max_results: Maximum trials to return

    Returns:
        List of trial data dicts
    """
    # Build date filter
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=lookback_days)

    params = {
        "query.spons": CT_SPONSOR,
        "filter.lastUpdatePostDate": f"MIN:{start_date.strftime('%Y-%m-%d')}",
        "pageSize": min(max_results, 100),
        "format": "json",
        "fields": "NCTId,BriefTitle,OverallStatus,LastUpdatePostDate",
    }

    logger.info(f"Searching ClinicalTrials.gov for recent updates (last {lookback_days} days)")
    response = requests.get(CT_API_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    studies = data.get("studies", [])

    logger.info(f"Found {len(studies)} recently updated trials")
    return studies


@retry_with_backoff(max_retries=3, base_delay=1.0)
def get_pdf_url(pmc_id: str) -> Optional[str]:
    """Get PDF download URL for a PMC ID."""
    params = {"id": pmc_id}
    response = requests.get(PMC_OA_URL, params=params, timeout=30)
    response.raise_for_status()

    root = ElementTree.fromstring(response.content)
    error = root.find(".//error")
    if error is not None:
        return None

    for link in root.findall(".//link"):
        if link.get("format") == "pdf":
            return link.get("href")

    return None


@retry_with_backoff(max_retries=3, base_delay=2.0)
def download_paper_to_s3(pmc_id: str, bucket: str) -> bool:
    """
    Download a paper PDF and upload to S3.

    Args:
        pmc_id: PMC ID
        bucket: S3 bucket name

    Returns:
        True if successful
    """
    pdf_url = get_pdf_url(pmc_id)
    if not pdf_url:
        logger.warning(f"No PDF URL for {pmc_id}")
        return False

    # Handle FTP URLs
    if pdf_url.startswith("ftp://"):
        pdf_url = pdf_url.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")

    # Download PDF
    response = requests.get(pdf_url, timeout=120, stream=True)
    response.raise_for_status()

    # Upload to S3
    s3 = get_s3_client()
    s3_key = f"{RAW_PREFIX}papers/{pmc_id}.pdf"

    s3.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=response.content,
        ContentType="application/pdf",
    )

    logger.info(f"Uploaded {pmc_id} to s3://{bucket}/{s3_key}")
    return True


@retry_with_backoff(max_retries=3, base_delay=1.0)
def download_trial_to_s3(nct_id: str, bucket: str) -> bool:
    """
    Download trial data and upload to S3.

    Args:
        nct_id: NCT ID
        bucket: S3 bucket name

    Returns:
        True if successful
    """
    # Fetch full trial data
    params = {
        "query.term": nct_id,
        "format": "json",
    }

    response = requests.get(CT_API_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    studies = data.get("studies", [])

    if not studies:
        logger.warning(f"Trial not found: {nct_id}")
        return False

    trial_data = studies[0]

    # Upload JSON to S3
    s3 = get_s3_client()
    s3_key = f"{RAW_PREFIX}trials/{nct_id}.json"

    s3.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=json.dumps(trial_data, ensure_ascii=False, indent=2),
        ContentType="application/json",
    )

    logger.info(f"Uploaded {nct_id} to s3://{bucket}/{s3_key}")
    return True


def handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for scheduled data refresh.

    Triggered by EventBridge rule on weekly schedule.

    Args:
        event: EventBridge event
        context: Lambda context

    Returns:
        Response dict with refresh statistics
    """
    try:
        logger.info("Starting scheduled data refresh")
        logger.info(f"Event: {json.dumps(event)}")

        # Get existing documents
        existing_docs = get_existing_documents(S3_BUCKET)
        logger.info(f"Found {len(existing_docs)} existing documents")

        stats = {
            "papers_checked": 0,
            "papers_downloaded": 0,
            "papers_skipped": 0,
            "trials_checked": 0,
            "trials_downloaded": 0,
            "trials_skipped": 0,
            "errors": [],
        }

        # Check for new papers
        try:
            recent_papers = search_recent_papers(LOOKBACK_DAYS, MAX_NEW_PAPERS * 2)
            stats["papers_checked"] = len(recent_papers)

            for pmc_id in recent_papers:
                if pmc_id in existing_docs:
                    stats["papers_skipped"] += 1
                    continue

                if stats["papers_downloaded"] >= MAX_NEW_PAPERS:
                    break

                try:
                    if download_paper_to_s3(pmc_id, S3_BUCKET):
                        stats["papers_downloaded"] += 1
                except Exception as e:
                    logger.error(f"Failed to download {pmc_id}: {e}")
                    stats["errors"].append(f"Paper {pmc_id}: {str(e)}")

        except Exception as e:
            logger.error(f"Failed to search for papers: {e}")
            stats["errors"].append(f"Paper search: {str(e)}")

        # Check for updated trials
        try:
            recent_trials = search_recent_trials(LOOKBACK_DAYS, MAX_NEW_TRIALS * 2)
            stats["trials_checked"] = len(recent_trials)

            for trial in recent_trials:
                protocol = trial.get("protocolSection", {})
                id_module = protocol.get("identificationModule", {})
                nct_id = id_module.get("nctId", "")

                if not nct_id:
                    continue

                # For trials, we re-download even if they exist (to get updates)
                if stats["trials_downloaded"] >= MAX_NEW_TRIALS:
                    break

                try:
                    if download_trial_to_s3(nct_id, S3_BUCKET):
                        if nct_id in existing_docs:
                            stats["trials_skipped"] += 1  # Updated, not new
                        else:
                            stats["trials_downloaded"] += 1
                except Exception as e:
                    logger.error(f"Failed to download {nct_id}: {e}")
                    stats["errors"].append(f"Trial {nct_id}: {str(e)}")

        except Exception as e:
            logger.error(f"Failed to search for trials: {e}")
            stats["errors"].append(f"Trial search: {str(e)}")

        # Log summary
        logger.info(f"Refresh complete: {json.dumps(stats)}")

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Refresh complete",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "stats": stats,
            }),
        }

    except Exception as e:
        logger.exception("Error during scheduled refresh")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }
