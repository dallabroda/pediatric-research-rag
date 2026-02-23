#!/usr/bin/env python3
"""
Download clinical trial data from ClinicalTrials.gov API v2.

Fetches trials sponsored by St. Jude Children's Research Hospital.

Supports direct upload to S3 with --upload-to-s3 flag.
"""
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import boto3
import requests
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lambdas.shared.retry import retry_with_backoff

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ClinicalTrials.gov API v2
CT_API_URL = "https://clinicaltrials.gov/api/v2/studies"

# Rate limiting
REQUEST_DELAY = 0.5


def get_s3_client():
    """Get boto3 S3 client."""
    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )


def get_existing_nct_ids(s3_bucket: str, prefix: str = "raw/trials/") -> set[str]:
    """
    List NCT IDs already present in S3 bucket.

    Args:
        s3_bucket: S3 bucket name
        prefix: S3 key prefix for trials

    Returns:
        Set of NCT IDs (e.g., {"NCT01234567", "NCT07654321"})
    """
    s3 = get_s3_client()
    nct_ids = set()

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Extract NCT ID from filename like "raw/trials/NCT01234567.json"
            filename = key.split("/")[-1]
            if filename.endswith(".json") and not filename.endswith("_text.txt"):
                nct_id = filename.replace(".json", "")
                if nct_id.startswith("NCT"):
                    nct_ids.add(nct_id)

    return nct_ids


def upload_trial_to_s3(
    trial: "TrialMetadata",
    text_content: str,
    s3_bucket: str,
    prefix: str = "raw/trials/",
) -> bool:
    """
    Upload trial JSON and text to S3.

    Args:
        trial: TrialMetadata object
        text_content: Text representation for RAG
        s3_bucket: S3 bucket name
        prefix: S3 key prefix

    Returns:
        True if successful, False otherwise
    """
    s3 = get_s3_client()

    try:
        # Upload JSON metadata
        json_key = f"{prefix}{trial.nct_id}.json"
        s3.put_object(
            Bucket=s3_bucket,
            Key=json_key,
            Body=json.dumps(asdict(trial), indent=2, ensure_ascii=False),
            ContentType="application/json",
        )

        # Upload text for RAG
        text_key = f"{prefix}{trial.nct_id}_text.txt"
        s3.put_object(
            Bucket=s3_bucket,
            Key=text_key,
            Body=text_content.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )

        return True

    except Exception as e:
        logger.error(f"Failed to upload {trial.nct_id} to S3: {e}")
        return False


@dataclass
class TrialMetadata:
    """Metadata for a clinical trial."""
    nct_id: str
    title: str
    official_title: Optional[str]
    sponsor: str
    status: str
    phase: Optional[str]
    conditions: list[str]
    interventions: list[str]
    enrollment: Optional[int]
    start_date: Optional[str]
    completion_date: Optional[str]
    brief_summary: str
    detailed_description: Optional[str]
    eligibility_criteria: Optional[str]
    primary_outcomes: list[str] = field(default_factory=list)
    secondary_outcomes: list[str] = field(default_factory=list)
    source_url: str = ""


@retry_with_backoff(max_retries=3, base_delay=1.0)
def _fetch_trials_page(params: dict) -> dict:
    """
    Fetch a single page of trials from ClinicalTrials.gov API.

    Args:
        params: API query parameters

    Returns:
        JSON response data
    """
    time.sleep(REQUEST_DELAY)
    response = requests.get(CT_API_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def search_trials(sponsor: str, max_results: int = 10) -> list[dict]:
    """
    Search ClinicalTrials.gov for trials by sponsor.

    Args:
        sponsor: Sponsor organization name
        max_results: Maximum number of trials to return

    Returns:
        List of raw study data dictionaries
    """
    params = {
        "query.spons": sponsor,
        "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING,COMPLETED",
        "pageSize": min(max_results, 100),
        "format": "json",
        "fields": ",".join([
            "NCTId",
            "BriefTitle",
            "OfficialTitle",
            "LeadSponsorName",
            "OverallStatus",
            "Phase",
            "Condition",
            "InterventionName",
            "EnrollmentCount",
            "StartDate",
            "CompletionDate",
            "BriefSummary",
            "DetailedDescription",
            "EligibilityCriteria",
            "PrimaryOutcomeMeasure",
            "SecondaryOutcomeMeasure",
        ]),
    }

    logger.info(f"Searching ClinicalTrials.gov for sponsor: {sponsor}")

    studies = []
    next_page_token = None

    while len(studies) < max_results:
        if next_page_token:
            params["pageToken"] = next_page_token

        data = _fetch_trials_page(params)
        batch = data.get("studies", [])
        studies.extend(batch)

        next_page_token = data.get("nextPageToken")
        if not next_page_token or not batch:
            break

    logger.info(f"Found {len(studies)} trials")
    return studies[:max_results]


def parse_study(raw_study: dict) -> TrialMetadata:
    """
    Parse raw ClinicalTrials.gov study data into TrialMetadata.

    Args:
        raw_study: Raw study dictionary from API

    Returns:
        Parsed TrialMetadata object
    """
    protocol = raw_study.get("protocolSection", {})
    id_module = protocol.get("identificationModule", {})
    status_module = protocol.get("statusModule", {})
    sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
    design_module = protocol.get("designModule", {})
    conditions_module = protocol.get("conditionsModule", {})
    arms_module = protocol.get("armsInterventionsModule", {})
    description_module = protocol.get("descriptionModule", {})
    eligibility_module = protocol.get("eligibilityModule", {})
    outcomes_module = protocol.get("outcomesModule", {})

    # Extract interventions
    interventions = []
    for intervention in arms_module.get("interventions", []):
        name = intervention.get("name", "")
        int_type = intervention.get("type", "")
        if name:
            interventions.append(f"{int_type}: {name}" if int_type else name)

    # Extract primary outcomes
    primary_outcomes = []
    for outcome in outcomes_module.get("primaryOutcomes", []):
        measure = outcome.get("measure", "")
        if measure:
            primary_outcomes.append(measure)

    # Extract secondary outcomes
    secondary_outcomes = []
    for outcome in outcomes_module.get("secondaryOutcomes", []):
        measure = outcome.get("measure", "")
        if measure:
            secondary_outcomes.append(measure)

    # Get lead sponsor
    lead_sponsor = sponsor_module.get("leadSponsor", {})

    # Parse dates
    start_date_struct = status_module.get("startDateStruct", {})
    completion_date_struct = status_module.get("completionDateStruct", {})

    nct_id = id_module.get("nctId", "")

    return TrialMetadata(
        nct_id=nct_id,
        title=id_module.get("briefTitle", ""),
        official_title=id_module.get("officialTitle"),
        sponsor=lead_sponsor.get("name", ""),
        status=status_module.get("overallStatus", ""),
        phase=",".join(design_module.get("phases", [])) or None,
        conditions=conditions_module.get("conditions", []),
        interventions=interventions,
        enrollment=design_module.get("enrollmentInfo", {}).get("count"),
        start_date=start_date_struct.get("date"),
        completion_date=completion_date_struct.get("date"),
        brief_summary=description_module.get("briefSummary", ""),
        detailed_description=description_module.get("detailedDescription"),
        eligibility_criteria=eligibility_module.get("eligibilityCriteria"),
        primary_outcomes=primary_outcomes,
        secondary_outcomes=secondary_outcomes,
        source_url=f"https://clinicaltrials.gov/study/{nct_id}",
    )


def extract_text_for_rag(trial: TrialMetadata) -> str:
    """
    Convert trial metadata to text suitable for RAG ingestion.

    Args:
        trial: TrialMetadata object

    Returns:
        Formatted text representation of the trial
    """
    sections = []

    # Title and ID
    sections.append(f"Clinical Trial: {trial.title}")
    sections.append(f"NCT ID: {trial.nct_id}")
    sections.append(f"Status: {trial.status}")
    if trial.phase:
        sections.append(f"Phase: {trial.phase}")
    sections.append(f"Sponsor: {trial.sponsor}")
    sections.append("")

    # Official title if different
    if trial.official_title and trial.official_title != trial.title:
        sections.append(f"Official Title: {trial.official_title}")
        sections.append("")

    # Conditions
    if trial.conditions:
        sections.append("Conditions:")
        for condition in trial.conditions:
            sections.append(f"  - {condition}")
        sections.append("")

    # Interventions
    if trial.interventions:
        sections.append("Interventions:")
        for intervention in trial.interventions:
            sections.append(f"  - {intervention}")
        sections.append("")

    # Summary
    if trial.brief_summary:
        sections.append("Summary:")
        sections.append(trial.brief_summary)
        sections.append("")

    # Detailed description
    if trial.detailed_description:
        sections.append("Detailed Description:")
        sections.append(trial.detailed_description)
        sections.append("")

    # Eligibility
    if trial.eligibility_criteria:
        sections.append("Eligibility Criteria:")
        sections.append(trial.eligibility_criteria)
        sections.append("")

    # Outcomes
    if trial.primary_outcomes:
        sections.append("Primary Outcomes:")
        for outcome in trial.primary_outcomes:
            sections.append(f"  - {outcome}")
        sections.append("")

    if trial.secondary_outcomes:
        sections.append("Secondary Outcomes:")
        for outcome in trial.secondary_outcomes:
            sections.append(f"  - {outcome}")
        sections.append("")

    # Enrollment and dates
    if trial.enrollment:
        sections.append(f"Enrollment: {trial.enrollment} participants")
    if trial.start_date:
        sections.append(f"Start Date: {trial.start_date}")
    if trial.completion_date:
        sections.append(f"Completion Date: {trial.completion_date}")

    sections.append("")
    sections.append(f"Source: {trial.source_url}")

    return "\n".join(sections)


def download_trials(
    sponsor: str,
    count: int,
    output_dir: str,
) -> list[TrialMetadata]:
    """
    Download clinical trials from ClinicalTrials.gov.

    Args:
        sponsor: Sponsor organization to search for
        count: Number of trials to download
        output_dir: Directory to save trial data

    Returns:
        List of TrialMetadata for downloaded trials
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Search for trials
    raw_studies = search_trials(sponsor, max_results=count)

    trials = []

    for raw_study in raw_studies:
        try:
            trial = parse_study(raw_study)
            if not trial.nct_id:
                continue

            logger.info(f"Processing {trial.nct_id}: {trial.title[:50]}...")

            # Save JSON metadata
            json_file = output_path / f"{trial.nct_id}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(asdict(trial), f, indent=2, ensure_ascii=False)

            # Save text for RAG
            text_file = output_path / f"{trial.nct_id}_text.txt"
            text_content = extract_text_for_rag(trial)
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(text_content)

            trials.append(trial)
            logger.info(f"Saved: {json_file} and {text_file}")

        except Exception as e:
            logger.error(f"Failed to process trial: {e}")
            continue

    logger.info(f"Downloaded {len(trials)} trials")
    return trials


def download_trials_to_s3(
    sponsor: str,
    count: int,
    s3_bucket: str,
    skip_existing: bool = True,
) -> dict:
    """
    Download trials from ClinicalTrials.gov and upload directly to S3.

    Args:
        sponsor: Sponsor organization to search for
        count: Number of trials to download
        s3_bucket: S3 bucket name
        skip_existing: Skip trials already in S3

    Returns:
        Dict with download statistics
    """
    # Get existing NCT IDs if skip_existing
    existing_ids = set()
    if skip_existing:
        logger.info(f"Checking existing trials in s3://{s3_bucket}/raw/trials/...")
        existing_ids = get_existing_nct_ids(s3_bucket)
        logger.info(f"Found {len(existing_ids)} existing trials in S3")

    # Search for more trials than needed (some may already exist)
    search_count = count * 2 if skip_existing else count
    raw_studies = search_trials(sponsor, max_results=search_count)

    stats = {
        "searched": len(raw_studies),
        "skipped_existing": 0,
        "downloaded": 0,
        "failed": 0,
    }

    # Process trials with progress bar
    with tqdm(total=count, desc="Downloading trials", unit="trial") as pbar:
        for raw_study in raw_studies:
            if stats["downloaded"] >= count:
                break

            try:
                trial = parse_study(raw_study)
                if not trial.nct_id:
                    stats["failed"] += 1
                    continue

                # Skip if already exists
                if skip_existing and trial.nct_id in existing_ids:
                    stats["skipped_existing"] += 1
                    continue

                # Generate text for RAG
                text_content = extract_text_for_rag(trial)

                # Upload to S3
                if upload_trial_to_s3(trial, text_content, s3_bucket):
                    stats["downloaded"] += 1
                    pbar.update(1)
                    pbar.set_postfix({"last": trial.nct_id})
                else:
                    stats["failed"] += 1

            except Exception as e:
                logger.error(f"Failed to process trial: {e}")
                stats["failed"] += 1
                continue

    logger.info(f"Download complete: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download clinical trials from ClinicalTrials.gov"
    )
    parser.add_argument(
        "--sponsor",
        type=str,
        default="St. Jude Children's Research Hospital",
        help="Sponsor organization to search for",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of trials to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sample/trials",
        help="Output directory for trial data (local mode only)",
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
        help="Skip trials already in S3 (default: True)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Re-download trials even if they exist in S3",
    )

    args = parser.parse_args()

    if args.upload_to_s3:
        # S3 mode
        s3_bucket = args.s3_bucket or os.environ.get("S3_BUCKET", "pediatric-research-rag")
        print(f"\nDownloading trials to s3://{s3_bucket}/raw/trials/")
        print(f"Skip existing: {args.skip_existing}\n")

        stats = download_trials_to_s3(
            sponsor=args.sponsor,
            count=args.count,
            s3_bucket=s3_bucket,
            skip_existing=args.skip_existing,
        )

        print(f"\nDownload Summary:")
        print(f"  Searched: {stats['searched']}")
        print(f"  Skipped (already in S3): {stats['skipped_existing']}")
        print(f"  Downloaded: {stats['downloaded']}")
        print(f"  Failed: {stats['failed']}")

    else:
        # Local mode (original behavior)
        trials = download_trials(
            sponsor=args.sponsor,
            count=args.count,
            output_dir=args.output_dir,
        )

        print(f"\nDownloaded {len(trials)} trials")
        for trial in trials:
            print(f"  - {trial.nct_id}: {trial.title[:60]}... ({trial.status})")


if __name__ == "__main__":
    main()
