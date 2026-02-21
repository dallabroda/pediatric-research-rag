#!/usr/bin/env python3
"""
Download clinical trial data from ClinicalTrials.gov API v2.

Fetches trials sponsored by St. Jude Children's Research Hospital.
"""
import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import requests

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lambdas.shared.retry import retry_with_backoff

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ClinicalTrials.gov API v2
CT_API_URL = "https://clinicaltrials.gov/api/v2/studies"

# Rate limiting
REQUEST_DELAY = 0.5


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
        help="Output directory for trial data",
    )

    args = parser.parse_args()

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
