"""
Document parsers for PDFs and clinical trial JSON files.

Extracts text and metadata from source documents for RAG ingestion.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    """Parsed document with text and metadata."""
    document_id: str
    doc_type: str  # "paper" | "trial"
    title: str
    text: str
    metadata: dict = field(default_factory=dict)
    page_boundaries: list[int] = field(default_factory=list)  # Character positions
    section_titles: dict[int, str] = field(default_factory=dict)  # Position -> title


class PDFParser:
    """Parser for PDF documents using pypdf with pdfplumber fallback."""

    def parse(self, source: Path) -> ParsedDocument:
        """
        Parse a PDF file and extract text with page boundaries.

        Args:
            source: Path to the PDF file

        Returns:
            ParsedDocument with extracted text and metadata
        """
        document_id = source.stem  # e.g., "PMC1234567"

        # Try pypdf first (faster, usually works for text-based PDFs)
        text, page_boundaries = self._parse_with_pypdf(source)

        # If pypdf extraction is poor, try pdfplumber (better for complex layouts)
        if len(text.strip()) < 100:
            logger.info(f"pypdf extraction poor, trying pdfplumber for {document_id}")
            text, page_boundaries = self._parse_with_pdfplumber(source)

        # Clean up extracted text
        text = self._clean_text(text)

        # Extract section titles
        section_titles = self._extract_sections(text)

        # Load metadata if available
        metadata = self._load_metadata(source)

        return ParsedDocument(
            document_id=document_id,
            doc_type="paper",
            title=metadata.get("title", document_id),
            text=text,
            metadata=metadata,
            page_boundaries=page_boundaries,
            section_titles=section_titles,
        )

    def _parse_with_pypdf(self, source: Path) -> tuple[str, list[int]]:
        """Parse PDF using pypdf."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(source))
            text_parts = []
            page_boundaries = [0]
            current_pos = 0

            for page in reader.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
                current_pos += len(page_text) + 1  # +1 for newline
                page_boundaries.append(current_pos)

            return "\n".join(text_parts), page_boundaries

        except Exception as e:
            logger.warning(f"pypdf failed: {e}")
            return "", [0]

    def _parse_with_pdfplumber(self, source: Path) -> tuple[str, list[int]]:
        """Parse PDF using pdfplumber (better for complex layouts)."""
        try:
            import pdfplumber

            text_parts = []
            page_boundaries = [0]
            current_pos = 0

            with pdfplumber.open(str(source)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text_parts.append(page_text)
                    current_pos += len(page_text) + 1
                    page_boundaries.append(current_pos)

            return "\n".join(text_parts), page_boundaries

        except Exception as e:
            logger.error(f"pdfplumber failed: {e}")
            return "", [0]

    def _clean_text(self, text: str) -> str:
        """Clean extracted PDF text."""
        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r"\n\d+\s*\n", "\n", text)

        # Fix common OCR issues
        text = re.sub(r"ﬁ", "fi", text)
        text = re.sub(r"ﬂ", "fl", text)

        return text.strip()

    def _extract_sections(self, text: str) -> dict[int, str]:
        """Extract section titles and their positions."""
        sections = {}

        # Common section patterns in scientific papers
        section_patterns = [
            r"^(Abstract|ABSTRACT)$",
            r"^(Introduction|INTRODUCTION)$",
            r"^(Methods|METHODS|Materials and Methods|MATERIALS AND METHODS)$",
            r"^(Results|RESULTS)$",
            r"^(Discussion|DISCUSSION)$",
            r"^(Conclusion|CONCLUSION|Conclusions|CONCLUSIONS)$",
            r"^(References|REFERENCES)$",
            r"^(Acknowledgements|ACKNOWLEDGEMENTS|Acknowledgments|ACKNOWLEDGMENTS)$",
            r"^\d+\.?\s+(Introduction|Methods|Results|Discussion|Conclusion)$",
        ]

        combined_pattern = "|".join(f"({p})" for p in section_patterns)

        for match in re.finditer(combined_pattern, text, re.MULTILINE):
            sections[match.start()] = match.group().strip()

        return sections

    def _load_metadata(self, source: Path) -> dict:
        """Load metadata from sidecar JSON file if available."""
        metadata_path = source.parent / f"{source.stem}_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        return {}


class ClinicalTrialParser:
    """Parser for ClinicalTrials.gov JSON files."""

    def parse(self, source: Path) -> ParsedDocument:
        """
        Parse a clinical trial JSON file.

        Args:
            source: Path to the JSON file (not the _text.txt file)

        Returns:
            ParsedDocument with structured trial information
        """
        # Load JSON metadata
        with open(source, "r", encoding="utf-8") as f:
            trial_data = json.load(f)

        document_id = trial_data.get("nct_id", source.stem)

        # Try to load pre-generated text file
        text_path = source.parent / f"{document_id}_text.txt"
        if text_path.exists():
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            # Generate text from JSON
            text = self._generate_text(trial_data)

        # Extract section positions
        section_titles = self._extract_sections(text)

        return ParsedDocument(
            document_id=document_id,
            doc_type="trial",
            title=trial_data.get("title", document_id),
            text=text,
            metadata=trial_data,
            page_boundaries=[0, len(text)],  # Trials are single "page"
            section_titles=section_titles,
        )

    def _generate_text(self, trial_data: dict) -> str:
        """Generate RAG-friendly text from trial data."""
        sections = []

        # Title and ID
        sections.append(f"Clinical Trial: {trial_data.get('title', 'Unknown')}")
        sections.append(f"NCT ID: {trial_data.get('nct_id', 'Unknown')}")
        sections.append(f"Status: {trial_data.get('status', 'Unknown')}")

        if trial_data.get("phase"):
            sections.append(f"Phase: {trial_data['phase']}")
        if trial_data.get("sponsor"):
            sections.append(f"Sponsor: {trial_data['sponsor']}")
        sections.append("")

        # Conditions
        conditions = trial_data.get("conditions", [])
        if conditions:
            sections.append("Conditions:")
            for condition in conditions:
                sections.append(f"  - {condition}")
            sections.append("")

        # Interventions
        interventions = trial_data.get("interventions", [])
        if interventions:
            sections.append("Interventions:")
            for intervention in interventions:
                sections.append(f"  - {intervention}")
            sections.append("")

        # Summary
        if trial_data.get("brief_summary"):
            sections.append("Summary:")
            sections.append(trial_data["brief_summary"])
            sections.append("")

        # Detailed description
        if trial_data.get("detailed_description"):
            sections.append("Detailed Description:")
            sections.append(trial_data["detailed_description"])
            sections.append("")

        # Eligibility
        if trial_data.get("eligibility_criteria"):
            sections.append("Eligibility Criteria:")
            sections.append(trial_data["eligibility_criteria"])
            sections.append("")

        # Outcomes
        primary_outcomes = trial_data.get("primary_outcomes", [])
        if primary_outcomes:
            sections.append("Primary Outcomes:")
            for outcome in primary_outcomes:
                sections.append(f"  - {outcome}")
            sections.append("")

        secondary_outcomes = trial_data.get("secondary_outcomes", [])
        if secondary_outcomes:
            sections.append("Secondary Outcomes:")
            for outcome in secondary_outcomes:
                sections.append(f"  - {outcome}")
            sections.append("")

        # Source
        if trial_data.get("source_url"):
            sections.append(f"Source: {trial_data['source_url']}")

        return "\n".join(sections)

    def _extract_sections(self, text: str) -> dict[int, str]:
        """Extract section titles from trial text."""
        sections = {}

        section_names = [
            "Conditions:",
            "Interventions:",
            "Summary:",
            "Detailed Description:",
            "Eligibility Criteria:",
            "Primary Outcomes:",
            "Secondary Outcomes:",
        ]

        for section_name in section_names:
            pos = text.find(section_name)
            if pos != -1:
                sections[pos] = section_name.rstrip(":")

        return sections


def parse_document(source: Path) -> ParsedDocument:
    """
    Parse a document, auto-detecting the type.

    Args:
        source: Path to the document file

    Returns:
        ParsedDocument with extracted text and metadata

    Raises:
        ValueError: If document type cannot be determined
    """
    suffix = source.suffix.lower()

    if suffix == ".pdf":
        return PDFParser().parse(source)
    elif suffix == ".json":
        # Check if it's a clinical trial JSON (has nct_id)
        try:
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "nct_id" in data:
                return ClinicalTrialParser().parse(source)
            else:
                raise ValueError(f"JSON file does not appear to be a clinical trial: {source}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON file: {source}")
    else:
        raise ValueError(f"Unsupported document type: {suffix}")
