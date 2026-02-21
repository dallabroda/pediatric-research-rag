"""Tests for the parsers module."""
import json
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lambdas.ingest.parsers import (
    ClinicalTrialParser,
    PDFParser,
    ParsedDocument,
    parse_document,
)


class TestClinicalTrialParser:
    """Tests for clinical trial JSON parsing."""

    def test_parses_basic_trial(self, tmp_path):
        trial_data = {
            "nct_id": "NCT12345678",
            "title": "Test Clinical Trial",
            "sponsor": "St. Jude",
            "status": "RECRUITING",
            "conditions": ["Pediatric Cancer"],
            "brief_summary": "This is a test trial.",
            "source_url": "https://clinicaltrials.gov/study/NCT12345678",
        }

        json_file = tmp_path / "NCT12345678.json"
        with open(json_file, "w") as f:
            json.dump(trial_data, f)

        parser = ClinicalTrialParser()
        result = parser.parse(json_file)

        assert isinstance(result, ParsedDocument)
        assert result.document_id == "NCT12345678"
        assert result.doc_type == "trial"
        assert result.title == "Test Clinical Trial"
        assert "test trial" in result.text.lower()

    def test_handles_missing_fields(self, tmp_path):
        trial_data = {
            "nct_id": "NCT99999999",
            "title": "Minimal Trial",
        }

        json_file = tmp_path / "NCT99999999.json"
        with open(json_file, "w") as f:
            json.dump(trial_data, f)

        parser = ClinicalTrialParser()
        result = parser.parse(json_file)

        assert result.document_id == "NCT99999999"
        assert result.title == "Minimal Trial"

    def test_extracts_sections(self, tmp_path):
        trial_data = {
            "nct_id": "NCT11111111",
            "title": "Trial With Sections",
            "brief_summary": "Summary here",
            "detailed_description": "Details here",
            "eligibility_criteria": "Criteria here",
        }

        json_file = tmp_path / "NCT11111111.json"
        with open(json_file, "w") as f:
            json.dump(trial_data, f)

        parser = ClinicalTrialParser()
        result = parser.parse(json_file)

        # Should extract section titles
        assert len(result.section_titles) > 0


class TestParseDocument:
    """Tests for auto-detection parsing."""

    def test_detects_clinical_trial_json(self, tmp_path):
        trial_data = {
            "nct_id": "NCT00000001",
            "title": "Auto-detected Trial",
        }

        json_file = tmp_path / "NCT00000001.json"
        with open(json_file, "w") as f:
            json.dump(trial_data, f)

        result = parse_document(json_file)

        assert result.doc_type == "trial"

    def test_raises_for_non_trial_json(self, tmp_path):
        other_data = {
            "key": "value",
            "not_a_trial": True,
        }

        json_file = tmp_path / "other.json"
        with open(json_file, "w") as f:
            json.dump(other_data, f)

        with pytest.raises(ValueError, match="does not appear to be a clinical trial"):
            parse_document(json_file)

    def test_raises_for_unsupported_type(self, tmp_path):
        txt_file = tmp_path / "file.txt"
        txt_file.write_text("Some text")

        with pytest.raises(ValueError, match="Unsupported document type"):
            parse_document(txt_file)
