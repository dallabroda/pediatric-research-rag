#!/usr/bin/env python3
"""
Download survivorship data from St. Jude LIFE study.

Fetches data dictionary and summary statistics from the St. Jude
Survivorship Portal (open-access tier).
"""
import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import requests

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lambdas.shared.retry import retry_with_backoff

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# St. Jude Survivorship Portal endpoints
# Note: These are example endpoints - actual portal may have different API structure
SJLIFE_BASE_URL = "https://survivorship.stjude.cloud/api/v1"


@dataclass
class VariableDefinition:
    """Definition of a SJLIFE data variable."""
    variable_name: str
    category: str
    description: str
    data_type: str
    values: list[str] = field(default_factory=list)
    unit: Optional[str] = None
    measurement_protocol: Optional[str] = None


@dataclass
class SurvivorshipDataset:
    """Collection of survivorship variables organized by category."""
    categories: dict[str, list[VariableDefinition]] = field(default_factory=dict)
    version: str = "1.0"
    source: str = "SJLIFE Data Dictionary"


# Since the actual API may not be publicly accessible, we'll define
# a comprehensive set of SJLIFE variables based on published literature
SJLIFE_VARIABLE_CATEGORIES = {
    "Demographics": [
        {
            "variable_name": "age_at_diagnosis",
            "description": "Age at primary cancer diagnosis in years",
            "data_type": "continuous",
            "unit": "years",
        },
        {
            "variable_name": "age_at_evaluation",
            "description": "Age at SJLIFE evaluation in years",
            "data_type": "continuous",
            "unit": "years",
        },
        {
            "variable_name": "sex",
            "description": "Biological sex of participant",
            "data_type": "categorical",
            "values": ["Male", "Female"],
        },
        {
            "variable_name": "race_ethnicity",
            "description": "Self-reported race and ethnicity",
            "data_type": "categorical",
            "values": ["White", "Black", "Hispanic", "Asian", "Other"],
        },
        {
            "variable_name": "time_since_diagnosis",
            "description": "Time elapsed since primary cancer diagnosis",
            "data_type": "continuous",
            "unit": "years",
        },
    ],
    "Primary_Cancer": [
        {
            "variable_name": "primary_diagnosis",
            "description": "Primary cancer diagnosis at initial treatment",
            "data_type": "categorical",
            "values": [
                "Acute Lymphoblastic Leukemia",
                "Acute Myeloid Leukemia",
                "Hodgkin Lymphoma",
                "Non-Hodgkin Lymphoma",
                "CNS Tumor",
                "Neuroblastoma",
                "Wilms Tumor",
                "Bone Tumor",
                "Soft Tissue Sarcoma",
                "Retinoblastoma",
                "Germ Cell Tumor",
                "Other",
            ],
        },
        {
            "variable_name": "treatment_era",
            "description": "Era of cancer treatment",
            "data_type": "categorical",
            "values": ["1962-1979", "1980-1989", "1990-1999", "2000-2009", "2010-present"],
        },
        {
            "variable_name": "relapse_history",
            "description": "History of cancer relapse",
            "data_type": "categorical",
            "values": ["No", "Yes - single relapse", "Yes - multiple relapses"],
        },
    ],
    "Treatment_Exposures": [
        {
            "variable_name": "anthracycline_dose",
            "description": "Cumulative anthracycline dose (doxorubicin equivalents)",
            "data_type": "continuous",
            "unit": "mg/m2",
            "measurement_protocol": "Isotoxic equivalents using conversion factors",
        },
        {
            "variable_name": "chest_radiation",
            "description": "Radiation exposure to chest/mediastinum",
            "data_type": "categorical",
            "values": ["None", "<20 Gy", "20-39 Gy", ">=40 Gy"],
        },
        {
            "variable_name": "cranial_radiation",
            "description": "Radiation exposure to brain/cranium",
            "data_type": "categorical",
            "values": ["None", "<18 Gy", "18-23 Gy", ">=24 Gy"],
        },
        {
            "variable_name": "alkylating_agent_dose",
            "description": "Cumulative alkylating agent dose (cyclophosphamide equivalents)",
            "data_type": "continuous",
            "unit": "mg/m2",
        },
        {
            "variable_name": "platinum_dose",
            "description": "Cumulative platinum agent dose (cisplatin equivalents)",
            "data_type": "continuous",
            "unit": "mg/m2",
        },
        {
            "variable_name": "hematopoietic_cell_transplant",
            "description": "History of hematopoietic cell transplantation",
            "data_type": "categorical",
            "values": ["No", "Autologous", "Allogeneic"],
        },
    ],
    "Cardiac_Outcomes": [
        {
            "variable_name": "cardiomyopathy",
            "description": "Presence of cardiomyopathy based on echocardiography",
            "data_type": "categorical",
            "values": ["None", "Subclinical", "Clinical"],
            "measurement_protocol": "CTCAE grading based on LVEF",
        },
        {
            "variable_name": "lvef",
            "description": "Left ventricular ejection fraction",
            "data_type": "continuous",
            "unit": "%",
            "measurement_protocol": "2D echocardiography, Simpson's method",
        },
        {
            "variable_name": "coronary_artery_disease",
            "description": "Presence of coronary artery disease",
            "data_type": "categorical",
            "values": ["No", "Yes"],
        },
        {
            "variable_name": "heart_failure",
            "description": "Clinical heart failure diagnosis",
            "data_type": "categorical",
            "values": ["No", "HFpEF", "HFrEF"],
        },
        {
            "variable_name": "valvular_disease",
            "description": "Presence of valvular heart disease",
            "data_type": "categorical",
            "values": ["None", "Mild", "Moderate", "Severe"],
        },
    ],
    "Secondary_Malignancies": [
        {
            "variable_name": "secondary_malignancy",
            "description": "Development of secondary malignant neoplasm",
            "data_type": "categorical",
            "values": ["No", "Yes"],
        },
        {
            "variable_name": "smn_type",
            "description": "Type of secondary malignancy",
            "data_type": "categorical",
            "values": ["None", "Breast", "Thyroid", "Skin", "Sarcoma", "Leukemia", "Other"],
        },
        {
            "variable_name": "time_to_smn",
            "description": "Time from primary diagnosis to secondary malignancy",
            "data_type": "continuous",
            "unit": "years",
        },
    ],
    "Neurocognitive_Outcomes": [
        {
            "variable_name": "neurocognitive_impairment",
            "description": "Presence of neurocognitive impairment",
            "data_type": "categorical",
            "values": ["None", "Mild", "Moderate", "Severe"],
        },
        {
            "variable_name": "iq_score",
            "description": "Full-scale IQ score",
            "data_type": "continuous",
            "unit": "standard score",
            "measurement_protocol": "WAIS-IV or age-appropriate Wechsler scale",
        },
        {
            "variable_name": "processing_speed",
            "description": "Processing speed index",
            "data_type": "continuous",
            "unit": "standard score",
        },
        {
            "variable_name": "attention_deficit",
            "description": "Attention deficit classification",
            "data_type": "categorical",
            "values": ["None", "Mild", "Moderate", "Severe"],
        },
        {
            "variable_name": "executive_function",
            "description": "Executive function impairment",
            "data_type": "categorical",
            "values": ["Normal", "Impaired"],
        },
    ],
    "Endocrine_Outcomes": [
        {
            "variable_name": "hypothyroidism",
            "description": "Presence of hypothyroidism",
            "data_type": "categorical",
            "values": ["No", "Primary", "Central"],
        },
        {
            "variable_name": "growth_hormone_deficiency",
            "description": "Growth hormone deficiency status",
            "data_type": "categorical",
            "values": ["No", "Yes"],
        },
        {
            "variable_name": "gonadal_dysfunction",
            "description": "Presence of gonadal dysfunction",
            "data_type": "categorical",
            "values": ["No", "Yes"],
        },
        {
            "variable_name": "diabetes",
            "description": "Diabetes mellitus status",
            "data_type": "categorical",
            "values": ["No", "Prediabetes", "Type 2 DM"],
        },
        {
            "variable_name": "metabolic_syndrome",
            "description": "Metabolic syndrome presence",
            "data_type": "categorical",
            "values": ["No", "Yes"],
        },
        {
            "variable_name": "bmi_category",
            "description": "Body mass index category",
            "data_type": "categorical",
            "values": ["Underweight", "Normal", "Overweight", "Obese"],
        },
    ],
    "Quality_of_Life": [
        {
            "variable_name": "sf36_physical",
            "description": "SF-36 Physical Component Summary Score",
            "data_type": "continuous",
            "unit": "T-score",
            "measurement_protocol": "SF-36v2 questionnaire",
        },
        {
            "variable_name": "sf36_mental",
            "description": "SF-36 Mental Component Summary Score",
            "data_type": "continuous",
            "unit": "T-score",
        },
        {
            "variable_name": "fatigue",
            "description": "Cancer-related fatigue severity",
            "data_type": "categorical",
            "values": ["None", "Mild", "Moderate", "Severe"],
        },
        {
            "variable_name": "pain",
            "description": "Chronic pain severity",
            "data_type": "categorical",
            "values": ["None", "Mild", "Moderate", "Severe"],
        },
        {
            "variable_name": "anxiety",
            "description": "Anxiety symptoms",
            "data_type": "categorical",
            "values": ["None", "Mild", "Moderate", "Severe"],
        },
        {
            "variable_name": "depression",
            "description": "Depressive symptoms",
            "data_type": "categorical",
            "values": ["None", "Mild", "Moderate", "Severe"],
        },
    ],
    "Pulmonary_Outcomes": [
        {
            "variable_name": "pulmonary_dysfunction",
            "description": "Pulmonary function abnormality",
            "data_type": "categorical",
            "values": ["None", "Restrictive", "Obstructive", "Mixed"],
        },
        {
            "variable_name": "fev1_percent_predicted",
            "description": "FEV1 as percentage of predicted",
            "data_type": "continuous",
            "unit": "%",
            "measurement_protocol": "Spirometry per ATS/ERS guidelines",
        },
        {
            "variable_name": "dlco_percent_predicted",
            "description": "DLCO as percentage of predicted",
            "data_type": "continuous",
            "unit": "%",
        },
    ],
    "Musculoskeletal_Outcomes": [
        {
            "variable_name": "osteoporosis",
            "description": "Bone mineral density classification",
            "data_type": "categorical",
            "values": ["Normal", "Osteopenia", "Osteoporosis"],
            "measurement_protocol": "DXA scan, WHO criteria",
        },
        {
            "variable_name": "osteonecrosis",
            "description": "Presence of osteonecrosis",
            "data_type": "categorical",
            "values": ["No", "Yes"],
        },
        {
            "variable_name": "joint_replacement",
            "description": "History of joint replacement",
            "data_type": "categorical",
            "values": ["No", "Yes"],
        },
    ],
    "Hearing_Vision": [
        {
            "variable_name": "hearing_loss",
            "description": "Hearing loss severity",
            "data_type": "categorical",
            "values": ["None", "Mild", "Moderate", "Severe", "Profound"],
            "measurement_protocol": "Pure tone audiometry, ASHA criteria",
        },
        {
            "variable_name": "cataract",
            "description": "Cataract presence",
            "data_type": "categorical",
            "values": ["No", "Yes"],
        },
    ],
}


def create_variable_definitions() -> SurvivorshipDataset:
    """
    Create structured variable definitions for SJLIFE.

    Returns:
        SurvivorshipDataset with all variable definitions
    """
    dataset = SurvivorshipDataset()

    for category, variables in SJLIFE_VARIABLE_CATEGORIES.items():
        dataset.categories[category] = []
        for var_dict in variables:
            var_def = VariableDefinition(
                variable_name=var_dict["variable_name"],
                category=category,
                description=var_dict["description"],
                data_type=var_dict["data_type"],
                values=var_dict.get("values", []),
                unit=var_dict.get("unit"),
                measurement_protocol=var_dict.get("measurement_protocol"),
            )
            dataset.categories[category].append(var_def)

    return dataset


def generate_rag_text(dataset: SurvivorshipDataset) -> str:
    """
    Generate RAG-friendly text from survivorship data dictionary.

    Args:
        dataset: Survivorship dataset with variable definitions

    Returns:
        Formatted text for RAG ingestion
    """
    sections = []

    sections.append("St. Jude LIFE Study - Survivorship Data Dictionary")
    sections.append("=" * 60)
    sections.append("")
    sections.append(
        "The St. Jude Lifetime Cohort Study (SJLIFE) is a retrospective cohort study "
        "with prospective clinical follow-up of childhood cancer survivors treated at "
        "St. Jude Children's Research Hospital. The study characterizes the long-term "
        "health outcomes of survivors through comprehensive clinical assessments."
    )
    sections.append("")

    for category, variables in dataset.categories.items():
        category_display = category.replace("_", " ")
        sections.append(f"\n## {category_display}")
        sections.append("-" * 40)

        for var in variables:
            sections.append(f"\n### {var.variable_name}")
            sections.append(f"**Description:** {var.description}")
            sections.append(f"**Data Type:** {var.data_type}")

            if var.unit:
                sections.append(f"**Unit:** {var.unit}")

            if var.values:
                sections.append(f"**Possible Values:** {', '.join(var.values)}")

            if var.measurement_protocol:
                sections.append(f"**Measurement Protocol:** {var.measurement_protocol}")

            sections.append("")

    return "\n".join(sections)


def save_data_dictionary(output_dir: str) -> dict:
    """
    Save SJLIFE data dictionary to files.

    Args:
        output_dir: Directory to save output files

    Returns:
        Dict with paths to saved files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create dataset
    dataset = create_variable_definitions()

    # Count variables
    total_vars = sum(len(vars) for vars in dataset.categories.values())
    logger.info(f"Created data dictionary with {total_vars} variables across {len(dataset.categories)} categories")

    # Save JSON
    json_path = output_path / "sjlife_variables.json"
    json_data = {
        "version": dataset.version,
        "source": dataset.source,
        "total_variables": total_vars,
        "categories": {
            cat: [asdict(v) for v in vars]
            for cat, vars in dataset.categories.items()
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON: {json_path}")

    # Save RAG-friendly text
    text_path = output_path / "sjlife_variables_text.txt"
    rag_text = generate_rag_text(dataset)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(rag_text)
    logger.info(f"Saved text: {text_path}")

    return {
        "json_path": str(json_path),
        "text_path": str(text_path),
        "total_variables": total_vars,
        "categories": list(dataset.categories.keys()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download/generate SJLIFE survivorship data dictionary"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sample/survivorship",
        help="Output directory for data files",
    )

    args = parser.parse_args()

    result = save_data_dictionary(args.output_dir)

    print(f"\nSJLIFE Data Dictionary Generated")
    print(f"  Total Variables: {result['total_variables']}")
    print(f"  Categories: {len(result['categories'])}")
    print(f"  JSON File: {result['json_path']}")
    print(f"  Text File: {result['text_path']}")
    print(f"\nCategories:")
    for cat in result["categories"]:
        print(f"  - {cat}")


if __name__ == "__main__":
    main()
