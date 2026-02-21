"""
Document quality validation for RAG ingestion.

Validates extracted content to ensure quality before embedding.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum requirements
MIN_DOCUMENT_CHARS = 500
MAX_WHITESPACE_RATIO = 0.5
MIN_WORD_LENGTH = 2.0
MAX_WORD_LENGTH = 15.0

# Required fields for different document types
REQUIRED_PAPER_SECTIONS = {"abstract"}  # At least abstract should be present
REQUIRED_TRIAL_FIELDS = {"nct_id", "title", "status"}

# OCR artifact patterns
OCR_ARTIFACT_PATTERNS = [
    (r"ﬁ", "fi ligature"),
    (r"ﬂ", "fl ligature"),
    (r"ﬀ", "ff ligature"),
    (r"ﬃ", "ffi ligature"),
    (r"ﬄ", "ffl ligature"),
    (r"[^\x00-\x7F]{5,}", "long non-ASCII sequence"),
    (r"(?<![A-Za-z])[Il1|]{4,}(?![A-Za-z])", "confused vertical chars"),
    (r"[0oO]{5,}", "confused zero/O chars"),
    (r"\b[A-Z]{15,}\b", "very long all-caps word"),
]


@dataclass
class ValidationError:
    """A single validation error."""
    code: str
    message: str
    severity: str  # "error", "warning", "info"
    location: Optional[str] = None


@dataclass
class QualityReport:
    """Quality report for a validated document."""
    document_id: str
    is_valid: bool
    quality_score: float  # 0.0 to 1.0
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def all_issues(self) -> list[ValidationError]:
        """Get all errors and warnings."""
        return self.errors + self.warnings


class DocumentValidator:
    """
    Validates document extraction quality.

    Checks for:
    - Minimum content length
    - Whitespace ratio
    - OCR artifacts
    - Section completeness (for papers)
    - Required fields (for trials)
    """

    def __init__(
        self,
        min_chars: int = MIN_DOCUMENT_CHARS,
        max_whitespace_ratio: float = MAX_WHITESPACE_RATIO,
    ):
        """
        Initialize validator with thresholds.

        Args:
            min_chars: Minimum character count for valid document
            max_whitespace_ratio: Maximum allowed whitespace ratio
        """
        self.min_chars = min_chars
        self.max_whitespace_ratio = max_whitespace_ratio

    def validate_extraction(
        self,
        text: str,
        document_id: str,
        doc_type: str,
        metadata: Optional[dict] = None,
    ) -> QualityReport:
        """
        Validate extracted document content.

        Args:
            text: Extracted document text
            document_id: Document identifier
            doc_type: Document type ("paper" or "trial")
            metadata: Optional metadata dict for additional validation

        Returns:
            QualityReport with validation results
        """
        errors = []
        warnings = []
        metrics = {}

        # Check minimum content
        if not self.check_minimum_content(text):
            errors.append(ValidationError(
                code="MIN_CONTENT",
                message=f"Document has less than {self.min_chars} characters ({len(text)} chars)",
                severity="error",
            ))

        # Compute text metrics
        char_count = len(text)
        words = text.split()
        word_count = len(words)
        whitespace_count = sum(1 for c in text if c.isspace())
        whitespace_ratio = whitespace_count / char_count if char_count > 0 else 0.0
        avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0

        metrics.update({
            "char_count": char_count,
            "word_count": word_count,
            "whitespace_ratio": whitespace_ratio,
            "avg_word_length": avg_word_length,
        })

        # Check whitespace ratio
        if whitespace_ratio > self.max_whitespace_ratio:
            warnings.append(ValidationError(
                code="HIGH_WHITESPACE",
                message=f"High whitespace ratio: {whitespace_ratio:.2f} (max: {self.max_whitespace_ratio})",
                severity="warning",
            ))

        # Check for OCR artifacts
        artifacts = self.detect_ocr_artifacts(text)
        if artifacts:
            metrics["ocr_artifacts"] = artifacts
            warnings.append(ValidationError(
                code="OCR_ARTIFACTS",
                message=f"Detected {len(artifacts)} OCR artifact(s): {', '.join(artifacts[:3])}",
                severity="warning",
            ))

        # Type-specific validation
        if doc_type == "paper":
            paper_errors, paper_warnings = self._validate_paper(text, metadata)
            errors.extend(paper_errors)
            warnings.extend(paper_warnings)
        elif doc_type == "trial":
            trial_errors, trial_warnings = self._validate_trial(metadata or {})
            errors.extend(trial_errors)
            warnings.extend(trial_warnings)

        # Compute quality score
        quality_score = self.compute_quality_score(
            text=text,
            errors=errors,
            warnings=warnings,
        )
        metrics["quality_score"] = quality_score

        return QualityReport(
            document_id=document_id,
            is_valid=len(errors) == 0,
            quality_score=quality_score,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )

    def check_minimum_content(self, text: str, min_chars: Optional[int] = None) -> bool:
        """
        Check if text meets minimum content requirements.

        Args:
            text: Text to check
            min_chars: Override minimum character count

        Returns:
            True if text meets minimum requirements
        """
        min_threshold = min_chars if min_chars is not None else self.min_chars
        return len(text.strip()) >= min_threshold

    def detect_ocr_artifacts(self, text: str) -> list[str]:
        """
        Detect OCR artifacts in text.

        Args:
            text: Text to check

        Returns:
            List of detected artifact types
        """
        detected = []
        for pattern, artifact_type in OCR_ARTIFACT_PATTERNS:
            if re.search(pattern, text):
                detected.append(artifact_type)
        return detected

    def compute_quality_score(
        self,
        text: str,
        errors: Optional[list[ValidationError]] = None,
        warnings: Optional[list[ValidationError]] = None,
    ) -> float:
        """
        Compute overall quality score for extracted text.

        Args:
            text: Extracted text
            errors: Validation errors (if already computed)
            warnings: Validation warnings (if already computed)

        Returns:
            Quality score from 0.0 to 1.0
        """
        if not text.strip():
            return 0.0

        score = 1.0

        # Penalize for errors
        if errors:
            score -= 0.3 * len(errors)

        # Penalize for warnings
        if warnings:
            score -= 0.1 * len(warnings)

        # Check text metrics
        char_count = len(text)
        words = text.split()
        word_count = len(words)

        # Penalize very short documents
        if char_count < self.min_chars:
            score -= 0.3

        # Penalize extreme average word length
        if word_count > 0:
            avg_word_length = sum(len(w) for w in words) / word_count
            if avg_word_length < MIN_WORD_LENGTH or avg_word_length > MAX_WORD_LENGTH:
                score -= 0.1

        # Penalize high whitespace ratio
        whitespace_count = sum(1 for c in text if c.isspace())
        whitespace_ratio = whitespace_count / char_count if char_count > 0 else 0.0
        if whitespace_ratio > self.max_whitespace_ratio:
            score -= 0.15

        # Penalize OCR artifacts
        artifacts = self.detect_ocr_artifacts(text)
        if artifacts:
            score -= 0.05 * min(len(artifacts), 4)

        return max(0.0, min(1.0, score))

    def validate_trial_schema(self, trial: dict) -> list[str]:
        """
        Validate clinical trial data against required schema.

        Args:
            trial: Trial data dictionary

        Returns:
            List of validation error messages
        """
        errors = []

        for field in REQUIRED_TRIAL_FIELDS:
            if not trial.get(field):
                errors.append(f"Missing required field: {field}")

        # Validate NCT ID format
        nct_id = trial.get("nct_id", "")
        if nct_id and not re.match(r"^NCT\d{8}$", nct_id):
            errors.append(f"Invalid NCT ID format: {nct_id}")

        # Validate status
        valid_statuses = {
            "RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED",
            "TERMINATED", "SUSPENDED", "WITHDRAWN", "NOT_YET_RECRUITING",
            "ENROLLING_BY_INVITATION", "UNKNOWN",
        }
        status = trial.get("status", "").upper()
        if status and status not in valid_statuses:
            errors.append(f"Invalid trial status: {status}")

        return errors

    def _validate_paper(
        self,
        text: str,
        metadata: Optional[dict],
    ) -> tuple[list[ValidationError], list[ValidationError]]:
        """Validate paper-specific requirements."""
        errors = []
        warnings = []

        # Check for abstract
        text_lower = text.lower()
        has_abstract = any(marker in text_lower for marker in ["abstract", "summary"])

        if not has_abstract:
            warnings.append(ValidationError(
                code="NO_ABSTRACT",
                message="No abstract section detected in paper",
                severity="warning",
            ))

        # Check metadata if available
        if metadata:
            if not metadata.get("title"):
                warnings.append(ValidationError(
                    code="NO_TITLE",
                    message="Paper metadata missing title",
                    severity="warning",
                ))

        return errors, warnings

    def _validate_trial(
        self,
        metadata: dict,
    ) -> tuple[list[ValidationError], list[ValidationError]]:
        """Validate trial-specific requirements."""
        errors = []
        warnings = []

        schema_errors = self.validate_trial_schema(metadata)
        for error_msg in schema_errors:
            if "required" in error_msg.lower():
                errors.append(ValidationError(
                    code="MISSING_FIELD",
                    message=error_msg,
                    severity="error",
                ))
            else:
                warnings.append(ValidationError(
                    code="INVALID_FIELD",
                    message=error_msg,
                    severity="warning",
                ))

        return errors, warnings


def validate_document(
    text: str,
    document_id: str,
    doc_type: str,
    metadata: Optional[dict] = None,
) -> QualityReport:
    """
    Convenience function to validate a document.

    Args:
        text: Document text
        document_id: Document identifier
        doc_type: Document type
        metadata: Optional metadata

    Returns:
        QualityReport with validation results
    """
    validator = DocumentValidator()
    return validator.validate_extraction(text, document_id, doc_type, metadata)
