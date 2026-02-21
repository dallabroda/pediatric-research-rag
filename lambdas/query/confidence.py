"""
Confidence scoring for RAG answers.

Computes confidence levels based on retrieval results and source analysis.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ConfidenceScore:
    """Confidence assessment for an answer."""
    level: str  # "high", "medium", "low"
    score: float  # 0.0 to 1.0
    reasoning: str
    factors: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result from similarity search (simplified for confidence scoring)."""
    chunk_id: str
    document_id: str
    chunk_text: str
    doc_title: str
    doc_type: str
    score: float
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    source_url: Optional[str] = None
    pub_date: Optional[str] = None


@dataclass
class RelatedFinding:
    """A related finding from cross-document analysis."""
    finding_type: str  # "complementary", "contradictory", "supporting"
    finding: str
    source: str
    document_id: str


# Confidence thresholds
HIGH_CONFIDENCE_MIN_SOURCES = 2
HIGH_CONFIDENCE_MIN_AVG_SCORE = 0.7
MEDIUM_CONFIDENCE_MIN_SOURCES = 1
MEDIUM_CONFIDENCE_MIN_AVG_SCORE = 0.5
LOW_CONFIDENCE_THRESHOLD = 0.3

# Recency bonus (papers within last 3 years get a boost)
RECENCY_BONUS = 0.1
RECENCY_YEARS = 3


def parse_pub_date(pub_date: Optional[str]) -> Optional[datetime]:
    """
    Parse publication date string to datetime.

    Args:
        pub_date: Date string in various formats (YYYY, YYYY-MM, YYYY-MM-DD)

    Returns:
        datetime object or None if parsing fails
    """
    if not pub_date:
        return None

    # Try different formats
    formats = ["%Y-%m-%d", "%Y-%m", "%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(pub_date[:len(fmt.replace("%", "").replace("-", "")) + pub_date.count("-")], fmt)
        except (ValueError, IndexError):
            continue

    # Try to extract year
    match = re.search(r"\d{4}", pub_date)
    if match:
        try:
            return datetime(int(match.group()), 1, 1)
        except ValueError:
            pass

    return None


def compute_recency_score(sources: list[RetrievalResult]) -> float:
    """
    Compute a recency score based on publication dates.

    Args:
        sources: List of retrieval results

    Returns:
        Score from 0.0 to 1.0 (higher = more recent sources)
    """
    now = datetime.now()
    recency_threshold = now.year - RECENCY_YEARS

    recent_count = 0
    dated_count = 0

    for source in sources:
        pub_date = parse_pub_date(source.pub_date)
        if pub_date:
            dated_count += 1
            if pub_date.year >= recency_threshold:
                recent_count += 1

    if dated_count == 0:
        return 0.5  # Neutral if no dates available

    return recent_count / dated_count


def compute_source_diversity_score(sources: list[RetrievalResult]) -> float:
    """
    Compute diversity score based on variety of source types.

    Args:
        sources: List of retrieval results

    Returns:
        Score from 0.0 to 1.0 (higher = more diverse)
    """
    if not sources:
        return 0.0

    doc_types = set(s.doc_type for s in sources)
    unique_docs = len(set(s.document_id for s in sources))

    # Diversity factors
    type_diversity = min(len(doc_types) / 2, 1.0)  # Max 2 types: paper, trial
    doc_diversity = min(unique_docs / 3, 1.0)  # Ideal: 3+ unique docs

    return (type_diversity + doc_diversity) / 2


def detect_contradictions(sources: list[RetrievalResult]) -> list[RelatedFinding]:
    """
    Detect potential contradictions between sources.

    This is a simple heuristic-based detection. For production,
    this would use NLI or more sophisticated analysis.

    Args:
        sources: List of retrieval results

    Returns:
        List of potential contradictions or conflicting findings
    """
    findings = []

    # Simple keyword-based contradiction detection
    contradiction_patterns = [
        (r"\bnot\b.*\beffective\b", r"\beffective\b"),
        (r"\bno\b.*\bsignificant\b", r"\bsignificant\b"),
        (r"\bnegative\b", r"\bpositive\b"),
        (r"\bfailed\b", r"\bsuccess\b"),
        (r"\bcontrary\b", r"\bconfirm\b"),
    ]

    # Group sources by document
    source_texts = {s.document_id: s.chunk_text.lower() for s in sources}

    for doc_id_1, text_1 in source_texts.items():
        for doc_id_2, text_2 in source_texts.items():
            if doc_id_1 >= doc_id_2:
                continue

            for pattern_1, pattern_2 in contradiction_patterns:
                if (re.search(pattern_1, text_1) and re.search(pattern_2, text_2)) or \
                   (re.search(pattern_2, text_1) and re.search(pattern_1, text_2)):
                    # Find the source titles
                    title_1 = next((s.doc_title for s in sources if s.document_id == doc_id_1), doc_id_1)
                    title_2 = next((s.doc_title for s in sources if s.document_id == doc_id_2), doc_id_2)

                    findings.append(RelatedFinding(
                        finding_type="contradictory",
                        finding=f"Potential disagreement detected between sources",
                        source=f"{title_1} vs {title_2}",
                        document_id=f"{doc_id_1},{doc_id_2}",
                    ))
                    break

    return findings


def find_complementary_findings(sources: list[RetrievalResult]) -> list[RelatedFinding]:
    """
    Identify complementary findings across sources.

    Args:
        sources: List of retrieval results

    Returns:
        List of complementary findings
    """
    findings = []

    # Group by document to find papers/trials covering same topics
    doc_types = {}
    for source in sources:
        if source.document_id not in doc_types:
            doc_types[source.document_id] = {
                "type": source.doc_type,
                "title": source.doc_title,
            }

    # Check for paper-trial complementary pairs
    papers = [d for d, info in doc_types.items() if info["type"] == "paper"]
    trials = [d for d, info in doc_types.items() if info["type"] == "trial"]

    if papers and trials:
        findings.append(RelatedFinding(
            finding_type="complementary",
            finding="Research papers and clinical trials provide complementary evidence",
            source=f"{len(papers)} paper(s), {len(trials)} trial(s)",
            document_id=",".join(papers[:2] + trials[:2]),
        ))

    return findings


def compute_answer_confidence(
    question: str,
    sources: list[RetrievalResult],
) -> ConfidenceScore:
    """
    Compute overall confidence score for an answer.

    Factors considered:
    - Number of supporting sources
    - Similarity scores from retrieval
    - Recency of sources
    - Source diversity (papers vs trials)
    - Explicit contradictions detected

    Args:
        question: The user's question
        sources: Retrieved source chunks

    Returns:
        ConfidenceScore with level, numeric score, and reasoning
    """
    if not sources:
        return ConfidenceScore(
            level="low",
            score=0.0,
            reasoning="No relevant sources found",
            factors={"source_count": 0},
        )

    # Count unique documents
    unique_docs = len(set(s.document_id for s in sources))

    # Compute average similarity score
    avg_score = sum(s.score for s in sources) / len(sources)
    max_score = max(s.score for s in sources)

    # Compute recency score
    recency_score = compute_recency_score(sources)

    # Compute diversity score
    diversity_score = compute_source_diversity_score(sources)

    # Detect contradictions
    contradictions = detect_contradictions(sources)
    contradiction_penalty = 0.15 * len(contradictions)

    # Compute weighted confidence
    # Base: average similarity score (40%)
    # Document count bonus (20%)
    # Max score (15%)
    # Recency (15%)
    # Diversity (10%)
    doc_count_factor = min(unique_docs / 3, 1.0)

    raw_score = (
        avg_score * 0.40 +
        doc_count_factor * 0.20 +
        max_score * 0.15 +
        recency_score * 0.15 +
        diversity_score * 0.10
    )

    # Apply contradiction penalty
    final_score = max(0.0, min(1.0, raw_score - contradiction_penalty))

    # Determine confidence level
    if final_score >= 0.7 and unique_docs >= HIGH_CONFIDENCE_MIN_SOURCES:
        level = "high"
    elif final_score >= 0.5 and unique_docs >= MEDIUM_CONFIDENCE_MIN_SOURCES:
        level = "medium"
    else:
        level = "low"

    # Build reasoning
    reasoning_parts = []
    if unique_docs >= 3:
        reasoning_parts.append(f"{unique_docs} sources agree")
    elif unique_docs == 2:
        reasoning_parts.append(f"2 sources support this")
    elif unique_docs == 1:
        reasoning_parts.append("Based on single source")
    else:
        reasoning_parts.append("No supporting sources")

    if avg_score >= 0.7:
        reasoning_parts.append("high relevance")
    elif avg_score >= 0.5:
        reasoning_parts.append("moderate relevance")
    else:
        reasoning_parts.append("marginal relevance")

    if recency_score >= 0.7:
        reasoning_parts.append("recent sources")
    elif recency_score <= 0.3:
        reasoning_parts.append("older sources")

    if contradictions:
        reasoning_parts.append(f"{len(contradictions)} potential contradiction(s)")

    return ConfidenceScore(
        level=level,
        score=final_score,
        reasoning="; ".join(reasoning_parts),
        factors={
            "source_count": unique_docs,
            "avg_similarity": round(avg_score, 3),
            "max_similarity": round(max_score, 3),
            "recency_score": round(recency_score, 3),
            "diversity_score": round(diversity_score, 3),
            "contradictions": len(contradictions),
        },
    )


def analyze_sources(sources: list[RetrievalResult]) -> dict:
    """
    Analyze sources for cross-document intelligence.

    Args:
        sources: Retrieved source chunks

    Returns:
        Analysis dict with related findings, contradictions, etc.
    """
    contradictions = detect_contradictions(sources)
    complementary = find_complementary_findings(sources)

    all_findings = contradictions + complementary

    return {
        "related_findings": [
            {
                "type": f.finding_type,
                "finding": f.finding,
                "source": f.source,
            }
            for f in all_findings
        ],
        "has_contradictions": len(contradictions) > 0,
        "has_complementary": len(complementary) > 0,
    }
