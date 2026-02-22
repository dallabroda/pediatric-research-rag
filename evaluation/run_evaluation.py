#!/usr/bin/env python3
"""
Evaluation script for Pediatric Research RAG system.

Runs gold standard test cases against the RAG pipeline and computes metrics.
"""
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lambdas.query.handler import handler as query_handler


@dataclass
class TestResult:
    """Result of a single test case."""
    test_id: int
    question: str
    category: str
    difficulty: str
    success: bool
    answer: str
    sources: list[dict]
    confidence_level: str
    confidence_score: float
    latency_ms: int
    retrieval_hit: bool  # Did we get relevant source types?
    answer_relevance: float  # Keyword match ratio
    error: Optional[str] = None


@dataclass
class EvaluationResults:
    """Aggregated evaluation results."""
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    avg_latency_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    avg_confidence: float
    retrieval_precision: float
    answer_relevance: float
    results_by_category: dict = field(default_factory=dict)
    results_by_difficulty: dict = field(default_factory=dict)
    individual_results: list[TestResult] = field(default_factory=list)


def load_gold_standard(path: str = None) -> dict:
    """Load gold standard test cases."""
    if path is None:
        path = Path(__file__).parent / "gold_standard.json"

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_single_test(test_case: dict) -> TestResult:
    """Run a single test case through the RAG pipeline."""
    question = test_case["question"]
    expected_keywords = test_case.get("expected_answer_contains", [])
    expected_source_types = test_case.get("expected_sources_types", [])

    # Create mock event for Lambda handler
    event = {
        "body": json.dumps({
            "question": question,
            "include_confidence": True,
        })
    }

    # Run query
    start_time = time.perf_counter()
    try:
        response = query_handler(event, None)
        end_time = time.perf_counter()

        latency_ms = int((end_time - start_time) * 1000)

        if response["statusCode"] != 200:
            return TestResult(
                test_id=test_case["id"],
                question=question,
                category=test_case.get("category", "unknown"),
                difficulty=test_case.get("difficulty", "unknown"),
                success=False,
                answer="",
                sources=[],
                confidence_level="error",
                confidence_score=0.0,
                latency_ms=latency_ms,
                retrieval_hit=False,
                answer_relevance=0.0,
                error=f"HTTP {response['statusCode']}",
            )

        body = json.loads(response["body"])
        answer = body.get("answer", "")
        sources = body.get("sources", [])
        confidence = body.get("confidence", {})

        # Check retrieval precision - did we get expected source types?
        retrieved_types = set(s.get("doc_type", "") for s in sources)
        expected_types_set = set(expected_source_types)
        retrieval_hit = bool(retrieved_types & expected_types_set) if expected_types_set else True

        # Check answer relevance - keyword matching
        answer_lower = answer.lower()
        keywords_found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        answer_relevance = keywords_found / len(expected_keywords) if expected_keywords else 1.0

        # Consider test passed if we got some keywords and reasonable confidence
        success = answer_relevance >= 0.5 and confidence.get("level", "low") != "low"

        return TestResult(
            test_id=test_case["id"],
            question=question,
            category=test_case.get("category", "unknown"),
            difficulty=test_case.get("difficulty", "unknown"),
            success=success,
            answer=answer,
            sources=sources,
            confidence_level=confidence.get("level", "unknown"),
            confidence_score=confidence.get("score", 0.0),
            latency_ms=latency_ms,
            retrieval_hit=retrieval_hit,
            answer_relevance=answer_relevance,
        )

    except Exception as e:
        end_time = time.perf_counter()
        return TestResult(
            test_id=test_case["id"],
            question=question,
            category=test_case.get("category", "unknown"),
            difficulty=test_case.get("difficulty", "unknown"),
            success=False,
            answer="",
            sources=[],
            confidence_level="error",
            confidence_score=0.0,
            latency_ms=int((end_time - start_time) * 1000),
            retrieval_hit=False,
            answer_relevance=0.0,
            error=str(e),
        )


def calculate_percentile(values: list[float], percentile: float) -> float:
    """Calculate percentile of a list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile / 100)
    index = min(index, len(sorted_values) - 1)
    return sorted_values[index]


def run_evaluation(gold_standard_path: str = None) -> EvaluationResults:
    """Run full evaluation against gold standard."""
    gold_standard = load_gold_standard(gold_standard_path)
    test_cases = gold_standard.get("test_cases", [])

    print(f"Running evaluation with {len(test_cases)} test cases...")
    print("-" * 60)

    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Testing: {test_case['question'][:50]}...")
        result = run_single_test(test_case)
        results.append(result)

        status = "PASS" if result.success else "FAIL"
        print(f"         {status} (latency: {result.latency_ms}ms, confidence: {result.confidence_level})")

    print("-" * 60)

    # Calculate aggregates
    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed
    latencies = [r.latency_ms for r in results]

    # Group by category
    by_category = {}
    for r in results:
        if r.category not in by_category:
            by_category[r.category] = {"total": 0, "passed": 0}
        by_category[r.category]["total"] += 1
        if r.success:
            by_category[r.category]["passed"] += 1

    # Group by difficulty
    by_difficulty = {}
    for r in results:
        if r.difficulty not in by_difficulty:
            by_difficulty[r.difficulty] = {"total": 0, "passed": 0}
        by_difficulty[r.difficulty]["total"] += 1
        if r.success:
            by_difficulty[r.difficulty]["passed"] += 1

    return EvaluationResults(
        timestamp=datetime.utcnow().isoformat() + "Z",
        total_tests=len(results),
        passed=passed,
        failed=failed,
        avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
        latency_p50_ms=calculate_percentile(latencies, 50),
        latency_p95_ms=calculate_percentile(latencies, 95),
        avg_confidence=sum(r.confidence_score for r in results) / len(results) if results else 0,
        retrieval_precision=sum(1 for r in results if r.retrieval_hit) / len(results) if results else 0,
        answer_relevance=sum(r.answer_relevance for r in results) / len(results) if results else 0,
        results_by_category=by_category,
        results_by_difficulty=by_difficulty,
        individual_results=results,
    )


def save_results(results: EvaluationResults, output_path: str = None):
    """Save evaluation results to JSON."""
    if output_path is None:
        output_path = Path(__file__).parent / "results.json"

    # Convert dataclasses to dicts for JSON serialization
    results_dict = {
        "timestamp": results.timestamp,
        "summary": {
            "total_tests": results.total_tests,
            "passed": results.passed,
            "failed": results.failed,
            "pass_rate": results.passed / results.total_tests if results.total_tests else 0,
        },
        "metrics": {
            "avg_latency_ms": round(results.avg_latency_ms, 2),
            "latency_p50_ms": results.latency_p50_ms,
            "latency_p95_ms": results.latency_p95_ms,
            "avg_confidence": round(results.avg_confidence, 3),
            "retrieval_precision": round(results.retrieval_precision, 3),
            "answer_relevance": round(results.answer_relevance, 3),
        },
        "by_category": results.results_by_category,
        "by_difficulty": results.results_by_difficulty,
        "individual_results": [
            {
                "test_id": r.test_id,
                "question": r.question,
                "category": r.category,
                "difficulty": r.difficulty,
                "success": r.success,
                "confidence_level": r.confidence_level,
                "confidence_score": r.confidence_score,
                "latency_ms": r.latency_ms,
                "retrieval_hit": r.retrieval_hit,
                "answer_relevance": round(r.answer_relevance, 3),
                "answer_preview": r.answer[:200] + "..." if len(r.answer) > 200 else r.answer,
                "error": r.error,
            }
            for r in results.individual_results
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def print_summary(results: EvaluationResults):
    """Print evaluation summary to console."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    pass_rate = results.passed / results.total_tests * 100 if results.total_tests else 0
    print(f"\nOverall: {results.passed}/{results.total_tests} passed ({pass_rate:.1f}%)")

    print(f"\nLatency:")
    print(f"  Average: {results.avg_latency_ms:.0f}ms")
    print(f"  P50:     {results.latency_p50_ms:.0f}ms")
    print(f"  P95:     {results.latency_p95_ms:.0f}ms")

    print(f"\nQuality Metrics:")
    print(f"  Retrieval Precision: {results.retrieval_precision:.1%}")
    print(f"  Answer Relevance:    {results.answer_relevance:.1%}")
    print(f"  Avg Confidence:      {results.avg_confidence:.3f}")

    print(f"\nBy Category:")
    for cat, stats in results.results_by_category.items():
        rate = stats["passed"] / stats["total"] * 100 if stats["total"] else 0
        print(f"  {cat}: {stats['passed']}/{stats['total']} ({rate:.0f}%)")

    print(f"\nBy Difficulty:")
    for diff, stats in results.results_by_difficulty.items():
        rate = stats["passed"] / stats["total"] * 100 if stats["total"] else 0
        print(f"  {diff}: {stats['passed']}/{stats['total']} ({rate:.0f}%)")

    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--gold-standard",
        "-g",
        default=None,
        help="Path to gold standard JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to save results JSON",
    )

    args = parser.parse_args()

    results = run_evaluation(args.gold_standard)
    print_summary(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
