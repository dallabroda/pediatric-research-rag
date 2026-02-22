# RAG Evaluation Framework

## Overview

This document describes the evaluation methodology for the Pediatric Research RAG system.

## Gold Standard Dataset

Located at `evaluation/gold_standard.json`, contains 10 test cases covering:

| Category | Count | Description |
|----------|-------|-------------|
| survival_statistics | 1 | Questions about survival rates |
| late_effects | 1 | Long-term effects of treatment |
| treatment_protocols | 4 | Treatment approaches and therapies |
| clinical_trials | 2 | Trial availability and eligibility |
| genetics | 1 | Genetic factors in pediatric cancer |
| supportive_care | 1 | Patient support services |

### Difficulty Distribution

| Difficulty | Count |
|------------|-------|
| Easy | 3 |
| Medium | 4 |
| Hard | 3 |

## Evaluation Metrics

### 1. Retrieval Precision

Measures whether retrieved chunks come from expected source types.

```
Retrieval Precision = (Tests with matching source types) / (Total tests)
```

Target: > 80%

### 2. Answer Relevance

Keyword-based relevance scoring.

```
Answer Relevance = (Expected keywords found in answer) / (Total expected keywords)
```

Target: > 70%

### 3. Confidence Calibration

Checks if confidence scores correlate with actual answer quality.

- High confidence should indicate correct answers
- Low confidence should indicate uncertain/missing information

### 4. Latency

Performance metrics:

| Metric | Target |
|--------|--------|
| P50 | < 3000ms |
| P95 | < 5000ms |
| Average | < 2500ms |

## Running the Evaluation

```bash
# Run full evaluation
python evaluation/run_evaluation.py

# With custom gold standard
python evaluation/run_evaluation.py -g path/to/custom.json

# Save results to specific file
python evaluation/run_evaluation.py -o results/run_001.json
```

## Results Interpretation

### Pass Criteria

A test case is considered PASSED if:
1. Answer relevance >= 50% (at least half of expected keywords found)
2. Confidence level is not "low"

### Output Files

Results are saved to `evaluation/results.json` with:
- Summary statistics
- Per-category breakdown
- Per-difficulty breakdown
- Individual test results with answer previews

## Failure Analysis

When tests fail, investigate:

1. **Low retrieval precision**: Check if documents cover the topic
2. **Low answer relevance**: Review prompt engineering
3. **High latency**: Check index size, consider caching
4. **Low confidence on correct answers**: Adjust confidence thresholds

## Continuous Improvement

1. Run evaluation after each significant change
2. Track metrics over time in `evaluation/history/`
3. Add new test cases as edge cases are discovered
4. Update expected keywords based on actual corpus content

## Sample Results

```
EVALUATION SUMMARY
============================================================

Overall: 7/10 passed (70.0%)

Latency:
  Average: 2100ms
  P50:     1850ms
  P95:     3500ms

Quality Metrics:
  Retrieval Precision: 85.0%
  Answer Relevance:    72.0%
  Avg Confidence:      0.721

By Category:
  survival_statistics: 1/1 (100%)
  late_effects: 1/1 (100%)
  treatment_protocols: 3/4 (75%)
  clinical_trials: 1/2 (50%)
  genetics: 0/1 (0%)
  supportive_care: 1/1 (100%)

By Difficulty:
  easy: 3/3 (100%)
  medium: 3/4 (75%)
  hard: 1/3 (33%)
```

## Known Limitations

1. Keyword matching is basic - doesn't handle synonyms
2. No semantic similarity scoring (would require embeddings)
3. No human evaluation component
4. Test set is small (10 cases)

## Future Enhancements

- [ ] Add semantic similarity using embeddings
- [ ] Implement human evaluation workflow
- [ ] Track regression over deployments
- [ ] Add more diverse test cases
- [ ] Compare against baseline (no RAG)
