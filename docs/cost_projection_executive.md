# Pediatric Research RAG: Cost Analysis

## Executive Summary

This RAG system enables natural-language Q&A over pediatric cancer research at **under $1/month for demo workloads** and scales to **~$30-50/month at institutional scale**.

---

## Current Demo Costs

| Component | Usage | Cost |
|-----------|-------|------|
| Bedrock Titan Embeddings | ~150K tokens (initial index) | $0.02 |
| Bedrock Claude Haiku | ~50 queries (dev/testing) | $0.05 |
| Bedrock Claude Sonnet | ~50 queries (demo) | $0.60 |
| Lambda | ~100 invocations | $0.00 (free tier) |
| S3 | ~50 MB storage | $0.00 (free tier) |
| API Gateway | ~100 calls | $0.00 (free tier) |
| DynamoDB | ~100 queries logged | $0.00 (free tier) |
| **Total** | | **< $1** |

---

## Scale Projection: St. Jude Institutional Use

Assumptions:
- 10,000 documents indexed
- 1,000 queries/day
- 30 days/month

| Component | Usage | Monthly Cost |
|-----------|-------|--------------|
| Titan Embeddings (re-index) | ~10M tokens | $2 |
| Claude Haiku (queries) | ~30K queries | $15 |
| Lambda | ~50K invocations | $0.50 |
| S3 | ~2 GB | $0.05 |
| API Gateway | ~30K calls | $3.50 |
| DynamoDB | ~30K writes | $2 |
| CloudWatch | Metrics & logs | $2 |
| **Total** | | **~$25/month** |

**With Claude Sonnet for all queries:** ~$45/month

---

## Cost vs. Value

### Research Time Saved

| Metric | Manual Search | RAG System |
|--------|---------------|------------|
| Find relevant paper | 15-30 min | < 1 min |
| Extract key findings | 10-20 min | Instant |
| Cross-reference sources | 30+ min | 5 sec |

**Estimated time savings:** 2-4 hours/researcher/week

### Alternative Solutions

| Solution | Monthly Cost | Notes |
|----------|--------------|-------|
| **This RAG system** | $25-45 | Self-hosted, full control |
| Amazon Kendra | $810+ | Managed, limited customization |
| Pinecone + OpenAI | $50-100 | Vendor lock-in |
| Elasticsearch + OpenAI | $150+ | Complex setup |

---

## Recommendations

1. **Development/Testing:** Use Claude Haiku ($0.25/1K queries)
2. **Demo/Interviews:** Use Claude Sonnet ($3/1K queries)
3. **Production:** Start with Haiku, upgrade for complex queries

### Cost Controls

- **Budget Alert:** Set at $10/month (already configured)
- **Rate Limiting:** 5 requests/second (prevents abuse)
- **Index Caching:** Avoid re-embedding unchanged documents

---

## ROI Summary

| Investment | Return |
|------------|--------|
| $25-50/month | Research efficiency gains |
| 1 week setup | Instant literature search |
| AWS credits ($200) | ~4+ years of demo usage |

**For a research institution:** The system pays for itself if it saves one researcher 2 hours per month.
