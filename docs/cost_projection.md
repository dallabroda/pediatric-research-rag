# Cost Projection: Pediatric Research RAG

## Executive Summary

This document provides cost analysis for the Pediatric Research RAG system at two scales:
- **Demo Scale**: ~25 documents, suitable for portfolio demonstration
- **St. Jude Scale**: ~10,000 documents, realistic production deployment

**Key Finding**: The system is extremely cost-effective. Demo scale costs <$1 total, while production scale would cost ~$30-50/month with Haiku or ~$180-200/month with Sonnet.

---

## AWS Free Tier Benefits

| Service | Free Tier Allowance | Our Usage (Demo) | Coverage |
|---------|---------------------|------------------|----------|
| Lambda | 1M requests + 400K GB-sec/month | ~100 requests | Free |
| S3 | 5 GB storage (12 months) | ~50 MB | Free |
| API Gateway | 1M REST calls (12 months) | ~100 calls | Free |
| CloudWatch | 10 custom metrics, 5 GB logs | Minimal | Free |
| SQS | 1M requests/month | ~10 messages | Free |
| EventBridge | First 14M events/month | 4 events/month | Free |

**Conclusion**: Infrastructure is essentially free for the first 12 months with a new AWS account.

---

## Bedrock Pricing (Pay-per-Use)

### Embedding Model: Titan Embeddings V2

| Metric | Price |
|--------|-------|
| Input tokens | $0.00002/1K tokens |
| Output dimensions | 1024 |

### LLM Models

| Model | Input Price | Output Price |
|-------|-------------|--------------|
| Claude 3 Haiku | $0.00025/1K tokens | $0.00125/1K tokens |
| Claude 3 Sonnet | $0.003/1K tokens | $0.015/1K tokens |

**Haiku is 12x cheaper than Sonnet** - use Haiku for development, Sonnet for demos.

---

## Demo Scale Analysis (~25 Documents)

### One-Time Ingestion

| Operation | Tokens | Cost |
|-----------|--------|------|
| Parse 25 documents | N/A | $0.00 |
| Chunk (~150 chunks) | N/A | $0.00 |
| Embed 150 chunks (~150K tokens) | 150,000 | $0.003 |
| **Ingestion Total** | | **~$0.01** |

### Monthly Query Operations (Assuming 50 queries/month)

| Operation | Tokens per Query | Monthly Tokens | Haiku Cost | Sonnet Cost |
|-----------|------------------|----------------|------------|-------------|
| Query embedding | ~50 | 2,500 | $0.0001 | $0.0001 |
| Context retrieval | 2,500 (5 chunks) | 125,000 | $0.03 | $0.38 |
| LLM response | ~500 output | 25,000 | $0.03 | $0.38 |
| **Monthly Query Total** | | | **~$0.06** | **~$0.76** |

### Demo Scale Total Cost

| Phase | Haiku | Sonnet |
|-------|-------|--------|
| One-time ingestion | $0.01 | $0.01 |
| Monthly operations | $0.06 | $0.76 |
| **Total (Month 1)** | **~$0.07** | **~$0.77** |

---

## St. Jude Scale Analysis (~10,000 Documents)

### One-Time Ingestion

| Operation | Estimate | Cost |
|-----------|----------|------|
| Documents | 10,000 | - |
| Average chunks per doc | 20 | - |
| Total chunks | 200,000 | - |
| Tokens per chunk | 500 | - |
| Total embedding tokens | 100M | $2.00 |
| **Ingestion Total** | | **~$2.00** |

### Monthly Query Operations (Assuming 5,000 queries/month)

| Operation | Monthly Tokens | Haiku Cost | Sonnet Cost |
|-----------|----------------|------------|-------------|
| Query embeddings | 250,000 | $0.01 | $0.01 |
| Context retrieval (input) | 12.5M | $3.13 | $37.50 |
| LLM responses (output) | 2.5M | $3.13 | $37.50 |
| **Monthly Query Total** | | **~$6.27** | **~$75.01** |

### Weekly Refresh Operations

| Operation | Weekly Tokens | Monthly Cost |
|-----------|---------------|--------------|
| Check for new papers | ~1,000 | <$0.01 |
| Download & embed ~5 new docs | 50,000 | ~$0.01 |
| **Monthly Refresh Total** | | **~$0.04** |

### St. Jude Scale Total Monthly Cost

| Component | Haiku | Sonnet |
|-----------|-------|--------|
| Query operations | $6.27 | $75.01 |
| Weekly refresh | $0.04 | $0.04 |
| Lambda/S3/API Gateway | Free* | Free* |
| **Total/Month** | **~$6-7** | **~$75** |

*Free tier for 12 months; minimal cost (~$5/month) after free tier expires.

---

## Cost Optimization Strategies

### 1. Use Haiku for Development
- 12x cheaper than Sonnet
- Sufficient quality for testing and iteration
- Switch to Sonnet only for demos/production

### 2. Implement Response Caching
- Cache common queries in Lambda memory
- Estimated 30-40% reduction in LLM calls
- **Potential savings**: $2-3/month (Haiku) or $22-30/month (Sonnet)

### 3. Batch Embedding Operations
- Process documents in batches during off-peak hours
- Use EventBridge scheduling for weekly refreshes
- Avoid real-time embedding when possible

### 4. Optimize Chunk Size
- Current: 500 tokens with 50 token overlap
- Consider: 750 tokens for dense documents
- **Trade-off**: Larger chunks = fewer embeddings but less precise retrieval

### 5. Progressive Top-K
- Start with Top-3 retrieval
- Expand to Top-5 only if confidence is low
- **Potential savings**: 20-30% reduction in context tokens

---

## Break-Even Analysis

### When to Move Beyond Free Tier

| Metric | Free Tier Limit | Approx. Usage Level |
|--------|-----------------|---------------------|
| Lambda requests | 1M/month | ~33K queries/day |
| S3 storage | 5 GB | ~100,000 documents |
| API Gateway | 1M calls/month | ~33K queries/day |

**Conclusion**: Free tier covers most use cases. Only high-traffic production deployments exceed limits.

---

## Comparison: Build vs. Buy

### Option 1: This RAG System
- **Initial Setup**: <$5
- **Monthly (Haiku, 5K queries)**: ~$7
- **Monthly (Sonnet, 5K queries)**: ~$75
- **Full control over data and prompts**

### Option 2: OpenAI + Pinecone
- **Pinecone**: $70/month (Starter)
- **OpenAI GPT-4 (5K queries)**: ~$150/month
- **Total**: ~$220/month

### Option 3: Managed RAG Service
- **Amazon Kendra + Bedrock**: ~$400/month minimum
- **Includes managed index and retrieval**

**Savings with our approach**: 3-60x cheaper than alternatives while maintaining full customization.

---

## Recommendations for St. Jude Interview

1. **Lead with Haiku costs** - Emphasize the ~$30-50/month production cost
2. **Highlight free tier leverage** - Infrastructure essentially free
3. **Show scaling math** - Demonstrate understanding of cost at 10K document scale
4. **Mention optimization strategies** - Shows production thinking
5. **Compare to alternatives** - 5-10x cheaper than managed solutions

---

## Appendix: Token Estimation Formulas

```
Embedding tokens ≈ chunk_text_chars / 4
LLM input tokens ≈ (system_prompt + context_chunks + question) / 4
LLM output tokens ≈ response_chars / 4
```

Standard assumptions:
- System prompt: ~500 tokens
- Each context chunk: ~500 tokens
- User question: ~50 tokens
- LLM response: ~500 tokens
