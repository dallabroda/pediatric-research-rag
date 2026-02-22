# Pediatric Research RAG

A production-grade RAG (Retrieval-Augmented Generation) system built on AWS that enables natural-language Q&A over pediatric cancer research from St. Jude Children's Research Hospital.

---

## The Mission

St. Jude Children's Research Hospital has achieved something remarkable: an **80% survival rate** for childhood cancers, up from just 20% when the hospital opened in 1962. This transformation didn't happen by chance—it came from decades of rigorous research, clinical trials, and a commitment to sharing findings freely with the world.

But there's a challenge: the research corpus is vast. Thousands of papers, hundreds of clinical trials, decades of survivorship data. Finding relevant information across these sources requires domain expertise and significant time.

---

## The Problem

Researchers and clinicians face a daily challenge:

- **Fragmented information**: Research papers in PMC, trial data in ClinicalTrials.gov, survivorship data in separate portals
- **Volume**: Hundreds of new pediatric oncology papers published monthly
- **Time pressure**: Clinical decisions can't wait for exhaustive literature reviews
- **Citation requirements**: Medical decisions require traceable, verifiable sources

Traditional search returns documents. What's needed is *answers*—synthesized, cited, and confidence-weighted.

---

## The Solution

This RAG system transforms how researchers interact with pediatric cancer literature:

```
Upload research papers and clinical trials
         │
         ▼
Parse → Validate → Chunk with lineage tracking
         │
         ▼
Embed with Titan → Build FAISS index
         │
         ▼
Ask questions → Retrieve relevant chunks → Generate cited answers
         │
         ├── REST API for programmatic access
         ├── MCP Server for Claude Desktop integration
         └── Streamlit Chat for interactive exploration
```

**Key capabilities:**
- Cross-document synthesis with contradiction detection
- Confidence scoring based on source agreement and recency
- Full data lineage from source document to chunk to answer
- Quality validation to catch OCR artifacts and extraction errors

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Sources                                  │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐    │
│  │ PMC Papers   │  │ ClinicalTrials   │  │ SJLIFE Survivorship│    │
│  │ (Open Access)│  │ (API v2)         │  │ (Data Dictionary)  │    │
│  └──────┬───────┘  └────────┬─────────┘  └─────────┬──────────┘    │
└─────────┼──────────────────┼───────────────────────┼────────────────┘
          │                   │                       │
          ▼                   ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    S3: Raw Documents                                 │
│                  raw/papers/  raw/trials/  raw/survivorship/        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ S3 Event
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Lambda: Ingest                                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │ PDFParser  │  │TrialParser │  │Survivorship│  │  Validator   │  │
│  │            │  │            │  │   Parser   │  │ (Quality QA) │  │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └──────────────┘  │
│        └───────────────┼───────────────┘                            │
│                        ▼                                             │
│              Chunker (with lineage metadata)                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Lambda: Embed                                     │
│  ┌────────────────────────┐     ┌─────────────────────────────────┐│
│  │ Bedrock Titan V2       │────▶│ FAISS Index + Metadata (S3)     ││
│  │ (1024-dim embeddings)  │     │ CloudWatch Metrics              ││
│  └────────────────────────┘     └─────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  REST API    │    │   MCP Server     │    │  Streamlit UI    │
│ (/query,     │    │ (search_research,│    │  (Chat interface │
│  /documents, │    │  ask_question,   │    │   with citations)│
│  /lineage)   │    │  list_trials)    │    │                  │
└──────────────┘    └──────────────────┘    └──────────────────┘
```

---

## Demo: Ask a Real Question

```bash
curl -X POST https://your-api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the long-term cardiac effects of anthracycline treatment in childhood leukemia survivors?"}'
```

**Response:**
```json
{
  "question": "What are the long-term cardiac effects of anthracycline treatment...",
  "answer": "Based on the SJLIFE cohort studies, childhood leukemia survivors treated with anthracyclines show significant long-term cardiac effects...",
  "confidence": {
    "level": "high",
    "score": 0.847,
    "reasoning": "3 sources agree; high relevance; recent sources"
  },
  "sources": [
    {
      "doc_title": "Cardiomyopathy in Childhood Cancer Survivors",
      "doc_type": "paper",
      "score": 0.89
    }
  ],
  "related_findings": [
    {
      "type": "complementary",
      "finding": "Research papers and clinical trials provide complementary evidence"
    }
  ]
}
```

---

## Why MCP Matters

The Model Context Protocol transforms this from a single-purpose API into a **universal research interface**:

```
┌────────────────────────────────────────────────────────────────┐
│                    One RAG Pipeline                             │
│                           │                                     │
│    ┌──────────────────────┼──────────────────────┐             │
│    ▼                      ▼                      ▼             │
│ Claude Desktop     VS Code Copilot      Custom Agents          │
│ (MCP stdio)        (MCP extension)      (MCP HTTP)             │
│                                                                 │
│ "Search St. Jude   "Find trials for    Automated literature    │
│  papers on CAR-T"   neuroblastoma"      review pipelines       │
└────────────────────────────────────────────────────────────────┘
```

**MCP Tools:**
- `search_research(query)` - Semantic search with relevance scores
- `ask_research_question(question)` - Full RAG with citations
- `list_clinical_trials(cancer_type)` - Filtered trial data
- `get_document(doc_id)` - Full document with lineage

Build once, deploy everywhere.

---

## Cost at Scale

| Component | Demo (~25 docs) | Production (~10K docs) |
|-----------|-----------------|------------------------|
| Titan Embeddings (ingest) | $0.01 | ~$2.00 |
| Titan Embeddings (queries) | $0.001/month | ~$0.30/month |
| Claude Haiku (queries) | $0.06/month | ~$6/month |
| Claude Sonnet (queries) | $0.76/month | ~$75/month |
| Lambda + S3 + API GW | Free tier | Free tier* |
| **Total (Haiku)** | **<$1** | **~$10/month** |
| **Total (Sonnet)** | **<$2** | **~$80/month** |

*Free tier covers most infrastructure for 12 months

**Compared to alternatives:**
- Pinecone + OpenAI: ~$220/month
- Amazon Kendra + Bedrock: ~$400/month minimum

See [docs/cost_projection.md](docs/cost_projection.md) for detailed analysis.

---

## Quick Start

### Prerequisites
- AWS Account with Bedrock access (Titan Embeddings + Claude)
- Python 3.12+
- AWS CLI configured (`aws configure`)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download sample data
```bash
python scripts/download_papers.py --count 15
python scripts/download_trials.py --sponsor "St. Jude" --count 10
python scripts/download_survivorship.py
```

### 3. Build index and run locally
```bash
# Ingest + embed sample data
python scripts/seed_index.py

# Start Streamlit chat
streamlit run streamlit_app/app.py
```

### 4. Deploy to AWS
```bash
cd deploy
./setup.sh          # Create S3, Lambda, API Gateway, EventBridge, DLQ
./deploy_lambdas.sh # Package and deploy Lambda functions
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Ask a question, get a cited answer with confidence |
| GET | `/documents` | List all ingested documents |
| GET | `/documents/{id}` | Get document details + chunks |
| GET | `/health` | System health check |
| GET | `/lineage/{chunk_id}` | Get full provenance for a chunk |
| GET | `/lineage?document_id={id}` | Get lineage for all chunks in a document |

---

## Production Features

### Data Quality & Lineage
- SHA256 document hashing for content tracking
- Parser versioning (PDFParser:v1.0.0, etc.)
- OCR artifact detection
- Quality scoring (0.0-1.0) per chunk
- Full provenance chain from source URL to embedded vector

### Cross-Document Intelligence
- Synthesis prompts for multi-source analysis
- Contradiction detection between sources
- Complementary finding identification
- Confidence scoring based on source agreement and recency

### Infrastructure
- EventBridge scheduled refresh (weekly by default)
- SQS Dead Letter Queue for failed ingestions
- CloudWatch metrics and dashboard (query latency, confidence, errors)
- DynamoDB query logging for analytics
- API rate limiting (5 req/sec)
- $10/month budget alarm
- Exponential backoff retry logic for all API calls
- Document deduplication via SHA256 hashing

---

## Future Directions

Aligned with St. Jude's strategic priorities:

1. **Expand data sources**: Integrate additional open-access research from COG (Children's Oncology Group)
2. **Temporal analysis**: Track evolution of treatment protocols over time
3. **Multi-modal support**: Add support for figures and tables extraction
4. **Federated learning**: Enable collaboration with other research institutions while keeping data local
5. **Real-time updates**: Move from weekly to event-driven refresh

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Compute | AWS Lambda (Python 3.12) |
| Storage | S3 (raw docs + FAISS index) |
| Embeddings | AWS Bedrock - Titan Embeddings V2 (1024 dim) |
| LLM | AWS Bedrock - Claude Haiku/Sonnet |
| Vector Store | FAISS (IndexFlatIP, cosine similarity) |
| API | API Gateway + Lambda |
| MCP Server | FastMCP (Python) |
| Chat UI | Streamlit |
| Scheduling | EventBridge |
| Monitoring | CloudWatch Metrics |
| Deploy | AWS CLI scripts (no Terraform/CDK) |

---

## Project Structure

```
pediatric-research-rag/
├── lambdas/
│   ├── ingest/           # Parse, validate, chunk
│   │   ├── handler.py
│   │   ├── parsers.py    # PDF, Trial, Survivorship parsers
│   │   ├── chunker.py    # Recursive char split with lineage
│   │   └── validators.py # Quality validation
│   ├── embed/            # Bedrock Titan → FAISS
│   ├── query/            # RAG pipeline with confidence
│   │   ├── handler.py
│   │   ├── retriever.py
│   │   ├── prompts.py    # Standard + synthesis prompts
│   │   └── confidence.py # Answer confidence scoring
│   ├── lineage/          # Provenance API
│   ├── refresh/          # Scheduled data refresh
│   └── shared/           # Retry logic, utilities
├── mcp_server/           # FastMCP implementation
├── streamlit_app/        # Chat UI
├── scripts/              # Download + seed scripts
├── deploy/               # AWS CLI deployment
├── docs/                 # Cost projection, architecture
└── tests/                # pytest + moto
```

---

## Documentation

- **[Architecture Diagram](docs/architecture.md)** - System architecture with Mermaid diagrams
- **[Cost Projection](docs/cost_projection_executive.md)** - Executive summary of costs
- **[Detailed Cost Analysis](docs/cost_projection.md)** - Full cost breakdown
- **[Evaluation Framework](docs/evaluation.md)** - RAG quality testing methodology

---

## License

MIT

---

*Built as a portfolio project demonstrating AWS data engineering, RAG architecture, and production ML system design.*
