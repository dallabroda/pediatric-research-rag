# 🏥 St. Jude Clinical Research RAG Assistant — Complete Project Plan

## 1. Project Vision

**What it is:** A serverless AWS pipeline that ingests publicly available pediatric cancer research documents (PDFs, CSVs), chunks and embeds them, stores vectors for semantic search, and exposes both a REST API and an MCP server — so any LLM client (Streamlit chat, Claude Desktop, etc.) can ask natural-language questions and receive grounded, cited answers from St. Jude's own published research.

**Why it matters for the interview:** Every required skill from the job posting maps directly to a component of this project. The domain (pediatric cancer research) shows you understand St. Jude's mission, not just the tech stack.

---

## 2. Data Sources — What We'll Actually Ingest

### ⚠️ Critical Reality Check

St. Jude Cloud's genomic data (BAM files, WGS/WES) requires a **Data Access Agreement** and is restricted to academic researchers. **We cannot use it for a demo project.** Here's what IS freely available:

### ✅ Freely Available Sources (No Approval Needed)

| Source | Type | Access | URL |
|--------|------|--------|-----|
| **St. Jude Published Research Papers** | PDFs (open-access via PubMed Central) | Free download | PMC articles from St. Jude authors |
| **St. Jude Survivorship Portal** — open-tier data | Summary statistics, variable dictionaries | Free, no login | https://survivorship.stjude.cloud |
| **ClinicalTrials.gov** — St. Jude trials | Structured XML/JSON | Free API | https://clinicaltrials.gov/api |
| **PeCan Knowledgebase** | Pediatric cancer variant data (public tier) | Free | https://pecan.stjude.cloud |
| **St. Jude Annual Reports / Fact Sheets** | PDFs | Free download | https://www.stjude.org |
| **NCI Childhood Cancer Data Catalog** | Metadata, dataset descriptions | Free | https://datacatalog.ccdi.cancer.gov |

### 📦 Recommended Starter Corpus (~20-30 documents)

For the demo, curate a focused set:

1. **10-15 open-access research papers** from PMC (search: `"St. Jude Children's Research Hospital" pediatric cancer` on PubMed, filter by Free Full Text)
   - The PCGP landmark paper (Nature Genetics, 2012)
   - St. Jude Cloud ecosystem paper (Cancer Discovery, 2021)
   - Survivorship Portal paper (Cancer Discovery, 2024)
   - Key papers on ALL, medulloblastoma, neuroblastoma
2. **5-8 ClinicalTrials.gov entries** for active St. Jude trials (JSON from their API)
3. **2-3 St. Jude fact sheets / annual report pages** (public PDFs)
4. **Survivorship Portal data dictionary** (publicly available metadata describing 1,600+ variables)

### Why This Corpus Works

- **No licensing/access issues** — everything is open-access
- **Rich enough for meaningful Q&A** — covers multiple cancer types, treatments, outcomes
- **Shows domain understanding** — you've curated actual St. Jude research, not random medical papers
- **Diverse formats** — PDFs, JSON, structured data → shows pipeline flexibility

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                        │
│                                                                     │
│  S3 Bucket (raw/)          S3 Bucket (processed/)                  │
│  ├── papers/*.pdf    ──►   ├── chunks/*.json                       │
│  ├── trials/*.json         └── embeddings/*.json                   │
│  └── reports/*.pdf                                                  │
│           │                         │                               │
│           ▼                         ▼                               │
│  Lambda: ingest_trigger     Lambda: embed_chunks                   │
│  (S3 event → parse →        (chunk text → call                     │
│   chunk → store)             Bedrock/OpenAI →                      │
│                              store vectors)                         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         STORAGE LAYER                                │
│                                                                     │
│  PostgreSQL (RDS) + pgvector          OR     FAISS on S3            │
│  ┌──────────────────────────┐                (simpler, cheaper)     │
│  │ documents                │                                       │
│  │  - id, title, source_url │                                       │
│  │  - doc_type, authors     │                                       │
│  │  - ingested_at           │                                       │
│  ├──────────────────────────┤                                       │
│  │ chunks                   │                                       │
│  │  - id, document_id       │                                       │
│  │  - chunk_text, chunk_idx │                                       │
│  │  - embedding vector(1536)│                                       │
│  │  - metadata (page, sec)  │                                       │
│  └──────────────────────────┘                                       │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       RETRIEVAL & API LAYER                         │
│                                                                     │
│  API Gateway + Lambda: /query                                       │
│  1. Receive question                                                │
│  2. Embed question                                                  │
│  3. Vector similarity search (top-K chunks)                        │
│  4. Build prompt with context                                       │
│  5. Call LLM (Bedrock Claude / OpenAI)                             │
│  6. Return answer + source citations                               │
│                                                                     │
│  API Gateway + Lambda: /documents                                   │
│  - GET /documents → list ingested docs                             │
│  - GET /documents/{id}/chunks → get chunks for a doc               │
│                                                                     │
│  MCP Server (Python, FastMCP)                                       │
│  - Tool: search_research(query) → semantic search                  │
│  - Tool: get_document(doc_id) → full doc context                   │
│  - Tool: list_trials(cancer_type) → filter clinical trials         │
│  - Resource: research://documents → browsable doc list             │
│  - Prompt: research_qa → structured Q&A template                   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       PRESENTATION LAYER                            │
│                                                                     │
│  Streamlit Chat App                                                 │
│  - Chat interface for asking questions                             │
│  - Shows source documents with links                               │
│  - Document explorer sidebar                                        │
│  - Ingestion status dashboard                                       │
│                                                                     │
│  Claude Desktop (via MCP)                                           │
│  - Connect to MCP server                                           │
│  - Use tools natively in conversation                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Breakdown — What to Build

### 4.1 S3 Bucket Structure

```
stjude-research-rag/
├── raw/
│   ├── papers/          # Original PDFs from PMC
│   ├── trials/          # ClinicalTrials.gov JSON files
│   └── reports/         # St. Jude fact sheets, annual reports
├── processed/
│   ├── chunks/          # Parsed and chunked text (JSON)
│   └── index/           # FAISS index files (if using FAISS)
└── config/
    └── prompt_templates/ # System prompts for RAG
```

### 4.2 Lambda Functions

| Function | Trigger | Purpose | Runtime |
|----------|---------|---------|---------|
| `ingest_trigger` | S3 PutObject (raw/) | Parse PDF/JSON → chunk → write to processed/ | Python 3.12 |
| `embed_chunks` | S3 PutObject (processed/chunks/) | Generate embeddings → store in pgvector or FAISS | Python 3.12 |
| `query_handler` | API Gateway POST /query | Embed question → retrieve → LLM → respond | Python 3.12 |
| `documents_handler` | API Gateway GET /documents | List/filter ingested documents | Python 3.12 |

### 4.3 Key Python Dependencies

```
# Lambda Layer
boto3           # AWS SDK
pypdf2          # PDF text extraction
tiktoken        # Token counting for chunking
openai          # Embeddings (or use boto3 for Bedrock)
psycopg2-binary # PostgreSQL (if using pgvector)
faiss-cpu       # Vector search (if using FAISS)
fastmcp         # MCP server framework
```

### 4.4 MCP Server — The Differentiator

This is what will set you apart from other candidates:

```python
# mcp_server.py (using FastMCP)
from fastmcp import FastMCP

mcp = FastMCP("St. Jude Research Assistant")

@mcp.tool()
def search_research(query: str, top_k: int = 5) -> list[dict]:
    """Search St. Jude pediatric cancer research papers by semantic similarity."""
    # Embed query → search pgvector/FAISS → return relevant chunks with metadata
    ...

@mcp.tool()
def get_document_context(document_id: str) -> dict:
    """Get full context for a specific research document including all chunks."""
    ...

@mcp.tool()
def list_clinical_trials(cancer_type: str = None, status: str = "active") -> list[dict]:
    """List St. Jude clinical trials, optionally filtered by cancer type."""
    ...

@mcp.tool()
def ask_research_question(question: str) -> dict:
    """Ask a question and get an answer grounded in St. Jude research with citations."""
    # Full RAG pipeline: embed → retrieve → LLM → cited answer
    ...

@mcp.resource("research://documents")
def list_documents() -> str:
    """Browse all ingested research documents."""
    ...

@mcp.prompt()
def research_qa_prompt(question: str) -> str:
    """Structured prompt template for research Q&A with citation requirements."""
    return f"""You are a research assistant specializing in St. Jude Children's Research Hospital 
    pediatric cancer studies. Answer the following question using ONLY the provided context. 
    Cite your sources using [Source: document_title] format.
    
    Question: {question}
    """
```

**Why this impresses:** It shows you understand MCP primitives (Tools, Resources, Prompts), not just the concept. It also makes the project usable from Claude Desktop or any MCP client, not just a custom UI.

---

## 5. Database Schema (PostgreSQL + pgvector)

```sql
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents metadata
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    source_url TEXT,
    doc_type VARCHAR(50) NOT NULL, -- 'paper', 'trial', 'report'
    authors TEXT[],
    publication_date DATE,
    cancer_types TEXT[],           -- e.g., ['ALL', 'medulloblastoma']
    s3_key TEXT NOT NULL,
    ingested_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB                 -- flexible additional fields
);

-- Document chunks with embeddings
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(1536),       -- OpenAI ada-002 dimension
    page_number INTEGER,
    section_title TEXT,
    token_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for vector similarity search
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Index for filtering
CREATE INDEX idx_documents_type ON documents(doc_type);
CREATE INDEX idx_documents_cancer ON documents USING GIN(cancer_types);
CREATE INDEX idx_chunks_document ON chunks(document_id);

-- Query log for analytics (nice-to-have)
CREATE TABLE query_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question TEXT NOT NULL,
    answer TEXT,
    source_chunks UUID[],
    latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Simpler alternative (FAISS):** If you want to avoid RDS costs for the demo, use FAISS with a JSON metadata sidecar on S3. Less impressive for the interview but faster to set up.

---

## 6. Infrastructure as Code (Terraform)

Key resources to provision:

```
├── terraform/
│   ├── main.tf              # Provider, backend
│   ├── s3.tf                # Buckets + event notifications
│   ├── lambda.tf            # 4 Lambda functions + layers
│   ├── api_gateway.tf       # REST API endpoints
│   ├── rds.tf               # PostgreSQL + pgvector (optional)
│   ├── iam.tf               # Roles and policies
│   ├── variables.tf         # Configurable params
│   └── outputs.tf           # API URL, bucket names
```

**Cost estimate for demo:** ~$5-15/month (Lambda free tier + small RDS or just FAISS on S3)

---

## 7. Folder Structure

```
stjude-research-rag/
├── README.md                 # Project overview, setup, architecture diagram
├── architecture.png          # Visual architecture diagram
├── terraform/                # IaC
├── lambdas/
│   ├── ingest/
│   │   ├── handler.py        # S3 trigger → parse → chunk
│   │   ├── parsers/
│   │   │   ├── pdf_parser.py
│   │   │   ├── json_parser.py  # ClinicalTrials.gov format
│   │   │   └── base.py
│   │   └── chunker.py        # Text chunking strategies
│   ├── embed/
│   │   └── handler.py        # Generate embeddings
│   ├── query/
│   │   ├── handler.py        # RAG query endpoint
│   │   ├── retriever.py      # Vector search
│   │   └── prompts.py        # Prompt templates
│   └── documents/
│       └── handler.py        # Document listing API
├── mcp_server/
│   ├── server.py             # FastMCP server
│   ├── tools.py              # Tool implementations
│   └── config.json           # MCP server config
├── streamlit_app/
│   ├── app.py                # Chat interface
│   ├── components/
│   │   ├── chat.py
│   │   ├── sidebar.py        # Document explorer
│   │   └── citations.py      # Source display
│   └── config.py
├── scripts/
│   ├── download_papers.py    # Fetch open-access papers from PMC
│   ├── download_trials.py    # Fetch from ClinicalTrials.gov API
│   ├── seed_db.py            # Initial data load
│   └── test_query.py         # Quick test script
├── tests/
│   ├── test_parsers.py
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_mcp_tools.py
├── data/
│   └── sample/               # 2-3 sample docs for local testing
├── docker-compose.yml        # Local PostgreSQL + pgvector
├── requirements.txt
├── Makefile                  # Common commands
└── .env.example
```

---

## 8. Implementation Timeline

### Day 1: Foundation (Data + Storage)
- [ ] Set up S3 bucket structure
- [ ] Write `download_papers.py` to fetch 10-15 open-access PDFs from PMC
- [ ] Write `download_trials.py` to fetch St. Jude trials from ClinicalTrials.gov
- [ ] Implement PDF parser + JSON parser
- [ ] Implement chunking logic (recursive character splitter, ~500 tokens per chunk)
- [ ] Set up local PostgreSQL + pgvector with Docker (or FAISS)
- [ ] Write `ingest_trigger` Lambda handler (local testing)

### Day 2: Embeddings + Retrieval
- [ ] Implement embedding generation (OpenAI ada-002 or Bedrock Titan)
- [ ] Write `embed_chunks` Lambda handler
- [ ] Implement vector similarity search function
- [ ] Write `query_handler` Lambda with RAG pipeline
- [ ] Test full pipeline locally: upload PDF → chunks → embeddings → query → answer
- [ ] Write prompt templates with citation requirements

### Day 3: MCP Server + Chat UI
- [ ] Implement MCP server with all 4 tools + resource + prompt
- [ ] Test MCP server with Claude Desktop
- [ ] Build Streamlit chat app
- [ ] Add citation display (source doc, page, link)
- [ ] Add document explorer sidebar

### Day 4: Infrastructure + Polish
- [ ] Write Terraform configs (or SAM template)
- [ ] Create architecture diagram (draw.io or Mermaid)
- [ ] Write comprehensive README
- [ ] Add error handling, logging, retries
- [ ] Record a 2-min demo video (optional but powerful)
- [ ] Prepare "scale" talking points (see Section 10)

---

## 9. Demo Script — What to Show in the Interview

### Opening (30 seconds)
> "I built a RAG system that lets researchers ask questions about St. Jude's published pediatric cancer research and get grounded, cited answers. It runs entirely on AWS serverless."

### Demo Flow (3-5 minutes)

1. **Upload a document** → Drop a PDF into S3 → Show Lambda trigger in CloudWatch
2. **Ask a question in Streamlit:**
   - "What are the survival rates for pediatric ALL based on St. Jude studies?"
   - Show the answer with citations
   - Show which source documents were retrieved
3. **Ask a cross-document question:**
   - "How has St. Jude's approach to medulloblastoma treatment evolved based on genomic findings?"
4. **Show MCP integration:**
   - Open Claude Desktop connected to MCP server
   - Ask the same question → show tools being called
   - "This is the same pipeline, but now accessible from ANY MCP-compatible client"
5. **Show the API:**
   - Quick curl to `/query` endpoint
   - Quick curl to `/documents` endpoint

### Key Talking Points
- "The MCP server means this isn't locked to one UI — any LLM client can use it"
- "The pipeline is event-driven and serverless — it scales automatically"
- "All data sources are publicly available open-access research"

---

## 10. "Scale" Answers — Be Ready for These Questions

| Question | Your Answer |
|----------|-------------|
| "How would this handle 100x more documents?" | Partition pgvector by doc_type, use HNSW index instead of IVFFlat, add a Redis cache for frequent queries |
| "What about concurrent users?" | Lambda auto-scales, API Gateway handles throttling, add SQS queue for ingestion to handle bursts |
| "How would you ensure answer quality?" | Add a reranking step (cross-encoder), implement confidence scoring, log all queries for evaluation |
| "What about data freshness?" | EventBridge scheduled rule to poll PMC/ClinicalTrials.gov APIs for new St. Jude publications weekly |
| "Cost at scale?" | Embeddings are the main cost (~$0.10/1M tokens). Cache embeddings aggressively. Use Bedrock Titan for cheaper option |
| "Security considerations?" | IAM least privilege, VPC for RDS, API keys for public endpoints, no PHI/PII in the demo (all open-access data) |

---

## 11. Pre-requisites & Costs

### You'll Need
- AWS Account (free tier covers most of this)
- OpenAI API key OR AWS Bedrock access (for embeddings + LLM)
- Python 3.12
- Docker (for local pgvector)
- Node.js (optional, for MCP testing with Claude Desktop)

### Estimated AWS Costs (Demo)
| Service | Estimated Monthly |
|---------|------------------|
| Lambda | $0 (free tier: 1M requests) |
| S3 | < $1 |
| API Gateway | $0 (free tier: 1M calls) |
| RDS (db.t3.micro) | ~$15 (or $0 if using FAISS on S3) |
| **Total** | **$0-15/month** |

---

## 12. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| PDF parsing quality varies | Use multiple parsers (pypdf2 + pdfplumber as fallback), add manual review flag |
| Embedding costs for large corpus | Start with 20-30 docs, batch embedding calls, cache aggressively |
| pgvector setup complexity | Have FAISS fallback ready — works with S3 only, no RDS needed |
| MCP server complexity | Start with 1-2 tools, expand. FastMCP makes it straightforward |
| Interview timeline pressure | Prioritize: Pipeline > API > MCP > Streamlit UI. Even without UI, the API + MCP is impressive |

---

## 13. What Makes This Stand Out vs Other Candidates

1. **Domain relevance** — You used St. Jude's actual published research, not generic data
2. **MCP implementation** — Most candidates won't even attempt this; it's explicitly in the job requirements
3. **Full pipeline** — Not just a notebook; it's a deployable, event-driven system
4. **Multiple interfaces** — REST API + MCP + Chat UI shows versatility
5. **IaC included** — Shows you think about deployment, not just code
6. **Scalability answers ready** — You've thought beyond the demo
