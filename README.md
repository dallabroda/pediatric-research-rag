# 🏥 pediatric-research-rag

A serverless RAG (Retrieval-Augmented Generation) system built on AWS that enables natural-language Q&A over publicly available pediatric cancer research from St. Jude Children's Research Hospital.

Upload research papers and clinical trial data → the pipeline automatically chunks, embeds, and indexes them → ask questions via REST API, MCP server, or Streamlit chat → get grounded, cited answers.

## Architecture

```
S3 (raw PDFs/JSON)
  │
  ▼  Lambda: ingest
Parse → Chunk → Store chunks (S3)
  │
  ▼  Lambda: embed
Bedrock Titan Embeddings → FAISS index (S3)
  │
  ▼  Lambda: query (API Gateway)
Embed question → FAISS search → Build prompt → Bedrock Claude → Cited answer
  │
  ├── REST API (/query, /documents)
  ├── MCP Server (FastMCP — tools, resources, prompts)
  └── Streamlit Chat UI
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Compute | AWS Lambda (Python 3.12) |
| Storage | S3 (raw docs + FAISS index) |
| Embeddings | AWS Bedrock — Titan Embeddings V2 |
| LLM | AWS Bedrock — Claude Sonnet |
| Vector Store | FAISS (file-based, stored on S3) |
| API | API Gateway + Lambda |
| MCP Server | FastMCP (Python) |
| Chat UI | Streamlit |
| Deploy | AWS CLI scripts |

## Data Sources (Free, Open Access)

| Source | Type | License |
|--------|------|---------|
| PubMed Central Open Access | Research paper PDFs/XML | Creative Commons |
| ClinicalTrials.gov API v2 | Clinical trial JSON | Public Domain (US Gov) |
| St. Jude Survivorship Portal | Summary statistics, data dictionary | Open Access |

## Project Structure

```
pediatric-research-rag/
├── README.md
├── requirements.txt
├── lambdas/
│   ├── ingest/            # S3 trigger → parse → chunk
│   │   ├── handler.py
│   │   ├── parsers.py     # PDF + JSON parsers
│   │   └── chunker.py     # Text chunking logic
│   ├── embed/             # Chunk → Bedrock Titan → FAISS
│   │   └── handler.py
│   ├── query/             # RAG: embed question → retrieve → LLM → answer
│   │   ├── handler.py
│   │   ├── retriever.py   # FAISS search
│   │   └── prompts.py     # Prompt templates
│   └── documents/         # Document listing API
│       └── handler.py
├── mcp_server/            # FastMCP server
│   ├── server.py
│   └── tools.py
├── streamlit_app/         # Chat UI
│   ├── app.py
│   └── components/
├── scripts/               # Data download scripts
│   ├── download_papers.py
│   ├── download_trials.py
│   └── seed_index.py
├── deploy/                # AWS CLI deployment
│   ├── setup.sh           # Create S3, Lambda, API Gateway
│   ├── deploy_lambdas.sh
│   └── teardown.sh
├── config/
│   └── settings.py        # Shared configuration
├── tests/
└── data/sample/           # Sample docs for local testing
```

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
```

### 3. Run locally
```bash
# Ingest + embed sample data
python scripts/seed_index.py

# Start MCP server
python mcp_server/server.py

# Start Streamlit chat
streamlit run streamlit_app/app.py
```

### 4. Deploy to AWS
```bash
cd deploy
./setup.sh          # Create infrastructure
./deploy_lambdas.sh # Package and deploy Lambda functions
```

## MCP Server

The MCP server exposes pediatric cancer research as tools accessible from Claude Desktop or any MCP-compatible client.

**Tools:**
- `search_research(query)` — Semantic search over research papers
- `ask_research_question(question)` — Full RAG pipeline with citations
- `list_clinical_trials(cancer_type)` — Filter clinical trials
- `get_document(doc_id)` — Get full document context

**Resources:**
- `research://documents` — Browse all ingested documents
- `research://trials` — Browse clinical trials

**Prompts:**
- `research_qa` — Structured Q&A template with citation requirements

### Connect to Claude Desktop
```json
{
  "mcpServers": {
    "pediatric-research": {
      "command": "python",
      "args": ["mcp_server/server.py"]
    }
  }
}
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Ask a question, get a cited answer |
| GET | `/documents` | List all ingested documents |
| GET | `/documents/{id}` | Get document details + chunks |

## License

MIT
