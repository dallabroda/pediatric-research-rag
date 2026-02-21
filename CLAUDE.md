# pediatric-research-rag

## Project Overview
A serverless RAG (Retrieval-Augmented Generation) system on AWS that enables natural-language Q&A over publicly available pediatric cancer research from St. Jude Children's Research Hospital. Built as a portfolio project demonstrating AWS data engineering, RAG architecture, and MCP integration.

## Tech Stack
- **Language:** Python 3.12
- **AWS Services:** Lambda, S3, API Gateway, Bedrock
- **Embeddings:** AWS Bedrock — Titan Embeddings V2 (`amazon.titan-embed-text-v2:0`, 1024 dimensions)
- **LLM:** AWS Bedrock — Claude Haiku for dev (`anthropic.claude-3-haiku-20240307-v1:0`), Sonnet for demo (`anthropic.claude-3-sonnet-20240229-v1:0`)
- **Vector Store:** FAISS (file-based, stored on S3 as binary + JSON metadata sidecar)
- **MCP Server:** FastMCP (Python)
- **Chat UI:** Streamlit
- **Infrastructure:** AWS CLI scripts (no Terraform, no CDK)
- **Testing:** pytest + moto (AWS mocking)

## Critical Rules

### Architecture
- Event-driven pipeline: S3 upload → Lambda ingest → Lambda embed → FAISS index on S3
- RAG flow: embed question → FAISS cosine similarity search → top-K chunks → build prompt → Bedrock Claude → cited answer
- MCP server exposes tools, resources, and prompts via FastMCP
- REST API via API Gateway + Lambda for /query and /documents endpoints

### Secrets Management
- **ALWAYS store credentials in GitHub Secrets** — never commit passwords, API keys, or credentials to the repo
- AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) must be stored in GitHub Secrets for CI/CD
- Any API keys, tokens, or sensitive config go in GitHub Secrets, not in code or `.env` files committed to repo
- For local development, use `.env` file (already in `.gitignore`)
- When setting up GitHub Actions, reference secrets via `${{ secrets.SECRET_NAME }}`

### Code Style
- **NO frameworks:** Do not use LangChain, LlamaIndex, or any RAG framework. All RAG logic is implemented directly with boto3 + FAISS.
- Always use boto3 for AWS calls, never subprocess/aws cli in Python code
- Type hints on all functions
- Docstrings on all public functions
- Config in `config/settings.py`, secrets via environment variables only
- No hardcoded AWS credentials, regions, or bucket names

### Lambda Handlers
- All handlers follow the pattern: `def handler(event, context) -> dict`
- Return format: `{"statusCode": 200, "body": json.dumps(response)}`
- Always include error handling with proper HTTP status codes
- Log with `logging` module, not print()

### FAISS Index
- Index stored as binary file (`faiss_index.bin`) on S3
- Metadata stored as JSON sidecar (`faiss_metadata.json`) on S3
- Metadata structure per chunk: `{chunk_id, document_id, chunk_text, chunk_index, page_number, section_title, source_url, doc_title}`
- Use `faiss.IndexFlatIP` (inner product) with L2-normalized vectors for cosine similarity
- Rebuild index on each new document ingestion (acceptable for demo scale of ~20-30 docs)

### Bedrock Calls
- Titan Embeddings: `bedrock-runtime` client, `invoke_model` with `amazon.titan-embed-text-v2:0`
- Claude LLM: `bedrock-runtime` client, `invoke_model`
- **Default to Haiku during development** (`anthropic.claude-3-haiku-20240307-v1:0`) — 12x cheaper than Sonnet
- Switch to Sonnet for demo/interview (`anthropic.claude-3-sonnet-20240229-v1:0`) via env var `LLM_MODEL_ID`
- Always set `max_tokens` for Claude calls
- Handle throttling with exponential backoff

### AWS Free Tier & Cost Strategy
- **Bedrock has NO free tier** — it's pay-per-use from the first token
- New AWS accounts get $200 in credits which covers this entire project (~200x over)
- Lambda: Always Free — 1M requests + 400K GB-sec/month (we use ~100 invocations)
- S3: 5 GB free for 12 months (we use ~50 MB)
- API Gateway: 1M REST calls free for 12 months (we use ~100 calls)
- Titan Embeddings V2: ~$0.02 for entire corpus (~150K tokens)
- Claude Haiku: ~$0.05 for all dev/testing (~50 queries)
- Claude Sonnet: ~$0.60 for demo queries (~50 queries)
- **Total project cost: < $1**
- **Set a $5 billing alert** in AWS Budgets as a safety net

### Chunking Strategy
- Recursive character splitting
- Target chunk size: ~500 tokens (~2000 chars)
- Overlap: 50 tokens (~200 chars)
- Preserve paragraph boundaries where possible
- Each chunk keeps metadata: document_id, chunk_index, page_number, section_title

### MCP Server (FastMCP)
- Located in `mcp_server/`
- Tools: `search_research`, `ask_research_question`, `list_clinical_trials`, `get_document`
- Resources: `research://documents`, `research://trials`
- Prompts: `research_qa` (structured Q&A with citation requirements)
- Must work both locally (stdio transport for Claude Desktop) and could be adapted for remote (Streamable HTTP)

### Data Sources
All data is freely available, open access, no approvals needed:
- **PubMed Central (PMC):** Open-access research papers via E-utilities API + OA Web Service
- **ClinicalTrials.gov:** Clinical trial data via API v2 (REST, JSON)
- **St. Jude Survivorship Portal:** Open-tier summary statistics and data dictionary

### Testing
- Use `pytest` for all tests
- Use `moto` to mock AWS services (S3, Bedrock)
- Test files in `tests/` directory, named `test_*.py`
- Mock Bedrock responses for embedding and LLM calls
- Test chunking with sample text, not real PDFs

## Project Structure
```
pediatric-research-rag/
├── CLAUDE.md                  # This file
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   └── settings.py            # Shared configuration
├── lambdas/
│   ├── ingest/
│   │   ├── handler.py         # S3 trigger → parse → chunk → store
│   │   ├── parsers.py         # PDF + ClinicalTrials JSON parsers
│   │   └── chunker.py         # Text chunking logic
│   ├── embed/
│   │   └── handler.py         # Chunk → Bedrock Titan → FAISS index
│   ├── query/
│   │   ├── handler.py         # API Gateway → RAG pipeline → response
│   │   ├── retriever.py       # Load FAISS index, search, return chunks
│   │   └── prompts.py         # System prompts and prompt templates
│   └── documents/
│       └── handler.py         # GET /documents and /documents/{id}
├── mcp_server/
│   ├── server.py              # FastMCP server entry point
│   └── tools.py               # Tool, resource, and prompt implementations
├── streamlit_app/
│   ├── app.py                 # Main Streamlit chat application
│   └── components/
│       ├── chat.py            # Chat interface component
│       ├── sidebar.py         # Document explorer sidebar
│       └── citations.py       # Source citation display
├── scripts/
│   ├── download_papers.py     # Fetch open-access papers from PMC
│   ├── download_trials.py     # Fetch St. Jude trials from ClinicalTrials.gov
│   └── seed_index.py          # Local: ingest + embed + build FAISS index
├── deploy/
│   ├── setup.sh               # Create S3 bucket, Lambda functions, API Gateway
│   ├── deploy_lambdas.sh      # Package and deploy Lambda code
│   └── teardown.sh            # Clean up all AWS resources
├── tests/
│   ├── test_parsers.py
│   ├── test_chunker.py
│   ├── test_retriever.py
│   ├── test_prompts.py
│   └── test_mcp_tools.py
└── data/
    └── sample/                # Sample docs for local development
```

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download sample data
python scripts/download_papers.py --count 15
python scripts/download_trials.py --sponsor "St. Jude" --count 10

# Build FAISS index locally
python scripts/seed_index.py

# Run MCP server locally
python mcp_server/server.py

# Run Streamlit app
streamlit run streamlit_app/app.py

# Run tests
pytest tests/ -v

# Deploy to AWS
cd deploy && ./setup.sh && ./deploy_lambdas.sh
```

## Environment Variables

```
AWS_REGION=us-east-1
S3_BUCKET=pediatric-research-rag
EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
LLM_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0       # dev (cheap)
# LLM_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0    # demo (better quality)
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
```
