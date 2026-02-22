# System Architecture

## High-Level Architecture

```mermaid
flowchart TB
    subgraph "Data Sources"
        PMC[PubMed Central<br/>Open Access Papers]
        CT[ClinicalTrials.gov<br/>Trial Registry]
        SJLIFE[SJLIFE Portal<br/>Survivorship Data]
    end

    subgraph "Data Ingestion"
        DL[Download Scripts<br/>Python]
        S3Raw[(S3: raw/)]
        IngestLambda[Ingest Lambda<br/>Parse & Chunk]
        S3Chunks[(S3: processed/chunks/)]
    end

    subgraph "Embedding Pipeline"
        EmbedLambda[Embed Lambda<br/>Titan Embeddings V2]
        Bedrock1[AWS Bedrock<br/>Titan V2]
        FAISS[(FAISS Index<br/>S3: processed/index/)]
    end

    subgraph "Query Pipeline"
        API[API Gateway<br/>/query, /health]
        QueryLambda[Query Lambda<br/>RAG Pipeline]
        Bedrock2[AWS Bedrock<br/>Claude Haiku/Sonnet]
        DynamoDB[(DynamoDB<br/>Query Logs)]
    end

    subgraph "User Interfaces"
        Streamlit[Streamlit App<br/>Chat UI]
        MCP[MCP Server<br/>Claude Desktop]
        REST[REST API<br/>Direct Integration]
    end

    subgraph "Monitoring"
        CW[CloudWatch<br/>Dashboard & Metrics]
        Budget[AWS Budget<br/>$10 Alert]
    end

    PMC --> DL
    CT --> DL
    SJLIFE --> DL
    DL --> S3Raw
    S3Raw -->|S3 Trigger| IngestLambda
    IngestLambda --> S3Chunks
    S3Chunks --> EmbedLambda
    EmbedLambda --> Bedrock1
    EmbedLambda --> FAISS

    Streamlit --> API
    MCP --> API
    REST --> API
    API --> QueryLambda
    QueryLambda --> FAISS
    QueryLambda --> Bedrock2
    QueryLambda --> DynamoDB
    QueryLambda --> CW

    EmbedLambda --> CW
```

## Data Flow

### 1. Ingestion Flow

```mermaid
sequenceDiagram
    participant U as User/Cron
    participant S as Download Script
    participant S3 as S3 (raw/)
    participant IL as Ingest Lambda
    participant C as S3 (chunks/)

    U->>S: Run download
    S->>S3: Upload PDF/JSON + metadata
    S3->>IL: S3 Trigger
    IL->>IL: Parse document
    IL->>IL: Check duplicate (hash)
    IL->>IL: Chunk text
    IL->>C: Store chunks JSON
```

### 2. Embedding Flow

```mermaid
sequenceDiagram
    participant C as S3 (chunks/)
    participant EL as Embed Lambda
    participant BR as Bedrock Titan
    participant I as S3 (index/)

    C->>EL: New chunks available
    EL->>BR: Get embeddings
    BR-->>EL: 1024-dim vectors
    EL->>EL: Build FAISS index
    EL->>I: Save index + metadata
```

### 3. Query Flow

```mermaid
sequenceDiagram
    participant U as User
    participant A as API Gateway
    participant QL as Query Lambda
    participant I as FAISS Index
    participant BR as Claude (Bedrock)
    participant DB as DynamoDB

    U->>A: POST /query
    A->>QL: Invoke
    QL->>I: Load index
    QL->>BR: Embed question
    QL->>I: Similarity search (top-K)
    QL->>QL: Build prompt with context
    QL->>BR: Generate answer
    QL->>DB: Log query metrics
    QL-->>A: Response with citations
    A-->>U: JSON response
```

## Component Details

### Lambda Functions

| Function | Trigger | Memory | Timeout | Purpose |
|----------|---------|--------|---------|---------|
| ingest | S3 upload | 1024 MB | 5 min | Parse & chunk documents |
| embed | Manual/scheduled | 2048 MB | 10 min | Generate embeddings, build index |
| query | API Gateway | 1024 MB | 2 min | RAG query pipeline |
| documents | API Gateway | 512 MB | 1 min | List/get documents |
| health | API Gateway | 256 MB | 30 sec | Health check |

### S3 Structure

```
pediatric-research-rag/
├── raw/
│   ├── papers/
│   │   ├── PMC12345.pdf
│   │   └── PMC12345_metadata.json
│   └── trials/
│       └── NCT01234567.json
├── processed/
│   ├── chunks/
│   │   ├── PMC12345_chunks.json
│   │   └── NCT01234567_chunks.json
│   └── index/
│       ├── faiss_index.bin
│       └── faiss_metadata.json
```

### API Endpoints

| Endpoint | Method | Rate Limit | Description |
|----------|--------|------------|-------------|
| /query | POST | 5/sec | RAG question answering |
| /documents | GET | 5/sec | List all documents |
| /documents/{id} | GET | 5/sec | Get document details |
| /health | GET | 5/sec | System health check |

## Security

- API Gateway: No authentication (public demo)
- Rate limiting: 5 req/sec (burst: 10)
- Budget alert: $10/month
- No PII/PHI in demo data (public papers only)

## Scalability Notes

Current design handles demo scale (~30 docs, ~50 queries/day).

For production scale:
- Replace FAISS with managed vector DB (OpenSearch, Pinecone)
- Add API Gateway authentication (Cognito, API keys)
- Implement request queuing for burst traffic
- Add CloudFront caching for document retrieval
