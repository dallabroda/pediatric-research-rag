"""
Shared configuration for pediatric-research-rag.
All settings can be overridden via environment variables.
"""
import os


# AWS
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "pediatric-research-rag")

# S3 prefixes
S3_RAW_PREFIX = "raw/"
S3_CHUNKS_PREFIX = "processed/chunks/"
S3_INDEX_PREFIX = "processed/index/"

# Bedrock Models
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
EMBEDDING_DIMENSION = 1024  # Titan V2 output dimension

# LLM — Amazon Nova Pro (no approval required, works immediately)
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "amazon.nova-pro-v1:0")
LLM_MODEL_ID_CHEAP = "us.anthropic.claude-3-5-haiku-20241022-v1:0"  # cheaper alternative

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))         # tokens per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))     # overlap tokens
MAX_CHUNK_CHARS = CHUNK_SIZE * 4                           # rough char estimate

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))                      # chunks to retrieve
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# FAISS
FAISS_INDEX_FILE = "faiss_index.bin"
FAISS_METADATA_FILE = "faiss_metadata.json"

# Paths (local development)
LOCAL_DATA_DIR = os.getenv("LOCAL_DATA_DIR", "data/sample")
LOCAL_INDEX_DIR = os.getenv("LOCAL_INDEX_DIR", "data/index")
