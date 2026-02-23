"""
Streamlit chat application for Pediatric Research RAG.

A research intelligence platform for pediatric cancer research that provides:
- Natural language Q&A over research papers and clinical trials
- Cross-document comparison and synthesis
- Data quality and lineage transparency
- Audit-ready source citations

Run with: streamlit run streamlit_app/app.py
"""
import json
import os
import sys
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


def get_config(key: str, default: str = "") -> str:
    """
    Get configuration value from environment or Streamlit secrets.

    Checks in order:
    1. Environment variables (for Docker/local)
    2. Streamlit secrets (for Streamlit Cloud)
    3. Default value
    """
    # Check environment first
    value = os.environ.get(key)
    if value:
        return value

    # Check Streamlit secrets (Streamlit Cloud)
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            # Also set as environment variable for boto3 to pick up
            value = st.secrets[key]
            os.environ[key] = value
            return value
    except Exception:
        pass

    return default


# Configuration - supports both env vars and Streamlit secrets
S3_BUCKET = get_config("S3_BUCKET", "pediatric-research-rag")
LOCAL_INDEX_DIR = get_config("LOCAL_INDEX_DIR", "data/index")
LLM_MODEL_ID = get_config("LLM_MODEL_ID", "anthropic.claude-sonnet-4-5-20250929-v1:0")
MAX_TOKENS = int(get_config("MAX_TOKENS", "1024"))

# Ensure AWS credentials are loaded from secrets if present
get_config("AWS_ACCESS_KEY_ID")
get_config("AWS_SECRET_ACCESS_KEY")
get_config("AWS_REGION", "us-east-1")

from lambdas.query.prompts import ContextChunk, build_no_context_response, build_user_prompt, SYSTEM_PROMPT
from lambdas.query.retriever import FAISSRetriever
from lambdas.query.multi_query import is_comparison_question, handle_comparison_question
from streamlit_app.components.citations import render_citations
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.data_quality import render_data_quality_dashboard

# Page config
st.set_page_config(
    page_title="Pediatric Research RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for chat UI
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Chat container styling */
    .stChatFloatingInputContainer {
        bottom: 20px;
        background-color: var(--background-color);
    }

    /* Make chat messages scrollable */
    [data-testid="stChatMessageContainer"] {
        max-height: calc(100vh - 250px);
        overflow-y: auto;
    }

    /* Improve message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }

    /* Stats bar styling */
    .stats-container {
        display: flex;
        gap: 1rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid #333;
        margin-bottom: 1rem;
    }

    .stat-item {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .stat-value {
        font-size: 1.2rem;
        font-weight: bold;
    }

    .stat-label {
        font-size: 0.8rem;
        color: #888;
    }

    /* Example buttons */
    .example-btn {
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_retriever() -> FAISSRetriever:
    """Get cached retriever instance."""
    try:
        return FAISSRetriever.from_local(LOCAL_INDEX_DIR)
    except FileNotFoundError:
        return FAISSRetriever.from_s3(S3_BUCKET)


@st.cache_resource
def get_bedrock_client():
    """Get cached Bedrock client."""
    import boto3

    region = get_config("AWS_REGION", "us-east-1")

    return boto3.client(
        "bedrock-runtime",
        region_name=region,
    )


def get_documents() -> list[dict]:
    """Get list of all documents."""
    try:
        retriever = get_retriever()
        doc_ids = retriever.get_document_ids()

        documents = []
        for doc_id in doc_ids:
            chunks = retriever.get_document_chunks(doc_id)
            if chunks:
                first_chunk = chunks[0]
                documents.append({
                    "document_id": doc_id,
                    "title": first_chunk.get("doc_title", doc_id),
                    "doc_type": first_chunk.get("doc_type", "unknown"),
                    "source_url": first_chunk.get("source_url"),
                    "chunks_count": len(chunks),
                })

        return documents
    except Exception as e:
        st.error(f"Failed to load documents: {e}")
        return []


def ask_question(question: str, top_k: int = 5) -> tuple[str, list[dict], dict]:
    """
    Ask a question and get an answer with sources.

    Automatically detects comparison questions and routes to multi-query handler.

    Args:
        question: User's question
        top_k: Number of context chunks

    Returns:
        Tuple of (answer, sources, metadata)
    """
    retriever = get_retriever()
    bedrock = get_bedrock_client()

    # Check if this is a comparison question
    if is_comparison_question(question):
        result = handle_comparison_question(
            question=question,
            retriever=retriever,
            bedrock_client=bedrock,
            model_id=LLM_MODEL_ID,
            top_k_per_entity=3,
        )

        if result.get("is_comparison"):
            return (
                result["answer"],
                result["sources"],
                {
                    "is_comparison": True,
                    "entities": result.get("entities", []),
                    "comparison_type": result.get("comparison_type"),
                    "confidence": result.get("confidence", "medium"),
                    "entity_coverage": result.get("entity_coverage", {}),
                }
            )

    # Standard RAG flow
    results = retriever.search(question, top_k=top_k)

    if not results:
        return build_no_context_response(question), [], {"is_comparison": False}

    # Build context
    context_chunks = [
        ContextChunk(
            text=r.chunk_text,
            doc_title=r.doc_title,
            doc_type=r.doc_type,
            source_url=r.source_url,
            section_title=r.section_title,
            page_number=r.page_number,
        )
        for r in results
    ]

    # Build prompt
    user_prompt = build_user_prompt(question, context_chunks)

    # Build request body based on model type
    if "anthropic" in LLM_MODEL_ID or "claude" in LLM_MODEL_ID:
        # Claude/Anthropic format
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
        })
    else:
        # Amazon Nova format
        body = json.dumps({
            "system": [{"text": SYSTEM_PROMPT}],
            "messages": [{"role": "user", "content": [{"text": user_prompt}]}],
            "inferenceConfig": {"maxTokens": MAX_TOKENS},
        })

    response = bedrock.invoke_model(
        modelId=LLM_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())

    # Parse response based on model type
    if "anthropic" in LLM_MODEL_ID or "claude" in LLM_MODEL_ID:
        answer = response_body["content"][0]["text"]
    else:
        # Amazon Nova format
        answer = response_body["output"]["message"]["content"][0]["text"]

    # Format sources
    sources = [
        {
            "document_id": r.document_id,
            "doc_title": r.doc_title,
            "doc_type": r.doc_type,
            "score": r.score,
            "source_url": r.source_url,
            "section": r.section_title,
            "page": r.page_number,
        }
        for r in results
    ]

    # Compute confidence based on scores
    avg_score = sum(r.score for r in results) / len(results) if results else 0
    unique_docs = len(set(r.document_id for r in results))

    if avg_score >= 0.7 and unique_docs >= 2:
        confidence = "high"
    elif avg_score >= 0.5:
        confidence = "medium"
    else:
        confidence = "low"

    return answer, sources, {
        "is_comparison": False,
        "confidence": confidence,
        "avg_relevance": round(avg_score, 2),
        "unique_sources": unique_docs,
    }


def render_confidence_badge(metadata: dict) -> None:
    """Render a confidence badge based on answer metadata."""
    confidence = metadata.get("confidence", "medium")
    sources = metadata.get("unique_sources", 0)

    if confidence == "high":
        st.caption(f"✅ High confidence · {sources} sources")
    elif confidence == "medium":
        st.caption(f"ℹ️ Medium confidence · {sources} sources")
    else:
        st.caption(f"⚠️ Low confidence · Limited evidence")


def main():
    """Main application entry point."""
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Check for index
    try:
        retriever = get_retriever()
        doc_count = len(retriever.get_document_ids())
        vector_count = retriever.index.ntotal

        if vector_count == 0:
            st.warning(
                "No documents indexed yet. Run the seed_index.py script first."
            )
            return

    except Exception as e:
        st.error(f"Failed to load index: {e}")
        return

    # Get documents for sidebar
    documents = get_documents()
    papers = len([d for d in documents if d.get("doc_type") == "paper"])
    trials = len([d for d in documents if d.get("doc_type") == "trial"])

    # Header with stats
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🔬 Pediatric Cancer Research")
    with col2:
        st.caption(f"📄 {papers} papers · 🧪 {trials} trials · 📊 {vector_count:,} vectors")

    # Sidebar for settings and document browser
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Sources to retrieve", 3, 10, 5)
        st.session_state["top_k"] = top_k

        st.divider()

        st.header("Documents")
        selected_doc = render_sidebar(documents)

        if selected_doc:
            chunks = retriever.get_document_chunks(selected_doc)
            if chunks:
                first = chunks[0]
                st.markdown(f"**{first.get('doc_title', selected_doc)[:50]}...**")
                if first.get("source_url"):
                    st.markdown(f"[View Original]({first['source_url']})")

    # Main chat area
    chat_container = st.container()

    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show sources for assistant messages
                if message["role"] == "assistant" and message.get("sources"):
                    with st.expander(f"📚 Sources ({len(message['sources'])})"):
                        render_citations(message["sources"])

                # Show metadata
                if message.get("metadata"):
                    render_confidence_badge(message["metadata"])

    # Show example questions if no messages
    if not st.session_state.messages:
        st.markdown("### 💡 Try asking:")

        col1, col2 = st.columns(2)

        examples = [
            "What is acute lymphoblastic leukemia?",
            "What are survival rates for pediatric cancers?",
            "Compare ALL and AML treatment approaches",
            "What clinical trials are available for neuroblastoma?",
        ]

        with col1:
            if st.button(examples[0], use_container_width=True):
                st.session_state.pending_question = examples[0]
                st.rerun()
            if st.button(examples[2], use_container_width=True):
                st.session_state.pending_question = examples[2]
                st.rerun()

        with col2:
            if st.button(examples[1], use_container_width=True):
                st.session_state.pending_question = examples[1]
                st.rerun()
            if st.button(examples[3], use_container_width=True):
                st.session_state.pending_question = examples[3]
                st.rerun()

    # Chat input at the bottom
    if prompt := st.chat_input("Ask about pediatric cancer research..."):
        process_question(prompt)

    # Handle pending question from example buttons
    if st.session_state.get("pending_question"):
        question = st.session_state.pop("pending_question")
        process_question(question)


def process_question(question: str):
    """Process a user question and generate response."""
    top_k = st.session_state.get("top_k", 5)

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    # Generate and display response
    with st.chat_message("assistant"):
        spinner_text = "🔄 Comparing sources..." if is_comparison_question(question) else "🔍 Searching research..."

        with st.spinner(spinner_text):
            try:
                answer, sources, metadata = ask_question(question, top_k=top_k)

                st.markdown(answer)
                render_confidence_badge(metadata)

                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        render_citations(sources)

                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "metadata": metadata,
                })

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                })


if __name__ == "__main__":
    main()
