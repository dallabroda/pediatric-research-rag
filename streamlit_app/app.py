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
# Use Amazon Nova Lite by default (no model access approval needed)
# Set LLM_MODEL_ID to anthropic.claude-3-haiku-20240307-v1:0 if you have Claude access
LLM_MODEL_ID = get_config("LLM_MODEL_ID", "amazon.nova-lite-v1:0")
MAX_TOKENS = int(get_config("MAX_TOKENS", "1024"))

# Ensure AWS credentials are loaded from secrets if present
get_config("AWS_ACCESS_KEY_ID")
get_config("AWS_SECRET_ACCESS_KEY")
get_config("AWS_REGION", "us-east-1")

from lambdas.query.prompts import ContextChunk, build_no_context_response, build_user_prompt, SYSTEM_PROMPT
from lambdas.query.retriever import FAISSRetriever
from lambdas.query.multi_query import is_comparison_question, handle_comparison_question
from streamlit_app.components.chat import add_assistant_message, add_user_message, clear_chat, render_chat
from streamlit_app.components.citations import render_citations
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.data_quality import render_data_quality_dashboard

# Page config
st.set_page_config(
    page_title="Pediatric Research RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


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

    if confidence == "high":
        st.success(f"Confidence: High | {metadata.get('unique_sources', 0)} sources")
    elif confidence == "medium":
        st.info(f"Confidence: Medium | {metadata.get('unique_sources', 0)} sources")
    else:
        st.warning(f"Confidence: Low | Limited evidence")

    if metadata.get("is_comparison"):
        entities = metadata.get("entities", [])
        coverage = metadata.get("entity_coverage", {})
        st.caption(f"Comparison: {' vs '.join(entities)} | Coverage: {coverage}")


def render_about_section() -> None:
    """Render the About section explaining system capabilities."""
    st.markdown("---")
    st.subheader("About This System")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **What This System Does:**
        - Natural language Q&A over pediatric cancer research
        - Cross-document comparison and synthesis
        - Automatic source citation and provenance tracking
        - Confidence scoring based on evidence quality

        **Data Sources:**
        - PubMed Central open-access research papers
        - ClinicalTrials.gov trial registrations
        - St. Jude Children's Research Hospital publications
        """)

    with col2:
        st.markdown("""
        **Key Differentiators:**
        - No RAG frameworks - direct implementation for full control
        - Data lineage tracking from source to response
        - Extraction confidence and quality metrics
        - Contradiction detection across sources
        - Audit-ready query logging

        **Try a Comparison:**
        - "Compare ALL and AML treatment outcomes"
        - "What's the difference between chemotherapy and immunotherapy?"
        """)


def main():
    """Main application entry point."""
    # Header
    st.title("Pediatric Cancer Research Intelligence")
    st.markdown(
        "A research intelligence platform for pediatric oncology. Ask questions, compare treatments, "
        "and explore research with full source transparency and data quality metrics."
    )

    # Check for index
    try:
        retriever = get_retriever()
        doc_count = len(retriever.get_document_ids())
        vector_count = retriever.index.ntotal

        if vector_count == 0:
            st.warning(
                "No documents indexed yet. Run the following to build the index:\n\n"
                "```bash\n"
                "python scripts/download_papers.py --count 5\n"
                "python scripts/download_trials.py --count 5\n"
                "python scripts/seed_index.py\n"
                "```"
            )
            return

    except Exception as e:
        st.error(
            f"Failed to load index: {e}\n\n"
            "Make sure to run the seed_index.py script first."
        )
        return

    # Sidebar
    documents = get_documents()
    selected_doc = render_sidebar(documents)

    # Create tabs for different views
    tab_chat, tab_quality = st.tabs(["Research Assistant", "Data Quality"])

    with tab_chat:
        # Corpus stats bar
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents", doc_count)
        with col2:
            st.metric("Vectors", f"{vector_count:,}")
        with col3:
            papers = len([d for d in documents if d.get("doc_type") == "paper"])
            trials = len([d for d in documents if d.get("doc_type") == "trial"])
            st.metric("Coverage", f"{papers} papers, {trials} trials")

        st.divider()

        # Handle document selection
        if selected_doc:
            st.subheader(f"Document: {selected_doc}")
            chunks = retriever.get_document_chunks(selected_doc)
            if chunks:
                first = chunks[0]
                st.markdown(f"**{first.get('doc_title', selected_doc)}**")
                st.caption(f"Type: {first.get('doc_type', 'unknown')}")
                if first.get("source_url"):
                    st.markdown(f"[View Original]({first['source_url']})")

                with st.expander("View Content", expanded=True):
                    full_text = "\n\n".join(c.get("chunk_text", "") for c in chunks)
                    st.text_area("Document Text", full_text[:5000], height=300)

            st.divider()

        # Chat interface
        render_chat()

        # Chat input
        top_k = st.session_state.get("top_k", 5)

        if prompt := st.chat_input("Ask a question about pediatric cancer research..."):
            # Add user message
            add_user_message(prompt)

            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                # Show appropriate spinner based on question type
                spinner_text = "Comparing sources..." if is_comparison_question(prompt) else "Searching research database..."

                with st.spinner(spinner_text):
                    try:
                        answer, sources, metadata = ask_question(prompt, top_k=top_k)

                        # Show comparison badge if applicable
                        if metadata.get("is_comparison"):
                            st.caption(f"Cross-document comparison: {' vs '.join(metadata.get('entities', []))}")

                        st.markdown(answer)

                        # Confidence badge
                        render_confidence_badge(metadata)

                        if sources:
                            with st.expander(f"View Sources ({len(sources)})"):
                                render_citations(sources)

                        # Add to history
                        add_assistant_message(answer, sources)

                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        add_assistant_message(error_msg, [])

        # Example questions
        if not st.session_state.get("messages"):
            st.markdown("### Example Questions")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Standard Queries:**")
                if st.button("What is acute lymphoblastic leukemia?"):
                    st.session_state.example_question = "What is acute lymphoblastic leukemia?"
                    st.rerun()

                if st.button("What are the survival rates for pediatric cancers?"):
                    st.session_state.example_question = "What are the survival rates for pediatric cancers?"
                    st.rerun()

            with col2:
                st.markdown("**Comparison Queries:**")
                if st.button("Compare ALL and AML treatment approaches"):
                    st.session_state.example_question = "Compare ALL and AML treatment approaches"
                    st.rerun()

                if st.button("What's the difference between chemotherapy and immunotherapy?"):
                    st.session_state.example_question = "What's the difference between chemotherapy and immunotherapy for pediatric cancer?"
                    st.rerun()

        # Handle example question selection
        if st.session_state.get("example_question"):
            question = st.session_state.pop("example_question")
            add_user_message(question)

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                spinner_text = "Comparing sources..." if is_comparison_question(question) else "Searching research database..."

                with st.spinner(spinner_text):
                    answer, sources, metadata = ask_question(question, top_k=top_k)

                    if metadata.get("is_comparison"):
                        st.caption(f"Cross-document comparison: {' vs '.join(metadata.get('entities', []))}")

                    st.markdown(answer)
                    render_confidence_badge(metadata)

                    if sources:
                        with st.expander(f"View Sources ({len(sources)})"):
                            render_citations(sources)

                    add_assistant_message(answer, sources)

        # About section at bottom
        render_about_section()

    with tab_quality:
        # Data Quality Dashboard
        render_data_quality_dashboard(retriever)


if __name__ == "__main__":
    main()
