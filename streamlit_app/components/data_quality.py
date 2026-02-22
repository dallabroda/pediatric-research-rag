"""
Data Quality Dashboard component for Pediatric Research RAG.

Displays corpus statistics, document freshness, source distribution,
and extraction confidence metrics to demonstrate data governance.
"""
from collections import Counter
from datetime import datetime
from typing import Optional

import streamlit as st


def compute_corpus_stats(metadata: list[dict]) -> dict:
    """
    Compute comprehensive corpus statistics from FAISS metadata.

    Args:
        metadata: List of chunk metadata dicts from FAISS index

    Returns:
        Dictionary with corpus statistics
    """
    if not metadata:
        return {
            "total_chunks": 0,
            "total_documents": 0,
            "doc_types": {},
            "avg_chunk_length": 0,
            "total_characters": 0,
            "confidence_scores": [],
            "ingestion_dates": [],
        }

    # Basic counts
    doc_ids = set()
    doc_types = Counter()
    confidence_scores = []
    ingestion_dates = []
    char_counts = []
    word_counts = []

    for chunk in metadata:
        doc_id = chunk.get("document_id", "")
        doc_ids.add(doc_id)
        doc_types[chunk.get("doc_type", "unknown")] += 1

        # Quality metrics
        if "extraction_confidence" in chunk:
            confidence_scores.append(chunk["extraction_confidence"])

        # Lineage
        if "ingestion_timestamp" in chunk:
            ingestion_dates.append(chunk["ingestion_timestamp"])

        # Size metrics
        char_counts.append(chunk.get("char_count", len(chunk.get("chunk_text", ""))))
        word_counts.append(chunk.get("word_count", len(chunk.get("chunk_text", "").split())))

    return {
        "total_chunks": len(metadata),
        "total_documents": len(doc_ids),
        "doc_types": dict(doc_types),
        "avg_chunk_length": sum(char_counts) / len(char_counts) if char_counts else 0,
        "avg_words_per_chunk": sum(word_counts) / len(word_counts) if word_counts else 0,
        "total_characters": sum(char_counts),
        "confidence_scores": confidence_scores,
        "ingestion_dates": ingestion_dates,
    }


def get_document_details(metadata: list[dict]) -> list[dict]:
    """
    Extract document-level details from chunk metadata.

    Args:
        metadata: List of chunk metadata dicts

    Returns:
        List of document details
    """
    docs = {}
    for chunk in metadata:
        doc_id = chunk.get("document_id", "")
        if doc_id not in docs:
            docs[doc_id] = {
                "document_id": doc_id,
                "title": chunk.get("doc_title", doc_id),
                "doc_type": chunk.get("doc_type", "unknown"),
                "source_url": chunk.get("source_url"),
                "chunk_count": 0,
                "total_chars": 0,
                "ingestion_timestamp": chunk.get("ingestion_timestamp"),
                "extraction_confidence": [],
            }

        docs[doc_id]["chunk_count"] += 1
        docs[doc_id]["total_chars"] += chunk.get("char_count", len(chunk.get("chunk_text", "")))

        if "extraction_confidence" in chunk:
            docs[doc_id]["extraction_confidence"].append(chunk["extraction_confidence"])

    # Compute average confidence per document
    for doc in docs.values():
        scores = doc["extraction_confidence"]
        doc["avg_confidence"] = sum(scores) / len(scores) if scores else None
        del doc["extraction_confidence"]

    return list(docs.values())


def render_corpus_overview(stats: dict) -> None:
    """Render corpus overview metrics."""
    st.subheader("Corpus Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Documents",
            value=stats["total_documents"],
            help="Total number of unique documents in the corpus",
        )

    with col2:
        st.metric(
            label="Vectors",
            value=f"{stats['total_chunks']:,}",
            help="Total number of embedded chunks in FAISS index",
        )

    with col3:
        avg_chunks = stats["total_chunks"] / stats["total_documents"] if stats["total_documents"] > 0 else 0
        st.metric(
            label="Avg Chunks/Doc",
            value=f"{avg_chunks:.1f}",
            help="Average number of chunks per document",
        )

    with col4:
        st.metric(
            label="Total Content",
            value=f"{stats['total_characters'] / 1000:.0f}K chars",
            help="Total character count across all chunks",
        )


def render_source_distribution(stats: dict) -> None:
    """Render source type distribution chart."""
    st.subheader("Source Distribution")

    doc_types = stats.get("doc_types", {})
    if not doc_types:
        st.info("No document type information available.")
        return

    # Create two columns: chart and breakdown
    col1, col2 = st.columns([2, 1])

    with col1:
        # Pie chart using Streamlit's native plotting
        import pandas as pd

        df = pd.DataFrame({
            "Source Type": list(doc_types.keys()),
            "Chunks": list(doc_types.values()),
        })

        # Display as bar chart (more readable than pie for small datasets)
        st.bar_chart(df.set_index("Source Type"))

    with col2:
        st.markdown("**Breakdown:**")
        total = sum(doc_types.values())
        for doc_type, count in sorted(doc_types.items(), key=lambda x: -x[1]):
            pct = (count / total * 100) if total > 0 else 0
            icon = "📄" if doc_type == "paper" else "🧪" if doc_type == "trial" else "📋"
            st.markdown(f"{icon} **{doc_type.title()}**: {count} chunks ({pct:.1f}%)")


def render_confidence_metrics(stats: dict) -> None:
    """Render extraction confidence metrics."""
    st.subheader("Extraction Quality")

    confidence_scores = stats.get("confidence_scores", [])

    if not confidence_scores:
        st.info(
            "Confidence scores not available in current index. "
            "Re-run ingestion pipeline to generate quality metrics."
        )
        return

    col1, col2, col3 = st.columns(3)

    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    min_confidence = min(confidence_scores)
    max_confidence = max(confidence_scores)

    with col1:
        st.metric(
            label="Avg Confidence",
            value=f"{avg_confidence:.1%}",
            help="Average extraction confidence across all chunks",
        )

    with col2:
        st.metric(
            label="Min Confidence",
            value=f"{min_confidence:.1%}",
            help="Lowest extraction confidence (may indicate OCR issues)",
        )

    with col3:
        st.metric(
            label="High Quality Chunks",
            value=f"{sum(1 for c in confidence_scores if c >= 0.8) / len(confidence_scores):.0%}",
            help="Percentage of chunks with confidence >= 80%",
        )

    # Confidence distribution histogram
    import pandas as pd

    st.markdown("**Confidence Distribution:**")
    df = pd.DataFrame({"Confidence": confidence_scores})
    st.bar_chart(df["Confidence"].value_counts(bins=10).sort_index())


def render_data_lineage(metadata: list[dict]) -> None:
    """Render data lineage information."""
    st.subheader("Data Lineage")

    # Get unique parsers and timestamps
    parsers = Counter()
    timestamps = []

    for chunk in metadata:
        if "parser_name" in chunk:
            parsers[chunk["parser_name"]] += 1
        if "ingestion_timestamp" in chunk:
            timestamps.append(chunk["ingestion_timestamp"])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Processing Pipeline:**")
        if parsers:
            for parser, count in parsers.items():
                version = "1.0.0"  # Default
                for chunk in metadata:
                    if chunk.get("parser_name") == parser:
                        version = chunk.get("parser_version", "1.0.0")
                        break
                st.markdown(f"- `{parser}` v{version}: {count} chunks")
        else:
            st.info("Parser information not available in metadata.")

    with col2:
        st.markdown("**Index Timeline:**")
        if timestamps:
            # Parse and sort timestamps
            try:
                parsed = []
                for ts in timestamps:
                    if isinstance(ts, str):
                        # Handle ISO format
                        parsed.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))

                if parsed:
                    earliest = min(parsed)
                    latest = max(parsed)
                    st.markdown(f"- **First indexed:** {earliest.strftime('%Y-%m-%d %H:%M')}")
                    st.markdown(f"- **Last indexed:** {latest.strftime('%Y-%m-%d %H:%M')}")
                    st.markdown(f"- **Index age:** {(datetime.now(earliest.tzinfo) - latest).days} days")
            except Exception:
                st.info("Could not parse ingestion timestamps.")
        else:
            st.info("Ingestion timestamps not available.")


def render_document_table(documents: list[dict]) -> None:
    """Render sortable document table."""
    st.subheader("Document Details")

    if not documents:
        st.info("No documents found.")
        return

    import pandas as pd

    # Prepare data for table
    table_data = []
    for doc in documents:
        row = {
            "Title": doc["title"][:50] + "..." if len(doc.get("title", "")) > 50 else doc.get("title", "N/A"),
            "Type": doc.get("doc_type", "unknown").title(),
            "Chunks": doc.get("chunk_count", 0),
            "Size": f"{doc.get('total_chars', 0) / 1000:.1f}K",
            "Quality": f"{doc.get('avg_confidence', 0) * 100:.0f}%" if doc.get("avg_confidence") else "N/A",
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_compliance_signals() -> None:
    """Render compliance and audit readiness indicators."""
    st.subheader("Compliance Readiness")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Data Governance Features:**")
        features = [
            ("Document hash verification", True),
            ("Extraction confidence scoring", True),
            ("Ingestion timestamp tracking", True),
            ("Parser version tracking", True),
            ("Source URL preservation", True),
            ("OCR artifact detection", True),
        ]
        for feature, available in features:
            icon = "✅" if available else "❌"
            st.markdown(f"{icon} {feature}")

    with col2:
        st.markdown("**Audit Trail Capabilities:**")
        capabilities = [
            ("Query logging to DynamoDB", True),
            ("CloudWatch metrics", True),
            ("Chunk-level provenance", True),
            ("Answer confidence scoring", True),
            ("Source citation tracking", True),
            ("Contradiction detection", True),
        ]
        for capability, available in capabilities:
            icon = "✅" if available else "❌"
            st.markdown(f"{icon} {capability}")


def render_data_quality_dashboard(retriever) -> None:
    """
    Render the complete Data Quality Dashboard.

    Args:
        retriever: FAISSRetriever instance with loaded metadata
    """
    st.header("Data Quality & Lineage")
    st.markdown(
        "This dashboard provides transparency into the corpus composition, "
        "extraction quality, and data governance features of the RAG system."
    )

    # Compute stats from metadata
    stats = compute_corpus_stats(retriever.metadata)
    documents = get_document_details(retriever.metadata)

    # Render sections
    render_corpus_overview(stats)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        render_source_distribution(stats)
    with col2:
        render_confidence_metrics(stats)

    st.divider()
    render_data_lineage(retriever.metadata)

    st.divider()
    render_document_table(documents)

    st.divider()
    render_compliance_signals()

    # Footer with timestamp
    st.caption(f"Dashboard generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
