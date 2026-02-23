"""
Sidebar component for document explorer.
"""
import streamlit as st


def render_sidebar(documents: list[dict]) -> str | None:
    """
    Render document list for sidebar.

    Args:
        documents: List of document metadata dicts

    Returns:
        Selected document ID or None
    """
    # Document counts
    papers = [d for d in documents if d.get("doc_type") == "paper"]
    trials = [d for d in documents if d.get("doc_type") == "trial"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Papers", len(papers))
    with col2:
        st.metric("Trials", len(trials))

    st.divider()

    # Filter options
    doc_type_filter = st.selectbox(
        "Filter by type",
        options=["All", "Papers", "Trials"],
        index=0,
        label_visibility="collapsed",
    )

    # Filter documents
    if doc_type_filter == "Papers":
        filtered_docs = papers
    elif doc_type_filter == "Trials":
        filtered_docs = trials
    else:
        filtered_docs = documents

    # Document list (scrollable)
    selected_doc = None

    for doc in filtered_docs[:20]:  # Limit to 20 for performance
        doc_id = doc.get("document_id", "")
        title = doc.get("title", doc_id)
        doc_type = doc.get("doc_type", "unknown")

        # Truncate long titles
        display_title = title[:35] + "..." if len(title) > 35 else title
        icon = "📄" if doc_type == "paper" else "🧪"

        if st.button(f"{icon} {display_title}", key=f"doc_{doc_id}", use_container_width=True):
            selected_doc = doc_id

    if len(filtered_docs) > 20:
        st.caption(f"Showing 20 of {len(filtered_docs)} documents")

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    return selected_doc
