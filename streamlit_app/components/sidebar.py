"""
Sidebar component for document explorer.
"""
import streamlit as st


def render_sidebar(documents: list[dict]) -> str | None:
    """
    Render the sidebar with document explorer.

    Args:
        documents: List of document metadata dicts

    Returns:
        Selected document ID or None
    """
    with st.sidebar:
        st.header("Research Documents")

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
        )

        # Filter documents
        if doc_type_filter == "Papers":
            filtered_docs = papers
        elif doc_type_filter == "Trials":
            filtered_docs = trials
        else:
            filtered_docs = documents

        st.divider()

        # Document list
        st.subheader("Documents")

        selected_doc = None

        for doc in filtered_docs:
            doc_id = doc.get("document_id", "")
            title = doc.get("title", doc_id)
            doc_type = doc.get("doc_type", "unknown")

            # Truncate long titles
            display_title = title[:40] + "..." if len(title) > 40 else title

            # Create clickable expander for each doc
            with st.expander(f"{doc_id}", expanded=False):
                st.markdown(f"**{display_title}**")
                st.caption(f"Type: {doc_type}")

                if doc.get("source_url"):
                    st.markdown(f"[View Source]({doc['source_url']})")

                if doc.get("chunks_count"):
                    st.caption(f"Chunks: {doc['chunks_count']}")

                if st.button("View Details", key=f"view_{doc_id}"):
                    selected_doc = doc_id

        st.divider()

        # Clear chat button
        if st.button("Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.rerun()

        # Settings
        with st.expander("Settings"):
            top_k = st.slider(
                "Results per query",
                min_value=1,
                max_value=10,
                value=5,
            )
            st.session_state.top_k = top_k

    return selected_doc
