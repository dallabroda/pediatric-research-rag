"""
Citations display component.
"""
import streamlit as st


def render_citations(sources: list[dict]) -> None:
    """
    Render source citations in a formatted display.

    Args:
        sources: List of source metadata dicts
    """
    if not sources:
        return

    st.subheader("Sources")

    for i, source in enumerate(sources, 1):
        title = source.get("doc_title", source.get("title", "Unknown"))
        doc_type = source.get("doc_type", "document")
        score = source.get("score", 0)
        url = source.get("source_url")
        section = source.get("section")
        page = source.get("page")

        # Create citation card
        with st.container():
            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"**[{i}] {title}**")

                details = []
                if doc_type:
                    details.append(f"Type: {doc_type}")
                if section:
                    details.append(f"Section: {section}")
                if page:
                    details.append(f"Page: {page}")

                if details:
                    st.caption(" | ".join(details))

            with col2:
                st.caption(f"Score: {score:.3f}")

            if url:
                st.markdown(f"[View Source]({url})")

            st.divider()


def render_inline_citations(answer: str, sources: list[dict]) -> str:
    """
    Add inline citation links to answer text.

    This is a simple implementation that looks for [Source: ...] patterns
    and converts them to links.

    Args:
        answer: Answer text with citations
        sources: List of source metadata

    Returns:
        Answer with formatted citations
    """
    # For now, just return the answer as-is
    # A more sophisticated implementation would parse and link citations
    return answer
