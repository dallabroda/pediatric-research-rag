"""
Chat interface component for the Streamlit app.
"""
import streamlit as st


def render_chat() -> None:
    """Render the chat interface with message history."""
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if available
            if message.get("sources"):
                with st.expander("View Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- **{source['doc_title']}** (score: {source['score']:.3f})")
                        if source.get("source_url"):
                            st.markdown(f"  [{source['source_url']}]({source['source_url']})")


def add_user_message(content: str) -> None:
    """Add a user message to the chat history."""
    st.session_state.messages.append({
        "role": "user",
        "content": content,
    })


def add_assistant_message(content: str, sources: list[dict] = None) -> None:
    """Add an assistant message to the chat history."""
    st.session_state.messages.append({
        "role": "assistant",
        "content": content,
        "sources": sources or [],
    })


def clear_chat() -> None:
    """Clear the chat history."""
    st.session_state.messages = []
