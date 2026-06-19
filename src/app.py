"""
Stage 5: Streamlit App
------------------------------------------------
The user-facing web app. Lets users upload multiple PDFs, builds the
advanced RAG pipeline (hybrid retrieval + re-ranking), and provides
a conversational chat interface with source citations.

Run with: streamlit run app.py
"""

import os
import shutil
import streamlit as st

from ingest import ingest_pdfs
from retrieval import build_vector_store, build_hybrid_retriever
from rag_chain import ConversationalRAGChain

DATA_DIR = "../data"
CHROMA_DIR = "../chroma_db"

st.set_page_config(page_title="Multi-PDF RAG Chatbot", page_icon="📚", layout="wide")


def init_session_state():
    """Sets up persistent state across Streamlit reruns."""
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "messages" not in st.session_state:
        st.session_state.messages = []  # for displaying chat in the UI
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []


def save_uploaded_files(uploaded_files) -> list[str]:
    """Saves uploaded PDFs to the data folder, returns list of filenames saved."""
    os.makedirs(DATA_DIR, exist_ok=True)
    saved_names = []
    for uploaded_file in uploaded_files:
        filepath = os.path.join(DATA_DIR, uploaded_file.name)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_names.append(uploaded_file.name)
    return saved_names


def build_pipeline():
    """
    Runs the full pipeline: ingest -> embed -> hybrid retriever -> RAG chain.
    Called only when PDFs are newly uploaded/changed, not on every interaction.
    """
    with st.spinner("Reading and chunking PDFs..."):
        chunks = ingest_pdfs(data_folder=DATA_DIR)

    with st.spinner("Embedding chunks with Gemini (this may take a moment)..."):
        # Fresh build each time files change; for a persistent-across-restarts
        # version, you could check CHROMA_DIR and call load_existing_vector_store
        # instead, as long as the PDF set hasn't changed.
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        vector_store = build_vector_store(chunks, persist_directory=CHROMA_DIR)

    with st.spinner("Setting up hybrid retrieval..."):
        hybrid_retriever = build_hybrid_retriever(chunks, vector_store)

    rag_chain = ConversationalRAGChain(hybrid_retriever)
    return rag_chain


def main():
    init_session_state()

    st.title("📚 Multi-PDF RAG Chatbot")
    st.caption("Powered by Gemini + LangChain — hybrid retrieval, re-ranking, source citations")

    # ---------------- Sidebar: PDF upload ----------------
    with st.sidebar:
        st.header("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        process_clicked = st.button("Process PDFs", type="primary", use_container_width=True)

        if process_clicked:
            if not uploaded_files:
                st.warning("Please upload at least one PDF first.")
            else:
                saved_names = save_uploaded_files(uploaded_files)
                st.session_state.rag_chain = build_pipeline()
                st.session_state.processed_files = saved_names
                st.session_state.messages = []  # reset chat on new PDF set
                st.success(f"Processed {len(saved_names)} file(s): {', '.join(saved_names)}")

        if st.session_state.processed_files:
            st.divider()
            st.subheader("Active documents")
            for name in st.session_state.processed_files:
                st.text(f"📄 {name}")

            if st.button("Clear chat history", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.rag_chain:
                    st.session_state.rag_chain.reset_memory()
                st.rerun()

    # ---------------- Main: Chat interface ----------------
    if st.session_state.rag_chain is None:
        st.info("👈 Upload PDFs and click 'Process PDFs' to get started.")
        return

    # Display existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📖 Sources"):
                    for source, page in msg["sources"]:
                        st.text(f"{source} — page {page}")

    # Chat input
    user_question = st.chat_input("Ask a question about your PDFs...")

    if user_question:
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate and show assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.rag_chain.ask(user_question)
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("📖 Sources"):
                        for source, page in result["sources"]:
                            st.text(f"{source} — page {page}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })


if __name__ == "__main__":
    main()
