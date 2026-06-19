"""
Stage 2: Multi-PDF Ingestion + Smart Chunking
------------------------------------------------
Loads all PDFs from a folder, tags each chunk with metadata
(source filename + page number), and splits text intelligently
using LangChain's RecursiveCharacterTextSplitter.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_pdfs(data_folder: str) -> list[Document]:
    """
    Loads every PDF in `data_folder` using PyPDFLoader.
    Each page becomes a separate Document object with metadata:
        - source: the PDF filename
        - page: page number (0-indexed by default)

    Returns a flat list of Document objects (one per page, across all PDFs).
    """
    all_documents = []

    pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{data_folder}'. Add some PDFs first.")

    print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")

    for filename in pdf_files:
        filepath = os.path.join(data_folder, filename)
        print(f"  Loading: {filename}")

        loader = PyPDFLoader(filepath)
        pages = loader.load()  # list of Document, one per page

        # Ensure clean, consistent metadata for every page
        for page_doc in pages:
            page_doc.metadata["source"] = filename
            # PyPDFLoader already sets "page", but we normalize to 1-indexed
            # for human-friendly citations later (page 1 instead of page 0)
            page_doc.metadata["page"] = page_doc.metadata.get("page", 0) + 1

        all_documents.extend(pages)

    print(f"Loaded {len(all_documents)} total pages across all PDFs.\n")
    return all_documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Splits documents into smaller chunks using recursive character splitting.

    Why RecursiveCharacterTextSplitter (the "smart" part):
    It tries to split on paragraph breaks first ("\\n\\n"), then sentences,
    then words, then characters as a last resort. This keeps semantically
    related text together as much as possible, instead of cutting mid-sentence.

    chunk_size=1000, chunk_overlap=200 is a strong default for RAG with Gemini:
    - Large enough to preserve context
    - Overlap ensures we don't lose meaning at chunk boundaries
      (e.g. a sentence that starts in chunk A and finishes in chunk B
      will still appear in full in at least one chunk)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Each chunk inherits the parent page's metadata (source, page) automatically.
    # We add a chunk_id for easier debugging/tracing later.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    print(f"Split into {len(chunks)} chunks (chunk_size=1000, overlap=200).\n")
    return chunks


def ingest_pdfs(data_folder: str = "data") -> list[Document]:
    """
    Full ingestion pipeline: load PDFs -> split into chunks.
    This is the function the rest of the app will import and call.
    """
    documents = load_pdfs(data_folder)
    chunks = split_documents(documents)
    return chunks


if __name__ == "__main__":
    # Quick manual test: run `python ingest.py` from inside src/
    # after placing PDFs in the ../data folder.
    chunks = ingest_pdfs(data_folder="../data")

    print("--- Sample chunk ---")
    print("Content preview:", chunks[0].page_content[:300])
    print("Metadata:", chunks[0].metadata)
