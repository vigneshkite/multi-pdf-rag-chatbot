"""
Stage 3: Embeddings + Hybrid Retrieval
------------------------------------------------
Builds:
  1. A Chroma vector store using Gemini embeddings (semantic search)
  2. A BM25 retriever (keyword search)
  3. An EnsembleRetriever that blends both (hybrid search)

The vector store is persisted to disk, so you don't need to
re-embed your PDFs every time you restart the app.
"""

import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document

load_dotenv()  # reads GOOGLE_API_KEY from .env

CHROMA_PERSIST_DIR = "../chroma_db"


def get_embeddings_model():
    """
    Gemini's embedding model. 'models/embedding-001' converts text
    into a high-dimensional vector capturing semantic meaning.
    """
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def build_vector_store(chunks: list[Document], persist_directory: str = CHROMA_PERSIST_DIR) -> Chroma:
    """
    Embeds all chunks and stores them in a local Chroma vector database.

    persist_directory: Chroma saves to disk here. On future runs, you can
    load the existing store instead of re-embedding (saves time + API calls).
    """
    embeddings = get_embeddings_model()

    print(f"Embedding {len(chunks)} chunks with Gemini... (this calls the API, may take a bit)")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    print(f"Vector store built and persisted to '{persist_directory}'.\n")
    return vector_store


def load_existing_vector_store(persist_directory: str = CHROMA_PERSIST_DIR) -> Chroma:
    """
    Loads a previously-built Chroma store from disk without re-embedding.
    Use this on app restarts if PDFs haven't changed.
    """
    embeddings = get_embeddings_model()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def build_bm25_retriever(chunks: list[Document], k: int = 5) -> BM25Retriever:
    """
    BM25 is a classic keyword-ranking algorithm (the same family of
    algorithm traditional search engines used before vector search existed).
    It scores chunks by term frequency, rewarding exact/near-exact word matches.

    k = number of top chunks to return per query.
    """
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k
    return bm25_retriever


def build_hybrid_retriever(
    chunks: list[Document],
    vector_store: Chroma,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    k: int = 5,
) -> EnsembleRetriever:
    """
    Combines vector search + BM25 into one retriever using weighted
    reciprocal rank fusion under the hood.

    vector_weight / bm25_weight: how much to trust each method.
    0.6/0.4 favors semantic search slightly, since most natural-language
    questions benefit more from meaning-based matching, while still letting
    BM25 catch exact keyword hits vector search might miss.

    Tune these based on your documents:
      - Technical docs with lots of codes/IDs -> raise bm25_weight
      - Narrative/conceptual docs -> raise vector_weight
    """
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    bm25_retriever = build_bm25_retriever(chunks, k=k)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[vector_weight, bm25_weight],
    )

    print(f"Hybrid retriever ready (vector_weight={vector_weight}, bm25_weight={bm25_weight}).\n")
    return ensemble_retriever


if __name__ == "__main__":
    # Quick manual test: run from inside src/, after ingest.py works.
    from ingest import ingest_pdfs

    chunks = ingest_pdfs(data_folder="../data")
    vector_store = build_vector_store(chunks)
    hybrid_retriever = build_hybrid_retriever(chunks, vector_store)

    # Test query
    test_query = "What is this document about?"
    results = hybrid_retriever.invoke(test_query)

    print(f"--- Top results for: '{test_query}' ---")
    for i, doc in enumerate(results):
        print(f"\n[{i+1}] Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}")
        print(doc.page_content[:200])
