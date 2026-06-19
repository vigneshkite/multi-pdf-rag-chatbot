"""
Stage 4a: Re-ranking
------------------------------------------------
Takes the candidate chunks from hybrid retrieval and re-scores them
using a cross-encoder model, which evaluates (query, chunk) pairs
together for much more precise relevance than embedding similarity alone.
"""

from sentence_transformers import CrossEncoder
from langchain.schema import Document

# Loaded once and reused (loading the model is the slow part, ~few seconds)
_cross_encoder = None


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print("Loading cross-encoder re-ranking model (first time only)...")
        # bge-reranker-base is small, fast, and strong for general-purpose re-ranking
        _cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
    return _cross_encoder


def rerank_documents(
    query: str,
    documents: list[Document],
    top_n: int = 4,
) -> list[Document]:
    """
    Re-scores retrieved documents against the query using a cross-encoder,
    then returns only the top_n most relevant ones.

    Why this works better than trusting retrieval order alone:
    Embedding-based retrieval scores query and document independently,
    then compares vectors. A cross-encoder feeds the query and document
    text TOGETHER into the model, letting it directly judge "does this
    chunk actually answer this question?" — much more accurate, but slower,
    which is why we only run it on the small candidate set from retrieval,
    not the entire document collection.
    """
    if not documents:
        return []

    cross_encoder = get_cross_encoder()

    # Build (query, chunk_text) pairs for scoring
    pairs = [(query, doc.page_content) for doc in documents]
    scores = cross_encoder.predict(pairs)

    # Pair each document with its score, sort descending, take top_n
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    top_docs = [doc for doc, score in scored_docs[:top_n]]

    print(f"Re-ranked {len(documents)} candidates -> kept top {len(top_docs)}.")
    for doc, score in scored_docs[:top_n]:
        print(f"  score={score:.3f} | {doc.metadata.get('source')} p.{doc.metadata.get('page')}")

    return top_docs


if __name__ == "__main__":
    # Quick manual test — requires retrieval.py to be working first
    from ingest import ingest_pdfs
    from retrieval import build_vector_store, build_hybrid_retriever

    chunks = ingest_pdfs(data_folder="../data")
    vector_store = build_vector_store(chunks)
    hybrid_retriever = build_hybrid_retriever(chunks, vector_store)

    query = "What is this document about?"
    candidates = hybrid_retriever.invoke(query)
    print(f"\nRetrieved {len(candidates)} candidates before re-ranking.\n")

    top_docs = rerank_documents(query, candidates, top_n=4)

    print("\n--- Final re-ranked chunks ---")
    for doc in top_docs:
        print(f"\nSource: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}")
        print(doc.page_content[:200])
