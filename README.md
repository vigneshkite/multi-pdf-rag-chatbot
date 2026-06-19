# Multi-PDF RAG Chatbot using Gemini & LangChain

A Retrieval-Augmented Generation (RAG) chatbot that answers questions across multiple PDF documents, built with an advanced retrieval pipeline rather than a basic vector-search-only approach.

## Features

- **Multi-PDF ingestion** with page-level metadata tracking for accurate source citation
- **Smart recursive chunking** that preserves sentence/paragraph boundaries instead of cutting text arbitrarily
- **Hybrid retrieval**: combines semantic vector search (Gemini embeddings + ChromaDB) with keyword search (BM25) via a weighted Ensemble Retriever
- **Cross-encoder re-ranking** (`BAAI/bge-reranker-base`) to precisely score and filter retrieved chunks before generation, reducing noise and hallucination
- **Conversational memory** with history-aware query rewriting, so follow-up questions work naturally
- **Source-cited answers** — every response references the specific PDF and page it came from
- **Streamlit web UI** for PDF upload and chat

## Architecture

```
PDFs --> Chunking --> [Vector Store (Chroma) + BM25 Index]
                              |
                      Hybrid Retrieval (Ensemble)
                              |
                      Cross-Encoder Re-ranking
                              |
                    Gemini (answer generation + citations)
                              |
                         Streamlit UI
```

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Google Gemini (gemini-1.5-flash) |
| Embeddings | Gemini Embeddings (models/embedding-001) |
| Orchestration | LangChain |
| Vector DB | ChromaDB |
| Keyword Search | BM25 (rank_bm25) |
| Re-ranking | Cross-encoder (sentence-transformers) |
| UI | Streamlit |

## Setup

1. Clone the repo and create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   source venv/bin/activate  # Mac/Linux
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey) and add it to a `.env` file:
   ```
   GOOGLE_API_KEY=your_key_here
   ```

4. Run the app:
   ```bash
   cd src
   streamlit run app.py
   ```

5. Upload PDFs in the sidebar, click "Process PDFs," and start chatting.

## Why Hybrid Retrieval + Re-ranking?

Standard RAG tutorials use vector similarity search alone, which struggles with exact keyword/term matches (codes, names, specific figures). This project combines:
- **Vector search** for semantic understanding
- **BM25** for precise keyword matching
- **Cross-encoder re-ranking** to filter the combined candidates down to the most relevant chunks before they reach the LLM

This mirrors retrieval techniques used in production-grade RAG systems, improving answer accuracy and reducing hallucination compared to single-method retrieval.

## Project Structure

```
multi-pdf-rag-chatbot/
├── data/                # uploaded PDFs (gitignored)
├── chroma_db/           # persisted vector store (gitignored)
├── src/
│   ├── ingest.py         # PDF loading + chunking
│   ├── retrieval.py       # embeddings + hybrid retriever
│   ├── rerank.py          # cross-encoder re-ranking
│   ├── rag_chain.py       # conversational RAG chain
│   └── app.py             # Streamlit UI
├── .env                 # API key (gitignored)
├── .gitignore
└── requirements.txt
```

## Future Improvements

- Cache embeddings across sessions to avoid re-processing unchanged PDFs
- Add a confidence/relevance score display per answer
- Support additional file types (DOCX, TXT)
- Deploy to Streamlit Community Cloud or HuggingFace Spaces
