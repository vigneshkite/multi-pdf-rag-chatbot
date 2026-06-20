"""
Stage 4b: Conversational RAG Chain
------------------------------------------------
Ties everything together:
  1. Rewrites follow-up questions into standalone questions using chat history
  2. Retrieves candidates via hybrid retriever
  3. Re-ranks candidates via cross-encoder
  4. Generates a final answer with Gemini, citing sources

This is structured as a class so the app (Stage 5) can hold one
instance per session and just call .ask(question).
"""

from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from rerank import rerank_documents

# Explicitly locate .env at the project root (one level above src/),
# regardless of which folder this script is run from.
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


# Prompt to rewrite a follow-up question into a standalone one,
# using chat history for context.
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation history and a follow-up question, rewrite the "
     "follow-up question to be a standalone question that contains all "
     "necessary context. Do NOT answer the question, only rewrite it. "
     "If the question is already standalone, return it unchanged."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# Prompt for the final answer generation, instructed to cite sources
# and to refuse gracefully when the context doesn't contain the answer.
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant answering questions based ONLY on the "
     "provided document excerpts. Each excerpt is labeled with its source "
     "file and page number.\n\n"
     "Rules:\n"
     "- Answer using only the given context. Do not use outside knowledge.\n"
     "- If the context doesn't contain the answer, say so clearly — do not guess.\n"
     "- After your answer, list the sources you used in this format:\n"
     "  Sources: [filename, page X], [filename, page Y]\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


class ConversationalRAGChain:
    def __init__(self, hybrid_retriever, model_name: str = "gemini-2.5-flash", rerank_top_n: int = 4):
        """
        hybrid_retriever: the EnsembleRetriever from retrieval.py
        model_name: Gemini model to use for both query rewriting and answering.
                    gemini-2.5-flash is fast, cheap, and current as of mid-2026
                    (the older gemini-1.5-flash has been fully deprecated/shut down).
                    Use gemini-2.5-pro for harder reasoning if needed.
        rerank_top_n: how many chunks survive re-ranking before going to the LLM
        """
        self.retriever = hybrid_retriever
        self.rerank_top_n = rerank_top_n
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
        self.chat_history: list = []  # list of HumanMessage / AIMessage

        self.condense_chain = CONDENSE_QUESTION_PROMPT | self.llm | StrOutputParser()
        self.answer_chain = ANSWER_PROMPT | self.llm | StrOutputParser()

    def _format_context(self, docs) -> str:
        """Formats re-ranked chunks into a labeled context block for the prompt."""
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            formatted.append(f"[{source}, page {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def ask(self, question: str) -> dict:
        """
        Main entry point. Returns a dict with:
          - answer: the generated answer text (includes cited sources)
          - sources: list of (source, page) tuples actually used
        """
        # Step 1: rewrite follow-up into standalone question, if there's history
        if self.chat_history:
            standalone_question = self.condense_chain.invoke({
                "chat_history": self.chat_history,
                "question": question,
            })
        else:
            standalone_question = question

        print(f"[DEBUG] Standalone question sent to retriever: {repr(standalone_question)}")

        # Step 2: hybrid retrieval, with retries for transient Gemini API errors.
        # NOTE: Google's free-tier API has a low requests-per-minute limit.
        # When you exceed it, Google sometimes returns a generic 500 INTERNAL
        # error instead of a proper 429 Too Many Requests — so we treat any
        # failure here as "possibly rate limited" and back off accordingly.
        import time
        max_attempts = 5
        candidates = None
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                candidates = self.retriever.invoke(standalone_question)
                break
            except Exception as e:
                last_error = e
                wait = 5 * attempt  # 5s, 10s, 15s, 20s, 25s — longer waits for rate limits
                print(f"[DEBUG] Retrieval attempt {attempt}/{max_attempts} failed: {e}")
                if attempt < max_attempts:
                    print(f"[DEBUG] Waiting {wait}s before retrying (possible rate limit)...")
                    time.sleep(wait)

        if candidates is None:
            raise RuntimeError(
                f"Retrieval failed after {max_attempts} attempts. "
                f"This is likely the Gemini free-tier rate limit (requests per minute). "
                f"Wait about a minute before asking another question. "
                f"Last error: {last_error}"
            )

        # Step 3: re-rank candidates, keep only the best
        top_docs = rerank_documents(standalone_question, candidates, top_n=self.rerank_top_n)

        # Step 4: generate the final answer with citations
        context = self._format_context(top_docs)
        answer = self.answer_chain.invoke({
            "context": context,
            "chat_history": self.chat_history,
            "question": question,
        })

        # Update memory
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        sources = [(doc.metadata.get("source"), doc.metadata.get("page")) for doc in top_docs]

        return {"answer": answer, "sources": sources}

    def reset_memory(self):
        """Clears conversation history (e.g. when user starts a new chat)."""
        self.chat_history = []


if __name__ == "__main__":
    # End-to-end manual test
    from ingest import ingest_pdfs
    from retrieval import build_vector_store, build_hybrid_retriever

    chunks = ingest_pdfs(data_folder="../data")
    vector_store = build_vector_store(chunks)
    hybrid_retriever = build_hybrid_retriever(chunks, vector_store)

    rag_chain = ConversationalRAGChain(hybrid_retriever)

    print("\n=== Ask questions (type 'exit' to quit) ===")
    while True:
        q = input("\nYou: ")
        if q.lower() == "exit":
            break
        try:
            result = rag_chain.ask(q)
            print(f"\nBot: {result['answer']}")
        except Exception as e:
            # Don't let one flaky API call kill the whole session —
            # just report it and let the user try again.
            print(f"\n[Error] That question failed: {e}")
            print("This is usually a temporary Gemini API hiccup — just try asking again.")