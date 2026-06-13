"""Policy & Returns Agent — retrieves and interprets refund/return eligibility.

Uses Agentic RAG: decides whether retrieval is needed (vs. answering from
conversation context), retrieves relevant policy/FAQ chunks, and generates
a grounded answer citing specific policy details.

Falls back to canned responses if RAG dependencies are unavailable (e.g., no
Chroma DB or no API key), so the graph remains runnable for demos.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage

from src.graph.state import State

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env")
_CHROMA_DIR = _PROJECT_ROOT / "chroma_db"


def policy_returns_node(state: State) -> dict:
    """Respond to policy/returns questions using RAG over policy documents.

    Agentic RAG logic:
    1. Check if the question can be answered from existing retrieved_context.
    2. If not, retrieve relevant chunks using the configured strategy.
    3. Generate a grounded answer from retrieved context.
    """
    last_message = state["messages"][-1].content
    logger.info("PolicyReturns: processing policy/returns query")

    # Check if RAG is available (Chroma DB exists and API key is set)
    if not _rag_available():
        logger.warning("PolicyReturns: RAG not available, using fallback")
        return _fallback_response(last_message)

    try:
        reply, contexts = _rag_answer(last_message)
        return {
            "messages": [AIMessage(content=reply)],
            "retrieved_context": contexts,
            "resolved": True,
        }
    except Exception as e:
        logger.error(f"PolicyReturns: RAG failed ({e}), using fallback")
        return _fallback_response(last_message)


def _rag_available() -> bool:
    """Check if RAG infrastructure is ready."""
    return _CHROMA_DIR.exists() and os.getenv("OPENAI_API_KEY")


def _rag_answer(question: str) -> tuple[str, list[str]]:
    """Retrieve context and generate a grounded answer."""
    from src.rag.ingest import load_vectorstore
    from src.rag.retrievers import retrieve

    vectorstore = load_vectorstore()

    # Use metadata filter for policy questions to narrow the search space.
    # For product questions, search the full corpus.
    question_lower = question.lower()
    metadata_filter = None
    strategy = "rerank"

    # Decide whether to use metadata filtering based on intent signals
    if any(kw in question_lower for kw in ["ship", "delivery", "tracking"]):
        metadata_filter = {"doc_type": "policy"}
    elif any(kw in question_lower for kw in ["return", "refund", "exchange", "warranty"]):
        metadata_filter = {"doc_type": "policy"}

    # For hybrid/rerank strategies, we need the full corpus for BM25
    corpus_chunks = None
    if strategy in ("hybrid", "rerank"):
        from src.rag.pipeline import get_corpus_chunks
        corpus_chunks = get_corpus_chunks()

    # If metadata filter is set, use metadata strategy; otherwise use rerank
    if metadata_filter:
        docs = retrieve(question, vectorstore, "metadata", k=5, metadata_filter=metadata_filter)
        # Supplement with reranked results from full corpus if we got few results
        if len(docs) < 3:
            docs = retrieve(question, vectorstore, strategy, k=5, corpus_chunks=corpus_chunks)
    else:
        docs = retrieve(question, vectorstore, strategy, k=5, corpus_chunks=corpus_chunks)

    contexts = [doc.page_content for doc in docs]
    context_str = "\n\n---\n\n".join(contexts)

    # Generate grounded answer
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)

    prompt = f"""You are a TechMart customer support agent. Answer the customer's question
using ONLY the information in the retrieved policy/product context below.

Rules:
- Cite specific numbers, timeframes, and conditions from the policy.
- If the context doesn't contain the answer, say "I don't have that information in our
  current policies" — never make up details.
- Be concise and helpful.

Context:
{context_str}

Customer question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content, contexts


def _fallback_response(message: str) -> dict:
    """Canned fallback when RAG is unavailable (keeps graph runnable)."""
    msg_lower = message.lower()
    if "refund" in msg_lower or "money back" in msg_lower:
        reply = (
            "Our refund policy allows returns within 30 days of delivery for a full refund, "
            "provided the item is in its original condition. Refunds are processed within "
            "5-7 business days after we receive the returned item."
        )
    elif "exchange" in msg_lower or "replacement" in msg_lower:
        reply = (
            "We offer free exchanges within 30 days of delivery. If the replacement item "
            "costs more, you'll be charged the difference."
        )
    else:
        reply = (
            "I can help with returns and policy questions! Our standard return window is "
            "30 days from delivery. Could you tell me more about what you need?"
        )

    return {
        "messages": [AIMessage(content=reply)],
        "resolved": True,
    }
