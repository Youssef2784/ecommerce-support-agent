"""RAG pipeline — retrieves context and generates grounded answers.

Used by the Policy & Returns agent for agentic RAG: the agent decides
whether retrieval is needed, retrieves if so, then answers grounded
in the retrieved chunks.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document

from src.rag.ingest import load_vectorstore
from src.rag.retrievers import Strategy, retrieve

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")


def rag_query(
    question: str,
    strategy: Strategy = "rerank",
    k: int = 5,
    metadata_filter: dict | None = None,
    corpus_chunks: list[Document] | None = None,
) -> dict:
    """Run a RAG query: retrieve context, then generate a grounded answer.

    Returns:
        dict with keys: answer, contexts (list of chunk texts), strategy
    """
    vectorstore = load_vectorstore()

    # Retrieve relevant chunks
    docs = retrieve(
        query=question,
        vectorstore=vectorstore,
        strategy=strategy,
        k=k,
        metadata_filter=metadata_filter,
        corpus_chunks=corpus_chunks,
    )

    contexts = [doc.page_content for doc in docs]
    context_str = "\n\n---\n\n".join(contexts)

    # Generate answer grounded in retrieved context
    answer = _generate_grounded_answer(question, context_str)

    return {
        "answer": answer,
        "contexts": contexts,
        "strategy": strategy,
        "num_chunks": len(docs),
    }


def _generate_grounded_answer(question: str, context: str) -> str:
    """Generate an answer using the LLM, grounded in retrieved context."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0,
    )

    prompt = f"""You are a TechMart customer support agent. Answer the customer's question
using ONLY the information provided in the context below. If the context does not contain
enough information to answer, say so honestly — do not make up information.

Be concise, helpful, and cite specific policy details (numbers, timeframes, costs) when available.

Context:
{context}

Customer question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content


def get_corpus_chunks() -> list[Document]:
    """Load all chunks from the vectorstore for BM25 indexing.

    Call once and pass to retrieve() to avoid reloading per query.
    """
    vectorstore = load_vectorstore()
    # Get all documents from Chroma
    results = vectorstore.get(include=["documents", "metadatas"])
    chunks = []
    for doc_text, metadata in zip(results["documents"], results["metadatas"]):
        chunks.append(Document(page_content=doc_text, metadata=metadata or {}))
    logger.info(f"Loaded {len(chunks)} chunks for BM25 corpus")
    return chunks
