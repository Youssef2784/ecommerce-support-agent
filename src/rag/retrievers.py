"""Swappable retrieval strategies behind a common interface.

All retrievers return List[Document] given a query string.
Strategies:
  - naive:    dense top-k from Chroma (baseline)
  - hybrid:   BM25 (lexical) + dense, merged via Reciprocal Rank Fusion
  - rerank:   hybrid candidates -> cross-encoder reranker
  - metadata: dense top-k filtered by doc_type metadata

The eval harness swaps strategies while keeping everything else fixed,
so the RAGAS delta is attributable to retrieval strategy alone.
"""

import logging
from typing import Literal

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

Strategy = Literal["naive", "hybrid", "rerank", "metadata"]


def retrieve(
    query: str,
    vectorstore,
    strategy: Strategy = "naive",
    k: int = 5,
    metadata_filter: dict | None = None,
    corpus_chunks: list[Document] | None = None,
) -> list[Document]:
    """Retrieve documents using the specified strategy.

    Args:
        query: User question.
        vectorstore: Chroma vectorstore instance.
        strategy: One of "naive", "hybrid", "rerank", "metadata".
        k: Number of results to return.
        metadata_filter: For "metadata" strategy — e.g. {"doc_type": "policy"}.
        corpus_chunks: Full chunk list needed for BM25 in hybrid/rerank strategies.
                       Pre-load once and pass in to avoid reloading per query.
    """
    if strategy == "naive":
        return _naive_retrieve(query, vectorstore, k)
    elif strategy == "hybrid":
        return _hybrid_retrieve(query, vectorstore, k, corpus_chunks)
    elif strategy == "rerank":
        return _rerank_retrieve(query, vectorstore, k, corpus_chunks)
    elif strategy == "metadata":
        return _metadata_retrieve(query, vectorstore, k, metadata_filter)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _naive_retrieve(query: str, vectorstore, k: int) -> list[Document]:
    """Dense embedding similarity search — the baseline."""
    results = vectorstore.similarity_search(query, k=k)
    logger.debug(f"Naive: returned {len(results)} docs")
    return results


def _metadata_retrieve(
    query: str, vectorstore, k: int, metadata_filter: dict | None
) -> list[Document]:
    """Dense search with metadata pre-filter (e.g., restrict to policy docs only)."""
    if metadata_filter is None:
        metadata_filter = {}
    results = vectorstore.similarity_search(query, k=k, filter=metadata_filter)
    logger.debug(f"Metadata-filtered: returned {len(results)} docs with filter {metadata_filter}")
    return results


def _hybrid_retrieve(
    query: str, vectorstore, k: int, corpus_chunks: list[Document] | None
) -> list[Document]:
    """BM25 (lexical) + dense (semantic), merged via Reciprocal Rank Fusion.

    RRF assigns each doc a score of 1/(rank + 60) from each ranker,
    then sums scores. The constant 60 is standard (from the RRF paper).
    This balances exact keyword matches (BM25) with semantic similarity (dense).
    """
    if corpus_chunks is None:
        raise ValueError("hybrid strategy requires corpus_chunks for BM25")

    # Dense results
    dense_results = vectorstore.similarity_search(query, k=k * 2)

    # BM25 results
    bm25_results = _bm25_search(query, corpus_chunks, k=k * 2)

    # Reciprocal Rank Fusion
    fused = _reciprocal_rank_fusion(
        rankings=[dense_results, bm25_results],
        k=k,
    )
    logger.debug(f"Hybrid: fused {len(fused)} docs from dense + BM25")
    return fused


def _rerank_retrieve(
    query: str, vectorstore, k: int, corpus_chunks: list[Document] | None
) -> list[Document]:
    """Hybrid retrieval + cross-encoder reranking.

    Step 1: Get candidates from hybrid search (wider net, k*3).
    Step 2: Score each candidate with a cross-encoder model.
    Step 3: Return top-k by cross-encoder score.

    The cross-encoder sees (query, passage) pairs jointly, so it captures
    nuances that bi-encoder similarity misses — the key quality lever.
    """
    # Get more candidates than needed, then rerank to top-k
    candidates = _hybrid_retrieve(query, vectorstore, k=k * 3, corpus_chunks=corpus_chunks)

    if not candidates:
        return []

    reranked = _cross_encoder_rerank(query, candidates, k=k)
    logger.debug(f"Rerank: {len(candidates)} candidates -> {len(reranked)} reranked")
    return reranked


# ---------------------------------------------------------------------------
# BM25 lexical search
# ---------------------------------------------------------------------------

_bm25_index = None
_bm25_corpus = None


def _bm25_search(query: str, corpus_chunks: list[Document], k: int) -> list[Document]:
    """BM25 keyword search over the corpus chunks."""
    global _bm25_index, _bm25_corpus
    from rank_bm25 import BM25Okapi

    # Build/cache the BM25 index (tokenize on whitespace + lowercase)
    if _bm25_corpus is not corpus_chunks:
        tokenized = [doc.page_content.lower().split() for doc in corpus_chunks]
        _bm25_index = BM25Okapi(tokenized)
        _bm25_corpus = corpus_chunks

    tokenized_query = query.lower().split()
    scores = _bm25_index.get_scores(tokenized_query)

    # Get top-k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [corpus_chunks[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    rankings: list[list[Document]], k: int, rrf_k: int = 60
) -> list[Document]:
    """Merge multiple ranked lists using RRF (Cormack et al., 2009).

    Score = sum over rankers of 1 / (rank + rrf_k).
    Higher rrf_k reduces the influence of high-ranked items.
    """
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            # Use page_content as identity key (chunks are unique by content)
            doc_id = doc.page_content[:200]
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (rank + rrf_k)
            doc_map[doc_id] = doc

    sorted_ids = sorted(doc_scores, key=lambda x: doc_scores[x], reverse=True)[:k]
    return [doc_map[doc_id] for doc_id in sorted_ids]


# ---------------------------------------------------------------------------
# Cross-encoder reranking
# ---------------------------------------------------------------------------

_cross_encoder = None


def _cross_encoder_rerank(query: str, docs: list[Document], k: int) -> list[Document]:
    """Rerank documents using a cross-encoder model.

    Uses ms-marco-MiniLM-L-6-v2 — a lightweight cross-encoder trained on
    MS MARCO passage ranking. It scores (query, passage) pairs jointly,
    capturing token-level interactions that bi-encoders miss.
    """
    global _cross_encoder
    from sentence_transformers import CrossEncoder

    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Loaded cross-encoder model: ms-marco-MiniLM-L-6-v2")

    # Score each (query, doc) pair
    pairs = [(query, doc.page_content) for doc in docs]
    scores = _cross_encoder.predict(pairs)

    # Sort by score descending, take top-k
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:k]]
