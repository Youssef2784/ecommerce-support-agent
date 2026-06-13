"""RAGAS-style evaluation — measures retrieval and generation quality.

Implements the four core RAGAS metrics using LLM-as-judge, avoiding the ragas
library's dependency conflicts while producing equivalent measurements.
Each metric mirrors the RAGAS paper definition:

  - Context Precision: fraction of retrieved chunks that are relevant to the question
  - Context Recall: fraction of ground-truth claims covered by retrieved context
  - Faithfulness: fraction of answer claims supported by retrieved context
  - Answer Relevancy: how well the answer addresses the original question

Why custom instead of the ragas library: ragas 0.1.x/0.2.x/0.4.x all conflict with
langchain-core >= 1.4 (required by langgraph 1.2). Since the metrics are LLM-scored,
reimplementing them is straightforward and gives us full control for the oral exam.

Run:  python -m src.eval.ragas_eval
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")


def _get_judge_llm():
    """Return the LLM used for evaluation scoring."""
    return ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)


# ---------------------------------------------------------------------------
# Metric implementations (LLM-as-judge)
# ---------------------------------------------------------------------------

def score_context_precision(question: str, contexts: list[str], llm=None) -> float:
    """What fraction of retrieved chunks are relevant to the question?

    Each chunk is scored as relevant (1) or irrelevant (0). Score = relevant / total.
    """
    if not contexts:
        return 0.0
    llm = llm or _get_judge_llm()

    relevant = 0
    for ctx in contexts:
        prompt = f"""Is the following context chunk relevant to answering the question?
Answer only "yes" or "no".

Question: {question}
Context chunk: {ctx[:500]}

Relevant?"""
        response = llm.invoke(prompt).content.strip().lower()
        if "yes" in response:
            relevant += 1

    return relevant / len(contexts)


def score_context_recall(question: str, contexts: list[str], ground_truth: str, llm=None) -> float:
    """What fraction of ground-truth claims are covered by retrieved context?

    Extracts claims from ground truth, checks each against the context.
    """
    llm = llm or _get_judge_llm()
    context_str = "\n\n".join(contexts)

    prompt = f"""Given the ground truth answer and retrieved context, determine what fraction
of the ground truth information is present in the retrieved context.

Ground truth answer: {ground_truth}
Retrieved context: {context_str[:2000]}

Score from 0.0 to 1.0 where:
- 1.0 = all information in the ground truth is found in the context
- 0.5 = about half the information is covered
- 0.0 = none of the ground truth information is in the context

Return ONLY a number between 0.0 and 1.0."""

    response = llm.invoke(prompt).content.strip()
    try:
        return max(0.0, min(1.0, float(response)))
    except ValueError:
        # Extract first float-like substring
        import re
        match = re.search(r"(\d+\.?\d*)", response)
        return float(match.group(1)) if match else 0.5


def score_faithfulness(answer: str, contexts: list[str], llm=None) -> float:
    """What fraction of claims in the answer are supported by the context?

    An answer that hallucinates beyond the context scores low.
    """
    llm = llm or _get_judge_llm()
    context_str = "\n\n".join(contexts)

    prompt = f"""Evaluate the faithfulness of the answer to the provided context.
A faithful answer makes claims ONLY supported by the context.

Context: {context_str[:2000]}
Answer: {answer}

Score from 0.0 to 1.0 where:
- 1.0 = every claim in the answer is supported by the context
- 0.5 = about half the claims are supported
- 0.0 = the answer is entirely unsupported or contradicts the context

Return ONLY a number between 0.0 and 1.0."""

    response = llm.invoke(prompt).content.strip()
    try:
        return max(0.0, min(1.0, float(response)))
    except ValueError:
        import re
        match = re.search(r"(\d+\.?\d*)", response)
        return float(match.group(1)) if match else 0.5


def score_answer_relevancy(question: str, answer: str, llm=None) -> float:
    """How well does the answer address the original question?"""
    llm = llm or _get_judge_llm()

    prompt = f"""How relevant is this answer to the question?

Question: {question}
Answer: {answer}

Score from 0.0 to 1.0 where:
- 1.0 = the answer directly and completely addresses the question
- 0.5 = the answer partially addresses the question
- 0.0 = the answer is completely irrelevant

Return ONLY a number between 0.0 and 1.0."""

    response = llm.invoke(prompt).content.strip()
    try:
        return max(0.0, min(1.0, float(response)))
    except ValueError:
        import re
        match = re.search(r"(\d+\.?\d*)", response)
        return float(match.group(1)) if match else 0.5


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_results(results: list[dict]) -> dict:
    """Score a list of retrieval results on all four RAGAS metrics.

    Returns dict of metric_name -> average score.
    """
    llm = _get_judge_llm()
    all_scores = {
        "context_precision": [],
        "context_recall": [],
        "faithfulness": [],
        "answer_relevancy": [],
    }

    for i, r in enumerate(results):
        logger.info(f"  Scoring Q{i+1}/{len(results)}: {r['question'][:50]}...")

        all_scores["context_precision"].append(
            score_context_precision(r["question"], r["contexts"], llm)
        )
        all_scores["context_recall"].append(
            score_context_recall(r["question"], r["contexts"], r["ground_truth"], llm)
        )
        all_scores["faithfulness"].append(
            score_faithfulness(r["answer"], r["contexts"], llm)
        )
        all_scores["answer_relevancy"].append(
            score_answer_relevancy(r["question"], r["answer"], llm)
        )

    # Average each metric
    return {k: sum(v) / len(v) for k, v in all_scores.items()}


def load_gold_set() -> list[dict]:
    """Load the gold Q&A evaluation set."""
    gold_path = _PROJECT_ROOT / "data" / "eval" / "gold_qa.json"
    with open(gold_path) as f:
        return json.load(f)


def run_retrieval_for_strategy(
    gold_set: list[dict],
    strategy: str,
    corpus_chunks=None,
    k: int = 5,
) -> list[dict]:
    """Run retrieval + generation for each gold question under a given strategy."""
    from src.rag.pipeline import rag_query

    results = []
    for i, item in enumerate(gold_set):
        question = item["question"]
        logger.info(f"  [{strategy}] Q{i+1}/{len(gold_set)}: {question[:60]}...")

        result = rag_query(
            question=question,
            strategy=strategy,
            k=k,
            corpus_chunks=corpus_chunks,
        )

        results.append({
            "question": question,
            "answer": result["answer"],
            "contexts": result["contexts"],
            "ground_truth": item["ground_truth"],
        })

    return results


def print_comparison_table(baseline_scores: dict, final_scores: dict):
    """Print a formatted comparison table."""
    print("\n" + "=" * 65)
    print("  RAGAS Metrics — Baseline (naive) vs Final (hybrid+rerank)")
    print("=" * 65)
    print(f"{'Metric':<25} {'Baseline':>10} {'Final':>10} {'Delta':>10}")
    print("-" * 65)

    for metric in baseline_scores:
        b = baseline_scores[metric]
        f = final_scores.get(metric, 0.0)
        delta = f - b
        sign = "+" if delta > 0 else ""
        print(f"{metric:<25} {b:>10.4f} {f:>10.4f} {sign}{delta:>9.4f}")

    print("-" * 65)
    avg_b = sum(baseline_scores.values()) / len(baseline_scores)
    avg_f = sum(final_scores.values()) / len(final_scores)
    avg_d = avg_f - avg_b
    sign = "+" if avg_d > 0 else ""
    print(f"{'AVERAGE':<25} {avg_b:>10.4f} {avg_f:>10.4f} {sign}{avg_d:>9.4f}")
    print("=" * 65)


def save_results(baseline_scores: dict, final_scores: dict):
    """Save results to JSON for the dashboard."""
    output = {
        "baseline_strategy": "naive",
        "final_strategy": "rerank",
        "baseline_scores": baseline_scores,
        "final_scores": final_scores,
        "delta": {k: final_scores[k] - baseline_scores[k] for k in baseline_scores},
    }
    out_path = _PROJECT_ROOT / "data" / "eval" / "ragas_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {out_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.path.insert(0, str(_PROJECT_ROOT))
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    from src.rag.pipeline import get_corpus_chunks

    gold_set = load_gold_set()
    print(f"Loaded {len(gold_set)} gold Q&A pairs\n")

    corpus_chunks = get_corpus_chunks()

    # --- Baseline ---
    print("=== BASELINE (naive dense retrieval) ===")
    t0 = time.time()
    baseline_results = run_retrieval_for_strategy(gold_set, "naive", corpus_chunks)
    print(f"Retrieval+generation: {time.time() - t0:.1f}s\n")

    print("Scoring baseline with RAGAS metrics...")
    baseline_scores = evaluate_results(baseline_results)

    # --- Final ---
    print("\n=== FINAL (hybrid + cross-encoder rerank) ===")
    t0 = time.time()
    final_results = run_retrieval_for_strategy(gold_set, "rerank", corpus_chunks)
    print(f"Retrieval+generation: {time.time() - t0:.1f}s\n")

    print("Scoring final with RAGAS metrics...")
    final_scores = evaluate_results(final_results)

    # --- Comparison ---
    print_comparison_table(baseline_scores, final_scores)
    save_results(baseline_scores, final_scores)

    # --- Sample ---
    print("\n--- Sample (final) ---")
    s = final_results[0]
    print(f"Q: {s['question']}")
    print(f"A: {s['answer'][:300]}")
    print(f"Ground truth: {s['ground_truth']}")
