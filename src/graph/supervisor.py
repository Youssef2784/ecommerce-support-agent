"""Supervisor node — classifies customer intent and routes to the right specialist.

Pillar 2 upgrade: LLM-based intent classification replaces keyword matching.
The LLM also receives the customer's interaction history (from the memory store)
so it can personalise routing (e.g. detect repeat complaints).

Falls back to keyword matching if no OPENAI_API_KEY is set, keeping the graph
runnable in offline / demo environments.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.store.base import BaseStore

from src.graph.state import State
from src.memory.store import format_memory_context, get_customer_profile

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Keyword fallback (used when no API key is available)
# ---------------------------------------------------------------------------
_INTENT_KEYWORDS: dict[str, list[str]] = {
    "escalation": [
        "escalat", "manager", "supervisor", "complain", "unacceptable",
        "speak to", "human", "real person", "lawyer", "sue", "legal",
        "furious", "outraged", "disgusted",
    ],
    "policy": [
        "return", "refund", "policy", "exchange", "warranty", "cancel",
        "money back", "replacement", "damaged", "broken", "defective",
    ],
    "product": [
        "do you sell", "do you have", "do you carry", "in stock", "available",
        "headphone", "earbud", "speaker", "laptop", "tablet", "charger",
        "webcam", "mouse", "keyboard", "monitor", "recommend", "specs",
        "catalog", "what products", "wireless",
    ],
    "order": [
        "order", "track", "shipping", "delivery", "where is", "status",
        "ord-", "when will", "arriving", "package", "tracking",
    ],
}


def supervisor_node(state: State, *, store: BaseStore) -> dict:
    """Classify the latest customer message and set the routing intent.

    Uses an LLM classifier when an API key is available; falls back to
    keyword matching otherwise.  Customer memory is included in the LLM
    prompt so the model can detect patterns like repeat unresolved issues.
    """
    last_message = state["messages"][-1]
    customer_id = state.get("customer_id", "unknown")
    text = last_message.content

    # Load long-term customer memory
    profile = get_customer_profile(store, customer_id)
    state["customer_memory"] = profile  # pass to rest of graph

    if os.getenv("OPENAI_API_KEY"):
        intent = _llm_classify(text, profile)
    else:
        intent = _keyword_classify(text.lower())

    route = _intent_to_route(intent)
    logger.info("Supervisor: intent=%s, route=%s", intent, route)

    return {
        "intent": intent,
        "route": route,
        "customer_memory": profile,
    }


# ---------------------------------------------------------------------------
# LLM classifier
# ---------------------------------------------------------------------------

def _llm_classify(text: str, profile: dict) -> str:
    """Ask the LLM to classify the customer message into one of four intents."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)

    memory_ctx = format_memory_context(profile)

    prompt = f"""You are a routing classifier for a customer support system.
Classify the customer message into exactly one of these intents:

  • order      — status, tracking, shipping or delivery of an EXISTING order
  • policy     — returns, refunds, exchanges, warranties, or shipping policies
  • product    — product availability, catalog, specs, recommendations, or
                 "do you sell / do you have / what products" questions
  • escalation — expressing strong frustration, demanding a manager/human,
                 threatening legal action, or flagging a repeated unresolved issue
  • general    — greetings or genuinely unclear messages

Customer memory (for context):
{memory_ctx}

Customer message: {text}

Reply with ONLY the intent word (order / policy / product / escalation / general)."""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip().lower()

    # Normalise — accept any response that *starts with* a valid intent
    for intent in ("escalation", "policy", "product", "order", "general"):
        if raw.startswith(intent):
            return intent

    logger.warning("LLM classifier returned unexpected value '%s', falling back", raw)
    return _keyword_classify(text.lower())


# ---------------------------------------------------------------------------
# Keyword fallback
# ---------------------------------------------------------------------------

def _keyword_classify(text_lower: str) -> str:
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return intent
    return "general"


def _intent_to_route(intent: str) -> str:
    return {
        "order": "order_lookup",
        "policy": "policy_returns",
        "product": "policy_returns",   # product/catalog questions use Agentic RAG
        "escalation": "escalation",
        "general": "order_lookup",
    }.get(intent, "order_lookup")
