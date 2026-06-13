"""Escalation Agent — drafts a structured handoff summary for human agents.

Pillar 2 upgrade: uses an LLM to generate a rich, context-aware handoff
summary from the full conversation history and the customer's long-term
memory profile.  Falls back to a template-based summary when no API key
is available.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.state import State
from src.memory.store import format_memory_context, get_customer_profile
from langgraph.store.base import BaseStore

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env")


def escalation_node(state: State, *, store: BaseStore) -> dict:
    """Draft a structured escalation summary and inform the customer."""
    customer_id = state.get("customer_id", "unknown")
    messages = state.get("messages", [])
    profile = get_customer_profile(store, customer_id)

    if os.getenv("OPENAI_API_KEY"):
        escalation_summary = _llm_summary(customer_id, messages, profile)
    else:
        escalation_summary = _template_summary(customer_id, messages)

    logger.info("Escalation: drafted handoff for customer %s", customer_id)

    reply = (
        "I understand this needs special attention. I've prepared a detailed summary "
        "of your case and I'm connecting you with a specialist who can help further. "
        "A human agent will review your case shortly — you won't need to repeat yourself. "
        "Thank you for your patience."
    )

    return {
        "messages": [AIMessage(content=reply)],
        "resolved": False,
        "escalation_summary": escalation_summary,
    }


# ---------------------------------------------------------------------------
# LLM summary
# ---------------------------------------------------------------------------

def _llm_summary(customer_id: str, messages: list, profile: dict) -> str:
    """Use an LLM to generate a concise, actionable escalation summary."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)

    # Build conversation transcript
    transcript_lines = []
    for m in messages[:-1]:  # exclude the escalation trigger message
        role = "Customer" if isinstance(m, HumanMessage) else "Agent"
        transcript_lines.append(f"{role}: {m.content[:300]}")
    transcript = "\n".join(transcript_lines) if transcript_lines else "(first message)"

    trigger = messages[-1].content if messages else ""
    memory_ctx = format_memory_context(profile)

    prompt = f"""You are a customer support escalation specialist.
Create a concise handoff summary for a human agent.

Customer ID: {customer_id}
Customer history:
{memory_ctx}

Conversation transcript:
{transcript}

Escalation trigger (latest message): {trigger}

Write a structured handoff summary with these sections:
1. ISSUE: One-sentence description of the core problem
2. URGENCY: Low / Medium / High — and why
3. HISTORY: Relevant prior interactions (from memory context)
4. SUGGESTED ACTION: What the human agent should do first
5. SENTIMENT: Customer's emotional state

Keep it under 200 words."""

    response = llm.invoke([HumanMessage(content=prompt)])
    llm_summary = response.content.strip()

    return (
        f"=== ESCALATION HANDOFF ===\n"
        f"Customer ID: {customer_id}\n"
        f"Conversation turns: {len(messages)}\n"
        f"Status: Awaiting human agent\n\n"
        f"{llm_summary}\n"
        f"=========================="
    )


# ---------------------------------------------------------------------------
# Template fallback
# ---------------------------------------------------------------------------

def _template_summary(customer_id: str, messages: list) -> str:
    human_messages = [m.content for m in messages if isinstance(m, HumanMessage)]
    issue_summary = human_messages[-1] if human_messages else "No details provided"

    return (
        "=== ESCALATION HANDOFF ===\n"
        f"Customer ID: {customer_id}\n"
        f"Issue: {issue_summary}\n"
        f"Conversation turns: {len(messages)}\n"
        "Status: Awaiting human agent\n"
        "=========================="
    )
