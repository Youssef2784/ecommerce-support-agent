"""Agent state schema — the single source of truth passed between all graph nodes.

Every node in the LangGraph graph reads from and writes to this State.
Nodes return *partial* dicts; LangGraph merges them using the annotated reducers.
"""

from typing import Annotated, Literal, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    """Typed state for the e-commerce support agent graph.

    Fields are grouped by purpose: conversation, identity, routing, RAG,
    resolution, memory, and guardrails.
    """

    # --- Conversation history ---
    # add_messages reducer: new messages are *appended*, not overwritten.
    messages: Annotated[list[AnyMessage], add_messages]

    # --- Customer identity ---
    # Set at session start; used to namespace long-term memory lookups.
    customer_id: str

    # --- Routing (set by supervisor, consumed by conditional edges) ---
    intent: Literal["order", "policy", "escalation", "general", "unknown"]
    route: Literal["order_lookup", "policy_returns", "escalation"]

    # --- RAG context (populated by retriever when needed) ---
    retrieved_context: list[str]

    # --- Resolution tracking ---
    resolved: bool
    escalation_summary: str

    # --- Pillar 2: Long-term memory ---
    # Customer profile loaded from the LangGraph store by the supervisor.
    # Used by specialist agents for personalised responses.
    customer_memory: dict

    # --- Pillar 3: Guardrails ---
    # Set by the input_guard node; True if the message was blocked.
    guardrail_blocked: bool
