"""Memory write node — persists the customer interaction to long-term store.

Runs after every specialist agent, before END.  Reads the interaction
details from the final state and appends them to the customer profile.
"""

import logging

from langchain_core.messages import AIMessage

from src.graph.state import State
from src.memory.store import update_customer_profile
from langgraph.store.base import BaseStore

logger = logging.getLogger(__name__)


def memory_write_node(state: State, *, store: BaseStore) -> dict:
    """Write the current interaction to the customer's long-term memory profile."""
    customer_id = state.get("customer_id", "")
    if not customer_id or customer_id == "unknown":
        return {}

    # Extract the agent's reply as the interaction summary
    ai_messages = [m for m in state.get("messages", []) if isinstance(m, AIMessage)]
    summary = ai_messages[-1].content[:200] if ai_messages else ""

    update_customer_profile(
        store=store,
        customer_id=customer_id,
        intent=state.get("intent", "general"),
        summary=summary,
        resolved=state.get("resolved", False),
    )

    return {}
