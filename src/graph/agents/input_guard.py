"""Input guard node — first node in the graph; gates all incoming messages.

Pillar 3: Guardrails
Runs check_input() on the latest customer message before routing.
  • Blocked messages: adds the guardrail's AI reply to state and sets
    guardrail_blocked=True so the conditional edge skips to END.
  • Passed messages: replaces the message text with the PII-masked version
    and continues to the supervisor.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage

from src.graph.state import State
from src.guardrails.checks import check_input

logger = logging.getLogger(__name__)


def input_guard_node(state: State) -> dict:
    """Run input guardrails and either pass through or block with a safe reply."""
    last_message = state["messages"][-1]
    text = last_message.content

    result = check_input(text)

    if not result.passed:
        logger.info("InputGuard: message blocked — %s", result.reason[:60])
        return {
            "messages": [AIMessage(content=result.reason)],
            "guardrail_blocked": True,
        }

    # If PII was masked, update the message in-place by preserving its ID.
    # The add_messages reducer replaces a message when the ID matches.
    if result.pii_found:
        masked_message = HumanMessage(
            content=result.masked_text,
            id=last_message.id,  # same ID → reducer replaces instead of appending
        )
        return {
            "messages": [masked_message],
            "guardrail_blocked": False,
        }

    return {"guardrail_blocked": False}
