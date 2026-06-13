"""Order Lookup Agent — fetches order status via the mock order API.

Milestone 0: extracts an order ID from the message, calls the mock API,
and returns a formatted status response. No LLM call needed yet.
Will be upgraded with RAG context and LLM-generated responses in Pillar 2.
"""

import logging
import re

from langchain_core.messages import AIMessage

from src.graph.state import State
from src.tools.mock_order_api import get_customer_orders, lookup_order

logger = logging.getLogger(__name__)

# Regex to find order IDs like ORD-12345 in user messages
_ORDER_ID_PATTERN = re.compile(r"ORD-\d+", re.IGNORECASE)


def order_lookup_node(state: State) -> dict:
    """Look up order status and return a response message."""
    last_message = state["messages"][-1].content

    # Try to extract an explicit order ID from the message
    match = _ORDER_ID_PATTERN.search(last_message)
    if match:
        order_id = match.group(0).upper()
        result = lookup_order(order_id)
        if "error" in result:
            reply = f"I couldn't find order {order_id}. Could you double-check the order number?"
        else:
            reply = _format_order_status(result)
        logger.info(f"OrderLookup: looked up {order_id}")
    else:
        # Fall back to customer_id lookup if available
        customer_id = state.get("customer_id", "")
        if customer_id:
            orders = get_customer_orders(customer_id)
            if orders:
                recent = sorted(orders, key=lambda o: o["order_date"], reverse=True)[0]
                reply = f"Here's your most recent order:\n{_format_order_status(recent)}"
                logger.info(f"OrderLookup: found recent order for {customer_id}")
            else:
                reply = "I don't see any orders on your account. Could you provide your order number?"
        else:
            reply = (
                "I'd be happy to help check your order status! "
                "Could you provide your order number? It looks like ORD-XXXXX."
            )

    return {
        "messages": [AIMessage(content=reply)],
        "resolved": True,
    }


def _format_order_status(order: dict) -> str:
    """Format an order dict into a readable status message."""
    items_str = ", ".join(item["name"] for item in order["items"])
    lines = [
        f"**Order {order['order_id']}** — Status: **{order['status'].upper()}**",
        f"  Items: {items_str}",
        f"  Total: ${order['total']:.2f}",
        f"  Estimated delivery: {order['estimated_delivery']}",
    ]
    if tracking := order.get("tracking_number"):
        lines.append(f"  Tracking: {tracking}")
    if reason := order.get("return_reason"):
        lines.append(f"  Return reason: {reason}")
    if reason := order.get("cancellation_reason"):
        lines.append(f"  Cancellation reason: {reason}")
    return "\n".join(lines)
