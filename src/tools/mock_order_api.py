"""Mock order API — reads orders.json and returns order status by order_id.

Serves as the external tool called by the Order Lookup Agent.
In a real system this would be an HTTP call to an order management service.
"""

import json
from pathlib import Path

# Resolve path relative to project root (two levels up from this file)
_ORDERS_FILE = Path(__file__).resolve().parents[2] / "data" / "orders" / "orders.json"

_orders_cache: dict | None = None


def _load_orders() -> dict:
    """Load orders from disk, caching in memory after first read."""
    global _orders_cache
    if _orders_cache is None:
        with open(_ORDERS_FILE) as f:
            orders_list = json.load(f)
        # Index by order_id for O(1) lookup
        _orders_cache = {order["order_id"]: order for order in orders_list}
    return _orders_cache


def lookup_order(order_id: str) -> dict:
    """Fetch a single order by ID. Returns the order dict or an error dict."""
    orders = _load_orders()
    order = orders.get(order_id)
    if order is None:
        return {"error": f"Order '{order_id}' not found.", "order_id": order_id}
    return order


def get_customer_orders(customer_id: str) -> list[dict]:
    """Fetch all orders for a given customer. Returns a list (possibly empty)."""
    orders = _load_orders()
    return [o for o in orders.values() if o["customer_id"] == customer_id]
