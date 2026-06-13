"""Long-term customer memory using LangGraph InMemoryStore.

Pillar 2: Multi-agent + Memory
Stores customer profiles across sessions, namespaced by customer_id.
Each profile records interaction count, last contact date, and a rolling
window of the last 10 issue summaries.

The store is injected into graph nodes that declare a `store` keyword
argument, once the graph is compiled with `builder.compile(store=store)`.

Namespace layout:
    ("customers", <customer_id>) / "profile"  →  CustomerProfile dict
"""

import logging
from datetime import datetime, timezone

from langgraph.store.base import BaseStore

logger = logging.getLogger(__name__)

_NS_PREFIX = "customers"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_customer_profile(store: BaseStore, customer_id: str) -> dict:
    """Load a customer profile. Returns a blank profile if none exists yet."""
    item = store.get((_NS_PREFIX, customer_id), "profile")
    return item.value if item is not None else _empty_profile(customer_id)


def update_customer_profile(
    store: BaseStore,
    customer_id: str,
    intent: str,
    summary: str,
    resolved: bool,
) -> None:
    """Append an interaction record to the customer profile and persist it."""
    profile = get_customer_profile(store, customer_id)

    profile["interaction_count"] += 1
    profile["last_interaction"] = datetime.now(timezone.utc).isoformat()

    profile["past_issues"].append(
        {
            "date": profile["last_interaction"],
            "type": intent,
            "summary": summary[:200],
            "resolved": resolved,
        }
    )
    # Rolling window: keep the 10 most recent interactions
    profile["past_issues"] = profile["past_issues"][-10:]

    store.put((_NS_PREFIX, customer_id), "profile", profile)
    logger.info(
        "Memory: updated profile for %s (interaction #%d)",
        customer_id,
        profile["interaction_count"],
    )


def format_memory_context(profile: dict) -> str:
    """Render the customer profile as a short text block for LLM prompts."""
    if not profile or profile["interaction_count"] == 0:
        return "New customer — no prior interaction history."

    lines = [
        f"Customer {profile['customer_id']} — "
        f"{profile['interaction_count']} prior interaction(s).",
        f"Last contact: {profile['last_interaction'][:10]}",
    ]

    recent = profile["past_issues"][-3:]  # last 3 issues only
    if recent:
        lines.append("Recent issues:")
        for issue in recent:
            status = "resolved" if issue["resolved"] else "unresolved"
            lines.append(f"  • [{issue['type']}] {issue['summary']} ({status})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty_profile(customer_id: str) -> dict:
    return {
        "customer_id": customer_id,
        "interaction_count": 0,
        "last_interaction": None,
        "past_issues": [],
        "preferences": {},
    }
