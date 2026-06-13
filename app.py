"""TechMart Customer Support Agent — Streamlit Dashboard.

Pillar 4: Observability & Evaluation
Provides:
  - Branded, interactive chat interface backed by the full LangGraph agent
  - Sidebar: routing decisions, customer memory profile, session info
  - RAGAS evaluation results tab (loaded from data/eval/ragas_results.json)
  - Architecture overview tab

Run:  streamlit run app.py
"""

import json
import sys
import uuid
from pathlib import Path

import streamlit as st

# Ensure project root is on the path when launched from anywhere
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TechMart Support Agent",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# route key -> (label, accent colour, icon)
ROUTE_META = {
    "order_lookup":   ("Order Lookup",   "#3B82F6", "📦"),
    "policy_returns": ("Policy & Returns", "#10B981", "📄"),
    "escalation":     ("Escalation",     "#F43F5E", "🚨"),
}

# (button label, message sent) — shown on the empty-chat welcome screen
QUICK_ACTIONS = [
    ("📦  Track an order",   "Where is my order ORD-10001?"),
    ("📄  Return policy",    "What is your return and refund policy?"),
    ("🎧  Browse products",  "What wireless earbuds do you sell?"),
    ("🚨  Talk to a human",  "This is unacceptable, I want to speak to a manager!"),
]

USER_AVATAR = "🧑"
BOT_AVATAR = "🛒"

# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown, .stChatMessage, button, input, textarea {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    /* Hide Streamlit chrome for a cleaner, app-like look */
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {display: none;}
    #MainMenu {visibility: hidden;}
    [data-testid="stHeader"] {background: transparent;}

    .block-container {padding-top: 1.6rem; padding-bottom: 5rem; max-width: 1200px;}

    /* ---------- Hero banner ---------- */
    .hero {
        background: linear-gradient(120deg, #6D28D9 0%, #7C5CFC 45%, #3B82F6 100%);
        border-radius: 18px;
        padding: 26px 30px;
        margin-bottom: 18px;
        box-shadow: 0 12px 32px rgba(124, 92, 252, 0.28);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "🛒";
        position: absolute; right: 22px; top: 50%;
        transform: translateY(-50%);
        font-size: 84px; opacity: 0.16;
    }
    .hero-title {
        font-size: 30px; font-weight: 800; color: #fff;
        letter-spacing: -0.5px; margin: 0;
    }
    .hero-sub {
        font-size: 14.5px; color: rgba(255,255,255,0.86);
        margin-top: 4px; font-weight: 500;
    }
    .hero-badges {margin-top: 14px;}
    .hero-pill {
        display: inline-block; background: rgba(255,255,255,0.16);
        color: #fff; border: 1px solid rgba(255,255,255,0.25);
        border-radius: 999px; padding: 4px 12px; font-size: 12px;
        font-weight: 600; margin-right: 8px; backdrop-filter: blur(4px);
    }

    /* ---------- Chat bubbles ---------- */
    [data-testid="stChatMessage"] {
        border-radius: 14px; padding: 12px 16px; margin-bottom: 8px;
        border: 1px solid rgba(255,255,255,0.06);
        background: #141A29;
    }

    /* ---------- Route / status pills ---------- */
    .pill {
        display: inline-flex; align-items: center; gap: 6px;
        border-radius: 999px; padding: 3px 11px; font-size: 12px;
        font-weight: 600; margin-top: 8px;
    }
    .pii-note {
        display: inline-flex; align-items: center; gap: 6px;
        color: #FBBF24; font-size: 12px; font-weight: 600; margin-top: 6px;
    }

    /* ---------- Welcome card ---------- */
    .welcome {
        background: #141A29; border: 1px solid rgba(124,92,252,0.25);
        border-radius: 16px; padding: 22px 24px; margin-bottom: 14px;
    }
    .welcome h3 {margin: 0 0 6px 0; font-size: 19px; color: #E8EAF2;}
    .welcome p {margin: 0; color: #9AA3B8; font-size: 14px;}

    /* ---------- Quick-action buttons ---------- */
    .stButton > button {
        border-radius: 12px; border: 1px solid rgba(124,92,252,0.35);
        background: #161D2E; color: #E8EAF2; font-weight: 600;
        padding: 10px 12px; transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: #7C5CFC; background: #1C2540;
        transform: translateY(-1px); color: #fff;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: #0E1320;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    .side-card {
        background: #141A29; border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px; padding: 14px 16px; margin-bottom: 12px;
    }
    .side-label {
        font-size: 11px; font-weight: 700; letter-spacing: 0.6px;
        text-transform: uppercase; color: #6B7390; margin-bottom: 8px;
    }

    /* ---------- Metric cards (RAGAS tab) ---------- */
    [data-testid="stMetric"] {
        background: #141A29; border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px; padding: 14px 16px;
    }
    [data-testid="stMetricValue"] {font-size: 26px;}

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {gap: 6px;}
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0; padding: 8px 16px; font-weight: 600;
    }

    /* ---------- Chat input ---------- */
    [data-testid="stChatInput"] {
        border-radius: 14px; border: 1px solid rgba(124,92,252,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def route_pill_html(route: str, intent: str | None) -> str:
    """Coloured pill describing the routing decision for an assistant message."""
    label, color, icon = ROUTE_META.get(
        route, ("Routed", "#7C5CFC", "↳")
    )
    intent_txt = f" · intent: {intent}" if intent else ""
    return (
        f'<span class="pill" style="background:{color}1F; color:{color}; '
        f'border:1px solid {color}55;">{icon} {label}{intent_txt}</span>'
    )


def guardrail_pill_html() -> str:
    return (
        '<span class="pill" style="background:#F59E0B1F; color:#FBBF24; '
        'border:1px solid #F59E0B55;">⚠️ Guardrail triggered — blocked</span>'
    )


# ---------------------------------------------------------------------------
# Graph initialisation (lazy singleton)
# ---------------------------------------------------------------------------

def _init_graph():
    """Initialise the compiled graph, checkpointer, and long-term store."""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.store.memory import InMemoryStore

    from src.graph.build_graph import build_graph

    conn = sqlite3.connect(
        str(_PROJECT_ROOT / "checkpoints.db"), check_same_thread=False
    )
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    store = InMemoryStore()
    graph = build_graph(checkpointer=checkpointer, store=store)
    return graph, store, conn


def _get_graph():
    """Lazy singleton: initialise once and cache in st.session_state."""
    if "graph" not in st.session_state:
        with st.spinner("Booting the support agent (first load takes a moment)…"):
            try:
                g, s, c = _init_graph()
                st.session_state["graph"] = g
                st.session_state["store"] = s
                st.session_state["_conn"] = c
            except Exception as e:
                st.error(f"**Agent initialisation failed:** {e}")
                st.stop()
    return st.session_state["graph"], st.session_state["store"]


def _default_session():
    return {
        "thread_id": str(uuid.uuid4()),
        "customer_id": "CUST-1001",
        "messages": [],       # list of (role, content, meta) tuples
        "last_route": None,
        "last_intent": None,
        "guardrail_blocked": False,
        "resolved": None,
    }


if "session" not in st.session_state:
    st.session_state.session = _default_session()

session = st.session_state.session


# ---------------------------------------------------------------------------
# Core: run one turn through the graph
# ---------------------------------------------------------------------------

def handle_prompt(prompt: str) -> None:
    """Send a customer message through the graph and record the turn."""
    from langchain_core.messages import HumanMessage

    from src.guardrails.checks import check_output

    graph, _ = _get_graph()

    # Mask PII in what we DISPLAY (the graph also masks internally before the LLM)
    display_prompt = check_output(prompt)
    pii_masked = display_prompt != prompt
    session["messages"].append(("user", display_prompt, {"pii_masked": pii_masked}))

    config = {"configurable": {"thread_id": session["thread_id"]}}
    with st.spinner("Agent is thinking…"):
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=prompt)],
                "customer_id": session["customer_id"],
            },
            config=config,
        )

    ai_reply = check_output(result["messages"][-1].content)
    route = result.get("route")
    intent = result.get("intent")
    blocked = result.get("guardrail_blocked", False)
    resolved = result.get("resolved")
    escalation_summary = result.get("escalation_summary")

    session["messages"].append((
        "assistant",
        ai_reply,
        {
            "route": route,
            "intent": intent,
            "guardrail_blocked": blocked,
            "resolved": resolved,
            "escalation_summary": escalation_summary,
        },
    ))
    session["last_route"] = route
    session["last_intent"] = intent
    session["guardrail_blocked"] = blocked
    session["resolved"] = resolved


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🛒 TechMart Agent")
    st.caption("CSAI 422 Capstone · LangGraph Multi-Agent System")
    st.divider()

    # --- Session ---
    st.markdown('<div class="side-label">Session</div>', unsafe_allow_html=True)
    new_cid = st.text_input(
        "Customer ID",
        value=session["customer_id"],
        placeholder="CUST-XXXX",
        label_visibility="collapsed",
    )
    if new_cid != session["customer_id"]:
        session["customer_id"] = new_cid

    if st.button("🔄  New Conversation", use_container_width=True):
        keep_cid = new_cid
        st.session_state.session = _default_session()
        st.session_state.session["customer_id"] = keep_cid
        st.rerun()

    c1, c2 = st.columns(2)
    c1.metric("Thread", session["thread_id"][:6] + "…")
    c2.metric("Turns", len([m for m in session["messages"] if m[0] == "user"]))

    st.divider()

    # --- Last routing decision ---
    st.markdown('<div class="side-label">Last Routing Decision</div>', unsafe_allow_html=True)
    if session["last_route"] or session["guardrail_blocked"]:
        if session.get("guardrail_blocked"):
            st.markdown(guardrail_pill_html(), unsafe_allow_html=True)
        if session["last_route"]:
            st.markdown(
                route_pill_html(session["last_route"], session["last_intent"]),
                unsafe_allow_html=True,
            )
        if session.get("resolved") is True:
            st.success("Resolved ✓", icon="✅")
        elif session.get("resolved") is False:
            st.warning("Escalated — awaiting human agent", icon="🚨")
    else:
        st.caption("No messages yet.")

    st.divider()

    # --- Customer memory ---
    st.markdown('<div class="side-label">Customer Memory · Long-term store</div>', unsafe_allow_html=True)
    if "store" in st.session_state:
        from src.memory.store import get_customer_profile

        profile = get_customer_profile(st.session_state["store"], session["customer_id"])
        if profile["interaction_count"] == 0:
            st.caption("🆕 New customer — no history yet.")
        else:
            m1, m2 = st.columns(2)
            m1.metric("Interactions", profile["interaction_count"])
            if profile["last_interaction"]:
                m2.metric("Last seen", profile["last_interaction"][:10])

            if profile["past_issues"]:
                with st.expander(f"📋 Past issues ({len(profile['past_issues'])})"):
                    for issue in reversed(profile["past_issues"][-5:]):
                        status_icon = "✅" if issue["resolved"] else "⏳"
                        st.markdown(
                            f"{status_icon} **[{issue['type']}]** {issue['summary'][:80]}…"
                        )
                        st.caption(issue["date"][:10])
    else:
        st.caption("Start a conversation to load customer history.")

    st.divider()
    st.caption("🟢 Agent online · gpt-4o-mini")


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">TechMart Support Agent</div>
        <div class="hero-sub">AI-powered multi-agent customer support — orders, returns, products &amp; escalations</div>
        <div class="hero-badges">
            <span class="hero-pill">⚡ LangGraph</span>
            <span class="hero-pill">🔍 Agentic RAG</span>
            <span class="hero-pill">🧠 Long-term memory</span>
            <span class="hero-pill">🛡️ Guardrails</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_chat, tab_eval, tab_arch = st.tabs(["💬  Chat", "📊  RAGAS Evaluation", "🏗️  Architecture"])

# ============================================================================
# TAB 1: Chat
# ============================================================================
with tab_chat:
    # --- Conversation history ---
    for role, content, meta in session["messages"]:
        avatar = USER_AVATAR if role == "user" else BOT_AVATAR
        with st.chat_message(role, avatar=avatar):
            st.markdown(content)
            if role == "user" and meta.get("pii_masked"):
                st.markdown(
                    '<span class="pii-note">🔒 Sensitive info detected and masked '
                    'before it reached the AI</span>',
                    unsafe_allow_html=True,
                )
            if role == "assistant":
                if meta.get("guardrail_blocked"):
                    st.markdown(guardrail_pill_html(), unsafe_allow_html=True)
                elif meta.get("route"):
                    st.markdown(
                        route_pill_html(meta["route"], meta.get("intent")),
                        unsafe_allow_html=True,
                    )
                if meta.get("escalation_summary"):
                    with st.expander("📋 Escalation handoff summary (for the human agent)"):
                        st.code(meta["escalation_summary"])

    # --- Welcome + quick actions (only on an empty chat) ---
    if not session["messages"]:
        st.markdown(
            """
            <div class="welcome">
                <h3>👋 Hi! I'm the TechMart support agent.</h3>
                <p>Ask me about your orders, returns &amp; refunds, shipping, or products —
                or pick a quick action below to get started.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cols = st.columns(len(QUICK_ACTIONS))
        for col, (label, msg) in zip(cols, QUICK_ACTIONS):
            if col.button(label, use_container_width=True, key=f"qa_{label}"):
                st.session_state["_pending_prompt"] = msg
                st.rerun()

    # --- Input handling (chat box + pending quick action) ---
    user_input = st.chat_input("Ask about orders, returns, shipping, or products…")
    pending = st.session_state.pop("_pending_prompt", None)
    prompt = user_input or pending
    if prompt:
        handle_prompt(prompt)
        st.rerun()


# ============================================================================
# TAB 2: RAGAS Evaluation Results
# ============================================================================
with tab_eval:
    st.subheader("RAGAS Evaluation — Pillar 1 (Advanced RAG)")
    st.caption(
        "LLM-as-judge evaluation across 20 gold Q&A pairs. "
        "Baseline = naive dense retrieval · Final = hybrid BM25 + cross-encoder rerank."
    )

    eval_path = _PROJECT_ROOT / "data" / "eval" / "ragas_results.json"
    if eval_path.exists():
        import pandas as pd

        with open(eval_path) as f:
            eval_data = json.load(f)

        baseline = eval_data["baseline_scores"]
        final = eval_data["final_scores"]
        delta = eval_data["delta"]

        metric_labels = {
            "context_precision": "Context Precision",
            "context_recall":    "Context Recall",
            "faithfulness":      "Faithfulness",
            "answer_relevancy":  "Answer Relevancy",
        }

        # --- Summary metric cards ---
        st.markdown("##### Final scores vs. baseline")
        cols = st.columns(4)
        for col, (key, label) in zip(cols, metric_labels.items()):
            d = delta[key]
            sign = "+" if d >= 0 else ""
            col.metric(label, f"{final[key]:.2f}", f"{sign}{d:.3f}")

        avg_b = sum(baseline.values()) / len(baseline)
        avg_f = sum(final.values()) / len(final)
        st.metric("Overall average (final)", f"{avg_f:.3f}", f"{avg_f - avg_b:+.3f} vs baseline")

        st.divider()

        # --- Grouped bar chart ---
        st.markdown("##### Baseline vs. final by metric")
        chart_df = pd.DataFrame(
            {
                "Baseline": [baseline[k] for k in metric_labels],
                "Final (hybrid+rerank)": [final[k] for k in metric_labels],
            },
            index=list(metric_labels.values()),
        )
        st.bar_chart(chart_df, height=320)

        # --- Detailed table ---
        with st.expander("📋 Full comparison table"):
            rows = [
                {
                    "Metric": label,
                    "Baseline": round(baseline[key], 4),
                    "Final": round(final[key], 4),
                    "Delta": round(delta[key], 4),
                }
                for key, label in metric_labels.items()
            ]
            rows.append({
                "Metric": "AVERAGE",
                "Baseline": round(avg_b, 4),
                "Final": round(avg_f, 4),
                "Delta": round(avg_f - avg_b, 4),
            })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # --- Strategy reference ---
        with st.expander("🔍 Retrieval strategies"):
            strat = {
                "Strategy": ["naive", "hybrid", "rerank", "metadata"],
                "Description": [
                    "Dense embedding top-k from Chroma",
                    "BM25 + dense, merged via Reciprocal Rank Fusion (RRF, k=60)",
                    "Hybrid candidates → cross-encoder rerank (ms-marco-MiniLM-L-6-v2)",
                    "Dense top-k with Chroma metadata filter (e.g. doc_type=policy)",
                ],
                "Role": ["Baseline", "Intermediate", "Final pipeline", "Policy-scoped queries"],
            }
            st.dataframe(pd.DataFrame(strat), use_container_width=True, hide_index=True)
    else:
        st.warning(
            "No RAGAS results found. Run the evaluation first:\n\n"
            "```bash\npython -m src.eval.ragas_eval\n```"
        )


# ============================================================================
# TAB 3: Architecture
# ============================================================================
with tab_arch:
    st.subheader("System Architecture")

    st.markdown("##### Agent graph")
    st.code(
        """
__start__
    │
input_guard   ← Pillar 3: injection block · PII masking · off-topic redirect
    │
┌──[blocked]──► __end__
│
[ok]
    │
supervisor    ← Pillar 2: LLM intent classifier + reads customer memory
    │
┌───────────┼───────────────┐
│           │               │
order_lookup  policy_returns  escalation
(mock API)    (Agentic RAG)   (LLM handoff summary)
└───────────┴───────────────┘
    │
memory_write  ← Pillar 2: persists the interaction to long-term store
    │
__end__
        """,
        language="text",
    )

    import pandas as pd

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Tech stack")
        st.dataframe(
            pd.DataFrame({
                "Component": ["Orchestration", "LLM", "Embeddings", "Vector store",
                              "Checkpointer", "Long-term memory", "Guardrails", "Eval", "Dashboard"],
                "Tool": ["LangGraph + LangChain", "OpenAI gpt-4o-mini", "text-embedding-3-small",
                         "Chroma (persistent)", "SqliteSaver", "LangGraph InMemoryStore",
                         "Rule-based (regex)", "RAGAS-style LLM-as-judge", "Streamlit"],
            }),
            use_container_width=True, hide_index=True,
        )
    with c2:
        st.markdown("##### Pillar status")
        st.dataframe(
            pd.DataFrame({
                "Pillar": ["M0: Runnable graph", "1: Advanced RAG", "2: Multi-agent + Memory",
                           "3: Guardrails", "4: Observability & Eval"],
                "Status": ["✅ Complete"] * 5,
            }),
            use_container_width=True, hide_index=True,
        )
