"""Input and output guardrails for the customer support agent.

Pillar 3: Guardrails
Four-layer protection:
  1. Prompt injection detection   — blocks attempts to override system instructions
  2. PII masking                  — redacts cards, SSNs, emails, phone numbers
  3. Off-topic detection          — redirects non-e-commerce messages
  4. Output scrubbing + length cap — strips leaked PII, truncates runaway responses

Design goals:
  - All checks are rule-based (regex/keyword) → zero extra latency, no LLM call.
  - Checks are composable: check_input() returns a GuardrailResult with
    the (possibly masked) text so the rest of the pipeline sees clean input.
  - check_output() is idempotent — safe to call even if no PII is present.
"""

import logging
import re

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

_CREDIT_CARD_RE = re.compile(
    r"\b(?:\d[ \-]?){13,16}\b"
)
_SSN_RE = re.compile(
    r"\b\d{3}[\-\s]?\d{2}[\-\s]?\d{4}\b"
)
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)
_PHONE_RE = re.compile(
    r"\b(?:\+?1[\-.\s]?)?\(?\d{3}\)?[\-.\s]?\d{3}[\-.\s]?\d{4}\b"
)


# ---------------------------------------------------------------------------
# Prompt injection patterns
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above)",
    r"you\s+are\s+now\s+(?!a\s+(techmart|customer\s+support))",
    r"act\s+as\s+(?!(a\s+)?(techmart|customer\s+support|helpful))",
    r"\bjailbreak\b",
    r"\bDAN\s+mode\b",
    r"system\s+prompt",
    r"forget\s+your\s+instructions",
    r"override\s+(your\s+)?(instructions|rules|guidelines)",
    r"pretend\s+you\s+(are|have\s+no)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Off-topic detection (allow-list of e-commerce signals)
# ---------------------------------------------------------------------------

_ON_TOPIC_KEYWORDS = [
    "order", "track", "ship", "deliver", "return", "refund", "exchange",
    "warranty", "product", "item", "package", "payment", "account",
    "cancel", "status", "help", "support", "customer", "techmart",
    "broken", "defective", "damaged", "late", "missing", "wrong",
    "price", "purchase", "buy", "bought", "receipt", "invoice",
    "cart", "checkout", "discount", "coupon", "promo", "address",
    "billing", "charge", "credit", "replace", "replacement", "fix",
    "where is", "when will", "arrive", "arriving", "received", "send",
    # Product / catalog signals — browsing is a support case
    "sell", "stock", "available", "carry", "offer", "recommend", "suggest",
    "headphone", "earbud", "earphone", "speaker", "laptop", "tablet",
    "phone", "smartphone", "charger", "cable", "mouse", "keyboard",
    "webcam", "monitor", "camera", "watch", "audio", "wireless",
    "electronic", "device", "gadget", "accessory", "model", "brand",
    "spec", "feature", "compatible", "battery", "color", "size",
    # Escalation signals — an angry customer IS a support case
    "unacceptable", "manager", "supervisor", "complain", "complaint",
    "escalat", "frustrated", "angry", "upset", "furious", "outraged",
    "speak to", "human", "real person", "this is", "i want", "i need",
    "lawyer", "sue", "legal", "disgusted", "ridiculous",
]

_ALWAYS_ON_TOPIC_PREFIXES = [
    "hello", "hi ", "hey", "thanks", "thank you", "ok", "okay",
    "yes", "no", "please", "sorry", "i need", "i have", "i want",
    "can you", "could you", "help me",
    # Customer-intent openers — imply a genuine request (not trivia).
    # Bare interrogatives (what/which/how) are deliberately NOT here:
    # a real product question carries a topical keyword instead, so trivia
    # like "what is the capital of France" is still redirected.
    "do you", "tell me", "show me", "looking for", "i'm looking", "i am looking",
]

# Strong on-topic signals: order / tracking IDs, or text where PII was masked
# (sharing a card or email implies an active transaction → a support context).
_ON_TOPIC_PATTERNS = re.compile(
    r"\bord[\-\s]?\d|\btrk[\-\s]?\w|\border\s*#|-redacted\]",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

class GuardrailResult:
    """Outcome of an input guardrail check."""

    __slots__ = ("passed", "reason", "masked_text", "pii_found")

    def __init__(
        self,
        passed: bool,
        reason: str = "",
        masked_text: str = "",
        pii_found: bool = False,
    ) -> None:
        self.passed = passed
        self.reason = reason
        self.masked_text = masked_text
        self.pii_found = pii_found


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def check_input(text: str) -> GuardrailResult:
    """Run all input guardrails. Returns a GuardrailResult.

    If passed=True, use result.masked_text downstream (PII may have been redacted).
    If passed=False, return result.reason to the customer directly.
    """
    # --- Layer 1: Prompt injection ---
    if _INJECTION_RE.search(text):
        logger.warning("Guardrail BLOCK [injection]: %.80s", text)
        return GuardrailResult(
            passed=False,
            reason=(
                "I'm not able to follow those instructions. I'm here to help with "
                "TechMart orders, returns, and product questions — how can I assist you?"
            ),
        )

    # --- Layer 2: PII masking ---
    masked, pii_found = _mask_pii(text)
    if pii_found:
        logger.info("Guardrail [PII masked]: sensitive data redacted from input")

    # --- Layer 3: Off-topic detection ---
    if _is_off_topic(masked):
        logger.info("Guardrail REDIRECT [off-topic]: %.60s", text)
        return GuardrailResult(
            passed=False,
            reason=(
                "I'm TechMart's customer support assistant. I can help with orders, "
                "shipping, returns, refunds, and product questions. "
                "What can I help you with today?"
            ),
        )

    return GuardrailResult(passed=True, masked_text=masked, pii_found=pii_found)


def check_output(text: str) -> str:
    """Scrub PII from agent responses and enforce a length cap.

    Safe to call on every response — returns the original text unchanged
    if no issues are found.
    """
    # Scrub any PII that might have leaked into the generated response
    cleaned, pii_found = _mask_pii(text)
    if pii_found:
        logger.warning("Guardrail [output PII scrubbed]")

    # Length cap: truncate runaway responses
    if len(cleaned) > 2000:
        cleaned = (
            cleaned[:1950]
            + "\n\n[Response truncated. Please ask a follow-up if you need more details.]"
        )
        logger.warning("Guardrail [output truncated]: response exceeded 2000 chars")

    return cleaned


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mask_pii(text: str) -> tuple[str, bool]:
    """Replace PII patterns with labelled placeholders. Returns (masked_text, found)."""
    original = text
    text = _CREDIT_CARD_RE.sub("[CARD-REDACTED]", text)
    text = _SSN_RE.sub("[SSN-REDACTED]", text)
    text = _EMAIL_RE.sub("[EMAIL-REDACTED]", text)
    text = _PHONE_RE.sub("[PHONE-REDACTED]", text)
    return text, text != original


def _is_off_topic(text: str) -> bool:
    """Return True if the message has no e-commerce support signal.

    Short messages and conversational openers are always on-topic to avoid
    false-positives on greetings like "Hi, I need help".
    """
    text_lower = text.lower().strip()

    # Very short messages are on-topic by default
    if len(text_lower) < 20:
        return False

    # Strong signals: order/tracking IDs or masked-PII markers → always on-topic
    if _ON_TOPIC_PATTERNS.search(text_lower):
        return False

    # Common conversation starters / question openers are always on-topic
    if any(text_lower.startswith(p) for p in _ALWAYS_ON_TOPIC_PREFIXES):
        return False

    # Accept if any e-commerce keyword appears
    return not any(kw in text_lower for kw in _ON_TOPIC_KEYWORDS)
