"""Follow-up detection; inheritance happens only in query_spec.merge_follow_up."""

import re
import logging

logger = logging.getLogger(__name__)

_FOLLOW_UP_RE = re.compile(
    r"^\s*(?:what about|how about|and|also|then)\b|"
    r"\b(?:their|those|that|it)\b|\b(?:last|this)\s+(?:month|week|year)\b\s*\??$",
    re.IGNORECASE,
)

def is_genuine_follow_up(question: str) -> bool:
    q = (question or "").strip()
    if re.search(r"\b(?:their|those|them|it|that)\b", q, re.I) and re.match(
        r"^(?:show|list|count|total|average|how many|how much|are)\b", q, re.I
    ):
        return True
    if re.search(r"\b(?:donations?|payments?|contributions?|donors?|givers?|ministr(?:y|ies))\b", q, re.I) and not re.match(r"^\s*what about\b", q, re.I):
        if re.search(r"\b(?:show|list|get|count|how many|how much|total|average)\b", q, re.I):
            return False
    if re.search(r"\b(?:donations?|payments?|contributions?)\s+(?:by|per)\s+ministr(?:y|ies)\b", q, re.I):
        return False
    if re.match(r"^(?:what|how) about\b", q, re.I):
        if re.search(r"\blist\s+of\s+(?:donors?|givers?)\b", q, re.I):
            return True
        return bool(re.search(r"\b(?:last|this|previous|today|yesterday|all time|count|total|average|paid|unpaid|details|ministr(?:y|ies)|donor|them|those|it|january|february|march|april|may|june|july|august|september|october|november|december)\b",q, re.I))
    if re.fullmatch(r"(?:and|also|then|now)?\s*(?:a |the )?(?:count|total|average|how many|how much)(?: instead)?\s*\??", q, re.I):
        return True
    return bool(re.search(r"^(?:(?:and|also|then|now)\s+)?(?:for |in )?(?:last|this|previous)\s+(?:month|week|year)\s*\??$", q, re.I))

def expand_question(question: str, history: list[dict]) -> str:
    """
    Rewrites follow-up questions using prior conversation context.
    If the follow-up is unrelated to the prior topic, returns it unchanged.
    If history is empty, returns original question.
    """
    if not history:
        return question

    if not is_genuine_follow_up(question):
        logger.info("Question is self-contained; skipping follow-up expansion")
        return question

    # Structured inheritance in query_spec.merge_follow_up is authoritative.
    # Preserve the user's continuation verbatim so an LLM rewrite cannot turn
    # an elliptical continuation into a different standalone request.
    return question
