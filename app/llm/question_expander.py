import re
import logging
from llm.gloo_client import get_llm
from langchain_core.messages import SystemMessage, HumanMessage
from data_masking import masker

logger = logging.getLogger(__name__)

_FOLLOW_UP_RE = re.compile(
    r"^\s*(?:what about|how about|and|also|then)\b|\b(?:their|those|that|it)\b|\b(?:last|this)\s+(?:month|week|year)\b\s*\??$",
    re.IGNORECASE,
)

def is_genuine_follow_up(question: str) -> bool:
    """A complete new analytics request must never inherit prior filters."""
    q = question.strip()
    if not q:
        return False
    # A concrete subject/entity/grouping makes the turn self-contained.
    if re.search(r"\b(?:donations?|payments?|contributions?|donors?|ministr(?:y|ies))\b", q, re.I) and not re.match(r"^\s*what about\b", q, re.I):
        return False
    return bool(_FOLLOW_UP_RE.search(q))

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

    # ── Mask PII in question + history before sending to LLM ──────────────
    # Fail-closed: if masking raises, never send unmasked data — return original.
    try:
        mapping: dict = {}
        masked_question, mapping = masker.mask_text(question, mapping)
        masked_turns: list[str] = []
        for i, t in enumerate(history):
            m_turn, mapping = masker.mask_text(t["question"], mapping)
            masked_turns.append(f"Turn {i+1}: Q: {m_turn}")
        history_text = "\n".join(masked_turns)
    except RuntimeError:
        logger.warning("PII masking failed — skipping question expansion, returning original")
        return question

    # ── Only genuine follow-ups reach the expansion model ──────────────
    try:
        system_prompt = (
            "You rewrite follow-up questions to resolve pronouns and implicit references using prior conversation context.\n"
            "CRITICAL RULES:\n"
            "1. Retain ALL specific numbers, limits, names, and explicit filters from the original follow-up (e.g. '158', 'next 5', 'top 10').\n"
            "2. If the user asks for 'some of the ministries from that 158', ensure '158 ministries' is explicitly in your output.\n"
            "3. ONLY rewrite if the follow-up is clearly continuing the prior topic.\n"
            "4. NEVER expand or rewrite fixed domain references like 'our ministry', 'our organisation', or 'us'. Leave them exactly as typed by the user, as the domain context handles them downstream.\n"
            "5. If the follow-up is a completely new/unrelated topic, return it unchanged.\n"
            "6. NEVER append organization names unless explicitly replacing a pronoun.\n\n"
            "Respond ONLY with the rewritten question. No explanation."
        )


        user_prompt = f"Prior Conversation:\n{history_text}\n\nFollow-up: {masked_question}"

        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        expanded = response.content.strip()
        expanded = masker.unmask_text(expanded, mapping)
        logger.info(f"Question expanded | orig={question!r} | expanded={expanded!r}")
        return expanded

    except Exception as e:
        logger.error(f"Question expansion failed: {e}")
        return question
