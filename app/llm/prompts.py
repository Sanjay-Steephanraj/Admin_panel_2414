"""
llm/prompts.py
───────────────
Generic prompt builder. ALL domain-specific content is read from components/.
This file never needs to change when switching databases.

Fix applied: user question is sanitised before injection into prompts to
prevent prompt-injection attacks (characters/phrases that could escape the
user-turn boundary and manipulate the system prompt).
"""

import re
import logging
from dynamic.schema_resolver import get_active_schema_prompt_block
from components.prompt_config import (
    ASSISTANT_PERSONA,
    DOMAIN_CONTEXT,
    SQL_RULES,
    get_summary_instruction,
    get_error_message,
)
from components.intent_config import get_intent_hint
from components.few_shot_examples import get_few_shot_examples, format_few_shot_block

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  PROMPT INJECTION SANITISER
# ─────────────────────────────────────────────

# Phrases that are common prompt-injection vectors
_INJECTION_PHRASES = re.compile(
    r"(ignore\s+(all\s+)?(previous|above|prior|earlier)\s+instructions?|"
    r"disregard\s+(all\s+)?(previous|above|prior|earlier)|"
    r"you\s+are\s+now\s+|"
    r"act\s+as\s+(a\s+)?|"
    r"new\s+instruction|"
    r"system\s*:\s*|"
    r"<\s*/?system\s*>|"
    r"<\s*/?prompt\s*>|"
    r"\[\s*system\s*\])",
    re.IGNORECASE,
)

# Characters used to break out of f-string / template context
_STRUCTURAL_CHARS = re.compile(r"[`\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_question(question: str) -> str:
    """
    Sanitises the user question before it is injected into any LLM prompt.

    Steps:
    1. Strip control characters and backticks (structural break chars).
    2. Detect and neutralise known prompt-injection phrases by wrapping
       them in angle brackets so the LLM cannot act on them.
    3. Truncate to a safe maximum length.
    """
    # 1. Remove dangerous control characters
    q = _STRUCTURAL_CHARS.sub("", question)

    # 2. Neutralise injection phrases — wrap with [BLOCKED:…] marker
    #    so the SQL-generation LLM sees them as inert text
    q = _INJECTION_PHRASES.sub(lambda m: f"[BLOCKED:{m.group(0).strip()}]", q)

    # 3. Hard length cap (well below the model's context limit)
    q = q[:500]

    if q != question:
        logger.warning(f"Question sanitised — original: {question!r} | sanitised: {q!r}")

    return q


# ─────────────────────────────────────────────
#  SESSION CONTEXT
# ─────────────────────────────────────────────
def format_history_block(history: list[dict]) -> str:
    if not history:
        return ""
    lines = ["### Conversation History (most recent last)"]
    for i, turn in enumerate(history, 1):
        lines.append(f"Turn {i}: Q: {turn['question']}")
        lines.append(f"         SQL: {turn['sql']}")
        lines.append(f"         Result summary: {turn['summary']}")
        lines.append("")
    lines.append("Use this history ONLY to resolve pronouns or filters in the current question.")
    lines.append("Do NOT repeat prior queries. Generate SQL only for the CURRENT question.\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────
#  CALL 1a — SQL GENERATION
# ─────────────────────────────────────────────
def build_sql_generation_prompt(
    question: str,
    intent: str = "",
    retry_feedback: str = "",
    history: list[dict] = None,
    query_spec: dict | None = None,
) -> list[dict]:

    safe_question = _sanitize_question(question)

    schema_block       = get_active_schema_prompt_block()
    intent_hint        = get_intent_hint(intent)
    few_shot_examples  = get_few_shot_examples(question=safe_question, intent=intent, n=2)
    few_shot_block     = format_few_shot_block(few_shot_examples)

    retry_section = (
        f"\n\n Previous attempt failed. Fix this issue:\n{retry_feedback}\n"
        if retry_feedback else ""
    )

    history_block = format_history_block(history or [])

    contract = query_spec or {}
    system = f"""{DOMAIN_CONTEXT}

You are an internal MySQL query engine for this system.
Convert the admin's natural language question into a precise MySQL SELECT query.

AUTHORITATIVE QUERY SPECIFICATION (do not reinterpret it):
{contract}
The specification's resolved ministry identity, entity role, metric, grouping,
status, and absolute start_date/exclusive_end_date are mandatory. Relative dates
have already been resolved. Do not broaden filters. Use sfpayments for recorded
donation/payment analytics; a receiving ministry joins sfpayments.ministryid to
sf_ministries.sfid. For all_time add no paymentdate predicate.

{schema_block}

{history_block}

{SQL_RULES}

{few_shot_block}
{intent_hint}{retry_section}"""

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Admin Question: {safe_question}"},
    ]


# ─────────────────────────────────────────────
#  CALL 1b — SQL VALIDATION (only on DB error)
# ─────────────────────────────────────────────
def build_sql_validation_prompt(
    question: str,
    generated_sql: str,
    db_error: str,
    db_result_sample: str,
    query_spec: dict | None = None,
) -> list[dict]:

    safe_question = _sanitize_question(question)
    schema_block  = get_active_schema_prompt_block()

    system = f"""You are an internal MySQL query reviewer.
A SQL query was executed and returned an error or failed semantic validation. Diagnose and suggest a fix.
The structured query specification is authoritative. Repair missing or changed
entity role, canonical ministry constraint, absolute period boundaries, aggregate,
grouping, status, or result shape; executable SQL is not automatically correct.

QUERY SPECIFICATION:
{query_spec or {}}

{schema_block}

Respond ONLY with this exact JSON — no markdown, no extra text:
{{"valid": true or false, "issues": "brief description or none", "suggested_fix": "corrected SQL or none"}}"""

    user = f"""Question: {safe_question}

SQL:
{generated_sql}

Error:
{db_error if db_error else db_result_sample}"""

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


# ─────────────────────────────────────────────
#  CALL 2 — SUMMARY GENERATION
# ─────────────────────────────────────────────
def build_summary_prompt(
    question: str,
    sql: str,
    db_result: list[dict],
    intent: str = "",
) -> list[dict]:

    safe_question  = _sanitize_question(question)
    result_sample  = db_result
    result_str     = "\n".join(str(row) for row in result_sample)
    truncation = ""

    intent_guidance = get_summary_instruction(intent)

    system = f"""{ASSISTANT_PERSONA}

Respond to the admin's question based on the data provided.

RESPONSE RULES:
1. Be helpful, direct and kind — admins need facts, not fluff
2. {intent_guidance}
3. Format currency with $ and commas (e.g. $12,500.00)
4. Format dates in readable form (e.g. January 15, 2025)
5. ALWAYS use the '-' character for bullet points when presenting lists (Markdown standard).
6. If no records found: say so clearly and suggest what to check
7. The application renders detail/grouped rows deterministically; never select or omit records.
8. Max 150 words for explanatory text unless the structured result requires more
9. NEVER mention SQL, queries, database, tables, columns, or technical terms
10. Speak directly and confidently — do not say "based on the data"
11. Be kind and interactive, talking to the admin as a helpful assistant, not a distant engine
12. Give the final response in valid markdown format, no spacing for ** when used for bold text
"""
    user = f"""Admin Question: {safe_question}

Data:{truncation}
{result_str}"""

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


# ─────────────────────────────────────────────
#  USER-FACING ERROR MESSAGES — from components
# ─────────────────────────────────────────────
def user_error_message(error_type: str) -> str:
    """Reads all error messages from components/prompt_config.py."""
    return get_error_message(error_type)


# ─────────────────────────────────────────────
#  AMBIGUITY CHECK — rule-based, no LLM
# ─────────────────────────────────────────────
_AMBIGUOUS_PATTERNS = [
    r"^(what|show|get|list|find|give|tell\s+me)\s*\??$",
    r"^(donors?|ministr\w*|payments?|opportunit\w*)\s*\??$",
    r"^.{1,6}$",
    r"^(hi|hello|hey|help|yes|no|ok|okay)\s*\??$",
]
_AMBIGUOUS_RE = re.compile("|".join(_AMBIGUOUS_PATTERNS), re.IGNORECASE)


def is_ambiguous(question: str) -> bool:
    return bool(_AMBIGUOUS_RE.match(question.strip()))
