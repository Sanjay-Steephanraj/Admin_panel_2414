"""
learning/few_shot_examples.py
──────────────────────────────
Few-shot examples with ROBUST name + ministry matching
"""

from typing import Optional
import re
import threading


# deliberate: optional dependency — fall back to accepting any non-empty
# intent string if intent_config is unavailable, to avoid a hard coupling.
try:
    from components.intent_config import get_intent_names as _get_intent_names
except Exception:  # pragma: no cover - import guard only
    _get_intent_names = None


FEW_SHOT_EXAMPLES: list[dict] = [

    # =========================================================
    # DONOR QUERIES (ROBUST NAME MATCHING)
    # =========================================================
    {
        "intent": "donor",
        "question": "What payments did John Walker make in January 2026 and to which ministries did he contribute?",
        "sql": """SELECT
    CONCAT(COALESCE(c.firstname, ''), ' ', COALESCE(c.lastname, '')) AS donor_name,
    p.paymentamount,
    p.paymentdate,
    m.name AS ministry_name
FROM sfpayments p
LEFT JOIN sf_ministries m ON p.ministryid = m.sfid
LEFT JOIN sf_opportunities o ON p.oppsfid = o.sfid
LEFT JOIN sf_contacts c ON o.primarycontact = c.sfid
WHERE LOWER(TRIM(c.firstname)) LIKE '%john%'
  AND LOWER(TRIM(c.lastname)) = 'walker'
  AND p.paymentdate >= '2026-01-01'
  AND p.paymentdate < '2026-02-01'
ORDER BY p.paymentdate DESC
LIMIT 100"""
    },

    {
        "intent": "donor",
        "question": "Give us the list of donors who have donated to The Ministry of Grant Richison (097MGR)",
        "sql": """SELECT DISTINCT
    c.sfid AS donor_id,
    CONCAT(COALESCE(c.firstname, ''), ' ', COALESCE(c.lastname, '')) AS donor_name,
    c.email,
    c.phone
FROM sfpayments p
JOIN sf_ministries m ON p.ministryid = m.sfid
JOIN sf_opportunities o ON p.oppsfid = o.sfid
JOIN sf_contacts c ON o.primarycontact = c.sfid
WHERE m.sfid = 'RESOLVED_MINISTRY_ID'
ORDER BY donor_name
LIMIT 100"""
    },
    
    # =========================================================
    # PAYMENT / MINISTRY CODE MATCHING
    # =========================================================
    {
        "intent": "payment",
        "question": "What were the contributions made to Ministry 801BOB in January 2026?",
        "sql": """SELECT
    p.paymentamount,
    p.paymentdate,
    m.name AS ministry_name
FROM sfpayments p
LEFT JOIN sf_ministries m ON p.ministryid = m.sfid
WHERE m.sfid = 'RESOLVED_MINISTRY_ID'
  AND p.paymentdate >= '2026-01-01'
  AND p.paymentdate < '2026-02-01'
ORDER BY p.paymentdate DESC
LIMIT 100"""
    },

    {
        "intent": "payment",
        "question": "say about payments done by Daniel karunakaran",
        "sql": """SELECT
    CONCAT(c.firstname, ' ', c.lastname) AS donor_name,
    p.paymentamount,
    p.paymentdate,
    p.payment_method,
    m.name AS ministry_name
FROM sfpayments p
LEFT JOIN sf_opportunities o ON p.oppsfid = o.sfid
LEFT JOIN sf_contacts c ON o.primarycontact = c.sfid
LEFT JOIN sf_ministries m ON p.ministryid = m.sfid
WHERE LOWER(TRIM(c.firstname)) LIKE '%daniel%'
  AND LOWER(TRIM(c.lastname)) = 'karunakaran'
ORDER BY p.paymentdate DESC
LIMIT 100"""
    },

    {
        "intent": "payment",
        "question": "say about payments done by National Philanthropic Trust",
        "sql": """SELECT
    CONCAT(COALESCE(c.firstname, ''), ' ', COALESCE(c.lastname, '')) AS donor_name,
    p.paymentamount,
    p.paymentdate,
    p.payment_method,
    m.name AS ministry_name
FROM sfpayments p
LEFT JOIN sf_opportunities o ON p.oppsfid = o.sfid
LEFT JOIN sf_contacts c ON o.primarycontact = c.sfid
LEFT JOIN sf_ministries m ON p.ministryid = m.sfid
WHERE (
    LOWER(TRIM(c.firstname)) LIKE '%national philanthropic trust%'
    OR LOWER(TRIM(c.lastname)) LIKE '%national philanthropic trust%'
  )
ORDER BY p.paymentdate DESC
LIMIT 100"""
    },

    {
        "intent": "payment",
        "question": "say about payments done by Fidelity Charitable",
        "sql": """SELECT
    CONCAT(COALESCE(c.firstname, ''), ' ', COALESCE(c.lastname, '')) AS donor_name,
    p.paymentamount,
    p.paymentdate,
    p.payment_method,
    m.name AS ministry_name
FROM sfpayments p
LEFT JOIN sf_opportunities o ON p.oppsfid = o.sfid
LEFT JOIN sf_contacts c ON o.primarycontact = c.sfid
LEFT JOIN sf_ministries m ON p.ministryid = m.sfid
WHERE (
    LOWER(TRIM(c.firstname)) LIKE '%fidelity%'
    OR LOWER(TRIM(c.lastname)) LIKE '%fidelity%'
  )
ORDER BY p.paymentdate DESC
LIMIT 100"""
    },

    {
        "intent": "payment",
        "question": "show payments done by Boyd",
        "sql": """SELECT
    CONCAT(COALESCE(c.firstname, ''), ' ', COALESCE(c.lastname, '')) AS donor_name,
    p.paymentamount,
    p.paymentdate,
    p.payment_method,
    m.name AS ministry_name
FROM sfpayments p
LEFT JOIN sf_opportunities o ON p.oppsfid = o.sfid
LEFT JOIN sf_contacts c ON o.primarycontact = c.sfid
LEFT JOIN sf_ministries m ON p.ministryid = m.sfid
WHERE (
    LOWER(TRIM(c.firstname)) LIKE '%boyd%'
    OR LOWER(TRIM(c.lastname)) LIKE '%boyd%'
  )
ORDER BY p.paymentdate DESC
LIMIT 100"""
    },

    {
        "intent": "payment",
        "question": "say about payments done by Advancing Native Missions",
        "sql": """SELECT
    CONCAT(COALESCE(c.firstname, ''), ' ', COALESCE(c.lastname, '')) AS donor_name,
    p.paymentamount,
    p.paymentdate,
    p.payment_method,
    m.name AS ministry_name
FROM sfpayments p
LEFT JOIN sf_opportunities o ON p.oppsfid = o.sfid
LEFT JOIN sf_contacts c ON o.primarycontact = c.sfid
LEFT JOIN sf_ministries m ON p.ministryid = m.sfid
WHERE (
    LOWER(TRIM(c.firstname)) LIKE '%advancing native missions%'
    OR LOWER(TRIM(c.lastname)) LIKE '%advancing native missions%'
  )
ORDER BY p.paymentdate DESC
LIMIT 100"""
    },

    {
        "intent": "payment",
        "question": "show donations to Advancing Native Missions",
        "sql": """SELECT
    CONCAT(COALESCE(c.firstname, ''), ' ', COALESCE(c.lastname, '')) AS donor_name,
    p.paymentamount,
    p.paymentdate,
    p.payment_method,
    m.name AS ministry_name
FROM sfpayments p
LEFT JOIN sf_ministries m ON p.ministryid = m.sfid
LEFT JOIN sf_opportunities o ON p.oppsfid = o.sfid
LEFT JOIN sf_contacts c ON o.primarycontact = c.sfid
WHERE LOWER(TRIM(m.name)) LIKE '%advancing native missions%'
ORDER BY p.paymentdate DESC
LIMIT 100"""
    },


    {
        "intent": "payment",
        "question": "Can you share the donations that were made in January 2026 to 095WMN?",
        "sql": """SELECT
    p.paymentamount,
    p.paymentdate,
    m.name AS ministry_name
FROM sfpayments p
LEFT JOIN sf_ministries m ON p.ministryid = m.sfid
WHERE m.sfid = 'RESOLVED_MINISTRY_ID'
  AND p.paymentdate >= '2026-01-01'
  AND p.paymentdate < '2026-02-01'
ORDER BY p.paymentdate DESC
LIMIT 100"""
    },

    {
        "intent": "payment",
        "question": "How many donations have been made to 095WMN?",
        "sql": """SELECT COUNT(p.id) AS record_count
FROM sfpayments p
LEFT JOIN sf_ministries m ON p.ministryid = m.sfid
WHERE m.sfid = 'RESOLVED_MINISTRY_ID'"""
    },

    {
        "intent": "payment",
        "question": "Can you give the total contributions to the ministry 801BOB?",
        "sql": """SELECT SUM(p.paymentamount) AS total_amount
FROM sfpayments p
LEFT JOIN sf_ministries m ON p.ministryid = m.sfid
WHERE m.sfid = 'RESOLVED_MINISTRY_ID'"""
    },

    {
        "intent": "payment",
        "question": "Can you give me the total donations of the ministry 470SFA?",
        "sql": """SELECT SUM(p.paymentamount) AS total_amount
FROM sfpayments p
LEFT JOIN sf_ministries m ON p.ministryid = m.sfid
WHERE m.sfid = 'RESOLVED_MINISTRY_ID'"""
    },

    # =========================================================
    # GENERIC DONOR QUERIES
    # =========================================================
    {
        "intent": "donor",
        "question": "List all active donors with their email and phone",
        "sql": """SELECT
    CONCAT(c.firstname, ' ', c.lastname) AS donor_name,
    c.email,
    c.phone,
    c.donortype
FROM sf_contacts c
WHERE c.donotcontact = 0
ORDER BY c.lastname ASC
LIMIT 100"""
    },

    {
        "intent": "donor",
        "question": "Show profile for donor #000016",
        "sql": """SELECT
    CONCAT(c.firstname, ' ', c.lastname) AS donor_name,
    c.email,
    c.phone,
    c.totaloppamount,
    c.largestamount,
    c.donortype,
    c.donorid
FROM sf_contacts c
WHERE c.donorid = '#000016'"""
    },
    {
        "intent": "donor",
        "question": "donations done by donor #000016",
        "sql": """SELECT
    CONCAT(c.firstname, ' ', c.lastname) AS donor_name,
    p.paymentamount,
    p.paymentdate,
    p.payment_method,
    m.name AS ministry_name
FROM sfpayments p
LEFT JOIN sf_opportunities o ON p.oppsfid = o.sfid
LEFT JOIN sf_contacts c ON o.primarycontact = c.sfid
LEFT JOIN sf_ministries m ON p.ministryid = m.sfid
WHERE p.paid = 1
  AND c.donorid = '#000016'
ORDER BY p.paymentdate DESC
LIMIT 100"""
    },

    # =========================================================
    # AGGREGATE
    # =========================================================
    {
        "intent": "aggregate",
        "question": "How many donors do we have in total",
        "sql": """SELECT
    COUNT(*) AS total_donors
FROM sf_contacts"""
    },

    {
        "intent": "aggregate",
        "question": "what is the highest donation for our organisation ?",
        "sql": """SELECT MAX(p.paymentamount) AS highest_donation
FROM sfpayments p
LEFT JOIN sf_ministries m ON p.ministryid = m.sfid
WHERE m.sfid = 'RESOLVED_MINISTRY_ID'"""
    }

]


# =========================================================
# FEW SHOT SELECTION
# =========================================================

def _tokenize(text: str) -> set[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return set(text.split())


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def get_few_shot_examples(question: str, intent: str, n: int = 3) -> list[dict]:
    q_tokens = _tokenize(question)

    # deliberate: snapshot runtime examples under the lock so appends during
    # scoring don't mutate the list we're iterating; selection logic unchanged.
    with _RUNTIME_LOCK:
        runtime_snapshot = list(_RUNTIME_EXAMPLES)

    candidates = [
        ex for ex in (FEW_SHOT_EXAMPLES + runtime_snapshot)
        if ex["intent"] == intent or ex["intent"] == "aggregate"
    ]

    scored = []
    for ex in candidates:
        ex_tokens = _tokenize(ex["question"])
        score = _jaccard(q_tokens, ex_tokens)

        if ex["intent"] == intent:
            score += 0.2

        scored.append((score, ex))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [ex for _, ex in scored[:n]]


def format_few_shot_block(examples: list[dict]) -> str:
    if not examples:
        return ""

    lines = ["### Reference Examples\n"]

    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"Q: {ex['question']}")
        lines.append(f"SQL:\n{ex['sql']}\n")

    return "\n".join(lines)


# =========================================================
# RUNTIME FEW-SHOT EXAMPLES (admin-confirmed feedback)
# =========================================================
# deliberate: in-process list, per-process and lost on restart, mirroring
# the conversation-memory model used when Redis is disabled. No disk layer
# exists in this codebase, so persistence is intentionally out of scope.
_RUNTIME_EXAMPLES: list[dict] = []
_RUNTIME_LOCK = threading.Lock()
MAX_RUNTIME_EXAMPLES = 50

_Q_MAX = 500
_SQL_MAX = 2000


def _normalize_question(question: str) -> str:
    """Lowercase + collapse internal whitespace for duplicate detection."""
    return " ".join(question.lower().split())


def _strip_sql(sql: str) -> str:
    """Remove surrounding markdown fences and whitespace for SELECT checks."""
    s = sql.strip()
    # deliberate: handle ```sql ... ``` fences admins may paste from chat UIs.
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


def add_example(question: str, sql: str, intent: str) -> bool:
    """
    Records an admin-confirmed question→SQL pair as a runtime few-shot example.

    Examples are appended to an in-process list that get_few_shot_examples()
    also draws from, so confirmed corrections influence subsequent SQL
    generation without a restart. Storage is intentionally in-memory: it is
    per-process and lost on restart, matching the conversation-memory model
    used when Redis is disabled.

    Returns True when the example was stored, False when it was rejected
    (blank fields, non-SELECT SQL, unknown intent, or duplicate).
    """
    if not question or not sql or not intent:
        return False

    question = question.strip()
    sql = sql.strip()
    intent = intent.strip()
    if not question or not sql or not intent:
        return False

    # Intent validation — only when the intent registry is available.
    if _get_intent_names is not None:
        try:
            known = set(_get_intent_names())
        except Exception:
            known = None
        if known is not None and intent not in known:
            return False

    # SQL must be a read query — never store DML/DDL.
    if not _strip_sql(sql).upper().startswith("SELECT"):
        return False

    # Cap stored lengths to bound prompt growth.
    question = question[:_Q_MAX]
    sql = sql[:_SQL_MAX]

    norm_q = _normalize_question(question)

    with _RUNTIME_LOCK:
        # Reject duplicates against the curated set and the runtime set.
        for ex in FEW_SHOT_EXAMPLES:
            if _normalize_question(ex["question"]) == norm_q:
                return False
        for ex in _RUNTIME_EXAMPLES:
            if _normalize_question(ex["question"]) == norm_q:
                return False

        # FIFO: drop oldest when at capacity before appending.
        if len(_RUNTIME_EXAMPLES) >= MAX_RUNTIME_EXAMPLES:
            del _RUNTIME_EXAMPLES[0]

        _RUNTIME_EXAMPLES.append(
            {"intent": intent, "question": question, "sql": sql}
        )
    return True


def runtime_example_count() -> int:
    """Number of stored runtime examples (for tests/observability)."""
    with _RUNTIME_LOCK:
        return len(_RUNTIME_EXAMPLES)
