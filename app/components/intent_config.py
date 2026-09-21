
"""
components/intent_config.py
────────────────────────────
Robust intent configuration + scoring-based resolver
WITH backward compatibility
"""

import re
from collections import defaultdict


# =========================================================
# INTENT DEFINITIONS (WITH PRIORITY)
# =========================================================

INTENTS: list[dict] = [

    # ─────────────────────────────
    # ENTITY LOOKUP
    # ─────────────────────────────
    {
        "name": "entity_lookup",
        "priority": 5,
        "keywords": [
            "who is", "details of", "details for", "information about",
            "info about", "tell me about", "show details", "find details",
            "lookup", "profile", "bio"
        ],
        "tables": ["sf_contacts"],
        "join_hint": "sf_contacts.sfid = sf_opportunities.primarycontact",
    },

    # ─────────────────────────────
    # DONOR (HIGH PRIORITY)
    # ─────────────────────────────
    {
        "name": "donor",
        "priority": 4,
        "keywords": [
            "donor", "donors", "contact", "contacts",
            "giver", "givers", "supporter", "person",
            "firstname", "lastname", "email", "phone",

            #  critical patterns
            "contribution of", "donations by", "payments by",
            "given by", "contributed by",

            # churn / inactivity (safety net — normally routed to churn analyzer)
            "churn", "churning", "lapsed", "at risk", "inactive donors"
        ],
        "tables": ["sf_contacts"],
        "join_hint": "sf_contacts.sfid = sf_opportunities.primarycontact",
    },

    # ─────────────────────────────
    # PAYMENT
    # ─────────────────────────────
    {
        "name": "payment",
        "priority": 3,
        "keywords": [
            "payment", "payments", "transaction", "transactions",
            "donation", "donations", "giving", "contribution",
            "paymentamount", "paymentdate"
        ],
        "tables": ["sfpayments"],
        "join_hint": "sfpayments.oppsfid = sf_opportunities.sfid",
    },

    # ─────────────────────────────
    # MINISTRY
    # ─────────────────────────────
    {
        "name": "ministry",
        "priority": 2,
        "keywords": [
            "ministry", "ministries", "fund", "funds",
            "giftcode", "program", "department",
            "church", "organization", "charity",
            "campaign", "campaigns"
        ],
        "tables": ["sf_ministries"],
        "join_hint": "sf_ministries.sfid = sfpayments.ministryid",
    },

    # ─────────────────────────────
    # AGGREGATE (LOWEST PRIORITY)
    # ─────────────────────────────
    {
        "name": "aggregate",
        "priority": 1,
        "keywords": [
            "total", "sum", "count", "average",
            "highest", "lowest",
            "monthly", "yearly",
            "report", "summary", "breakdown",
            "compare", "rank", "revenue", "income", "raised"
        ],
        "tables": [],
        "join_hint": "",
    },
]


# =========================================================
#  NAME DETECTION (CRITICAL)
# =========================================================

_ORG_SUFFIX_WORDS = re.compile(
    r"\b(missions?|trust|foundation|charitable|incorporated|inc|"
    r"ministries|ministry|church|organization|society|association|"
    r"services?|enterprises?|group|fund|charity|international|global)\b",
    re.IGNORECASE,
)

_BY_PREPOSITION_RE = re.compile(r"\bby\s+", re.IGNORECASE)
_TO_FOR_PREPOSITION_RE = re.compile(r"\b(to|for)\s+", re.IGNORECASE)


def detect_person_name(query: str) -> str | None:
    """
    Detects names like 'John Walker' — but EXCLUDES org/charity phrases.

    Rules:
    - If the name candidate contains an org-suffix word (Ministry, Trust,
      Foundation, Charitable, Inc, Missions…) → treat as org, not a person.
    - Only returns a hit when the preposition is 'by' (donor context).
      Prepositions like 'to' or 'for' indicate a ministry/recipient, not a donor.
    """
    # Only apply in "payments/donations BY <name>" context
    if not _BY_PREPOSITION_RE.search(query):
        return None

    match = re.search(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", query)
    if match:
        full_match = match.group(0)
        # Reject if the match or surrounding phrase looks like an org name
        if _ORG_SUFFIX_WORDS.search(query):
            return None
        return full_match
    return None


# =========================================================
# TOKENIZER
# =========================================================

def _tokenize(text: str) -> set[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return set(text.split())


# =========================================================
#  MAIN INTENT RESOLVER
# =========================================================

def resolve_intent(query: str) -> str:
    """
    Robust intent detection using:
    1. Name override
    2. Keyword scoring
    3. Priority weighting
    """

    query_lower = query.lower()

    # -----------------------------------------------------
    # STEP 1: NAME OVERRIDE
    # -----------------------------------------------------
    if detect_person_name(query):
        return "donor"

    # -----------------------------------------------------
    # STEP 2: KEYWORD SCORING
    # -----------------------------------------------------
    scores = defaultdict(int)

    for intent in INTENTS:
        intent_name = intent["name"]

        for kw in intent["keywords"]:
            if kw in query_lower:
                scores[intent_name] += 1

    # -----------------------------------------------------
    # STEP 3: PRIORITY WEIGHTING
    # -----------------------------------------------------
    weighted_scores = {}

    for intent in INTENTS:
        name = intent["name"]
        priority = intent["priority"]
        weighted_scores[name] = scores[name] * priority

    # -----------------------------------------------------
    # STEP 4: BEST INTENT
    # -----------------------------------------------------
    best_intent = max(weighted_scores, key=weighted_scores.get)

    return best_intent


# =========================================================
#  BACKWARD COMPATIBILITY (FIXES YOUR ERROR)
# =========================================================

def get_intent_keyword_map() -> dict[str, list[str]]:
    """
    Used by domain_guard (legacy dependency)
    """
    return {intent["name"]: intent["keywords"] for intent in INTENTS}


def get_intent_names() -> list[str]:
    return [intent["name"] for intent in INTENTS]


def get_intent_hint(intent_name: str) -> str:
    for intent in INTENTS:
        if intent["name"] == intent_name:
            tables = intent.get("tables", [])
            join_hint = intent.get("join_hint", "")

            parts = []
            if tables:
                parts.append(f"Primary table(s): {', '.join(tables)}.")
            if join_hint:
                parts.append(f"Key join: {join_hint}.")

            return " ".join(parts)

    return ""

