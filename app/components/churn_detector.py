"""
components/churn_detector.py
─────────────────────────────
Detects donor churn/inactivity-risk questions so they can be routed to the
deterministic churn analysis path (analysis/churn_analyzer.py) instead of the
LLM NL2SQL pipeline.

Conservative by design: only clearly churn-related questions match.
"""
import re

# Page-size default when a pagination follow-up gives no explicit number.
# Mirrors analysis.churn_analyzer.RESULT_LIMIT; kept local to avoid an
# import-cycle / cross-module coupling for a single constant.
DEFAULT_PAGE_SIZE = 15

# Word-boundary, case-insensitive regex patterns.
CHURN_KEYWORDS: list[str] = [
    r"\bchurn(?:ing|ed)?\b",
    r"\bat risk\b",
    r"\brisk of churn\b",
    r"\blapsed donors?\b",
    r"\binactive donors?\b",
    r"\bdonors?\s+(?:who\s+are\s+)?inactive\b",
    r"\binactive\b\s+donors?",  # extra safety: inactive before donors
    r"\bhaven'?t donated\b",
    r"\bhave not donated\b",
    r"\bhaven'?t given\b",
    r"\bhave not given\b",
    r"\bstopped giving\b",
    r"\bstopped donating\b",
    r"\binactivity\b",
    r"\bno longer giving\b",
    r"\bno longer donating\b",
    r"\bdonors?\s+we might lose\b",
    r"\bdonors?\s+at risk\b",
]

_PATTERN = re.compile("|".join(CHURN_KEYWORDS), re.IGNORECASE)


def detect_churn_intent(question: str) -> bool:
    """Return True if the question is clearly about donor churn/inactivity risk."""
    if not question:
        return False
    return bool(_PATTERN.search(question))


# Pagination follow-up patterns. The MATCH must extend to the end of the
# question ($-anchored) but the prefix is not anchored — short lead-ins like
# "say me", "tell me", "can you show" are tolerated because the question is
# also required to be terse (<= 8 words) and free of churn/content words.
_FOLLOWUP_PATTERNS = [
    re.compile(
        r"next\s*(\d+)?\s*(?:donors|results|ones|records|entries)?\s*\??\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\d+)\s+more(?:\s+(?:donors|results|records))?\s*\??\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"more(?:\s+(?:donors|results|records))?\s*\??\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"the\s+rest\s*\??\s*$", re.IGNORECASE),
    re.compile(
        r"remaining(?:\s+(?:donors|results))?\s*\??\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"next\s+batch\s*\??\s*$", re.IGNORECASE),
]

# Content words that indicate a real question, not a bare pagination request.
# deliberate: a fixed word list — lighter than a full POS scan and sufficient
# because pagination follow-ups are content-free by definition.
_CONTENT_WORDS = re.compile(
    r"\b(risk|churn(?:ing|ed)?|lapsed|inactive|inactivity|donated|donating|"
    r"payment|payments|ministry|ministries|contributions|giving|stopped|"
    r"haven|active|opportunities|total)\b",
    re.IGNORECASE,
)


def detect_churn_followup(question: str) -> int | None:
    """
    Detects pagination follow-ups like "give me the next 15", "show more",
    "next", "show the rest", "remaining donors", "say me next 5",
    "tell me the next 20 donors", "can you show next 10".
    Returns the requested count (int) when a number is present,
    DEFAULT_PAGE_SIZE when no number is given, or None if the question
    is not a bare pagination follow-up.
    NOTE: caller must verify the previous conversation turn was a churn
    answer — this function only checks the question's surface form.
    """
    if not question or not question.strip():
        return None
    # Strip trailing punctuation/whitespace and lowercase.
    q = question.strip().rstrip("?.!,;:").strip()
    if not q:
        return None
    # Pagination follow-ups are terse; guard against matching long real
    # questions that happen to end in "next 5".
    if len(q.split()) > 8:
        return None
    # A full churn question is not a pagination follow-up, and content words
    # indicate a fresh question rather than a bare "more" request.
    if detect_churn_intent(q) or _CONTENT_WORDS.search(q):
        return None
    num = None
    for pat in _FOLLOWUP_PATTERNS:
        m = pat.search(q)
        if m:
            for g in m.groups():
                if g is not None and g.isdigit():
                    num = int(g)
                    break
            break
    if num is None and not any(pat.search(q) for pat in _FOLLOWUP_PATTERNS):
        return None
    if num is None:
        return DEFAULT_PAGE_SIZE
    return min(max(num, 1), 50)


# Patterns for extracting an explicit result-count from a fresh churn
# question. Word boundaries (\b) ensure ministry codes like "801BOB" or
# "098WRLD" are not counted: there is no word boundary between the digits
# and the adjacent letters inside such a token.
_COUNT_PATTERNS = [
    re.compile(
        r"(?:top|first|next|show|list|give me|say me)\s+(\d{1,3})\b",
        re.IGNORECASE,
    ),
    re.compile(r"(\d{1,3})\s+(?:donors|results)\b", re.IGNORECASE),
]


def extract_requested_count(question: str) -> int | None:
    """
    Extracts an explicit result-count request from a question, e.g.
    "top 5 donors at risk", "show 20 lapsed donors", "first 10 donors
    who stopped giving". Returns the count clamped to 1..50, or None
    when the question doesn't specify one.
    """
    if not question:
        return None
    for pat in _COUNT_PATTERNS:
        m = pat.search(question)
        if m:
            return min(max(int(m.group(1)), 1), 50)
    return None


def resolve_churn_route(
    raw_question: str,
    expanded_question: str,
    normalized_question: str,
    history: list[dict] | None,
) -> dict | None:
    """
    Decides whether a request should take the deterministic churn path.

    Priority:
      1. Pagination follow-up — the RAW question looks like "next 15"/"show more"
         AND the immediately-previous turn was a churn answer
         (sql == "CHURN_ANALYSIS"). Continues from the prior offset and reuses
         the prior ministry scope. Checked FIRST because the question expander
         may rewrite bare follow-ups into full churn questions, which would
         otherwise restart at page 1.
      2. Fresh churn intent — normalized or expanded question matches churn
         keywords. Starts at offset 0.

    Returns None (not a churn request) or:
      {"offset": int, "limit": int | None, "entities_override": dict | None}
    """
    last_turn = history[-1] if history else None
    followup_count = detect_churn_followup(raw_question)
    if followup_count is not None and last_turn and last_turn.get("sql") == "CHURN_ANALYSIS":
        return {
            "offset": int(last_turn.get("churn_offset", 0) or 0) + int(last_turn.get("churn_shown", 0) or 0),
            "limit": followup_count,
            "entities_override": last_turn.get("churn_entities") or None,
        }
    if detect_churn_intent(normalized_question) or detect_churn_intent(expanded_question):
        limit = (extract_requested_count(raw_question)
                 or extract_requested_count(normalized_question))
        return {"offset": 0, "limit": limit, "entities_override": None}
    return None


if __name__ == "__main__":
    positives = [
        "who are the donors who are in risk of churning",
        "show me lapsed donors",
        "which donors are at risk",
        "show inactive donors",
        "donors who haven't donated in a while",
        "donors who have not given recently",
        "donors who stopped giving",
        "show donors we might lose",
        "donors at risk of churn",
        "any donor inactivity trends",
        "donors no longer donating",
    ]
    negatives = [
        "how many donors do we have in total",
        "show payments by Daniel",
        "total contributions to 801BOB",
        "list all active donors",
        "which ministries are inactive",
        "show me all opportunities",
    ]
    print("positives:")
    for q in positives:
        print(f"  {detect_churn_intent(q)!s:5} {q}")
    print("negatives:")
    for q in negatives:
        print(f"  {detect_churn_intent(q)!s:5} {q}")
