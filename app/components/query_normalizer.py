"""
components/query_normalizer.py
────────────────────────────────
Pre-processes the user's natural language question BEFORE it hits the LLM.

Entity extraction strategy (cost-efficient two-layer approach):
  Layer 1 — regex extraction (free, fast)
             Returns entities with a confidence score.
  Layer 2 — LLM extraction — called ONLY when regex confidence is low
             (a single cheap call, ~100 tokens, no schema injected)

This prevents silently corrupted ministry filters from bad regex matches
while keeping the LLM call count at zero for well-structured questions.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# SYNONYMS
# ─────────────────────────────────────────────
SYNONYM_MAP: list[tuple[str, str]] = [
    (r"\bcontributions?\b", "payments"),
    (r"\bdonations?\b",     "payments"),
    (r"\bgifts?\b",         "payments"),
    (r"\btransactions?\b",  "payments"),
    # domain synonym: campaign(s) → ministry/ministries (plurality preserved)
    (r"\bcampaigns\b",      "ministries"),
    (r"\bcampaign\b",       "ministry"),
]

_COMPILED_SYNONYMS = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in SYNONYM_MAP
]

# ─────────────────────────────────────────────
# PATTERN NORMALIZATIONS
# ─────────────────────────────────────────────
PATTERN_NORMALIZATIONS = [
    (r"how many (.+?) are there", r"count of \1"),
    (r"what is the total (.+)",   r"total \1"),
    (r"can you (show|list|get)\s", r"\1 "),
]

_COMPILED_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in PATTERN_NORMALIZATIONS
]

# ─────────────────────────────────────────────
# STOP WORDS — stripped from name candidates
# ─────────────────────────────────────────────
_LEADING_STOPWORDS = re.compile(
    r"^(?:by|for|from|of|in|at|the|and|or|is|are|was|were|"
    r"made|what|show|get|list|all|about|regarding|related|to|"
    r"did|do|does|contributions?|payments?|donations?|gifts?|transactions?)\s+",
    flags=re.IGNORECASE,
)

# ─────────────────────────────────────────────
# REGEX ENTITY EXTRACTION  (Layer 1)
# ─────────────────────────────────────────────

# Ministry code: alphanumeric tokens with BOTH letters and digits
_CODE_PAREN_RE = re.compile(
    r"\(\s*(?!\d+(?:st|nd|rd|th)\b)([A-Za-z]*\d+[A-Za-z]+|[A-Za-z]+\d+[A-Za-z]*)\s*\)",
    re.IGNORECASE,
)
_CODE_BARE_RE = re.compile(
    r"\b(?!\d+(?:st|nd|rd|th)\b)([A-Za-z]*\d+[A-Za-z]+|[A-Za-z]+\d+[A-Za-z]*)\b",
    re.IGNORECASE,
)

# Ministry name: 1-4 words directly before "ministry"
_NAME_RE = re.compile(
    r"\b((?:[A-Za-z]+\s+){1,4}?ministry)\b",
    re.IGNORECASE,
)

# Tokens that, if they end up as the entire "name", are meaningless
_STOPWORD_NAMES = {
    "ministry", "the ministry", "a ministry", "our ministry",
    "this ministry", "that ministry", "their ministry",
}

_ORG_LIKE_RE_LOCAL = re.compile(
    r"\b(missions?|trust|foundation|charitable|inc|incorporated|"
    r"church|organization|society|association|"
    r"services?|group|fund|charity|international)\b",
    re.IGNORECASE,
)


def _regex_extract(question: str) -> dict:
    """
    Returns:
        {
            "ministry_name": str | None,
            "ministry_code": str | None,
            "confidence":    "high" | "low"
        }

    Confidence is "high" only when the extracted values look unambiguous:
      - code found inside parentheses  → high
      - name candidate is a clean phrase (≤3 words before 'ministry') → high
      - anything else                  → low
    """
    # Apply synonym normalisation so "church" → "ministry" before matching
    normed = question
    for pattern, replacement in _COMPILED_SYNONYMS:
        normed = pattern.sub(replacement, normed)

    code       = None
    code_high  = False
    name       = None
    name_high  = False

    # ── Code extraction ──
    paren_match = _CODE_PAREN_RE.search(normed)
    if paren_match:
        code      = paren_match.group(1).upper()
        code_high = True
    else:
        bare_match = _CODE_BARE_RE.search(normed)
        if bare_match:
            code      = bare_match.group(1).upper()
            code_high = False   # bare code is ambiguous

    # ── Name extraction ──
    name_match = _NAME_RE.search(normed)
    if name_match:
        candidate = name_match.group(1).strip()

        # Iteratively strip leading stop-words
        prev = None
        while prev != candidate:
            prev      = candidate
            candidate = _LEADING_STOPWORDS.sub("", candidate).strip()

        candidate_lower = candidate.lower()

        if candidate_lower and candidate_lower not in _STOPWORD_NAMES:
            name = candidate_lower
            # High confidence only if the phrase is ≤ 3 words
            name_high = len(candidate.split()) <= 3
        # else name stays None — don't inject a bad filter

    # ── Confidence decision ──
    if code and not name:
        # A bare code with NO competing name candidate is unambiguous —
        # there is nothing else to misinterpret. Mark high confidence.
        # Example: "900DAC" alone → definitely a ministry code.
        confidence = "high"
    elif name and not code:
        confidence = "high" if name_high else "low"
    elif code and name:
        # Both present: only high if the code was in parens (unambiguous anchor).
        # A bare code alongside a name phrase could be a false positive for either.
        confidence = "high" if code_high else "low"
    else:
        if _ORG_LIKE_RE_LOCAL.search(normed):
            confidence = "low" # Looks like an org, let LLM check
        else:
            confidence = "high"   # nothing found — nothing to be wrong about

    return {
        "ministry_name": name,
        "ministry_code": code,
        "confidence":    confidence,
    }


# ─────────────────────────────────────────────
# LLM ENTITY EXTRACTION  (Layer 2 — fallback)
# ─────────────────────────────────────────────

def _llm_extract(question: str) -> dict:
    """
    Cheap LLM call that extracts ministry entities from the question.
    Called ONLY when regex confidence is "low".
    Returns {"ministry_name": str|None, "ministry_code": str|None}.
    """
    # Mask PII before sending to LLM. Fail-closed: if masking raises, skip the
    # LLM fallback and return empty entities (callers keep the regex result).
    try:
        from data_masking import masker
        masked_question, mapping = masker.mask_text(question)
    except RuntimeError:
        logger.warning("PII masking failed — skipping LLM entity extraction, returning empty entities")
        return {"ministry_name": None, "ministry_code": None}

    try:
        import json
        from llm.gloo_client import get_llm
        from langchain_core.messages import SystemMessage, HumanMessage

        system_prompt = (
            "Extract ministry identifiers from the admin's question.\n"
            "A ministry code is an alphanumeric token like '532PHI', 'HOPE1', 'ABC123'.\n"
            "A ministry name is a proper noun phrase like 'Petals of Hope', 'Grace Community'.\n\n"
            "Respond with ONLY valid JSON, no markdown, no explanation:\n"
            '{"ministry_name": "<name or null>", "ministry_code": "<code or null>"}\n\n'
            "Rules:\n"
            "- ministry_name: the name WITHOUT the word 'ministry' or 'church', lowercase, or null\n"
            "- ministry_code: the alphanumeric code only, uppercase, or null\n"
            "- If nothing is mentioned, return null for both fields\n"
            "- CRITICAL RULE: If the user is asking for 'contact details', 'profile', 'email', 'phone', or explicitly calls the entity a 'donor', the name belongs to a donor/contact, NOT a ministry. In this case, return null for ministry_name.\n"
            "- Otherwise, if it is an organization like 'Riverstone church', extract 'riverstone' as the ministry_name."
        )

        llm  = get_llm()
        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=masked_question),
        ])
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        parsed = json.loads(raw)
        name   = parsed.get("ministry_name") or None
        code   = parsed.get("ministry_code") or None

        # Sanitise: strip surrounding quotes the LLM sometimes adds
        if isinstance(name, str):
            name = name.strip().strip('"\'').lower() or None
        if isinstance(code, str):
            code = code.strip().strip('"\'').upper() or None

        # Unmask: a masked ministry NAME may come back as a token (codes are
        # allowlisted and stay literal). Restore real values for downstream SQL.
        if name:
            name = masker.unmask_text(name, mapping)
        if code:
            code = masker.unmask_text(code, mapping)

        logger.info(f"LLM entity extraction: name={name!r} code={code!r}")
        return {"ministry_name": name, "ministry_code": code}

    except Exception as e:
        logger.error(f"LLM entity extraction failed: {e} — using empty entities")
        return {"ministry_name": None, "ministry_code": None}


# ─────────────────────────────────────────────
# PUBLIC: extract_ministry_entities
# ─────────────────────────────────────────────

def extract_ministry_entities(question: str) -> dict:
    """
    Two-layer extraction:
      1. Regex  (free, instant)  → if confidence == "high", use as-is
      2. LLM fallback            → only when regex confidence == "low"

    Always returns {"ministry_name": str|None, "ministry_code": str|None}
    (confidence key is stripped before returning to callers).
    """
    result = _regex_extract(question)
    confidence = result.pop("confidence")

    if confidence == "high":
        logger.info(
            f"Regex entity extraction (high confidence): "
            f"name={result['ministry_name']!r} code={result['ministry_code']!r}"
        )
        return result

    # Low confidence — escalate to LLM
    logger.info(
        f"Regex entity extraction low confidence "
        f"(name={result['ministry_name']!r}, code={result['ministry_code']!r}) "
        f"— escalating to LLM"
    )
    return _llm_extract(question)


# ─────────────────────────────────────────────
# PUBLIC: normalize_question
# ─────────────────────────────────────────────

def normalize_question(question: str) -> dict:
    """
    Returns:
        {
            "normalized_question": str,
            "entities": {
                "ministry_name": str | None,
                "ministry_code": str | None,
            }
        }
    """
    original   = question
    normalized = question.strip()

    # Step 1: pattern normalisation
    for pattern, replacement in _COMPILED_PATTERNS:
        normalized = pattern.sub(replacement, normalized)

    # Step 2: synonym normalisation
    for pattern, replacement in _COMPILED_SYNONYMS:
        normalized = pattern.sub(replacement, normalized)

    # Collapse extra whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # Entity extraction from the ORIGINAL question
    entities = extract_ministry_entities(original)

    if normalized != original:
        logger.info(
            f"Query normalized:\n  Original:   {original!r}\n"
            f"  Normalized: {normalized!r}"
        )

    logger.info(f"Extracted entities: {entities}")

    return {
        "normalized_question": normalized,
        "entities":            entities,
    }


# ─────────────────────────────────────────────
# DEBUG HELPER
# ─────────────────────────────────────────────

def get_normalization_diff(question: str) -> dict:
    result = normalize_question(question)
    return {
        "original":   question,
        "normalized": result["normalized_question"],
        "entities":   result["entities"],
        "changed":    result["normalized_question"] != question,
    }


# ─────────────────────────────────────────────
# AMBIGUITY DETECTION
# ─────────────────────────────────────────────

# Signals that the name is clearly the DONOR (the one paying)
_BY_RE = re.compile(
    r"\b(by|from|made by|done by|contributed by|given by)\b",
    re.IGNORECASE,
)

# Signals that the name is clearly the MINISTRY (the recipient)
_TO_RE = re.compile(
    r"\b(to|for|towards?|into|designated to|allocated to)\b",
    re.IGNORECASE,
)

# Org-suffix words — if present the name looks like a potential ministry too
_ORG_LIKE_RE = re.compile(
    r"\b(missions?|trust|foundation|charitable|inc|incorporated|"
    r"ministries|ministry|church|organization|society|association|"
    r"services?|group|fund|charity|international)\b",
    re.IGNORECASE,
)

# Name-like phrase: 2-5 Title-Case words after payment-related verbs
_NAME_PHRASE_RE = re.compile(
    r"(?:payments?|donations?|contributions?|gifts?|transactions?)"
    r"(?:\s+\w+){0,3}\s+"          # allow "done", "made" etc between
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})",  # capture TitleCase phrase
)


def detect_name_ambiguity(question: str) -> dict | None:
    """
    Returns a clarification dict when the name in the question could belong
    to EITHER a donor (sf_contacts) OR a ministry (sf_ministries), and no
    preposition makes it clear.

    Returns:
        {
            "ambiguous_name": str,          # the extracted name phrase
            "clarification_question": str,  # ready-to-send message to user
        }
        — or None if the question is unambiguous.

    A question is AMBIGUOUS when ALL three are true:
      1. There is NO "by/from" preposition  (would point to donor)
      2. There is NO "to/for" preposition   (would point to ministry)
      3. An org-suffix word OR a multi-word TitleCase phrase is present

    If either preposition is found, the question is already unambiguous
    and we let the existing logic handle it.
    """
    has_by = bool(_BY_RE.search(question))
    has_to = bool(_TO_RE.search(question))

    # Clear context — no ambiguity
    if has_by or has_to:
        return None

    # No preposition at all — check if there's even a name-like phrase
    match = _NAME_PHRASE_RE.search(question)
    if not match:
        return None

    name = match.group(1).strip()

    # Only flag as ambiguous if the name has org-like suffixes OR is multi-word
    # (single common words like "January" won't match)
    is_org_like  = bool(_ORG_LIKE_RE.search(name))
    is_multiword = len(name.split()) >= 2

    if not (is_org_like or is_multiword):
        return None

    clarification = (
        f'Did you mean payments **made by "{name}"** (a donor), '
        f'or payments **received by "{name}"** (a ministry/fund)?\n\n'
        f'Please reply with one of:\n'
        f'• "payments by {name}" — to see what that donor paid\n'
        f'• "payments to {name}" — to see what that fund received'
    )

    logger.info(f"Name ambiguity detected: {name!r}")
    return {
        "ambiguous_name":           name,
        "clarification_question":   clarification,
    }


# ─────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    cases = [
        "What are the contributions made by Petals of Hope Ministry (532PHI) in January",
        "Show payments for 532PHI in March",
        "List all donations from Grace Community Ministry last year",
        "What transactions did the New Life Church make in 2024",
        "Total contributions for ABC123 ministry",
        # Tricky cases the old regex got wrong:
        "show all payments from Grace Community last year",
        "contributions made by hope ministry this quarter",
    ]

    for q in cases:
        result = get_normalization_diff(q)
        print(f"\nQ : {result['original']}")
        print(f"N : {result['normalized']}")
        print(f"E : {result['entities']}")