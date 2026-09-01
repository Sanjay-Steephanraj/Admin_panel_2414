"""
security/domain_guard.py
─────────────────────────
Security gate with two layers of intent classification:

  Layer 1 — fast keyword match (zero cost, same as before)
  Layer 2 — LLM fallback ONLY when keywords produce "off_topic"
             (single cheap call, ~80 tokens, no schema injected)

SQL guard runs three validation layers:
  1. Dangerous-pattern block  — DML/DDL + exfiltration functions + system schemas
  2. Table allowlist          — full recursive scan catches subqueries / CTEs
  3. Column allowlist         — qualified alias.column refs validated against schema
"""
import re
import logging
from components.intent_config import get_intent_keyword_map
from dynamic.schema_resolver import get_active_table_names
from components.context_enrichment import ALLOWED_TABLES

logger = logging.getLogger(__name__)

# ── Dangerous SQL patterns ──────────────────────────────────────────────────
_DANGEROUS_SQL_PATTERNS = [
    r"\bDROP\b", r"\bTRUNCATE\b", r"\bDELETE\b", r"\bINSERT\b",
    r"\bUPDATE\s+\w+\s+SET\b", r"\bALTER\b", r"\bCREATE\b",
    r"\bGRANT\b", r"\bREVOKE\b", r"\bEXEC(?:UTE)?\b", r"\bxp_\w+",
    r"/\*.*?\*/",
    # Dangerous MySQL functions / exfiltration vectors
    r"\bLOAD_FILE\s*\(",
    r"\bINTO\s+(?:OUTFILE|DUMPFILE)\b",
    r"\bBENCHMARK\s*\(",
    r"\bSLEEP\s*\(",
    r"\bEXTRACTVALUE\s*\(",
    r"\bUPDATEXML\s*\(",
    # System schema references — covers alias tricks like
    # (SELECT ... FROM information_schema.tables) AS t
    r"\binformation_schema\b",
    r"\bperformance_schema\b",
    r"\bmysql\.[a-zA-Z_]\w*",   # mysql.user, mysql.tables_priv, etc.
    r"\bsys\.[a-zA-Z_]\w*",     # sys.processlist, etc.
]
_INJECTION_PATTERNS = [
    r";\s*\w+", r"--\s", r"/\*.*?\*/", r"\bxp_\w+", r"\bEXEC(?:UTE)?\b",
]

_DANGEROUS_SQL_RE = re.compile("|".join(_DANGEROUS_SQL_PATTERNS), re.IGNORECASE | re.DOTALL)
_INJECTION_RE     = re.compile("|".join(_INJECTION_PATTERNS),      re.IGNORECASE | re.DOTALL)

# Matches both single- and double-quoted SQL string literals, allowing for
# MySQL-style doubled quotes ('it''s') and backslash-escaped quotes ('it\'s').
_SQL_LITERAL_RE = re.compile(r"'(?:[^'\\]|\\.|'')*'|\"(?:[^\"\\]|\\.|\"\")*\"")


def _strip_sql_literals(sql: str) -> str:
    """
    Removes quoted SQL string literals from *sql* and returns the skeleton.

    MySQL doubled quotes ('') and backslash-escaped quotes are treated as
    part of the literal, so only real SQL structure remains for guarding.
    """
    return _SQL_LITERAL_RE.sub("", sql)

# Removed table regex since we parse it in _build_alias_map now:
# Commas parsing allows us to detect cross joins.

# Qualified column references: alias.column or table.column
_QUALIFIED_COL_RE = re.compile(
    r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b"
)


def _build_alias_map(sql: str) -> dict[str, str]:
    """
    Returns {alias_or_table_name: canonical_table_name} from all
    FROM/JOIN clauses in the SQL (including inside subqueries).
    e.g. "FROM sf_contacts c" → {"c": "sf_contacts", "sf_contacts": "sf_contacts"}
    """
    alias_map: dict[str, str] = {}
    
    clause_re = re.compile(r"\b(?:FROM|JOIN)\s+(.*?)(?=\b(?:WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|OFFSET|ON|LEFT|RIGHT|INNER|CROSS|OUTER|JOIN|;)\b|$)", re.IGNORECASE | re.DOTALL)
    
    for clause_match in clause_re.finditer(sql):
        content = clause_match.group(1)
        for part in content.split(","):
            part = part.strip()
            if not part: continue
            
            part = re.sub(r'--.*', '', part)
            part = re.sub(r'/\*.*?\*/', '', part, flags=re.DOTALL)
            
            tokens = part.split()
            if not tokens: continue
            
            table = tokens[0].replace("`", "").lower()
            if table.startswith("("): continue
            
            alias = table
            if len(tokens) >= 2:
                if tokens[1].lower() == "as" and len(tokens) >= 3:
                    alias = tokens[2].replace("`", "").lower()
                else:
                    alias = tokens[1].replace("`", "").lower()
            
            alias_map[alias] = table
            alias_map[table] = table

    return alias_map


def _check_columns(sql: str, alias_map: dict[str, str]) -> tuple[bool, str]:
    """
    Validates every alias.column reference in the SQL against the schema.
    Case-insensitive check.
    """
    allowed_columns_lower: dict[str, set[str]] = {
        tbl.lower(): {c.lower() for c in meta["columns"].keys()}
        for tbl, meta in ALLOWED_TABLES.items()
    }
    allowed_columns_orig: dict[str, list[str]] = {
        tbl.lower(): sorted(meta["columns"].keys())
        for tbl, meta in ALLOWED_TABLES.items()
    }

    for match in _QUALIFIED_COL_RE.finditer(sql):
        prefix = match.group(1).lower()
        column = match.group(2).lower()

        canonical_table = alias_map.get(prefix)
        if canonical_table is None:
            continue

        table_cols_lower = allowed_columns_lower.get(canonical_table)
        if table_cols_lower is None:
            continue

        if column not in table_cols_lower:
            return (
                False,
                f"SQL references disallowed column '{column}' on table '{canonical_table}'. "
                f"Allowed columns: {', '.join(allowed_columns_orig[canonical_table])}",
            )

    return True, ""


# ── Name detection fallback ─────────────────────────────────────────────────
def _looks_like_person_name(q: str) -> bool:
    tokens = q.strip().split()
    if not (1 <= len(tokens) <= 4):
        return False
    return all(token.isalpha() for token in tokens)


# ── Layer 1: keyword-based classifier ───────────────────────────────────────
def _classify_intent_keywords(question: str) -> str:
    q = question.lower()
    intent_keywords = get_intent_keyword_map()

    matched = [
        intent for intent, keywords in intent_keywords.items()
        if any(kw in q for kw in keywords)
    ]

    if matched:
        specific = [m for m in matched if m != "aggregate"]
        return specific[0] if specific else "aggregate"

    if _looks_like_person_name(q):
        logger.info(f"Name-based fallback triggered for: {q!r}")
        return "entity_lookup"

    return "off_topic"


# ── Layer 2: LLM fallback (called ONLY when keywords return off_topic) ───────
_VALID_INTENTS = {
    "donor", "ministry", "payment", "opportunity",
    "aggregate", "entity_lookup", "off_topic",
}


def _classify_intent_llm(question: str) -> str:
    """
    Single cheap LLM call to classify intent when keyword matching fails.
    No schema injected — minimal token cost (~80 tokens total).
    Returns one of the valid intent strings.
    """
    # Mask PII before sending to LLM. Fail-closed: if masking raises, mirror
    # the existing LLM-failure behavior (return off_topic) — never send raw.
    try:
        from data_masking import masker
        masked_question, _ = masker.mask_text(question)
    except RuntimeError:
        logger.warning("PII masking failed — skipping LLM intent classifier, defaulting off_topic")
        return "off_topic"

    try:
        from llm.gloo_client import get_llm
        from langchain_core.messages import SystemMessage, HumanMessage

        system_prompt = (
            "You classify admin questions about a donor CRM system.\n"
            "Reply with EXACTLY one word from this list:\n"
            "  donor | ministry | payment | opportunity | aggregate | off_topic\n\n"
            "Definitions:\n"
            "  donor       — questions about people, contacts, givers, supporters\n"
            "  ministry    — questions about funds, churches, organizations, programs\n"
            "  payment     — questions about transactions, amounts received, card/check/cash\n"
            "  opportunity — questions about pledges, pipeline, recurring donations\n"
            "  aggregate   — counting, totalling, listing, reporting, ranking any of the above\n"
            "  off_topic   — anything unrelated to donors, ministries, payments, or opportunities\n\n"
            "Output only the single word. No punctuation, no explanation."
        )

        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=masked_question),
        ])
        intent = response.content.strip().lower().rstrip(".")

        if intent not in _VALID_INTENTS:
            logger.warning(
                f"LLM intent classifier returned unknown value: {intent!r} — defaulting off_topic"
            )
            return "off_topic"

        logger.info(f"LLM intent classifier: {intent!r} | q={masked_question!r}")
        return intent

    except Exception as e:
        logger.error(f"LLM intent classification failed: {e} — defaulting off_topic")
        return "off_topic"


# ── Combined two-layer classifier ────────────────────────────────────────────
def _classify_intent(question: str) -> str:
    """
    Layer 1: fast keyword match (free).
    Layer 2: LLM fallback — only when layer 1 returns off_topic.
    """
    intent = _classify_intent_keywords(question)
    if intent != "off_topic":
        return intent

    logger.info(
        f"Keyword classifier returned off_topic — escalating to LLM for: {question!r}"
    )
    return _classify_intent_llm(question)


# ── Domain guard ─────────────────────────────────────────────────────────────
class DomainGuard:

    @staticmethod
    def check_question(question: str) -> tuple[bool, str, str]:
        """Returns (is_safe, reason, intent)."""
        q = question.strip()

        if len(q) < 3:
            return False, "Question is too short.", "off_topic"

        if _INJECTION_RE.search(q):
            logger.warning(f"Injection blocked: {q!r}")
            return False, "injection", "off_topic"

        intent = _classify_intent(q)

        if intent == "off_topic":
            logger.warning(f"Off-topic blocked: {q!r}")
            return False, "off_topic", "off_topic"

        logger.info(f"Guard passed | intent={intent} | q={q!r}")
        return True, "", intent

    @staticmethod
    def check_sql(sql: str) -> tuple[bool, str]:
        """
        Validates generated SQL through three layers:

          1. Dangerous-pattern block  — DML, DDL, exfiltration functions
             (LOAD_FILE, INTO OUTFILE/DUMPFILE, BENCHMARK, SLEEP, etc.),
             and system-schema references (information_schema, performance_schema,
             mysql.*, sys.*) — catches subquery-alias tricks.
          2. Table allowlist          — every FROM/JOIN table (including inside
             subqueries / CTEs / WHERE clauses) must be in ALLOWED_TABLES.
          3. Column allowlist         — every qualified alias.column reference must
             name a column that exists in that table's schema.
        """
        if not sql or not sql.strip():
            return False, "Generated SQL is empty."

        # Strip quoted string literals once; all SQL-structure scans operate on
        # the skeleton so data inside literals cannot spoof FROM/JOIN tables,
        # qualified column refs, or dangerous-pattern matches.
        sql_skeleton = _strip_sql_literals(sql)

        # Layer 1 — dangerous patterns (includes system schema names).
        if _DANGEROUS_SQL_RE.search(sql_skeleton):
            logger.warning(f"Dangerous SQL blocked:\n{sql}")
            return False, "Generated SQL contains disallowed operations."

        # Layer 2 — table allowlist
        allowed    = set(get_active_table_names())
        alias_map  = _build_alias_map(sql_skeleton)
        referenced = set(alias_map.values())          # canonical table names

        logger.info(f"Tables referenced: {referenced} | allowed: {allowed}")

        disallowed = referenced - allowed
        if disallowed:
            return (
                False,
                f"SQL references disallowed table(s): {', '.join(sorted(disallowed))}. "
                f"Allowed: {', '.join(sorted(allowed))}",
            )

        # Layer 3 — column allowlist
        col_ok, col_err = _check_columns(sql_skeleton, alias_map)
        if not col_ok:
            logger.warning(f"Disallowed column reference blocked:\n{sql}\n{col_err}")
            return False, col_err

        return True, ""


domain_guard = DomainGuard()