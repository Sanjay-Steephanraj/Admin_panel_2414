import logging
import re
from langchain_core.messages import HumanMessage, SystemMessage

from ..states.nlsql_state import NLSQLState
from llm.gloo_client import get_llm
from llm.prompts import build_sql_generation_prompt
from security.domain_guard import domain_guard

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

# ─────────────────────────────────────────────
# Scrub pattern — removes any ID/code filter the LLM hallucinated
# Matches:  AND <alias>.<id_column> = '<any value>'
# Note: m.name is intentionally excluded so legitimate name-based
#       filters are not stripped.
# ─────────────────────────────────────────────
_LLM_HALLUCINATION_PATTERN = re.compile(
    r"(?:LOWER\(|UPPER\()?\s*\w+\.(?:sfid|giftcode|ministry_?id)\s*\)?(?:\s*[=!<>]+\s*'[^']*'|\s+LIKE\s+'[^']*')",
    re.IGNORECASE,
)

# Hallucinated condition WITH its leading boolean connector (AND/OR).
# Replacing the whole match with a space removes both the condition and the
# connector, so OR-branch tautologies like "OR 1=1" can never be produced.
_HALLUCINATION_LEADING_CONNECTOR_RE = re.compile(
    r"\s+(?:AND|OR)\s+"
    r"(?:LOWER\(|UPPER\()?\s*\w+\.(?:sfid|giftcode|ministry_?id)\s*\)?"
    r"(?:\s*[=!<>]+\s*'[^']*'|\s+LIKE\s+'[^']*')",
    re.IGNORECASE,
)

# Hallucinated condition WITH a trailing boolean connector — handles the
# case where the hallucination is the FIRST condition after WHERE.
_HALLUCINATION_TRAILING_CONNECTOR_RE = re.compile(
    r"(?:LOWER\(|UPPER\()?\s*\w+\.(?:sfid|giftcode|ministry_?id)\s*\)?"
    r"(?:\s*[=!<>]+\s*'[^']*'|\s+LIKE\s+'[^']*')\s+(?:AND|OR)\s+",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────
# Entity sanitizers — defense at the interpolation boundary.
# Entities arrive from LLM extraction and are embedded directly into SQL
# string literals, so every value is reduced to a safe character class before
# interpolation. domain_guard.check_sql does NOT catch injected structure
# hidden inside a literal (its _strip_sql_literals swallows literals whole),
# so this is the trust boundary for these values.
# ─────────────────────────────────────────────
_SAFE_CODE_RE = re.compile(r"[^A-Za-z0-9_]")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9 \-&.,']")


def _sanitize_ministry_code(code: str) -> str | None:
    """
    Ministry codes are alphanumeric identifiers (e.g. '532PHI'). Strip every
    other character so no quote, comment marker, or SQL keyword separator can
    reach the query. Returns None when nothing usable remains.
    """
    # Entities arrive from LLM JSON; a hallucinating model can return a
    # non-string (int/bool/list/dict). Coercing those with str() produces
    # nonsense filters (True -> "TRUE"), so ignore them outright.
    if not isinstance(code, str):
        return None
    cleaned = _SAFE_CODE_RE.sub("", code).upper()[:32]
    return cleaned or None


def _sanitize_ministry_name(name: str) -> str | None:
    """
    Ministry display names may contain spaces, hyphens, ampersands, periods,
    commas and apostrophes. Everything else is removed, and any remaining
    apostrophe is escaped by doubling ('' ) for MySQL string literals, so the
    value cannot terminate the literal it is embedded in.
    Also strips LIKE wildcards (% and _) so user input cannot widen the match.
    Returns None when nothing usable remains.
    """
    # See _sanitize_ministry_code: ignore non-string entities rather than
    # coercing them into garbage filters.
    if not isinstance(name, str):
        return None
    s = name
    # Remove LIKE wildcards first so user input cannot widen the match.
    s = s.replace("%", "").replace("_", "")
    # Keep only the safe display-name character class.
    s = _SAFE_NAME_RE.sub("", s)
    # Collapse runs of whitespace and trim.
    s = re.sub(r"\s+", " ", s).strip()[:100]
    if not s:
        return None
    # Escape apostrophes for MySQL string literals AFTER the empty check so
    # the pre-escape emptiness test is meaningful.
    return s.replace("'", "''")


def _find_outer_where(sql: str) -> int | None:
    """
    Returns the index of the first paren-depth-0, quote-safe WHERE keyword,
    or None if none exists. Mirrors the depth+quote scan in _find_outer_clause.
    """
    depth = 0
    in_quote = False
    quote_char = None
    pattern = re.compile(r"(\bWHERE\b|\(|\)|'|\")", re.IGNORECASE)
    for match in pattern.finditer(sql):
        token = match.group(1).upper()
        if token in ("'", '"'):
            if not in_quote:
                in_quote = True
                quote_char = token
            elif quote_char == token:
                in_quote = False
        elif not in_quote:
            if token == "(":
                depth += 1
            elif token == ")":
                depth = max(0, depth - 1)
            elif depth == 0 and token == "WHERE":
                return match.start()
    return None


def _scrub_hallucinated_conditions(sql: str) -> str:
    """
    Removes hallucinated ministry/ID filters the LLM added, WITHOUT
    substituting a tautology that could OR-leak across ministries.

    - A hallucinated condition with a leading AND/OR is removed together with
      that connector (replaced with a single space).
    - A hallucinated condition that is the first condition after WHERE (trailing
      AND/OR) is removed together with its trailing connector.
    - A lone hallucinated condition that is the ONLY condition in its WHERE
      clause is replaced with `1=1` — a harmless tautology that keeps the
      statement syntactically valid (WHERE 1=1).

    Only the portion of the SQL from the first paren-depth-0 WHERE onward is
    scrubbed, so JOIN ... ON clauses are never touched.
    """
    where_idx = _find_outer_where(sql)
    if where_idx is None:
        return sql

    head = sql[:where_idx]
    tail = sql[where_idx:]

    # Pass 1: hallucination + leading connector.
    tail = _HALLUCINATION_LEADING_CONNECTOR_RE.sub(" ", tail)
    # Pass 2: hallucination + trailing connector (first-condition case).
    tail = _HALLUCINATION_TRAILING_CONNECTOR_RE.sub(" ", tail)
    # Pass 3: any remaining lone hallucination (only condition in WHERE) → 1=1.
    tail = _LLM_HALLUCINATION_PATTERN.sub(" 1=1 ", tail)

    return head + tail


def _find_outer_clause(sql: str) -> int | None:
    """
    Finds the first occurring GROUP BY, ORDER BY, LIMIT, or OFFSET keyword
    that is NOT nested inside parentheses or quotes.
    """
    depth = 0
    in_quote = False
    quote_char = None
    
    pattern = re.compile(r"(GROUP\s+BY|ORDER\s+BY|LIMIT|OFFSET|\(|\)|'|\")", re.IGNORECASE)
    
    for match in pattern.finditer(sql):
        token = match.group(1).upper()
        if token in ("'", '"'):
            if not in_quote:
                in_quote = True
                quote_char = token
            elif quote_char == token:
                in_quote = False
        elif not in_quote:
            if token == "(":
                depth += 1
            elif token == ")":
                depth = max(0, depth - 1)
            elif depth == 0 and token.startswith(("GROUP", "ORDER", "LIMIT", "OFFSET")):
                return match.start()
    return None



# ─────────────────────────────────────────────
# Ministry filter injection
# ─────────────────────────────────────────────
def _inject_ministry_filter(sql: str, entities: dict, query_spec: dict | None = None) -> str:
    """
    1. Scrubs any hallucinated ministry/ID filter the LLM added
    2. Injects deterministic ministry filters from extracted entities

    DB format:
      sf_ministries.giftcode = "532PHI"               <- ministry code column
      sf_ministries.name     = "Petals of Hope Ministry"  <- display name column

    Handles:
    - code only  → UPPER(m.giftcode) = '532PHI'
    - name only  → UPPER(m.name) LIKE '%PETALS OF HOPE MINISTRY%'
    - both       → OR-combined so either match succeeds
    """

    if not entities:
        return sql

    spec = query_spec or {}
    # A resolved canonical Salesforce id is authoritative; raw text is only a
    # fallback for legacy non-resolved requests.
    canonical_id = _sanitize_ministry_code(spec.get("entity_id"))
    code = _sanitize_ministry_code(entities.get("ministry_code"))
    name = _sanitize_ministry_name(entities.get("ministry_name"))

    # ── Step 1: Remove hallucinated LLM filters ──
    sql = _scrub_hallucinated_conditions(sql)
    sql = re.sub(r"\s{2,}", " ", sql).strip()

    # ── Step 2: Build conditions ──
    conditions = []

    if canonical_id:
        conditions.append(f"m.sfid = '{canonical_id}'")
    elif code:
        conditions.append(f"UPPER(m.giftcode) = '{code.upper()}'")

    if name:
        conditions.append(f"UPPER(m.name) LIKE '%{name.upper()}%'")

    if not conditions:
        logger.info("Ministry filter injected | code=None name=None")
        return sql

    logger.info(f"Ministry filter injected | code={code!r} name={name!r}")

    # OR when both present — tolerates partial user input
    ministry_filter = (
        f"({conditions[0]} OR {conditions[1]})"
        if len(conditions) == 2
        else conditions[0]
    )

    # ── Step 3: Inject into SQL ──
    where_idx = _find_outer_where(sql)
    if where_idx is not None:
        # Splice at the outer WHERE (paren-depth 0, not inside a string
        # literal). re.sub would match the first textual WHERE, which can be
        # inside a subquery or a string literal — the former silently drops
        # the ministry scope (cross-ministry leak), the latter corrupts the
        # literal. See tests/test_sql_injection.py.
        sql = (
            sql[:where_idx]
            + f"WHERE {ministry_filter} AND "
            + sql[where_idx + len("WHERE"):]
        )
    else:
        # No outer WHERE — insert a new one before the first depth-0
        # GROUP BY / ORDER BY / LIMIT / OFFSET.
        idx = _find_outer_clause(sql)
        if idx is not None:
            sql = f"{sql[:idx].rstrip()} \nWHERE {ministry_filter}\n{sql[idx:]}"
        else:
            sql += f" \nWHERE {ministry_filter}"

    return sql


# ─────────────────────────────────────────────
# Main node
# ─────────────────────────────────────────────
def sql_generation_node(state: NLSQLState) -> NLSQLState:
    """
    SQL Generation Node

    - Calls LLM to generate base SQL (no ministry filter)
    - Scrubs any hallucinated ministry filters from LLM output
    - Injects deterministic ministry filters from entities
    - Validates SQL via domain guard
    """

    question       = state["question"]
    intent         = state.get("intent", "")
    entities       = state.get("entities", {})
    query_spec     = state.get("query_spec", {})
    history        = state.get("session_history", [])
    retry_count    = state.get("retry_count", 0)
    retry_feedback = state.get("llm_validation_feedback", "") if retry_count > 0 else ""

    if retry_count > 0:
        logger.info(f"SQL generation retry #{retry_count} | feedback: {retry_feedback}")

    try:
        # ── Build prompt with raw question (no masking in SQL-gen path) ──
        # History is NOT passed to SQL generation because the question has
        # already been expanded/contextualized in the prior node (~500-1000
        # tokens saved).
        messages_raw = build_sql_generation_prompt(question, intent, retry_feedback, history=None, query_spec=query_spec)
        messages = [
            SystemMessage(content=messages_raw[0]["content"]),
            HumanMessage(content=messages_raw[1]["content"]),
        ]

        llm = get_llm()
        response = llm.invoke(messages)
        raw_sql = response.content.strip()

        # ── Strip markdown fences ──
        if raw_sql.startswith("```"):
            parts = raw_sql.split("```")
            raw_sql = parts[1] if len(parts) > 1 else raw_sql
            if raw_sql.lower().startswith("sql"):
                raw_sql = raw_sql[3:].strip()

        # ── Handle UNCLEAR ──
        if raw_sql.strip().upper() == "UNCLEAR":
            logger.warning(f"LLM flagged question as UNCLEAR: {question!r}")
            return {
                **state,
                "generated_sql":        None,
                "sql_generation_error": "UNCLEAR",
                "error":                "Question too ambiguous to generate SQL",
            }

        # ── Scrub hallucinations + inject ministry filter ──
        final_sql = _inject_ministry_filter(raw_sql, entities, query_spec)

        # ── Domain guard validation ──
        is_safe, reason = domain_guard.check_sql(final_sql)
        if not is_safe:
            logger.warning(f"Domain guard blocked SQL: {reason}")
            return {
                **state,
                "generated_sql":        None,
                "sql_generation_error": f"Security: {reason}",
                "error":                f"Generated SQL failed security check: {reason}",
            }

        logger.info(f"SQL generated (FINAL) | intent={intent}\n{final_sql}")

        return {
            **state,
            "generated_sql":        final_sql,
            "sql_generation_error": None,
        }

    except Exception as e:
        logger.exception("SQL generation failed")
        return {
            **state,
            "generated_sql":        None,
            "sql_generation_error": str(e),
            "error":                f"SQL generation error: {str(e)}",
        }
