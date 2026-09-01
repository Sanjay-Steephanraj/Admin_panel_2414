"""
graph/nodes/sql_validation_node.py
"""

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from ..states.nlsql_state import NLSQLState
from data_masking import masker
from llm.gloo_client import get_llm
from llm.prompts import build_sql_validation_prompt
from tools.db_tool import execute_query

logger = logging.getLogger(__name__)


# =========================================================
# REMOVE DATE FILTER (FALLBACK SUPPORT)
# =========================================================

# Comparison operators recognised in YEAR/MONTH/DAY(col) <op> <expr> shapes.
_DATE_OPS = r"(?:=|!=|<>|>=|<=|>|<)"

# Clauses that terminate the outer WHERE region (depth-0). HAVING is included
# even though it follows GROUP BY so a missing GROUP BY still stops the scan.
_CLAUSE_STOP_RE = re.compile(
    r"\b(GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|OFFSET|UNION)\b",
    re.IGNORECASE,
)


def _find_outer_where_region(sql: str) -> tuple[int, int] | None:
    """
    Locate the [start, end) span of the outermost WHERE clause body.

    Returns (where_kw_start, region_end) where `region_end` is the index of the
    first depth-0 GROUP BY / ORDER BY / HAVING / LIMIT / OFFSET / UNION keyword,
    or len(sql) if none. Returns None if there is no depth-0 WHERE.

    A small local depth+quote-aware scanner; intentionally NOT imported from
    sql_generation_node (another agent edits that file). Same spirit as
    _find_outer_clause there.
    """
    depth = 0
    in_quote = False
    quote_char = None
    where_start = None

    token_re = re.compile(r"\bWHERE\b|\b(GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|OFFSET|UNION)\b|\(|\)|'|\"", re.IGNORECASE)

    for m in token_re.finditer(sql):
        tok = m.group(0)
        # quote handling
        if tok in ("'", '"'):
            if not in_quote:
                in_quote = True
                quote_char = tok
            elif quote_char == tok:
                in_quote = False
            continue
        if in_quote:
            continue
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            if tok.upper() == "WHERE" and where_start is None:
                where_start = m.start()
            elif where_start is not None and _CLAUSE_STOP_RE.match(tok):
                return (where_start, m.start())

    if where_start is None:
        return None
    return (where_start, len(sql))


def _split_conditions(body: str) -> list[tuple[str | None, str]]:
    """
    Split a WHERE-clause body into (connector, condition) pairs at depth-0
    AND/OR connectors. The first condition has connector None (it follows
    WHERE directly). Quote- and paren-aware, and BETWEEN-aware: the AND
    inside `col BETWEEN 'a' AND 'b'` is NOT a split point.

    The BETWEEN-awareness is the only non-obvious bit: while scanning, if we
    are at depth 0, not in a quote, and the current fragment (from cur_start
    to the AND token) matches a BETWEEN pattern, we treat the AND as part of
    the BETWEEN and do not split.
    """
    pieces: list[tuple[str | None, str]] = []
    depth = 0
    in_quote = False
    quote_char = None
    cur_start = 0
    cur_conn: str | None = None
    token_re = re.compile(r"\b(AND|OR)\b|\(|\)|'|\"", re.IGNORECASE)

    _between_re = re.compile(
        r"[\w.]+\s+BETWEEN\s+'[^']*'\s*$", re.IGNORECASE,
    )

    def _emit(end: int, conn: str | None):
        text = body[cur_start:end]
        if text.strip():
            pieces.append((conn, text.strip()))

    for m in token_re.finditer(body):
        tok = m.group(0)
        if tok in ("'", '"'):
            if not in_quote:
                in_quote = True
                quote_char = tok
            elif quote_char == tok:
                in_quote = False
            continue
        if in_quote:
            continue
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and tok.upper() in ("AND", "OR"):
            # Skip this AND if it is the internal AND of a BETWEEN: the
            # fragment from cur_start up to (but not including) this AND
            # ends with `col BETWEEN 'date' ` (trailing whitespace).
            if tok.upper() == "AND" and _between_re.search(body[cur_start:m.start()].rstrip()):
                continue
            _emit(m.start(), cur_conn)
            cur_conn = tok.upper()
            cur_start = m.end()
    _emit(len(body), cur_conn)
    return pieces


def _is_date_condition(cond: str) -> bool:
    """True if a single AND/OR-split condition is a temporal filter we strip."""
    s = cond.strip()

    # 1. (YEAR|MONTH|DAY)(col) <op> <expr>  — RHS is everything to end of cond.
    if re.match(rf"(?:YEAR|MONTH|DAY)\s*\([^)]+\)\s*{_DATE_OPS}\s*.+", s, re.IGNORECASE):
        return True

    # 2. col BETWEEN 'date' AND 'date'
    if re.search(r"[\w.]+\s+BETWEEN\s+'[^']+'\s+AND\s+'[^']+'", s, re.IGNORECASE):
        return True

    # 3. DATE(col) <op> 'date'
    if re.match(rf"DATE\s*\(\s*[\w.]+\s*\)\s*{_DATE_OPS}\s*'.*?'", s, re.IGNORECASE):
        return True

    # 4. Bare date-column comparisons. _has_date_filter already treats
    #    `paymentdate`/`payment_date` as a date filter, so strip the
    #    associated condition here. Covers:
    #      p.paymentdate >= '2024-01-01'
    #      p.paymentdate < CURDATE()
    #      p.paymentdate >= DATE_SUB(CURDATE(), INTERVAL 48 MONTH)
    #    The churn analyzer's history-window clause shape is also matched,
    #    but that is fine: this function is only invoked on LLM-generated
    #    pipeline SQL; churn SQL bypasses the graph entirely and never
    #    reaches remove_date_filter. deliberate: strip churn-shape too —
    #    no risk because churn never flows through here.
    if re.match(r"[\w.]*payment_?date\s*" + _DATE_OPS + r"\s*.+", s, re.IGNORECASE):
        return True

    return False


def _strip_date_conditions(region: str) -> str:
    """
    Given the WHERE-clause region (starting at the 'WHERE' keyword, ending
    before the next outer clause keyword or EOS), return the region with all
    temporal conditions removed and connectors fixed up.

    If everything is dropped, replace the body with `1=1` so the SQL stays
    valid (`WHERE 1=1`).
    """
    where_kw_match = re.match(r"\s*WHERE\b", region, re.IGNORECASE)
    if not where_kw_match:
        return region
    body = region[where_kw_match.end():]
    head = region[:where_kw_match.end()]

    pieces = _split_conditions(body)

    kept = [(conn, cond) for conn, cond in pieces if not _is_date_condition(cond)]

    if not kept:
        # Only condition(s) were date filters → neutralise the WHERE body.
        # Trailing space keeps any following clause (GROUP BY etc.) separated.
        return head + " 1=1 "

    # Reassemble. The first kept condition loses any leading connector
    # (it becomes the first condition after WHERE); subsequent kept
    # conditions keep their connector. Always emit a trailing space so the
    # next clause (GROUP BY etc.) stays separated.
    out = head
    for idx, (conn, cond) in enumerate(kept):
        if idx == 0:
            out += " " + cond
        else:
            out += f" {conn} {cond}"
    out += " "
    return out


def remove_date_filter(sql: str) -> str:
    """
    Strips temporal filter conditions so the fallback query returns all-time
    data, while ALWAYS leaving syntactically valid SQL.

    Handles four condition shapes, in both leading-AND and sole-WHERE
    positions:
      - (YEAR|MONTH|DAY)(col) <op> <expr>   (op in =,!=,<>,>=,<=,>,<)
      - col BETWEEN 'date' AND 'date'
      - DATE(col) <op> 'date'
      - bare <datecol> <op> <expr>  (paymentdate / payment_date)

    Connector fix-up rules:
      - leading AND/OR removed with the condition
      - first-after-WHERE condition: trailing AND/OR removed instead
      - if only condition → `WHERE 1=1`
    Never leaves WHERE AND, WHERE OR, a dangling trailing connector, or a
    WHERE with no condition. Idempotent. Returns input unchanged if nothing
    matched.
    """
    region = _find_outer_where_region(sql)
    if region is None:
        return sql

    start, end = region
    head = sql[:start]
    where_region = sql[start:end]
    tail = sql[end:]

    new_region = _strip_date_conditions(where_region)

    if new_region == where_region:
        return sql  # nothing matched → byte-identical (callers rely on this)

    result = head + new_region + tail
    # Collapse accidental double spaces from surgery. Do NOT rstrip when a
    # trailing clause (GROUP BY/ORDER BY/...) follows — the region ended at
    # the clause keyword's first char, so new_region's trailing space is the
    # only separator between WHERE body and that clause.
    result = re.sub(r" {2,}", " ", result)
    if not tail:
        result = result.rstrip()
    return result


def _has_date_filter(sql: str) -> bool:
    """
    True if SQL contains a temporal filter that remove_date_filter would
    strip. Detection mirrors _is_date_condition's shapes so the fallback is
    attempted exactly when stripping is possible.
    """
    region = _find_outer_where_region(sql)
    if region is None:
        return False
    start, end = region
    where_region = sql[start:end]

    where_kw_match = re.match(r"\s*WHERE\b", where_region, re.IGNORECASE)
    if not where_kw_match:
        return False
    body = where_region[where_kw_match.end():]

    pieces = _split_conditions(body)
    return any(_is_date_condition(cond) for _, cond in pieces)


# =========================================================
# MAIN NODE
# =========================================================

def sql_validation_node(state: NLSQLState) -> NLSQLState:
    """
    Step 1 : Execute original SQL
    Step 2a: Rows found          → return success
    Step 2b: No rows + date filt → fallback (strip date filter)
    Step 2c: No rows at all      → return empty
    Step 3 : DB error            → LLM validation for fix
    """

    sql      = state.get("generated_sql")
    question = state["question"]

    # ── Guard ────────────────────────────────────────────
    if not sql:
        return {
            **state,
            "validation_passed": False,
            "fallback_used":     False,       # ← always explicit
            "message":           None,
            "db_error":          "No SQL to validate",
        }

    # ── STEP 1: Execute original query ───────────────────
    logger.info(f"[sql_validation] Executing SQL:\n{sql}")
    db_result, db_error = execute_query(sql)

    # ── STEP 2: DB success (no error) ────────────────────
    if db_error is None:

        # CASE A: rows found — happy path
        if db_result:
            logger.info(f"[sql_validation] Success | rows={len(db_result)}")
            return {
                **state,
                "db_result":              db_result,
                "db_error":               None,
                "validation_passed":      True,
                "fallback_used":          False,       # ← explicit
                "message":                None,
                "llm_validation_feedback": "DB execution successful — LLM validation skipped",
            }

        # CASE B: zero rows — attempt date-filter fallback
        logger.info("[sql_validation] Zero rows — checking for date filter fallback")

        if _has_date_filter(sql):
            fallback_sql = remove_date_filter(sql)
            logger.info(f"[sql_validation] Fallback SQL:\n{fallback_sql}")

            fallback_result, fallback_error = execute_query(fallback_sql)

            # Fallback SQL itself errored: do NOT pretend success. Log the
            # warning and fall through to the empty-result path so the user
            # sees the original zero-row result rather than a silent
            # "no data found" that masks a broken fallback query.
            if fallback_error is not None:
                logger.warning(
                    f"[sql_validation] Fallback SQL errored — reporting original "
                    f"zero-row result. fallback_error={fallback_error}"
                )
            elif fallback_result:
                logger.info(f"[sql_validation] Fallback success | rows={len(fallback_result)}")
                return {
                    **state,
                    "db_result":              fallback_result,
                    "db_error":               None,
                    "validation_passed":      True,
                    "fallback_used":          True,    # ← THE key flag
                    "message":                "No data found for the specified period. Showing overall contributions instead.",
                    "llm_validation_feedback": "Fallback applied — date filter removed",
                }
            else:
                logger.info("[sql_validation] Fallback also returned no rows")

        # CASE C: no data even after fallback
        logger.info("[sql_validation] No data found")
        return {
            **state,
            "db_result":              [],
            "db_error":               None,
            "validation_passed":      True,
            "fallback_used":          False,           # ← explicit
            "message":                "No contributions found for the given query.",
            "llm_validation_feedback": "Query valid but returned no data",
        }

    # ── STEP 3: DB error → ask LLM for a fix ─────────────
    logger.warning(f"[sql_validation] DB error — invoking LLM: {db_error}")

    try:
        # ── Mask PII before prompt (fail-closed: raises RuntimeError) ──
        # Thread ONE mapping across all calls so the same value keeps one token.
        mapping: dict = {}
        masked_q, mapping = masker.mask_text(question, mapping)
        masked_sql, mapping = masker.mask_text(sql, mapping)
        masked_err, mapping = masker.mask_text(db_error or "", mapping)

        masked_rows, mapping = masker.mask_rows(db_result[:3], mapping) if db_result else ([], mapping)
        db_result_sample = str(masked_rows) if db_result else "No rows"

        messages_raw = build_sql_validation_prompt(
            question=masked_q,
            generated_sql=masked_sql,
            db_error=masked_err,
            db_result_sample=db_result_sample,
        )

        messages = [
            SystemMessage(content=messages_raw[0]["content"]),
            HumanMessage(content=messages_raw[1]["content"]),
        ]

        llm      = get_llm()
        response = llm.invoke(messages)
        raw      = response.content.strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        parsed        = json.loads(raw)
        issues        = parsed.get("issues", "")
        suggested_fix = parsed.get("suggested_fix", "")

        logger.info(f"[sql_validation] LLM valid={parsed.get('valid', False)}")

        # ── Unmask LLM output so feedback carries real values ──
        issues        = masker.unmask_text(issues, mapping)
        suggested_fix = masker.unmask_text(suggested_fix, mapping)

        feedback_parts = [f"PREVIOUS SQL:\n{sql}", f"DB ERROR: {db_error}"]
        if issues:
            feedback_parts.append(f"ROOT CAUSE: {issues}")
        if suggested_fix:
            feedback_parts.append(f"SUGGESTED FIX:\n{suggested_fix}")

        return {
            **state,
            "db_result":              db_result,
            "db_error":               db_error,
            "validation_passed":      False,
            "fallback_used":          False,           # ← explicit
            "message":                None,
            "llm_validation_feedback": "\n\n".join(feedback_parts),
        }

    except Exception:
        logger.exception("[sql_validation] LLM validation failed")
        return {
            **state,
            "db_result":              [],
            "db_error":               db_error,
            "validation_passed":      False,
            "fallback_used":          False,           # ← explicit
            "message":                None,
            "llm_validation_feedback": f"SQL failed: {db_error}",
        }