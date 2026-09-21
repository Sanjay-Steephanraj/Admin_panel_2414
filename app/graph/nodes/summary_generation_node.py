"""
graph/nodes/summary_generation_node.py
"""

import logging
from decimal import Decimal, InvalidOperation
from langchain_core.messages import HumanMessage, SystemMessage

from ..states.nlsql_state import NLSQLState
from data_masking import masker
from llm.gloo_client import get_llm
from llm.prompts import build_summary_prompt, user_error_message

logger = logging.getLogger(__name__)


def summary_generation_node(state: NLSQLState) -> NLSQLState:
    """
    Node 3 — Summary Generation

    Routing:
      error + no sql  → hard stop (security/unclear)
      db_error        → generic failure message
      no db_result    → no-results message
      fallback_used   → deterministic template (NO LLM — prevents hallucination)
      normal          → LLM summary
    """

    question      = state["question"]
    sql           = state.get("generated_sql", "")
    db_result     = state.get("db_result") or []
    db_error      = state.get("db_error")
    intent        = state.get("intent", "")
    error         = state.get("error")
    fallback_used = state.get("fallback_used", False)
    fallback_msg  = state.get("message")
    query_spec    = state.get("query_spec") or {}

    # ── Debug log — confirms what actually arrived ────────
    logger.info(
        f"[summary_node] fallback_used={fallback_used} | "
        f"rows={len(db_result)} | "
        f"db_error={db_error} | "
        f"message={fallback_msg}"
    )

    # ── Hard stop (security / unclear intent) ────────────
    if error and not sql:
        return {**state, "summary": error}

    # ── DB failure ────────────────────────────────────────
    if db_error and not db_result:
        return {**state, "summary": user_error_message("db_failure")}

    # ── No data at all ────────────────────────────────────
    if not db_result:
        period = query_spec.get("period_name")
        suffix = f" for {period.replace('_', ' ')}" if period and period != "unspecified" else ""
        return {**state, "summary": f"No recorded payments found{suffix}."}

    # Detail and grouped reports are data products, not prose summaries.  The
    # full approved DB result is rendered deterministically so an LLM cannot
    # silently select the first 15 rows.
    if query_spec.get("result_shape") in {"detail", "grouped"}:
        return {**state, "summary": _markdown_table(db_result, query_spec)}

    if query_spec.get("result_shape") == "scalar":
        return {**state, "summary": _scalar_result(db_result, query_spec)}

    # ── Fallback: deterministic template, zero LLM ───────
    if fallback_used:
        logger.info("[summary_node] Fallback path → deterministic response")
        return {**state, "summary": _build_fallback_summary(db_result, fallback_msg)}

    # ── Normal path: LLM summary ─────────────────────────
    try:
        # ── Mask PII before prompt (fail-closed: raises RuntimeError) ──
        # Mask all rows so no unmasked row can reach the LLM; the builder
        # slices to [:15] internally and uses len(db_result) for the
        # "Showing top 15 of N" message, so pass the full masked list.
        # Thread ONE mapping: rows first so DB values get tokens that question
        # text reuses if the same value appears.
        mapping: dict = {}
        masked_rows, mapping = masker.mask_rows(db_result, mapping)
        masked_q, mapping = masker.mask_text(question, mapping)
        masked_sql, mapping = masker.mask_text(sql or "", mapping)

        messages_raw = build_summary_prompt(
            question=masked_q,
            sql=masked_sql,
            db_result=masked_rows,
            intent=intent,
        )

        messages = [
            SystemMessage(content=messages_raw[0]["content"]),
            HumanMessage(content=messages_raw[1]["content"]),
        ]

        llm      = get_llm()
        response = llm.invoke(messages)
        summary  = masker.unmask_text(response.content.strip(), mapping)

        logger.info(f"[summary_node] LLM summary OK | intent={intent} | rows={len(db_result)}")
        return {**state, "summary": summary}

    except Exception:
        logger.exception("[summary_node] LLM summary failed")
        return {**state, "summary": user_error_message("db_failure")}


# ── Helpers ───────────────────────────────────────────────

def _build_fallback_summary(rows: list, fallback_msg: str | None) -> str:
    """
    Builds a deterministic, hallucination-proof summary for the fallback case.
    Works for any result shape — donor totals, payment-method breakdowns, etc.
    """
    try:
        header = fallback_msg or "No data found for the specified period. Showing all-time data instead."

        # ── Donor+ministry shape ──────────────────────────────────────
        # Trigger only if there's exactly one donor name and ministry info is present.
        # This prevents leaderboards (multiple donors) from being mislabeled as a single donor's breakdown.
        unique_donors = {r.get("donor_name") for r in rows if r.get("donor_name")}
        
        if rows and len(unique_donors) == 1 and "ministry_name" in rows[0] and "total_contribution" in rows[0]:
            donor_name   = list(unique_donors)[0]
            total_amount = 0.0
            ministry_lines = []
            for row in rows:
                ministry = str(row.get("ministry_name") or "Unknown ministry").strip()
                amount   = float(row.get("total_contribution") or 0)
                total_amount += amount
                ministry_lines.append(f"  - {ministry}: ${amount:,.2f}")

            return (
                f"{header}\n"
                f"{donor_name} has contributed a total of ${total_amount:,.2f} "
                f"to the following {'ministry' if len(ministry_lines) == 1 else 'ministries'}:\n"
                + "\n".join(ministry_lines)
            )

        # ── Payment-method breakdown shape ───────────────────────────
        if rows and "payment_method" in rows[0] and "total_amount" in rows[0]:
            grand_total = sum(float(r.get("total_amount") or 0) for r in rows)
            lines = []
            for row in rows:
                method = str(row.get("payment_method") or "Unknown").strip()
                amount = float(row.get("total_amount") or 0)
                lines.append(f"  - {method}: ${amount:,.2f}")

            return (
                f"{header}\n"
                f"All-time payment totals by method (grand total: ${grand_total:,.2f}):\n"
                + "\n".join(lines)
            )

        # ── Generic fallback: render key-value rows ───────────────────
        lines = []
        for row in rows[:20]:
            row_parts = []
            
            # 1. Bold the Donor Name (without the label)
            name = row.get("donor_name")
            if name:
                row_parts.append(f"**{str(name).strip()}**")

            # 2. Process other columns
            for k, v in row.items():
                if v is None or k == "donor_name":
                    continue
                
                # Format label: 'total_donated' -> 'Total Donated'
                label = k.replace("_", " ").title()
                
                # Format numbers as currency if key suggests it
                amt_keys = ["amount", "donated", "contribution", "total", "spent", "balance"]
                if any(ak in k.lower() for ak in amt_keys):
                    try:
                        v = f"${float(v):,.2f}"
                    except (ValueError, TypeError):
                        pass
                
                row_parts.append(f"{label}: {v}")

            lines.append("  - " + " | ".join(row_parts))

        return f"{header}\n" + "\n".join(lines) if lines else header

    except Exception:
        logger.exception("[summary_node] _build_fallback_summary failed")
        return fallback_msg or "Data available but could not be formatted."


def _markdown_table(rows: list[dict], spec: dict) -> str:
    """Render every returned row. Query LIMIT, not the responder, is the cap."""
    headers = list(rows[0].keys()) if rows else []
    if not headers:
        return "No recorded payments found."
    title = "Payment details" if spec.get("result_shape") == "detail" else "Payments by ministry"
    def cell(value):
        if value is None: return ""
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = [f"{title} ({len(rows)} rows):", "", "| " + " | ".join(h.replace("_", " ").title() for h in headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(cell(row.get(h)) for h in headers) + " |" for row in rows)
    return "\n".join(lines)


def _scalar_result(rows: list[dict], spec: dict) -> str:
    row = rows[0] if rows else {}
    metric = spec.get("metric") or "result"
    # SQL may include a descriptive column before the aggregate, e.g.
    # ``ministry_name, SUM(...)``. Select the aggregate by its column name or
    # numeric value rather than assuming the first column is the answer.
    value = None
    metric_tokens = {
        "total": ("total", "sum", "amount", "contribution", "raised", "income"),
        "average": ("avg", "average", "amount"),
        "count": ("count", "number", "total"),
    }.get(metric, ())
    for key, candidate in row.items():
        key_lower = str(key).lower()
        if any(token in key_lower for token in metric_tokens):
            value = candidate
            break
    if value is None:
        for candidate in row.values():
            try:
                Decimal(str(candidate))
                value = candidate
                break
            except (InvalidOperation, TypeError, ValueError):
                continue
    if value is None:
        value = 0
    if metric in {"total", "average"}:
        try: value = f"${Decimal(str(value)):,.2f}"
        except Exception: pass
    return f"{metric.title()}: **{value}**"
