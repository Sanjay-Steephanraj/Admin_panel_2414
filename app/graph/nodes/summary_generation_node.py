"""
graph/nodes/summary_generation_node.py
"""

import logging
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from langchain_core.messages import HumanMessage, SystemMessage

from ..states.nlsql_state import NLSQLState
from data_masking import masker
from llm.gloo_client import get_llm
from llm.prompts import build_summary_prompt, user_error_message

logger = logging.getLogger(__name__)

def _no_results_message(spec: dict) -> str:
    name = spec.get("entity_name") or spec.get("entity_code")
    entity = ((f" from {name}" if spec.get("entity_role") == "donor" else f" for {name}") if name else "")
    period = spec.get("period_label") or str(spec.get("period_name") or "all_time").replace("_", " ")
    start, end = spec.get("start_date"), spec.get("exclusive_end_date")
    if start and end:
        period += f" ({start} through {(date.fromisoformat(end) - timedelta(days=1)).isoformat()})"
    status = " " + spec["status"] if spec.get("status") else ""
    filters = spec.get("filters") or {}
    extra = "; " + ", ".join(f"{key.replace('_', ' ')}: {value}" for key, value in filters.items()) if filters else ""
    return f"No recorded{status} payments were found{entity} during {period}{extra}."

def _markdown_table(rows: list[dict], query_spec: dict | None = None) -> str:
    if not rows:
        return ""
    keys = list(rows[0].keys())
    headers = [k.replace("_", " ").title() for k in keys]
    spec = query_spec or {}
    title = (
        "Payment details"
        if spec.get("result_shape") == "detail"
        else "Payments by " + str(spec.get("group_by") or "group")
    )
    out = [f"{title} ({len(rows)} rows):", "", "| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in keys) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(k, "") if row.get(k) is not None else "") for k in keys) + " |")
    return "\n".join(out)


def summary_generation_node(state: NLSQLState) -> NLSQLState:
    """
    Node 3 — Summary Generation

    Routing:
      error + no sql  → hard stop (security/unclear)
      db_error        → generic failure message
      no db_result    → no-results message
      fallback_used   → retrieval error (broadened data must not be shown)
      detail/grouped/scalar → deterministic result
      other result    → existing LLM summary path
    """

    question      = state["question"]
    sql           = state.get("generated_sql", "")
    db_result     = state.get("db_result") or []
    db_error      = state.get("db_error")
    intent        = state.get("intent", "")
    error         = state.get("error")
    fallback_used = state.get("fallback_used", False)
    fallback_msg  = state.get("message")

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

    if not state.get("validation_passed", False):
        return {**state, "summary": user_error_message("db_failure"),
                "query_outcome": "retrieval_error", "displayed_count": 0, "total_count": 0}

    # ── DB failure ────────────────────────────────────────
    if db_error and not db_result:
        return {**state, "summary": user_error_message("db_failure"),
                "validation_passed": False, "query_outcome": "retrieval_error",
                "displayed_count": 0, "total_count": 0}

    # ── No data at all ────────────────────────────────────
    if not db_result:
        return {**state, "summary": _no_results_message(state.get("query_spec") or {}),
                "query_outcome": "no_results"}

    # ── Fallback: deterministic template, zero LLM ───────
    if fallback_used:
        return {**state, "summary": user_error_message("db_failure"),
                "validation_passed": False, "query_outcome": "retrieval_error",
                "displayed_count": 0, "total_count": 0}

    query_spec = state.get("query_spec") or {}
    result_shape = query_spec.get("result_shape")
    if result_shape in {"detail", "grouped"}:
        displayed = state.get("displayed_count", len(db_result))
        total = state.get("total_count", displayed)
        summary = _markdown_table(db_result, query_spec)
        if total > displayed:
            summary = f"Showing {displayed} of {total} records.\n\n" + summary
        return {**state, "summary": summary, "query_outcome": "success"}
    if result_shape == "scalar":
        try:
            metric = query_spec.get("metric")
            alias = {"total": "total_amount", "count": "record_count", "average": "average_amount"}[metric]
            value = db_result[0][alias]
            if value is None or (metric == "count" and Decimal(str(value)) == 0):
                summary = ""
                if metric == "average":
                    summary = "An average could not be calculated from the matching payment records."
                return {**state, "summary": summary + ("\n\n" if summary else "") + _no_results_message(query_spec),
                        "query_outcome": "no_results", "displayed_count": 0, "total_count": 0}
            if metric == "total": summary = f"Total: **${Decimal(str(value)):,.2f}**"
            elif metric == "average": summary = f"Average: **${Decimal(str(value)):,.2f}**"
            else: summary = f"Count: **{int(value):,}**"
            return {**state, "summary": summary, "query_outcome": "success"}
        except (KeyError, IndexError, TypeError, ValueError, InvalidOperation):
            return {**state, "summary": user_error_message("db_failure"),
                    "validation_passed": False, "query_outcome": "retrieval_error",
                    "displayed_count": 0, "total_count": 0}

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
    """Compatibility helper: never display automatically broadened data."""
    return user_error_message("db_failure")
