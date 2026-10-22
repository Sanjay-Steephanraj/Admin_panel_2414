"""
graph/nodes/summary_generation_node.py
"""

import logging
import html
import re
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
    if spec.get("donor_constraint"):
        donor = spec["donor_constraint"].get("entity_name") or spec["donor_constraint"].get("entity_code")
        if donor:
            entity = f" from {donor}" + entity
    period = spec.get("period_label") or str(spec.get("period_name") or "all_time").replace("_", " ")
    start, end = spec.get("start_date"), spec.get("exclusive_end_date")
    if start and end:
        period += f" ({start} through {(date.fromisoformat(end) - timedelta(days=1)).isoformat()})"
    status = " " + spec["status"] if spec.get("status") else ""
    filters = spec.get("filters") or {}
    extra = "; " + ", ".join(f"{key.replace('_', ' ')}: {value}" for key, value in filters.items()) if filters else ""
    result_entity = spec.get("result_entity")
    if spec.get("subject") != "payment":
        label = {"donor": "donor", "ministry": "ministry"}.get(result_entity, "matching")
        return f"No {label} records were found{entity}."
    if result_entity == "donor":
        return f"No donors were found{entity} during {period}{extra}."
    if result_entity == "ministry":
        return f"No ministries were found{entity} during {period}{extra}."
    return f"No recorded{status} payments were found{entity} during {period}{extra}."

def _markdown_cell(value) -> str:
    # Entity encoding keeps delimiters and HTML inside their original cell.
    text = html.escape(str(value) if value is not None else "", quote=False)
    return text.replace("|", "&#124;").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")

_MONEY_COLUMNS = {"paymentamount", "amount", "donationamount", "totalamount", "averageamount"}


def _currency(value) -> str:
    """Display only: database values remain numeric for later calculations."""
    if value is None or isinstance(value, str) and not value.strip():
        return ""
    try:
        amount = Decimal(str(value))
        if not amount.is_finite():
            raise ValueError("Non-finite monetary result")
        return f"${amount:,.2f}"
    except (InvalidOperation, TypeError) as exc:
        raise ValueError("Invalid monetary result") from exc


def _display_cell(key: str, value) -> str:
    name = re.sub(r"[^a-z0-9]", "", key.lower())
    return _markdown_cell(_currency(value) if name in _MONEY_COLUMNS else value)


def _markdown_table(rows: list[dict], query_spec: dict | None = None) -> str:
    if not rows:
        return ""
    spec = query_spec or {}
    monthly = spec.get("group_by") == "month"
    has_undated = monthly and any(value is None for row in rows for key, value in row.items() if key.lower() == "month")
    if spec:
        from components.report_options import display_rows
        rows = display_rows(rows, spec)
    keys = list(rows[0].keys())
    if (spec.get("result_entity") == "ministry"
            and spec.get("operation") in {"ministry_list", "list"}
            and not spec.get("display_detail")):
        name_key = next((key for key in keys if key.lower() in {"ministry_name", "name"}), None)
        if name_key:
            names = [str(row.get(name_key) or "").strip().casefold() for row in rows]
            keys = [name_key]
            if len(set(names)) < len(names):
                code_key = next((key for key in rows[0] if key.lower() in {"giftcode", "gift_code", "ministry_code"}), None)
                id_key = next((key for key in rows[0] if key.lower() in {"ministry_id", "sfid"}), None)
                if code_key and all(row.get(code_key) for row in rows):
                    keys.append(code_key)
                elif id_key:
                    keys.append(id_key)
    headers = [_markdown_cell(k.replace("_", " ").title()) for k in keys]
    if monthly:
        metric, entity = spec.get("metric"), spec.get("result_entity")
        title = ("Monthly donation totals" if metric == "total" else
                 "Monthly average donation amounts" if metric == "average" else
                 "Monthly " + {"donor": "donor", "ministry": "ministry"}.get(entity, "payment") + " counts")
    else:
        title = (
            "Donor details" if spec.get("result_entity") == "donor" else
            "Ministry details" if spec.get("result_entity") == "ministry" else
            "Payment details"
            if spec.get("result_shape") == "detail"
            else "Payments by " + str(spec.get("group_by") or "group")
        )
    out = [f"{title} ({len(rows)} rows):", "", "| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in keys) + " |"]
    for row in rows:
        out.append("| " + " | ".join(_display_cell(k, row.get(k)) for k in keys) + " |")
    if monthly:
        out.extend(("", "Months with no matching payments are not shown; an omitted month does not establish a zero total."))
        if has_undated:
            out.extend(("", "‘Date not recorded’ contains matching payments without a recorded payment date."))
        if spec.get("metric") == "count" and spec.get("result_entity") in {"donor", "ministry"}:
            out.extend(("", "Each month counts distinct " + ("donors" if spec["result_entity"] == "donor" else "ministries") + "; the same one can appear in more than one month."))
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
        try:
            summary = _markdown_table(db_result, query_spec)
        except ValueError:
            return {**state, "summary": user_error_message("db_failure"),
                    "validation_passed": False, "query_outcome": "retrieval_error",
                    "displayed_count": 0, "total_count": 0}
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
            if metric == "total": summary = f"Total: **{_currency(value)}**"
            elif metric == "average": summary = f"Average: **{_currency(value)}**"
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
