"""Authoritative, deterministic contract for Admin payment analytics queries."""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any


def resolve_ministry(spec: dict) -> tuple[dict, str | None]:
    """Resolve a receiver to one canonical sf_ministries record before SQL.

    Returns ``(spec, clarification)``.  A database connectivity failure leaves
    the spec unresolved so the normal pipeline can report its retrieval error;
    it is never confused with a genuine no-match.
    """
    if spec.get("entity_role") != "receiving_ministry" or not (spec.get("entity_name") or spec.get("entity_code")):
        return spec, None
    from tools.db_tool import execute_query
    raw = str(spec.get("entity_code") or spec.get("entity_name")).replace("'", "''").strip()
    field = "giftcode" if spec.get("entity_code") else "name"
    # Exact normalized name/code matching prevents an LLM phrase from becoming
    # an authoritative SQL filter and makes ambiguity explicit.
    sql = ("SELECT sfid, name, giftcode FROM sf_ministries "
           f"WHERE LOWER({field}) = LOWER('{raw}') LIMIT 2")
    rows, error = execute_query(sql)
    if error:
        return spec, None
    if not rows:
        spec = dict(spec); spec["resolution_outcome"] = "no_match"
        return spec, None
    if len(rows) > 1:
        return spec, "More than one ministry matches that name. Please provide its gift code."
    resolved = rows[0]
    spec = dict(spec)
    spec.update(entity_id=resolved.get("sfid"), entity_name=resolved.get("name"), entity_code=resolved.get("giftcode"), resolution_outcome="resolved")
    return spec, None


def _period(question: str, today: date) -> tuple[str, str | None, str | None]:
    q = question.lower().replace("’", "'")
    if re.search(r"\ball[- ]time\b", q):
        return "all_time", None, None
    if re.search(r"\btoday('?s)?\b", q):
        return "today", today.isoformat(), (today + timedelta(days=1)).isoformat()
    if "this week" in q:
        start = today - timedelta(days=today.weekday())
        return "this_week", start.isoformat(), (start + timedelta(days=7)).isoformat()
    if "this month" in q:
        start = today.replace(day=1)
        end = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
        return "this_month", start.isoformat(), end.isoformat()
    if "last month" in q:
        end = today.replace(day=1)
        start = (end - timedelta(days=1)).replace(day=1)
        return "last_month", start.isoformat(), end.isoformat()
    return "unspecified", None, None


def build_query_spec(question: str, entities: dict | None = None, *, today: date | None = None) -> dict[str, Any]:
    """Create one query contract; relative dates are resolved here exactly once."""
    today = today or date.today()
    q = question.replace("’", "'")
    lower = q.lower()
    period_name, start_date, exclusive_end_date = _period(q, today)
    payment = bool(re.search(r"\b(donation|donations|payment|payments|contribution|contributions|gift|gifts)\b", lower))
    metric = "count" if re.search(r"\b(how many|count)\b", lower) else ("total" if re.search(r"\b(total|sum|revenue|income|raised)\b", lower) else ("average" if re.search(r"\b(average|avg)\b", lower) else None))
    group_by = "ministry" if re.search(r"\b(by|per) ministr(?:y|ies)\b", lower) else None
    operation = "grouped" if group_by else (metric or ("details" if payment else "list"))
    role = None
    entity_name = None
    entity_code = None
    if group_by:
        role = "receiving_ministry"
    # A named ministry in a payment request is always its receiver unless a donor is explicit.
    e = entities or {}
    if payment and (e.get("ministry_name") or e.get("ministry_code")):
        role, entity_name, entity_code = "receiving_ministry", e.get("ministry_name"), e.get("ministry_code")
    donor = re.search(r"\b(?:donations?|payments?|contributions?|gifts?)\s+(?:made\s+)?by\s+([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){1,3})", q)
    if donor and not re.search(r"\bby\s+ministr(?:y|ies)\b", lower):
        role, entity_name = "donor", donor.group(1).strip()
    return {
        "subject": "payment" if payment else "other", "operation": operation,
        "period_name": period_name, "start_date": start_date, "exclusive_end_date": exclusive_end_date,
        "status": None, "group_by": group_by, "entity_role": role,
        "entity_id": None, "entity_name": entity_name, "entity_code": entity_code,
        "result_shape": "grouped" if group_by else ("scalar" if metric else "detail"),
        "metric": metric, "filters": {},
    }


def merge_follow_up(current: dict, previous: dict | None) -> dict:
    """Only unspecified dimensions may be inherited from the last success."""
    if not previous:
        return current
    merged = dict(current)
    for key in ("entity_role", "entity_id", "entity_name", "entity_code", "status", "group_by", "metric", "result_shape", "operation"):
        if merged.get(key) is None or merged.get(key) == "list":
            merged[key] = previous.get(key, merged.get(key))
    return merged


def semantic_sql_errors(sql: str, spec: dict) -> list[str]:
    """Reject executable SQL that changed the query contract before execution."""
    s = re.sub(r"\s+", " ", sql.lower())
    errors: list[str] = []
    if spec.get("subject") == "payment" and "sfpayments" not in s:
        errors.append("payment subject requires sfpayments")
    if spec.get("entity_role") == "receiving_ministry":
        if "sf_ministries" not in s or "ministryid" not in s:
            errors.append("receiving ministry join/filter is missing")
        entity_id = spec.get("entity_id") or spec.get("entity_code")
        if entity_id and str(entity_id).lower() not in s:
            errors.append("canonical ministry constraint is missing")
    if spec.get("entity_role") == "donor" and "sf_contacts" not in s:
        errors.append("donor query requires sf_contacts join")
    start, end = spec.get("start_date"), spec.get("exclusive_end_date")
    if start and (start.lower() not in s or end.lower() not in s):
        errors.append("exact reporting-period boundaries are missing")
    if start and re.search(r"\bmonth\s*\(", s):
        errors.append("month-only date filtering is not allowed")
    if spec.get("period_name") == "all_time" and re.search(r"paymentdate\s*(?:>=|<=|<|>|=)", s):
        errors.append("all-time query introduced a date predicate")
    metric = spec.get("metric")
    if metric == "total" and "sum(" not in s: errors.append("total requires SUM")
    if metric == "count" and "count(" not in s: errors.append("count requires COUNT")
    if metric == "average" and "avg(" not in s: errors.append("average requires AVG")
    if spec.get("group_by") == "ministry" and not re.search(r"\bgroup by\b.*(?:m\.)?(?:name|sfid|id)", s):
        errors.append("receiving-ministry grouping is missing")
    # Valid ranking/list questions can be represented as grouped aggregates
    # (for example, top donors or donors by ministry). Reject only an
    # aggregate with no GROUP BY, which is the actual detail-contract error.
    if (spec.get("result_shape") == "detail"
            and re.search(r"\b(sum|count|avg)\s*\(", s)
            and not re.search(r"\bgroup\s+by\b", s)
            and " union " not in f" {s} "):
        errors.append("detail request was changed into an aggregate")
    return errors
