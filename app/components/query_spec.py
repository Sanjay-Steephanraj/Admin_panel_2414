"""Authoritative, deterministic contract for Admin payment analytics queries."""
from __future__ import annotations

import calendar
from copy import deepcopy
from datetime import date, datetime, timedelta
import re
from typing import Any
from zoneinfo import ZoneInfo

PAYMENT_WORDS = re.compile(
    r"\b(?:donations?|payments?|contributions?|gifts?|transactions?|"
    r"donate[ds]?|contribute[ds]?|paid|receive[ds]?)\b", re.I,
)
_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
_MONTH_RE = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + ")"
_ENTITY_STOP = re.compile(
    r"\s+(?:to|for|in|on|during|this|last|all|today|yesterday|with|whose|"
    r"receive[ds]?|donate[ds]?|contribute[ds]?|paid)\b|[,?!]", re.I,
)
_ENTITY_FIELDS = ("entity_role", "entity_id", "entity_name", "entity_code", "resolution_outcome")
_PERIOD_FIELDS = ("period_name", "period_label", "start_date", "exclusive_end_date", "period_ambiguity")
_GIVER_LIST_RE = re.compile(r"\b(?:list\s+(?:of\s+|all\s+)?|show\s+(?:me\s+|us\s+)?|"r"get\s+(?:the\s+)?|which\s+)givers?\b", re.I,)

def reporting_date() -> date:
    from config.settings import get_settings
    return datetime.now(ZoneInfo(get_settings().reporting_timezone)).date()

def resolve_ministry(spec: dict) -> tuple[dict, str | None]:
    """Resolve exactly one official receiver; DB errors never mean no match."""
    if spec.get("subject") != "payment" or spec.get("entity_role") != "receiving_ministry" or not (spec.get("entity_name") or spec.get("entity_code")):
        return spec, None
    if spec.get("entity_id") and spec.get("resolution_outcome") == "resolved":
        return spec, None
    from tools.db_tool import execute_query
    raw = str(spec.get("entity_code") or spec.get("entity_name")).strip()
    candidates = [raw]
    if not spec.get("entity_code"):
        short = re.sub(r"^ministry\s+(?:of|to|for)\s+|\s+ministry$", "", raw, flags=re.I).strip()
        if short and short.lower() != raw.lower():
            candidates.extend((short, short + " Ministry", "Ministry of " + short))
        elif short:
            candidates.extend((short + " Ministry", "Ministry of " + short))
    candidates = list(dict.fromkeys(candidates))
    field = "giftcode" if spec.get("entity_code") else "name"
    for candidate in candidates:
        value = candidate.replace("\\", "\\\\").replace("'", "''")
        rows, error = execute_query("SELECT sfid, name, giftcode FROM sf_ministries " f"WHERE LOWER(TRIM({field})) = LOWER('{value}') LIMIT 2")
        if error:
            return {**spec, "resolution_outcome": "retrieval_error"}, None
        if len(rows) > 1:
            return spec, "More than one ministry matches that name. Please provide its gift code."
        if rows:
            resolved = rows[0]
            if not resolved.get("sfid"):
                return {**spec, "resolution_outcome": "retrieval_error"}, None
            return {**spec, "entity_id": resolved["sfid"], "entity_name": resolved["name"], "entity_code": resolved.get("giftcode"), "resolution_outcome": "resolved"}, None
    return {**spec, "resolution_outcome": "no_match"}, None

def _next_month(start: date) -> date:
    return (start.replace(day=28) + timedelta(days=4)).replace(day=1)

def _resolve_period(question: str, today: date) -> dict:
    q = question.lower().replace("’", "'")
    name, start, end, label, ambiguity = "all_time", None, None, "all time", None
    explicit = True
    if re.search(r"\ball[- ]time\b", q):
        pass
    elif re.search(r"\btoday(?:'s)?\b|\byesterday\b", q):
        name = "yesterday" if "yesterday" in q else "today"; start = today - timedelta(days=int(name == "yesterday")); end, label = start + timedelta(days=1), name
    elif re.search(r"\b(?:this|last|previous) week\b", q):
        name = "this_week" if "this week" in q else "last_week"; start = today - timedelta(days=today.weekday() + (7 if name == "last_week" else 0)); end, label = start + timedelta(days=7), name.replace("_", " ")
    elif re.search(r"\b(?:this|last|previous) month\b", q):
        name = "this_month" if "this month" in q else "last_month"; start = today.replace(day=1)
        if name == "last_month": start = (start - timedelta(days=1)).replace(day=1)
        end, label = _next_month(start), name.replace("_", " ")
    elif re.search(r"\b(?:this|last|previous) year\b", q):
        name = "this_year" if "this year" in q else "last_year"; start = date(today.year - int(name == "last_year"), 1, 1); end, label = date(start.year + 1, 1, 1), name.replace("_", " ")
    else:
        patterns = ((r"\b\d{4}-\d{2}-\d{2}\b", "%Y-%m-%d"), (r"\b\d{1,2}/\d{1,2}/\d{4}\b", "%d/%m/%Y"), (rf"\b{_MONTH_RE}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,?\s*\d{{4}}\b", "month_first"), (rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_RE}\s*,?\s*\d{{4}}\b", "day_first"))
        found = []
        try:
            for pattern, fmt in patterns:
                for match in re.finditer(pattern, q):
                    text = re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", match[0]).replace(",", ""); parts = text.split()
                    if fmt == "month_first": parsed = date(int(parts[2]), _MONTHS[parts[0]], int(parts[1]))
                    elif fmt == "day_first": parsed = date(int(parts[2]), _MONTHS[parts[1]], int(parts[0]))
                    else: parsed = datetime.strptime(text, fmt).date()
                    found.append((match.start(), parsed))
            found.sort()
            if found:
                if len(found) > 2 or (len(found) == 2 and not re.search(r"\b(?:to|through|until|between|and)\b", q)): raise ValueError("ambiguous date range")
                start, end = found[0][1], found[-1][1] + timedelta(days=1)
                if end <= start: raise ValueError("reversed date range")
                name = "date" if len(found) == 1 else "date_range"; label = start.isoformat() if len(found) == 1 else f"{start.isoformat()} through {(end - timedelta(days=1)).isoformat()}"
            else:
                month = re.search(rf"\b({_MONTH_RE})\b(?:\s+(\d{{4}})\b)?", q); year = re.search(r"\b(?:in|for|during)\s+(\d{4})\b", q)
                if month and month[1] == "may" and re.search(r"\bmay\s+(?:i|we|you)\b", q): month = None
                if month:
                    name, label = "month", month[1].title()
                    if not month[2]: ambiguity = f"Which year should I use for {label}?"
                    else: start = date(int(month[2]), _MONTHS[month[1]], 1); end, label = _next_month(start), f"{calendar.month_name[start.month]} {start.year}"
                elif year:
                    start = date(int(year[1]), 1, 1); end, name, label = date(start.year + 1, 1, 1), "year", year[1]
                elif re.search(r"\b(?:last|past|next)\s+\d+\s+(?:days?|weeks?|months?|years?)\b", q): ambiguity = "Please provide the start and end dates for that reporting period."
                else: explicit = False
        except ValueError: ambiguity = "Please provide a valid date or date range, including the year."
    return {"period_name": name, "period_label": label, "start_date": start.isoformat() if start else None, "exclusive_end_date": end.isoformat() if end else None, "period_ambiguity": ambiguity, "period_explicit": explicit}

def _period(question: str, today: date) -> tuple[str, str | None, str | None]:
    period = _resolve_period(question, today)
    return period["period_name"], period["start_date"], period["exclusive_end_date"]

def _donor_name(question: str) -> str | None:
    # "donated/contributed/paid to ..." identifies the receiver, not the
    # donor.  Do not let the broad legacy fallback below overwrite a ministry
    # code/name extracted from the receiver clause (for example Q26).
    if re.search(r"\b(?:donate[ds]?|contribute[ds]?|paid)\s+(?:to|for)\b", question, re.I):
        return None
    patterns = (r"\b(?:donations?|payments?|contributions?|gifts?|transactions?)\s+(?:(?:made|done|given|received)\s+)?(?:by|from)\s+(.+)", r"\b(?:donated|contributed|paid)\s+by\s+(.+)", r"\b(?:did|has|have)\s+(.+?)\s+(?:donate[ds]?|contribute[ds]?|pay|paid)\b", r"^\s*(.+?)\s+(?:donated|contributed|paid)\b")
    for pattern in patterns:
        match = re.search(pattern, question, re.I)
        if match:
            name = _ENTITY_STOP.split(match[1], maxsplit=1)[0].strip(" .\"'")
            if name.split() and name.split()[0].lower() in {"show", "list", "get", "give", "how", "what", "are", "count", "total", "average", "all"}: continue
            if name and not re.fullmatch(r"ministr(?:y|ies)|donors?|payment method", name, re.I): return name
    return None

def build_query_spec(question: str, entities: dict | None = None, *, today: date | None = None) -> dict[str, Any]:
    q = question.replace("’", "'"); lower = q.lower(); period = _resolve_period(q, today if today is not None else reporting_date()); explicit = ["period"] if period.pop("period_explicit") else []
    payment = bool(PAYMENT_WORDS.search(q))
    if payment: explicit.append("subject")
    metric = None
    if re.search(r"\b(?:average|avg)\b", lower): metric = "average"
    elif re.search(r"\b(?:how many|count|number of)\b", lower): metric = "count"
    elif re.search(r"\b(?:how much|total|sum|revenue|income|raised)\b", lower): metric = "total"
    if metric: explicit.append("metric")
    group_by = "ministry" if re.search(r"\b(?:by|per) ministr(?:y|ies)\b", lower) else None
    if group_by: explicit.append("group_by")
    details = bool(re.search(r"\b(?:details?|list|show|breakdown)\b", lower)) and not metric and not group_by
    if details: explicit.append("detail")
    e = entities or {}; role, name, code = None, None, None
    if e.get("ministry_name") or e.get("ministry_code"): role, name, code = "receiving_ministry", e.get("ministry_name"), e.get("ministry_code")
    our_receiver = re.search(
        r"\bour (?:ministry|organisation|organization)\b|"
        r"\b(?:to|for)\s+us\b|\b(?:we|us)\s+(?:received?|raise[ds]?)\b", lower,
    )
    if payment and not role and our_receiver:
        role, name, code = "receiving_ministry", None, "098WRLD"
    donor = _donor_name(q)
    if donor: role, name, code = "donor", donor, None
    if role: explicit.append("entity")
    status = None
    if re.search(r"\b(?:unpaid|pending|outstanding)\b", lower): status = "unpaid"
    elif re.search(r"\b(?:completed|confirmed)\b|\bpaid\s+(?:donations?|payments?|contributions?)\b|\bstatus\s+(?:is\s+)?paid\b", lower): status = "paid"
    if status: explicit.append("status")
    spec = {"subject": "payment" if payment else "other", **period, "status": status, "group_by": group_by, "entity_role": role, "entity_id": None, "entity_name": name, "entity_code": code, "metric": metric, "filters": {}, "explicit_fields": explicit}
    if re.search(r"\bfailed\s+(?:donations?|payments?)\b", lower): spec["clarification"] = "The payment records distinguish paid and unpaid payments. Which should I use?"
    donor_rows = bool((re.search(r"\bdonors?\b", lower) and re.search(r"\b(?:list|show|get|give|which|who)\b", lower)) or _GIVER_LIST_RE.search(lower))
    if payment:
        spec["result_entity"] = "donor" if donor_rows and not metric and not group_by else "payment"
    elif re.search(r"\b(?:donors?|contacts?|profile|email|phone)\b", lower) or _GIVER_LIST_RE.search(lower):
        spec["result_entity"] = "donor"
    elif re.search(r"\bministr(?:y|ies)\b", lower):
        spec["result_entity"] = "ministry"
    else:
        spec["result_entity"] = "record"
    if donor_rows or re.search(r"\b(?:payment|donation|contribution) details?\b", lower):
        explicit.append("result_entity")
        if donor_rows:
            spec["operation"], spec["result_shape"] = "donor_list", "detail"
    else:
        spec["operation"] = ("donor_list" if spec.get("subject") == "payment" and
                              spec.get("result_entity") == "donor" else
                              ("details" if spec.get("subject") == "payment" else "list"))
        spec["result_shape"] = "detail"
    return _result_fields(spec)

def _result_fields(spec: dict) -> dict:
    if spec.get("group_by"): spec["metric"] = spec.get("metric") or "total"; spec["operation"], spec["result_shape"] = "grouped", "grouped"
    elif spec.get("metric"): spec["operation"], spec["result_shape"] = spec["metric"], "scalar"
    elif spec.get("subject") == "payment" and spec.get("result_entity") == "donor": spec["operation"], spec["result_shape"] = "donor_list", "detail"
    elif spec.get("subject") == "payment" and spec.get("result_entity") == "payment": spec["operation"], spec["result_shape"] = "details", "detail"
    elif spec.get("operation") == "donor_list": spec["result_shape"] = "detail"
    else: spec["operation"] = "details" if spec.get("subject") == "payment" else "list"; spec["result_shape"] = "detail"
    return spec

def merge_follow_up(current: dict, previous: dict | None) -> dict:
    """Called only for genuine follow-ups. Replace complete field bundles."""
    if not previous: return current
    merged = deepcopy(previous); explicit = set(current.get("explicit_fields", []))
    for field in ("subject", "status", "metric", "group_by", "filters", "result_entity"):
        if field in explicit: merged[field] = deepcopy(current.get(field))
    # Recalculate operation/result_shape from merged result_entity.
    # Preserve old entity and period filters unless explicitly changed.
    for bundle, fields in (("entity", _ENTITY_FIELDS), ("period", _PERIOD_FIELDS)):
        if bundle in explicit:
            for field in fields: merged[field] = current.get(field)
    if "entity" in explicit and current.get("entity_role") != "receiving_ministry": merged["group_by"] = None
    if "detail" in explicit: merged["metric"], merged["group_by"] = None, None
    merged["explicit_fields"] = list(current.get("explicit_fields", []))
    if current.get("clarification"): merged["clarification"] = current["clarification"]
    return _result_fields(merged)

def semantic_sql_errors(sql: str, spec: dict) -> list[str]:
    from components.sql_contract import semantic_sql_errors as validate
    return validate(sql, spec)
