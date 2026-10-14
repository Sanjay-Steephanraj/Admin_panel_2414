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

_REPEAT_DONORS_RE = re.compile(
    r"\b(?:repeat|repeated|returning)\s+(?:donors?|givers?)\b|"
    r"\b(?:donors?|givers?)\b.{0,65}\b(?:more than once|more than one\s+(?:donation|gift)|multiple\s+(?:donations|gifts))\b|"
    r"\b(?:donated|gave|given)\s+more than once\b", re.I,
)
_MULTI_MINISTRY_DONORS_RE = re.compile(
    r"\b(?:donors?|givers?)\b.{0,75}\b(?:multiple|different|more than one|at least two|two or more)\s+ministr(?:y|ies)\b",
    re.I,
)
_MISSING_CODE_RE = re.compile(
    r"\bministr(?:y|ies)\b.{0,55}\b(?:missing|without|no|blank)\s+(?:gift\s*)?codes?\b|"
    r"\b(?:missing|without|no|blank)\s+(?:gift\s*)?codes?\b.{0,55}\bministr(?:y|ies)\b|"
    r"\b(?:missing|without|no|blank)\s+ministr(?:y|ies)\s+(?:gift\s*)?codes?\b|"
    r"\b(?:missing|without|no|blank)\s+(?:gift\s*)?codes?\s+(?:for|of)\s+ministr(?:y|ies)\b", re.I,
)
_UNSUPPORTED_OUTCOME_RE = re.compile(
    r"\b(?:failed|refunded|reversed|cancelled|canceled|pending|outstanding)\s+"
    r"(?:(?:or|and)\s+(?:failed|refunded|reversed|cancelled|canceled|pending|outstanding)\s+)?"
    r"(?:donations?|payments?|transactions?|gifts?)\b|"
    r"\b(?:donations?|payments?|transactions?|gifts?)\s+"
    r"(?:(?:that|which)\s+)?(?:(?:were|are|have|has)\s+(?:been\s+)?)?"
    r"(?:failed|refunded|reversed|cancelled|canceled|pending|outstanding)\b|"
    r"\b(?:refunds?|reversals?|chargebacks?)\b", re.I,
)
_LOCATION_WORDS_RE = re.compile(r"\b(?:region|regions|regional|countries|country|continent|geograph(?:y|ic|ical))\b", re.I)
USER_FACING_LIMITS = {
    "ministry_geography": (
        "I can’t reliably list ministries by country or region because their location "
        "information is incomplete here. If you give me a ministry name or gift code, "
        "I can report its recorded donations."
    ),
    "quickbooks": (
        "QuickBooks sync information is not kept current here, so I can’t reliably "
        "report which donations have synced."
    ),
    "payment_outcome": (
        "I can show recorded donations, but I can’t reliably identify failed, "
        "refunded, reversed, pending, or cancelled payments here."
    ),
    "payment_frequency": (
        "I can show recorded payments, but I can’t yet reliably separate "
        "recurring from one-time donations."
    ),
}

_PAYMENT_FREQUENCY_RE = re.compile(
    r"\b(?:recurr(?:ing|ent|ence)?|non[-\s]?recurr(?:ing|ent)?|"
    r"one[-\s]?time|subscriptions?|single\s+(?:donations?|payments?|gifts?|contributions?))\b",
    re.I,
)

def _ministry_geography_requested(question: str) -> bool:
    if not re.search(r"\bministr(?:y|ies)\b", question, re.I):
        return False
    if _LOCATION_WORDS_RE.search(question):
        return True
    # A named ministry followed by a reporting month is not a location query.
    match = re.search(r"\bministr(?:y|ies)\s+(?:in|within|across)\s+(?:the\s+)?([A-Za-z]+)", question, re.I)
    if not match:
        return False
    word = match[1].lower()
    return word not in _MONTHS and word not in {
        "this", "last", "previous", "next", "past", "year", "week", "month",
        "database", "system", "report", "list",
    }

def unsupported_request_reason(question: str) -> str | None:
    """Known unavailable reports stop before SQL; ordinary donor geography does not."""
    if _PAYMENT_FREQUENCY_RE.search(question):
        return "payment_frequency"
    if re.search(r"\b(?:quickbooks|qbo)\b", question, re.I):
        return "quickbooks"
    if _UNSUPPORTED_OUTCOME_RE.search(question):
        return "payment_outcome"
    if _ministry_geography_requested(question):
        return "ministry_geography"
    return None

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
    numeric_dates = re.findall(r"\b\d{4,6}-\d{1,3}-\d{1,3}\b|\b\d{1,3}/\d{1,3}/\d{4,6}\b", q)
    malformed = any(not re.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}", token) for token in numeric_dates)
    malformed = malformed or bool(re.search(
        rf"\b{_MONTH_RE}\s+\d{{1,2}}(?:st|nd|rd|th)?[\s,]+\d{{5,}}\b|\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_RE}[\s,]+\d{{5,}}\b", q))
    if malformed:
        return {"period_name": "invalid", "period_label": "the requested period",
                "start_date": None, "exclusive_end_date": None,
                "period_ambiguity": "Please provide a valid date or date range, including the year.",
                "period_explicit": True}
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
        patterns = (
            (r"\b\d{4}-\d{2}-\d{2}\b", "%Y-%m-%d"),
            (r"\b\d{1,2}/\d{1,2}/\d{4}\b", "%d/%m/%Y"),
            (rf"\b(?P<month>{_MONTH_RE})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?(?:\s*,\s*|\s+)(?P<year>\d{{4}})\b", "named"),
            (rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month>{_MONTH_RE})(?:\s*,\s*|\s+)(?P<year>\d{{4}})\b", "named"),
        )
        found = []
        try:
            for pattern, fmt in patterns:
                for match in re.finditer(pattern, q):
                    if fmt == "named":
                        parsed = date(int(match["year"]), _MONTHS[match["month"]], int(match["day"]))
                    else:
                        parsed = datetime.strptime(match[0], fmt).date()
                    found.append((match.start(), parsed, match.end()))
            found.sort()
            if found:
                if len(found) > 2 or (len(found) == 2 and not re.search(r"\b(?:to|through|until|till|between|and)\b", q)): raise ValueError("ambiguous date range")
                start, end = found[0][1], found[-1][1] + timedelta(days=1)
                if end <= start: raise ValueError("reversed date range")
                name = "date" if len(found) == 1 else "date_range"; label = start.isoformat() if len(found) == 1 else f"{start.isoformat()} through {(end - timedelta(days=1)).isoformat()}"
                if len(found) == 1:
                    if re.search(r"\b(?:to|through|until|till|and)\s+(?:\d|"+_MONTH_RE+r"\b)", q[found[0][2]:]):
                        raise ValueError("unresolved range endpoint")
                    if re.search(r"\b(?:from|since)\s*$", q[:found[0][0]]):
                        name, end, ambiguity = "date_range", None, "Which end date should I use for this report?"
            else:
                month = re.search(rf"\b({_MONTH_RE})\b(?:\s+(\d{{4,}})\b)?", q); year = re.search(r"\b(?:in|for|during)\s+(\d{4,})\b", q)
                if month and month[1] == "may" and re.search(r"\bmay\s+(?:i|we|you)\b", q): month = None
                if month:
                    name, label = "month", month[1].title()
                    if not month[2]: ambiguity = f"Which year should I use for {label}?"
                    elif len(month[2]) != 4:
                        ambiguity = f"Please provide a valid four-digit year for {label}."
                    else: start = date(int(month[2]), _MONTHS[month[1]], 1); end, label = _next_month(start), f"{calendar.month_name[start.month]} {start.year}"
                elif year:
                    if len(year[1]) != 4: raise ValueError("invalid year")
                    start = date(int(year[1]), 1, 1); end, name, label = date(start.year + 1, 1, 1), "year", year[1]
                elif re.search(r"\b(?:last|past|next)\s+\d+\s+(?:days?|weeks?|months?|years?)\b|\bquarter\b", q): ambiguity = "Please provide the start and end dates for that reporting period."
                else: explicit = False
        except (ValueError, OverflowError):
            start, end = None, None
            ambiguity = "Please provide a valid date or date range, including the year."
    return {"period_name": name, "period_label": label, "start_date": start.isoformat() if start else None, "exclusive_end_date": end.isoformat() if end else None, "period_ambiguity": ambiguity, "period_explicit": explicit}

def _period(question: str, today: date) -> tuple[str, str | None, str | None]:
    period = _resolve_period(question, today)
    return period["period_name"], period["start_date"], period["exclusive_end_date"]

def _donor_name(question: str) -> str | None:
    # "donated/contributed/paid to ..." identifies the receiver, not the
    # donor.  Do not let the broad legacy fallback below overwrite a ministry
    # code/name extracted from the receiver clause (for example Q26).
    patterns = (
        r"\b(?:donations?|payments?|contributions?|gifts?|transactions?)\s+(?:(?:made|done|given|received)\s+)?(?:by|from)\s+(.+)",
        r"\b(?:donation|payment|contribution)\s+(?:history|records?|details?)\s+(?:of|by|from)\s+(.+)",
        r"\b(?:donated|contributed|paid)\s+by\s+(.+)",
        r"\b(?:did|has|have)\s+(.+?)\s+(?:donate[ds]?|contribute[ds]?|pay|paid)\b",
        r"^\s*(.+?)\s+(?:donated|contributed|paid)\b",
    )
    for index, pattern in enumerate(patterns):
        match = re.search(pattern, question, re.I)
        if match:
            name = _ENTITY_STOP.split(match[1], maxsplit=1)[0].strip(" .\"'")
            if (re.match(rf"\d|(?:{_MONTH_RE})\s+\d|(?:today|yesterday|this|last|all)\b", name, re.I)
                    and not re.fullmatch(r"#?\d+", name)):
                continue
            if name.split() and name.split()[0].lower() in {"show", "list", "get", "give", "how", "what", "which", "who", "are", "count", "total", "average", "all", "now", "also", "then", "please", "can", "could", "would", "donors", "givers"}: continue
            if index == 4 and re.search(r"\b(?:donors?|givers?|who|have|has)\b", name, re.I):
                continue
            if name and not re.fullmatch(r"ministr(?:y|ies)|donors?|payment method", name, re.I): return name
    return None

def payment_status(question: str, *, contextual: bool = False) -> tuple[str | None, bool]:
    """An explicit None clears a status restriction; absent wording does not."""
    q = question.lower().strip(" .!?")
    negative_pattern = (
        r"\bunpaid\b|\bnot\s+(?:yet\s+)?paid\b|\bnot\s+paid\s+yet\b|"
        r"\b(?:hasn['’]?t|haven['’]?t|hadn['’]?t|has\s+not|have\s+not|had\s+not)\s+"
        r"(?:(?:been|yet)\s+)?paid\b|\b(?:wasn['’]?t|weren['’]?t|was\s+not|were\s+not)\s+paid\b"
    )
    negative = bool(re.search(negative_pattern, q))
    positive = bool(re.search(r"\b(?:completed|confirmed)\b|\bpaid\s+(?:donations?|payments?|contributions?|transactions?)\b|\bstatus\s+(?:is\s+)?paid\b|\b(?:donations?|payments?)\s+(?:that\s+are\s+|are\s+)?paid\b", q))
    # Normalize already-supported negative wording for conjunction detection.
    # Do not let the 'paid' inside 'not paid' count as a positive status.
    status_text = re.sub(negative_pattern, "unpaid", q)
    both = bool(re.search(r"\bpaid\s*(?:and|or|&)\s*unpaid\b|\bunpaid\s*(?:and|or|&)\s*paid\b", status_text))
    if contextual:
        both = both or bool(re.fullmatch(r"(?:use\s+)?(?:both|combined|all statuses)", q))
        if re.search(r"\bpaid\b", q) and not negative and not re.search(r"\bpaid\s+(?:by|to|for)\b", q):
            positive = True
    if both:
        return None, True
    if negative:
        return "unpaid", True
    return ("paid", True) if positive else (None, False)

def entity_constraints(spec: dict) -> dict[str, dict]:
    """Read legacy single-role requests and dual-role requests in one place."""
    roles = {}
    if spec.get("entity_role") in {"donor", "receiving_ministry"}:
        roles[spec["entity_role"]] = {key: spec.get(key) for key in _ENTITY_FIELDS}
    if spec.get("donor_constraint"):
        roles["donor"] = spec["donor_constraint"]
    return roles

def resolve_pending_reply(pending: dict | None, reply: str) -> tuple[str, dict | None]:
    """Resolve a typed clarification without rebuilding or widening its report."""
    if not pending:
        return "new", None
    q = reply.strip()
    kind = pending.get("kind")
    spec = deepcopy(pending["query_spec"])
    if kind == "year":
        match = re.fullmatch(r"\s*(?:(?:use|in|year(?:\s+is)?|it\s+is)\s+)?(\d{4})\s*[.!]?\s*", q, re.I)
        if match:
            month = spec.get("period_label", "")
            period = _resolve_period(f"{month} {match[1]}", reporting_date())
            if period.get("period_ambiguity") or not period.get("start_date"):
                return "invalid", None
            for field in _PERIOD_FIELDS:
                spec[field] = period.get(field)
            return "resolved", spec
    elif kind == "status":
        if re.fullmatch(r"(?:(?:use|show|only|what about|how about)\s+)?(?:paid|unpaid|not\s+(?:yet\s+)?paid|both|combined|all statuses|(?:both\s+)?paid\s+and\s+unpaid)(?:\s+(?:ones|payments|donations))?\s*[.!?]?", q, re.I):
            status, explicit = payment_status(q, contextual=True)
            if explicit:
                spec["status"] = status
                spec["explicit_fields"] = list(set(spec.get("explicit_fields", [])) | {"status"})
                spec.pop("clarification", None)
                spec.pop("clarification_kind", None)
                return "resolved", spec
    elif kind == "ministry":
        if re.fullmatch(r"\s*(?:use\s+)?[A-Za-z0-9_-]+\s*[.!]?\s*", q, re.I):
            from components.query_normalizer import _regex_extract
            code = _regex_extract(q).get("ministry_code")
            if code:
                spec.update(entity_role="receiving_ministry", entity_code=code,
                            entity_name=None, entity_id=None, resolution_outcome=None)
                spec.pop("clarification", None)
                spec.pop("clarification_kind", None)
                return "resolved", spec
    elif kind == "period" and not PAYMENT_WORDS.search(q):
        allowed = rf"(?:{_MONTH_RE}|\d+(?:st|nd|rd|th)?|to|through|until|till|from|between|and|on|in|use|today|yesterday|this|last|previous|week|month|year|all[- ]time)"
        remainder = re.sub(rf"\b{allowed}\b", "", q.lower())
        if re.search(r"[^\s,./?!:\-]", remainder):
            if re.search(r"\b(?:donors?|givers?|ministries|show|list|count|select|drop|delete|ignore|weather|joke)\b", q, re.I):
                return "new", None
            return "invalid", None
        # Two dates on separate lines are meaningful in a pending range reply.
        period = _resolve_period(re.sub(r"\s*\n\s*", " to ", q), reporting_date())
        if not period.get("period_ambiguity") and period.get("start_date"):
            if spec.get("period_name") == "date_range" and spec.get("start_date") and not spec.get("exclusive_end_date"):
                if period["period_name"] == "date":
                    if period["exclusive_end_date"] <= spec["start_date"]:
                        return "invalid", None
                    period["start_date"] = spec["start_date"]
                    period["period_name"] = "date_range"
                    period["period_label"] = f'{spec["start_date"]} through {(date.fromisoformat(period["exclusive_end_date"]) - timedelta(days=1)).isoformat()}'
                elif period["period_name"] != "date_range":
                    return "invalid", None
            for field in _PERIOD_FIELDS:
                spec[field] = period.get(field)
            return "resolved", spec
    # Complete requests and unsafe text still traverse the existing guard.
    if PAYMENT_WORDS.search(q) or re.search(r"\b(?:donors?|givers?|ministries|show|list|count|select|drop|delete|ignore|weather|joke)\b", q, re.I) or len(q.split()) > 4:
        return "new", None
    return "invalid", None

def build_query_spec(question: str, entities: dict | None = None, *, today: date | None = None, contextual: bool = False) -> dict[str, Any]:
    q = question.replace("’", "'"); lower = q.lower(); period = _resolve_period(q, today if today is not None else reporting_date()); period_explicit = period.pop("period_explicit"); explicit = ["period"] if period_explicit else []
    payment = bool(PAYMENT_WORDS.search(q))
    if payment: explicit.append("subject")
    metric = None
    if re.search(r"\b(?:average|avg)\b", lower): metric = "average"
    elif re.search(r"\b(?:how many|count|number of)\b", lower): metric = "count"
    elif re.search(r"\b(?:how much|total|sum|revenue|income|raised)\b", lower): metric = "total"
    if metric: explicit.append("metric")
    ministry_output = bool(re.search(r"\b(?:which|list(?:\s+of)?|show(?:\s+me)?|total|count(?:\s+of)?|how many)\s+(?:the\s+|all\s+)?ministries\b", lower) and not re.search(r"\b(?:by|per) ministr(?:y|ies)\b", lower))
    if ministry_output:
        if metric == "total": metric = "count"
        if re.search(r"\blist\b", lower):
            metric = None
            explicit = [field for field in explicit if field != "metric"]
    if not payment and re.search(r"\bministr(?:y|ies)\b", lower):
        if metric == "total": metric = "count"
        if re.search(r"\blist\b", lower):
            metric = None
            explicit = [field for field in explicit if field != "metric"]
    group_by = "ministry" if re.search(r"\b(?:by|per) ministr(?:y|ies)\b", lower) else None
    if group_by: explicit.append("group_by")
    details = bool(re.search(r"\b(?:details?|list|show|breakdown)\b", lower)) and not metric and not group_by
    if details: explicit.append("detail")
    e = entities or {}; role, name, code = None, None, None
    if e.get("ministry_name") or e.get("ministry_code"): role, name, code = "receiving_ministry", e.get("ministry_name"), e.get("ministry_code")
    if (_MULTI_MINISTRY_DONORS_RE.search(q) and not code and name and
            re.fullmatch(r"(?:multiple|different|two|more than one|at least two|two or more)\s+ministr(?:y|ies)", name, re.I)):
        # The existing normalizer treats "multiple ministries" as one named
        # ministry. It is a quantifier here, not a receiver to resolve.
        role, name = None, None
    our_receiver = re.search(
        r"\bour (?:ministry|organisation|organization)\b|"
        r"\b(?:to|for)\s+us\b|\b(?:we|us)\s+(?:received?|raise[ds]?)\b", lower,
    )
    if payment and not role and our_receiver:
        role, name, code = "receiving_ministry", None, "098WRLD"
    donor = _donor_name(q)
    donor_constraint = None
    if donor and role == "receiving_ministry":
        donor_constraint = {"entity_role": "donor", "entity_name": donor, "entity_code": None, "entity_id": None}
    elif donor: role, name, code = "donor", donor, None
    if role: explicit.append("entity")
    if donor: explicit.append("donor")
    if role == "receiving_ministry": explicit.append("receiver")
    status, status_explicit = payment_status(q, contextual=contextual)
    if status_explicit: explicit.append("status")
    elif payment:
        # The approved default applies to a new report. A genuine follow-up
        # retains the earlier status because this field was not explicit.
        status = "paid"
    spec = {"subject": "payment" if payment else "other", **period, "status": status, "group_by": group_by, "entity_role": role, "entity_id": None, "entity_name": name, "entity_code": code, "metric": metric, "filters": {}, "explicit_fields": explicit}
    if donor_constraint: spec["donor_constraint"] = donor_constraint
    if _MISSING_CODE_RE.search(q):
        spec.update(subject="other", status=None, group_by=None,
                    result_entity="ministry", qualification="missing_ministry_code")
        explicit.extend(("subject", "status", "group_by", "result_entity", "qualification"))
    elif _MULTI_MINISTRY_DONORS_RE.search(q):
        spec.update(subject="payment", result_entity="donor", qualification="multi_ministry_donor")
        explicit.extend(("subject", "result_entity", "qualification"))
    elif _REPEAT_DONORS_RE.search(q):
        spec.update(subject="payment", result_entity="donor", qualification="repeat_donor")
        explicit.extend(("subject", "result_entity", "qualification"))
    elif contextual and re.search(r"\b(?:all|any)\s+(?:donors?|givers?)\b", lower):
        spec["qualification"] = None
        explicit.append("qualification")
    if spec.get("qualification") in {"repeat_donor", "multi_ministry_donor"} and not status_explicit:
        spec["status"] = "paid"
    if spec.get("qualification") in {"repeat_donor", "multi_ministry_donor"} and status_explicit and status != "paid":
        spec["clarification"] = "Repeat and multi-ministry donor reports count paid donations. Please ask for a paid-donation report or a separate unpaid-payment report."
        spec["clarification_kind"] = "definition"
    if spec.get("qualification") and metric in {"total", "average"}:
        spec["clarification"] = "Would you like a list or a count for this report?"
        spec["clarification_kind"] = "definition"
    if spec.get("qualification") == "missing_ministry_code" and period_explicit:
        spec["clarification"] = "The missing-code report is a current ministry list, not a date-based donation report. Please ask without a period."
        spec["clarification_kind"] = "definition"
    if spec.get("qualification") == "multi_ministry_donor" and role == "receiving_ministry":
        spec["clarification"] = "Should I find donors who supported this ministry and another ministry, or donors who supported any two ministries?"
        spec["clarification_kind"] = "definition"
    spec["unsupported_reason"] = unsupported_request_reason(q)
    if spec.get("unsupported_reason"):
        spec["period_ambiguity"] = None
        spec["clarification"] = USER_FACING_LIMITS[spec["unsupported_reason"]]
        spec["clarification_kind"] = "unsupported"
    from components.query_normalizer import _CODE_BARE_RE
    codes = list(dict.fromkeys(m[1].upper() for m in _CODE_BARE_RE.finditer(q)))
    if len(codes) > 1:
        spec["clarification"] = "Please choose one ministry code for this report."
        spec["clarification_kind"] = "ministry"
    donor_rows = bool((re.search(r"\bdonors?\b", lower) and re.search(r"\b(?:list|show|get|give|which|who)\b", lower)) or _GIVER_LIST_RE.search(lower))
    donor_count = metric == "count" and bool(re.search(r"\b(?:how many|count(?:\s+of)?|number of)\s+(?:the\s+|all\s+)?(?:donors?|givers?)\b", lower))
    if not spec.get("qualification"):
        if payment:
            spec["result_entity"] = "ministry" if ministry_output else ("donor" if (donor_count or donor_rows and not metric) and not group_by else "payment")
        elif re.search(r"\b(?:donors?|contacts?|profile|email|phone)\b", lower) or _GIVER_LIST_RE.search(lower):
            spec["result_entity"] = "donor"
        elif re.search(r"\bministr(?:y|ies)\b", lower):
            spec["result_entity"] = "ministry"
        else:
            spec["result_entity"] = "record"
    if spec.get("result_entity") == "ministry" and spec.get("qualification") != "missing_ministry_code" and re.search(r"\b(?:details?|full\s+profile|gift\s*codes?|ministry\s+codes?)\b", lower):
        spec["display_detail"] = True
        explicit.append("display_detail")
    if donor_count or donor_rows or ministry_output or re.search(r"\b(?:payment|donation|contribution) details?\b", lower):
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
    elif spec.get("subject") == "payment" and spec.get("result_entity") == "ministry": spec["operation"], spec["result_shape"] = "ministry_list", "detail"
    elif spec.get("subject") == "payment" and spec.get("result_entity") == "payment": spec["operation"], spec["result_shape"] = "details", "detail"
    elif spec.get("operation") == "donor_list": spec["result_shape"] = "detail"
    else: spec["operation"] = "details" if spec.get("subject") == "payment" else "list"; spec["result_shape"] = "detail"
    return spec

def merge_follow_up(current: dict, previous: dict | None) -> dict:
    """Called only for genuine follow-ups. Replace complete field bundles."""
    if not previous: return current
    merged = deepcopy(previous); explicit = set(current.get("explicit_fields", []))
    for field in ("subject", "status", "metric", "group_by", "filters", "result_entity", "qualification", "display_detail"):
        if field in explicit: merged[field] = deepcopy(current.get(field))
    # Recalculate operation/result_shape from merged result_entity.
    # Preserve old entity and period filters unless explicitly changed.
    if "donor" in explicit or "receiver" in explicit:
        roles = entity_constraints(previous)
        current_roles = entity_constraints(current)
        for flag, role in (("donor", "donor"), ("receiver", "receiving_ministry")):
            if flag in explicit:
                roles[role] = current_roles[role]
        primary = roles.get("receiving_ministry") or roles.get("donor") or {}
        for field in _ENTITY_FIELDS:
            merged[field] = primary.get(field)
        merged.pop("donor_constraint", None)
        if "receiving_ministry" in roles and "donor" in roles:
            merged["donor_constraint"] = deepcopy(roles["donor"])
    for bundle, fields in (("entity", _ENTITY_FIELDS), ("period", _PERIOD_FIELDS)):
        if bundle in explicit:
            if bundle == "entity" and ("donor" in explicit or "receiver" in explicit):
                continue
            for field in fields: merged[field] = current.get(field)
    if "entity" in explicit and current.get("entity_role") != "receiving_ministry": merged["group_by"] = None
    if "detail" in explicit: merged["metric"], merged["group_by"] = None, None
    merged["explicit_fields"] = list(current.get("explicit_fields", []))
    if current.get("clarification"):
        for field in ("clarification", "clarification_kind", "unresolved_condition"):
            if field in current:
                merged[field] = current[field]
    return _result_fields(merged)

def semantic_sql_errors(sql: str, spec: dict) -> list[str]:
    from components.sql_contract import semantic_sql_errors as validate
    return validate(sql, spec)
