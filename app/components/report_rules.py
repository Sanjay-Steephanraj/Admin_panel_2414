"""Fixed SQL for the three reports whose business meanings are now confirmed.

The ordinary NL2SQL route is unchanged. These reports use a fixed query so a
model cannot silently omit their qualifying HAVING or missing-code condition.
"""
from __future__ import annotations

from datetime import date


QUALIFIED_REPORTS = {"repeat_donor", "multi_ministry_donor", "missing_ministry_code"}


def _date_bounds(spec: dict) -> list[str]:
    start, end = spec.get("start_date"), spec.get("exclusive_end_date")
    if not start and not end:
        return []
    if not start or not end:
        raise ValueError("A report needs both date boundaries")
    start, end = date.fromisoformat(start).isoformat(), date.fromisoformat(end).isoformat()
    if start >= end:
        raise ValueError("The end date must be after the start date")
    return [f"p.paymentdate >= '{start}'", f"p.paymentdate < '{end}'"]


def special_report_sql(spec: dict) -> str | None:
    """Return a fixed SELECT for an agreed qualifying report, or None."""
    kind = spec.get("qualification")
    if kind not in QUALIFIED_REPORTS:
        return None
    metric = spec.get("metric")
    if metric not in {None, "count"}:
        raise ValueError("This qualifying report supports a list or a count")
    if kind == "missing_ministry_code":
        where = "(m.giftcode IS NULL OR TRIM(m.giftcode) = '')"
        from components.report_options import filter_predicates
        extra = filter_predicates(spec.get("filters") or {}, {"sf_ministries": "m"})
        if extra:
            where += " AND " + " AND ".join(extra)
        if metric == "count":
            return f"SELECT COUNT(*) AS record_count FROM sf_ministries m WHERE {where}"
        return ("SELECT m.sfid AS ministry_id, m.name AS ministry_name, m.giftcode AS gift_code "
                f"FROM sf_ministries m WHERE {where} ORDER BY m.name LIMIT 100")

    if spec.get("status") != "paid":
        raise ValueError("Qualifying donors are counted from paid donation records")
    from components.query_spec import entity_constraints
    from components.sql_contract import _donor_filter

    roles = entity_constraints(spec)
    receiver, donor = roles.get("receiving_ministry"), roles.get("donor")
    if kind == "multi_ministry_donor" and receiver:
        raise ValueError("A receiving-ministry filter needs clarification for this report")
    joins = [
        "JOIN sf_opportunities o ON p.oppsfid = o.sfid",
        "JOIN sf_contacts c ON o.primarycontact = c.sfid",
    ]
    ministry_needed = kind == "multi_ministry_donor" or receiver or any(
        key in (spec.get("filters") or {}) for key in ("description_missing", "ministry_active"))
    if ministry_needed:
        joins.append("JOIN sf_ministries m ON p.ministryid = m.sfid")
    conditions = ["p.paid = 1", *_date_bounds(spec)]
    from components.report_options import FIELD_COLUMNS, filter_predicates
    conditions.extend(filter_predicates(spec.get("filters") or {}, {
        "sfpayments": "p", "sf_contacts": "c", **({"sf_ministries": "m"} if ministry_needed else {})}))
    if receiver:
        official_id = receiver.get("entity_id")
        if not official_id:
            raise ValueError("The receiving ministry must be resolved first")
        conditions.append("m.sfid = '" + str(official_id).replace("'", "''") + "'")
    if donor:
        conditions.append("(" + _donor_filter(donor, "c").sql(dialect="mysql") + ")")
    source = ("FROM sfpayments p " + " ".join(joins) + " WHERE " + " AND ".join(conditions))
    grouped = " GROUP BY c.sfid, c.firstname, c.lastname"
    counted = "COUNT(DISTINCT p.sfid)" if kind == "repeat_donor" else "COUNT(DISTINCT m.sfid)"
    having = f" HAVING {counted} > 1"
    if metric == "count":
        inner = "SELECT c.sfid " + source + grouped + having
        return f"SELECT COUNT(*) AS record_count FROM ({inner}) AS qualifying_donors"
    count_alias = "donation_count" if kind == "repeat_donor" else "ministry_count"
    extra_columns = []
    for field in spec.get("display_columns") or []:
        if field == "donor_name":
            continue
        columns = FIELD_COLUMNS.get(field)
        if not columns or any(table != "sf_contacts" for table, _ in columns):
            raise ValueError("This donor report can only display donor attributes")
        for _, column in sorted(columns):
            if column not in {"sfid", "firstname", "lastname"}:
                extra_columns.append(f"c.{column} AS {column}")
    extras = ", " + ", ".join(dict.fromkeys(extra_columns)) if extra_columns else ""
    if extra_columns:
        grouped += ", " + ", ".join(item.split(" AS ")[0] for item in dict.fromkeys(extra_columns))
    return (f"SELECT c.sfid AS donor_id, c.firstname AS first_name, "
            f"c.lastname AS last_name, {counted} AS {count_alias}{extras} "
            + source + grouped + having + f" ORDER BY {count_alias} DESC LIMIT 100")
