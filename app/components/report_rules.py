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
        if metric == "count":
            return f"SELECT COUNT(*) AS record_count FROM sf_ministries m WHERE {where}"
        return ("SELECT m.sfid AS ministry_id, m.name AS ministry_name "
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
    if kind == "multi_ministry_donor" or receiver:
        joins.append("JOIN sf_ministries m ON p.ministryid = m.sfid")
    conditions = ["p.paid = 1", *_date_bounds(spec)]
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
    return (f"SELECT c.sfid AS donor_id, c.firstname AS first_name, "
            f"c.lastname AS last_name, {counted} AS {count_alias} "
            + source + grouped + having + f" ORDER BY {count_alias} DESC LIMIT 100")
