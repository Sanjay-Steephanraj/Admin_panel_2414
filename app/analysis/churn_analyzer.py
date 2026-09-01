"""
analysis/churn_analyzer.py
──────────────────────────
Deterministic frequency-based donor inactivity risk indicator.

Bypasses the LLM NL2SQL pipeline: builds raw payment-history SQL, groups by
donor (optionally per ministry), infers each donor's usual giving cadence from
the median inter-donation interval, and flags donors whose last donation
exceeds the cadence-specific inactivity threshold.

This is an inactivity indicator, not confirmed churn.
"""
import datetime
import logging
import re
import statistics

logger = logging.getLogger(__name__)

FREQUENCY_BANDS = [
    # (label, min_median_days, max_median_days, threshold_days)
    ("monthly",      23,  38,  90),   # no donation for 3 months
    ("quarterly",    68, 113, 180),   # no donation for 6 months
    ("six-monthly", 137, 228, 270),   # no donation for 9 months
    ("annual",      274, 456, 450),   # no donation for 15 months
]
MIN_PAYMENTS = 3          # minimum payments to infer a frequency
MAX_INTERVALS = 6         # use up to the 6 most recent intervals
CONSISTENCY_RATIO = 0.75  # with 3+ intervals, >=75% must match the median's band
RESULT_LIMIT = 15         # return the 15 most-overdue donors

FREQUENCY_HISTORY_DAYS = 1461  # use up to 48 months of donation history for frequency inference
RECENCY_LIMIT_DAYS     = 730   # only flag donors whose latest donation was within the past 24 months


def build_churn_sql(ministry_code: str | None = None, ministry_name: str | None = None) -> str:
    """Build the raw per-donor payment-history query."""
    sql = (
        "SELECT\n"
        "    CONCAT(COALESCE(c.firstname, ''), ' ', COALESCE(c.lastname, '')) AS donor_name,\n"
        "    c.sfid AS donor_sfid,\n"
        "    m.name AS ministry_name,\n"
        "    p.paymentdate,\n"
        "    p.paymentamount\n"
        "FROM sfpayments p\n"
        "LEFT JOIN sf_opportunities o ON p.oppsfid = o.sfid\n"
        "LEFT JOIN sf_contacts c ON o.primarycontact = c.sfid\n"
        "LEFT JOIN sf_ministries m ON p.ministryid = m.sfid\n"
        "WHERE p.paid = 1\n"
        "  AND p.paymentamount > 0\n"
        "  AND p.paymentdate IS NOT NULL\n"
        "  AND c.sfid IS NOT NULL\n"
        "  AND p.paymentdate >= DATE_SUB(CURDATE(), INTERVAL 48 MONTH)\n"
    )

    conditions = []
    if ministry_code:
        code = re.sub(r"[^A-Za-z0-9]", "", ministry_code).upper()
        if code:
            conditions.append(f"UPPER(m.giftcode) = '{code}'")
    if ministry_name:
        # Keep only a leading run of safe characters; discard the rest. Pure
        # char-filtering leaves alphanumeric SQL keywords (e.g. DROP) intact in
        # the LIKE literal, so we truncate at the first disallowed character to
        # fully neutralize injection payloads. deliberate: legitimate names with
        # apostrophes (e.g. "Women's") truncate at the apostrophe — acceptable
        # tradeoff; switch to escaping if ministry names need punctuation.
        m = re.match(r"[A-Za-z0-9 \-&.,]*", ministry_name)
        name = m.group(0) if m else ""
        name = re.sub(r"\s+", " ", name).strip()
        if name:
            conditions.append(f"UPPER(m.name) LIKE '%{name.upper()}%'")

    if conditions:
        joined = " OR ".join(conditions)
        if len(conditions) > 1:
            joined = f"({joined})"
        sql += f"  AND {joined}\n"

    sql += "ORDER BY c.sfid ASC, p.paymentdate ASC"
    return sql


def _find_band(interval_days):
    """Return (label, threshold) for the band whose [min,max] contains the interval, else None."""
    for label, mn, mx, threshold in FREQUENCY_BANDS:
        if mn <= interval_days <= mx:
            return label, threshold
    return None


def classify_donor_frequency(payment_dates: list, today=None) -> dict | None:
    """
    Infer a donor's giving frequency from their payment dates.

    Returns {"frequency", "threshold_days", "median_interval_days"} or None if
    history is insufficient or irregular.
    """
    if not payment_dates:
        return None

    dates = sorted(payment_dates)
    # Deduplicate identical dates (same-day multiple payments = one event).
    unique = []
    seen = set()
    for d in dates:
        key = d
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)

    if len(unique) < MIN_PAYMENTS:
        return None

    intervals = []
    for i in range(1, len(unique)):
        gap = (unique[i] - unique[i - 1]).days
        if gap > 0:
            intervals.append(gap)

    if not intervals:
        return None

    # Use the most recent MAX_INTERVALS intervals.
    intervals = intervals[-MAX_INTERVALS:]
    if len(intervals) < MIN_PAYMENTS - 1:
        # Need at least 2 intervals to infer a cadence.
        if len(intervals) < 2:
            return None

    median = statistics.median(intervals)
    median_band = _find_band(median)
    if median_band is None:
        return None
    median_label, median_threshold = median_band

    # Consistency: how many intervals fall in the median's band.
    in_band = sum(1 for iv in intervals if _find_band(iv) is not None and _find_band(iv)[0] == median_label)

    if len(intervals) == 2:
        if in_band != 2:
            return None
    else:
        if in_band / len(intervals) < CONSISTENCY_RATIO:
            return None

    return {
        "frequency": median_label,
        "threshold_days": median_threshold,
        "median_interval_days": int(round(median)),
    }


def _parse_date(value):
    """Parse an ISO date/datetime string or pass through a date/datetime object.

    Handles DATETIME columns whose values carry a time component
    (e.g. "2021-04-05T00:00:00" or "2021-04-05 00:00:00"), which
    datetime.date.fromisoformat rejects. datetime is checked before
    date because datetime is a subclass of date.
    """
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, str):
        try:
            return datetime.datetime.fromisoformat(value).date()
        except ValueError:
            return None
    return None


def classify_churn_risk(payment_rows: list[dict], ministry_scoped: bool = False, today=None) -> list[dict]:
    """
    Group payment rows by donor (optionally per ministry), infer frequency,
    and return donors whose last donation exceeds their cadence threshold.

    Returns the full list sorted by overdue_ratio descending; caller trims.
    """
    if today is None:
        today = datetime.date.today()

    groups = {}
    for row in payment_rows:
        d = _parse_date(row.get("paymentdate"))
        if d is None:
            continue
        donor_sfid = row.get("donor_sfid")
        donor_name = row.get("donor_name", "") or ""
        ministry_name = row.get("ministry_name")

        if ministry_scoped:
            key = (donor_sfid, ministry_name)
        else:
            key = donor_sfid

        entry = groups.setdefault(key, {
            "donor_name": donor_name,
            "ministry_name": ministry_name,
            "dates": [],
        })
        entry["dates"].append(d)

    results = []
    for key, entry in groups.items():
        dates = entry["dates"]
        # Defensive Python-side enforcement of the 48-month frequency window.
        # The SQL already filters to 48 months, but tests pass synthetic data
        # with an explicit `today`, so re-window here.
        windowed = [d for d in dates if (today - d).days <= FREQUENCY_HISTORY_DAYS]
        if not windowed:
            continue

        freq = classify_donor_frequency(windowed)
        if freq is None:
            continue

        last_donation_date = max(windowed)
        days_since_last = (today - last_donation_date).days
        threshold = freq["threshold_days"]

        # Donors whose latest donation is older than 24 months are excluded —
        # follow-up is unlikely to be useful.
        if days_since_last > RECENCY_LIMIT_DAYS:
            continue

        if days_since_last <= threshold:
            continue

        overdue_ratio = days_since_last / threshold
        frequency = freq["frequency"]
        median_interval_days = freq["median_interval_days"]

        if ministry_scoped:
            ministry_label = entry["ministry_name"] if entry["ministry_name"] is not None else "Unknown ministry"
        else:
            ministry_label = "All ministries"

        last_display = f"{last_donation_date.strftime('%B')} {last_donation_date.day}, {last_donation_date.year}"
        explanation = (
            f"Usual giving pattern is {frequency} (median interval {median_interval_days} days). "
            f"Last donation was on {last_display} — {days_since_last} days ago, "
            f"exceeding the {threshold}-day inactivity threshold for {frequency} donors."
        )

        results.append({
            "donor_name": entry["donor_name"].strip(),
            "ministry_name": ministry_label,
            "frequency": frequency,
            "last_donation_date": last_donation_date.isoformat(),
            "last_donation_date_display": last_display,
            "days_since_last": days_since_last,
            "threshold_days": threshold,
            "median_interval_days": median_interval_days,
            "overdue_ratio": round(overdue_ratio, 2),
            "days_past_threshold": days_since_last - threshold,
            "explanation": explanation,
        })

    # Donors who crossed their inactivity threshold most recently come FIRST:
    # ascending days_past_threshold, tie-break ascending by days_since_last.
    results.sort(key=lambda r: (r["days_past_threshold"], r["days_since_last"]))
    return results


def run_churn_analysis(entities: dict | None, offset: int = 0, limit: int | None = None) -> dict:
    """
    End-to-end deterministic churn analysis.

    Builds SQL, fetches rows via db_tool, classifies, and returns a result dict
    with a markdown summary built by churn_summary.build_churn_summary.

    offset: 0-based index of the first donor to return (pagination).
    limit: page size (defaults to RESULT_LIMIT, clamped to 1..50).
    """
    try:
        from tools.db_tool import execute_query
        from components.prompt_config import get_error_message
        from analysis.churn_summary import build_churn_summary

        if offset < 0:
            offset = 0
        if limit is None:
            limit = RESULT_LIMIT
        if limit < 1:
            limit = 1
        elif limit > 50:
            limit = 50

        entities = entities or {}
        ministry_code = entities.get("ministry_code")
        ministry_name = entities.get("ministry_name")
        ministry_scoped = bool(ministry_code or ministry_name)

        sql = build_churn_sql(ministry_code, ministry_name)
        rows, err = execute_query(sql)
        if err:
            logger.error("churn analysis db error: %s", err)
            return {
                "summary": get_error_message("db_failure"),
                "row_count": 0,
                "db_result": [],
                "generated_sql": sql,
                "error": err,
                "offset": offset,
                "shown": 0,
                "total_at_risk": 0,
            }

        if not rows:
            return {
                "summary": get_error_message("no_results"),
                "row_count": 0,
                "db_result": [],
                "generated_sql": sql,
                "error": None,
                "offset": offset,
                "shown": 0,
                "total_at_risk": 0,
            }

        all_at_risk = classify_churn_risk(rows, ministry_scoped=ministry_scoped)
        page = all_at_risk[offset : offset + limit]

        scope_label = ministry_name or ministry_code or None
        summary = build_churn_summary(
            page, scope_label, total_at_risk=len(all_at_risk),
            analysis_date=datetime.date.today(),
            offset=offset,
        )

        return {
            "summary": summary,
            "row_count": len(page),
            "db_result": page,
            "generated_sql": sql,
            "error": None,
            "offset": offset,
            "shown": len(page),
            "total_at_risk": len(all_at_risk),
        }
    except Exception:
        logger.exception("unexpected error in run_churn_analysis")
        try:
            from components.prompt_config import get_error_message
            msg = get_error_message("db_failure")
        except Exception:
            msg = "Analysis failed due to an unexpected error."
        return {
            "summary": msg,
            "row_count": 0,
            "db_result": [],
            "generated_sql": None,
            "error": "unexpected_error",
            "offset": offset,
            "shown": 0,
            "total_at_risk": 0,
        }
