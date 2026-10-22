"""Approved report meanings shared by interpretation and SQL generation.

This is application configuration, not a new database or a source of inferred
business facts. Unverified capabilities remain unavailable.
"""
from __future__ import annotations


REPORT_MEANINGS = {
    "continuation_dimensions": "Calculation, grouping and reporting period are independent dimensions. A calculation-only continuation such as Total amount instead preserves the current grouping, even if its calculation is already total. Only an explicit request to combine groups into one overall/single result removes grouping. Changing a scalar to month-by-month does not change or widen its date filter: a September-only report grouped by month is valid and has at most one dated month bucket. Do not require multiple months or a new year to permit this grouping. These rules apply equally to counts, totals and averages.",
    "ranking_size": "An output size such as top 5 or bottom 5 ministries is the supported ranking.limit, not a donation-amount/count threshold. Rank receiving ministries by the requested total, average or payment count, in the requested direction, with at most 100 rows. In contrast, donors giving over 500 dollars or ministries receiving at least 10 donations require an unsupported qualifying threshold. Do not refuse a supported ministry ranking because it contains a numeric list size.",
    "compiled_report_encoding": "In a COMPILED report, subject=other with result_entity donor/ministry means the corresponding contact/ministry profile or directory, not off-topic and not a missing subject. The interpreter OUTPUT vocabulary still uses subject=profile for that intent; Python compiles it to other. Payment activity uses subject=payment, while the existing usual-pattern analysis uses subject=inactivity. Do not reject a correct profile simply because its compiled subject is named other.",
    "calendar_or_schedule": "Distinguish a reporting bucket from a payment-schedule qualifier. A monthly donation total/breakdown or counts for each calendar month requests calendar buckets. A phrase describing monthly giving/payments without establishing calendar-report buckets can describe a donation schedule instead. If both readings remain possible, clarify rather than assume a calendar report or silently discard a schedule restriction. A proposed calendar group does not prove the user requested calendar rather than scheduled giving. Explicit schedule/recurring conditions remain payment_frequency limitations.",
    "dimension_priority": "Resolve output entity, calculation, grouping and filters independently. Explicit current instructions override inherited choices; unchanged choices are inherited only on a follow-up. Apply a default only when neither supplies a value. In particular, monthly grouping never overrides an explicit count or average with a total. Changing just grouping or period retains the active calculation.",
    "grouped_calculations": "For an aggregate table, group_by names the rows and result_entity names what is calculated inside each group. group_by=\"ministry\", result_entity=\"payment\", measure total/average/count means each ministry's payment amount/average/payment count, including top/bottom ministry rankings. It does not request payment-detail rows. result_entity=\"ministry\" with count would instead count distinct ministries. Similarly, month grouping with result_entity=\"donor\" and count means distinct donors within each month, not a donor-detail list. A supported fixed qualification encodes its approved predicate even when the generic filters map is empty.",
    "analytics_capabilities": "The supported general calculations are one count, total amount, average payment amount, or list per report, with no grouping or one grouping by ministry or calendar year-month. Existing fixed qualification and inactivity reports retain their documented rules. Unimplemented analyses must not be approximated by a nearby supported report: day/week/quarter/year grouping, cross-period comparison calculations, differences/growth/percentage shares, median/percentiles, cumulative/rolling measures, forecasting, donor ranking, arbitrary numeric thresholds and multi-measure reports. A date filter such as last week or in 2025 is not grouping by week or year. A normal chronological monthly table is not a cumulative or growth calculation. Stop with analytics_capability rather than silently discard an operation or condition. This is a current implementation boundary, not a claim that the database or product could never support those analyses.",
    "payments": "An individual recorded payment is one donation, including each instalment. Default to paid records; unpaid and both statuses require an explicit request.",
    "donors": "A donor is a distinct contact, not a payment row. A donation-based donor list/count is distinct people or organisations with qualifying payments. A donor profile/directory does not imply giving activity.",
    "ministries": "A receiving ministry is separate from a donor. Keep both restrictions when both are requested. A ministry directory is different from ministries with donation activity. Ministry activity lists, grouped aggregates and rankings contain actual linked ministry records with contributing payments, not a blank group of unmatched payments or zero-activity directory entries. Ordinary payment totals and histories still include unmatched payments unless a requested ministry filter excludes them.",
    "repeat_donor": "More than one paid payment per donor, within the requested period, or all time if no period is requested. Repeated giving does not establish a subscription.",
    "multi_ministry_donor": "Paid giving to at least two distinct ministries in the requested period, or all time. The existing report is any two ministries; requiring a particular ministry plus another is not supported.",
    "missing_ministry_code": "Current ministries whose gift code is NULL or blank after trimming; not unmatched payments. This report has no donation period.",
    "missing_values": "Supported missing checks are donor email and ministry description. Missing means NULL or blank after trimming. Ministry active/inactive uses the explicit ministry flag, not donor inactivity.",
    "inactivity": "The existing donor analysis compares usual giving patterns using its fixed historical rule. It is not recurring classification, an arbitrary date cutoff, or a named-donor report. Its count means donors flagged by that analysis.",
    "ranking": "Ministry ranking requires an explicit donation measure (total amount, payment count or average) and direction. If the measure is unclear, ask. Maximum displayed list size is 100.",
    "dates": "Resolve supported periods using the reporting date/timezone in code. Do not guess a missing/invalid year, repair malformed dates silently, or replace an empty period with all-time data.",
    "monthly_reports": "Monthly donation reporting groups payments by calendar year and month of paymentdate, not subscription frequency. Preserve the independently requested period, identities, status and conditions. Default monthly breakdowns to total paymentamount; existing counts and averages can also use month grouping. Donor/ministry counts are distinct IDs within each month and must not be added together to claim an all-period distinct count. Return month plus the numeric measure in chronological order, maximum 100 rows. Show only months with matching payments; missing months are not automatically zero. If an all-time report includes undated payments, show a separate Date not recorded row. Ministry-and-month grouping, monthly ranking and monthly qualifying-donor reports are not part of this report.",
    "our_ministry": "Our organisation, our ministry and donations to us identify gift code 098WRLD, not every ministry.",
}


def compiled_contract_labels(spec: dict) -> dict:
    """Describe existing encoded fields without changing or inferring a report.

    Only closed categorical descriptions are added to model context. Identity,
    date and condition values still come from the original compiled report.
    """
    subject, entity = spec.get("subject"), spec.get("result_entity")
    source = ("payment_records" if subject == "payment" else
              "usual_giving_pattern" if subject == "inactivity" else
              {"donor": "contact_directory", "ministry": "ministry_directory"}.get(entity, "unresolved"))
    grouping, metric = spec.get("group_by"), spec.get("metric")
    rows = ({"month": "calendar_year_month", "ministry": "receiving_ministry"}.get(grouping)
            if grouping else ("single_value" if spec.get("result_shape") == "scalar" else entity))
    calculation = ({"donor": "distinct_donor_count", "ministry": "distinct_ministry_count"}.get(entity, "payment_count")
                   if metric == "count" and subject == "payment" else
                   {"total": "sum_payment_amount", "average": "average_payment_amount", "count": "directory_or_analysis_count"}.get(metric, "list"))
    return {"source_kind": source, "rows_represent": rows, "value_represents": calculation,
            "ranking_limit_is_output_size": bool(spec.get("ranking"))}


def report_semantics_prompt() -> str:
    from components.report_options import FIELD_COLUMNS
    from components.query_spec import USER_FACING_LIMITS
    lines = ["APPROVED REPORT MEANINGS:"]
    lines.extend(f"- {name}: {meaning}" for name, meaning in REPORT_MEANINGS.items())
    lines.append("Supported displayed fields: " + ", ".join(FIELD_COLUMNS))
    lines.append("Unavailable meanings (never drop one to answer a broader report):")
    lines.extend(f"- {name}: {message}" for name, message in USER_FACING_LIMITS.items())
    return "\n".join(lines)
