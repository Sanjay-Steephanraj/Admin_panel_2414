"""
analysis/churn_summary.py
─────────────────────────
Deterministic markdown formatter for churn-risk results. No LLM calls.
"""
import datetime


def build_churn_summary(
    risk_donors: list[dict],
    ministry_scope: str | None,
    total_at_risk: int,
    analysis_date: datetime.date | None = None,
    offset: int = 0,
) -> str:
    """
    Build a deterministic markdown summary of churn-risk donors.

    risk_donors: list of donor risk dicts (already trimmed to display limit).
    ministry_scope: human label for the ministry filter, or None.
    total_at_risk: total number of donors flagged (before trimming).
    analysis_date: date the analysis was run (defaults to today).
    offset: 0-based index of the first donor in this page (for pagination).
    """
    try:
        if analysis_date is None:
            analysis_date = datetime.date.today()
        date_str = f"{analysis_date.strftime('%B')} {analysis_date.day}, {analysis_date.year}"
        scope_text = f"for {ministry_scope}" if ministry_scope else "across all ministries"

        history_rule = (
            "Only donors whose latest donation was within the past 24 months are included; "
            "giving frequency is inferred from up to 48 months of donation history."
        )

        if not risk_donors:
            if offset > 0:
                return (
                    f"As of {date_str}, there are no further donors to show {scope_text} "
                    f"— all {total_at_risk} flagged donors have already been listed. "
                    f"This is an inactivity-based risk indicator, not confirmed churn."
                )
            return (
                f"Good news — as of {date_str}, based on each donor's usual giving schedule "
                f"{scope_text}, no donors currently exceed the inactivity threshold for their "
                f"usual giving schedule. Donors with insufficient or irregular giving history "
                f"were excluded from this analysis. "
                f"{history_rule} This is an inactivity-based risk indicator, not confirmed churn."
            )

        n = len(risk_donors)

        if offset > 0:
            intro = (
                f"As of {date_str}, continuing the inactivity analysis {scope_text}, "
                f"here are donors {offset + 1}–{offset + n} of {total_at_risk} flagged, "
                f"ordered by who most recently crossed their inactivity threshold."
            )
        elif total_at_risk > n:
            intro = (
                f"As of {date_str}, based on each donor's usual giving schedule {scope_text}, "
                f"here are the {n} donors who most recently crossed their inactivity threshold, "
                f"out of {total_at_risk} donors flagged."
            )
        else:
            intro = (
                f"As of {date_str}, based on each donor's usual giving schedule {scope_text}, "
                f"here are the {n} donors who most recently crossed their inactivity threshold."
            )

        bullets = []
        for d in risk_donors:
            donor_name = d.get("donor_name", "")
            ministry_name = d.get("ministry_name", "")
            frequency = d.get("frequency", "")
            last_display = d.get("last_donation_date_display", "")
            days_since_last = d.get("days_since_last", "")
            threshold_days = d.get("threshold_days", "")

            if ministry_name == "All ministries":
                ministry_segment = ""
            else:
                ministry_segment = f"Ministry: {ministry_name} | "

            bullets.append(
                f"- **{donor_name}** — {ministry_segment}"
                f"Usual frequency: {frequency} | Last donation: {last_display} | "
                f"Risk indicator: {days_since_last} days since last donation, "
                f"exceeding the {threshold_days}-day threshold for {frequency} donors."
            )
        bullets_block = "\n".join(bullets)

        footer = (
            "This is an inactivity-based risk indicator, not confirmed churn. "
            "Donors with insufficient or irregular giving history were excluded from this analysis. "
            f"{history_rule}"
        )
        if offset > 0:
            if total_at_risk > offset + n:
                footer += f" Showing donors {offset + 1}–{offset + n} of {total_at_risk} in total."
            else:
                footer += " This is the end of the list; all flagged donors have now been shown."
        elif total_at_risk > n:
            footer += (
                f" The {n} donors who most recently crossed their inactivity threshold are shown; "
                f"{total_at_risk} donors met the criteria in total."
            )

        return "\n\n".join([intro, bullets_block, footer])
    except Exception:
        try:
            names = [d.get("donor_name", "unknown") for d in risk_donors]
            return "Donors flagged as churn risk:\n" + "\n".join(f"- {n}" for n in names)
        except Exception:
            return "Unable to build churn summary."
