"""Small, explicit vocabulary for supported conditions and displayed fields.

These options belong to the existing reports; they do not grant access to new
tables or define PCN engagement, subscription, or integration meanings.
"""
import re

FIELD_COLUMNS = {
    "donor_name": {("sf_contacts", "firstname"), ("sf_contacts", "lastname")},
    "email": {("sf_contacts", "email")},
    "phone": {("sf_contacts", "phone")},
    "donor_id": {("sf_contacts", "donorid")},
    "address": {("sf_contacts", c) for c in ("addressstreet1", "addresscity", "addressstate", "addresspostalcode", "addresscountry")},
    "payment_amount": {("sfpayments", "paymentamount")},
    "payment_date": {("sfpayments", "paymentdate")},
    "month": {("sfpayments", "paymentdate")},
    "payment_status": {("sfpayments", "paid")},
    "payment_method": {("sfpayments", "payment_method")},
    "ministry_name": {("sf_ministries", "name")},
    "gift_code": {("sf_ministries", "giftcode")},
    "description": {("sf_ministries", "description")},
}


def report_filters(question: str) -> dict:
    q = question.lower()
    result = {}
    for field, key in ((r"email(?:\s+(?:ids?|addresses?))?", "email_missing"),
                       (r"descriptions?", "description_missing")):
        if re.search(rf"\b(?:without|missing|no|blank|empty)\s+(?:an?\s+)?{field}\b|"
                     rf"\b{field}\s+(?:(?:is|are)\s+)?(?:missing|blank|empty)\b", q):
            result[key] = True
    if re.search(r"\b(?:inactive|not\s+active)\s+ministries\b|\bministries\s+(?:that\s+are\s+|are\s+)?inactive\b", q):
        result["ministry_active"] = False
    elif re.search(r"\bactive\s+ministries\b|\bministries\s+(?:that\s+are\s+|are\s+)?active\b", q):
        result["ministry_active"] = True
    return result


def requested_columns(question: str, entity: str) -> list[str] | None:
    q = question.lower()
    # Qualifier words are not an instruction to display that column.
    if report_filters(q) and not re.search(r"\b(?:only|columns?|fields?)\b", q):
        return None
    fields = []
    patterns = {
        "donor_name": r"\b(?:donor|giver|contact)\s+names?\b",
        "email": r"\bemail(?:\s+(?:ids?|addresses?))?\b",
        "phone": r"\b(?:phone|telephone)(?:\s+numbers?)?\b",
        "donor_id": r"\bdonor\s+ids?\b",
        "address": r"\b(?:mailing\s+|postal\s+)?addresses?\b",
        "payment_amount": r"\b(?:payment|donation)\s+amounts?\b",
        "payment_date": r"\b(?:payment|donation)\s+dates?\b",
        "payment_status": r"\b(?:payment|donation)\s+status\b",
        "payment_method": r"\bpayment\s+methods?\b",
        "ministry_name": r"\bministry\s+names?\b",
        "gift_code": r"\b(?:gift|ministr(?:y|ies))\s+codes?\b",
        "description": r"\bdescriptions?\b",
    }
    matches = [(m.start(), field) for field, pattern in patterns.items()
               if (m := re.search(pattern, re.sub(r"\bemail\s+addresses?\b", "email", q) if field == "address" else q))]
    if re.search(r"\bnames?\s+only\b|\bonly\s+names?\b", q):
        matches.append((0, "ministry_name" if entity == "ministry" else "donor_name"))
    for _, field in sorted(matches):
        if field not in fields:
            fields.append(field)
    # A profile or 'details' request without 'only' keeps its usual columns.
    if re.search(r"\b(?:profile|details?)\b", q) and not re.search(r"\b(?:only|columns?|fields?)\b", q):
        return None
    return fields or None


def filter_predicates(filters: dict, aliases: dict[str, str]) -> list[str]:
    """Return canonical conditions; missing tables or unknown options fail closed."""
    predicates = []
    for field, value in filters.items():
        table, column = {
            "email_missing": ("sf_contacts", "email"),
            "description_missing": ("sf_ministries", "description"),
            "ministry_active": ("sf_ministries", "isactive"),
        }.get(field, (None, None))
        if table not in aliases or not isinstance(value, bool):
            raise ValueError("Unsupported condition or missing report table")
        ref = f"{aliases[table]}.{column}"
        if field == "ministry_active":
            predicates.append(f"{ref} = {int(value)}")
        elif value:
            predicates.append(f"({ref} IS NULL OR TRIM({ref}) = '')")
        else:
            raise ValueError("Unsupported missing-value condition")
    return predicates


def display_rows(rows: list[dict], spec: dict) -> list[dict]:
    """Keep internal SQL identity separate from the user-visible columns."""
    entity = spec.get("result_entity")
    explicit = spec.get("display_columns")
    if spec.get("group_by") == "month":
        alias = {"total": "total_amount", "average": "average_amount", "count": "record_count"}.get(spec.get("metric"))
        if not alias:
            raise ValueError("Monthly report has no supported measure")
        projected = []
        for row in rows:
            values = {key.lower(): value for key, value in row.items()}
            if "month" not in values or alias not in values:
                raise ValueError("Monthly report did not return its month and measure")
            month = values["month"]
            if month is not None and not re.fullmatch(r"\d{4}-(?:0[1-9]|1[0-2])", str(month)):
                raise ValueError("Monthly report returned an invalid calendar month")
            out = {"month": month if month is not None else "Date not recorded", alias: values[alias]}
            if explicit:
                allowed = {"month": "month", "payment_amount": alias}
                if set(explicit) - allowed.keys() or (alias == "record_count" and "payment_amount" in explicit):
                    raise ValueError("Unsupported monthly display field")
                out = {allowed[field]: out[allowed[field]] for field in explicit}
            projected.append((month, out))
        # Calendar months sort across years. Undated records stay separate;
        # their absence is never used to label a payment one-time.
        return [out for _, out in sorted(projected, key=lambda item: (item[0] is None, str(item[0] or "")))]
    if not explicit and (entity == "ministry" or spec.get("result_shape") != "detail"):
        return rows  # preserve existing ministry disambiguation/grouped rules
    aliases = {
        "donor_name": ("donor_name", "contact_name", "name"),
        "email": ("email", "email_address"), "phone": ("phone", "phone_number"),
        "donor_id": ("donorid", "donor_number", "donor_id"),
        "payment_date": ("paymentdate", "payment_date"),
        "payment_amount": ("paymentamount", "payment_amount", "amount"),
        "payment_status": ("paid", "payment_status"),
        "payment_method": ("payment_method",),
        "ministry_name": ("ministry_name", "ministry"),
        "gift_code": ("giftcode", "gift_code", "ministry_code"),
        "description": ("description", "ministry_description"),
        "donation_count": ("donation_count",), "ministry_count": ("ministry_count",),
    }
    defaults = (["donor_name", "donor_id", "email", "phone", "address", "donation_count", "ministry_count"] if entity == "donor" else
                ["payment_date", "payment_amount", "donor_name", "ministry_name", "gift_code", "payment_status", "payment_method"])
    fields = explicit or defaults
    projected = []
    for row in rows:
        values = {re.sub(r"[^a-z0-9]", "", key.lower()): value for key, value in row.items()}
        def read(*names):
            for name in names:
                key = re.sub(r"[^a-z0-9]", "", name)
                if key in values:
                    return True, values[key]
            return False, None
        out = {}
        for field in fields:
            found, value = read(*aliases.get(field, ()))
            if field == "donor_name" and not found:
                first, a = read("firstname", "first_name")
                last, b = read("lastname", "last_name")
                found, value = first and last, " ".join(str(part) for part in (a, b) if part)
            if field == "address":
                parts = [read(name) for name in ("addressstreet1", "addresscity", "addressstate", "addresspostalcode", "addresscountry")]
                found = any(present for present, _ in parts)
                value = ", ".join(str(part) for _, part in parts if part)
            if found:
                out[field] = value
            elif explicit:
                raise ValueError("A requested display field was not returned: " + field)
        # Unknown SQL aliases are not a licence to display arbitrary fields.
        if not out:
            raise ValueError("The query did not return recognizable report columns")
        projected.append(out)
    return projected
