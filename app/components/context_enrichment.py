"""
context_enrichment.py — Single source of truth for all allowed tables, columns, and relationships.
Changing this file is all that's needed to update the LLM's schema knowledge.
The LLM will NEVER see or use tables not defined here.
"""

ALLOWED_TABLES: dict[str, dict] = {

    "sf_contacts": {
        "description": "Stores individual donor/contact records. Primary contact identity table.",
        "columns": {
            "id":                {"type": "INT",           "pk": True,  "description": "Primary key"},
            "sfid":              {"type": "VARCHAR(255)",  "pk": False, "description": "Salesforce ID — used to join with sf_opportunities.primarycontact"},
            "firstname":         {"type": "VARCHAR(100)",  "pk": False, "description": "Contact first name"},
            "lastname":          {"type": "VARCHAR(100)",  "pk": False, "description": "Contact last name"},
            "email":             {"type": "VARCHAR(255)",  "pk": False, "description": "Contact email address"},
            "phone":             {"type": "VARCHAR(50)",   "pk": False, "description": "Contact phone number"},
            "gender":            {"type": "VARCHAR(20)",   "pk": False, "description": "Gender of the contact"},
            "birthdate":         {"type": "DATE",          "pk": False, "description": "Date of birth"},
            "salutationname":    {"type": "VARCHAR(50)",   "pk": False, "description": "Salutation e.g. Mr, Mrs, Dr"},
            "title":             {"type": "VARCHAR(100)",  "pk": False, "description": "Job title or role"},
            "mailname":          {"type": "VARCHAR(255)",  "pk": False, "description": "Mailing name"},
            "donortype":         {"type": "VARCHAR(100)",  "pk": False, "description": "Type of donor"},
            "donorid":           {"type": "VARCHAR(100)",  "pk": False, "description": "Donor identifier"},
            "addressstreet1":    {"type": "VARCHAR(255)",  "pk": False, "description": "Street address line 1"},
            "addressstreet2":    {"type": "VARCHAR(255)",  "pk": False, "description": "Street address line 2"},
            "addresscity":       {"type": "VARCHAR(100)",  "pk": False, "description": "City"},
            "addressstate":      {"type": "VARCHAR(100)",  "pk": False, "description": "State or province"},
            "addresspostalcode": {"type": "VARCHAR(20)",   "pk": False, "description": "Postal / ZIP code"},
            "addresscountry":    {"type": "VARCHAR(100)",  "pk": False, "description": "Country"},
            "addressid":         {"type": "VARCHAR(100)",  "pk": False, "description": "Address identifier"},
            "totaloppamount":    {"type": "DECIMAL(15,2)", "pk": False, "description": "Total opportunity amount across all donations"},
            "largestamount":     {"type": "DECIMAL(15,2)", "pk": False, "description": "Largest single donation amount"},
            "lastoppamount":     {"type": "DECIMAL(15,2)", "pk": False, "description": "Most recent opportunity amount"},
            "oppamountthisyear": {"type": "DECIMAL(15,2)", "pk": False, "description": "Total opportunity amount in current year"},
            "stripecustomerid":  {"type": "VARCHAR(255)",  "pk": False, "description": "Stripe customer ID for payments"},
            "donotcall":         {"type": "TINYINT(1)",    "pk": False, "description": "1 = do not call this contact"},
            "donotcontact":      {"type": "TINYINT(1)",    "pk": False, "description": "1 = do not contact this contact"},
            "donotmail":         {"type": "TINYINT(1)",    "pk": False, "description": "1 = do not mail this contact"},
            "optedoutofemail":   {"type": "TINYINT(1)",    "pk": False, "description": "1 = contact opted out of emails"},
            "synctosf":          {"type": "TINYINT(1)",    "pk": False, "description": "1 = sync this record to Salesforce"},
            "created_at":        {"type": "DATETIME",      "pk": False, "description": "Record creation timestamp"},
            "updated_at":        {"type": "DATETIME",      "pk": False, "description": "Last update timestamp"},
            "created_by":        {"type": "VARCHAR(100)",  "pk": False, "description": "User who created the record"},
            "updated_by":        {"type": "VARCHAR(100)",  "pk": False, "description": "User who last updated the record"},
        },
    },

    "sf_ministries": {
        "description": "Represents ministry / organizational units that receive donations.",
        "columns": {
            "id":               {"type": "INT",          "pk": True,  "description": "Primary key"},
            "sfid":             {"type": "VARCHAR(255)", "pk": False, "description": "Salesforce ID — used to join with sfpayments.ministryid"},
            "name":             {"type": "VARCHAR(255)", "pk": False, "description": "Full ministry name — use for display only, NOT for filtering by ministry code"},
            "description":      {"type": "TEXT",         "pk": False, "description": "Ministry description"},
            "giftcode":         {"type": "VARCHAR(100)", "pk": False, "description": "Ministry code used to identify the ministry (e.g. 095WMN, 801BOB, 470SFA) — ALWAYS use this column when filtering by a ministry code"},
            "isActive":         {"type": "TINYINT(1)",   "pk": False, "description": "1 = ministry is active"},
            "qbo_class_id":     {"type": "VARCHAR(100)", "pk": False, "description": "QuickBooks Online class ID"},
            "stripe_product":   {"type": "VARCHAR(255)", "pk": False, "description": "Stripe product ID linked to ministry"},
            "stripe_prices":    {"type": "TEXT",         "pk": False, "description": "Stripe price IDs for this ministry"},
            "donation_html":    {"type": "TEXT",         "pk": False, "description": "HTML content for donation page"},
            "created_at":       {"type": "DATETIME",     "pk": False, "description": "Record creation timestamp"},
            "updated_at":       {"type": "DATETIME",     "pk": False, "description": "Last update timestamp"},
            "created_by":       {"type": "VARCHAR(100)", "pk": False, "description": "User who created the record"},
            "updated_by":       {"type": "VARCHAR(100)", "pk": False, "description": "User who last updated the record"},
        },
    },

    "sf_opportunities": {
        "description": "Tracks fundraising opportunities / donation pledges linked to contacts.",
        "columns": {
            "id":                 {"type": "INT",           "pk": True,  "description": "Primary key"},
            "sfid":               {"type": "VARCHAR(255)",  "pk": False, "description": "Salesforce ID — used to join with sfpayments.oppsfid"},
            "name":               {"type": "VARCHAR(255)",  "pk": False, "description": "Opportunity name / title"},
            "primarycontact":     {"type": "VARCHAR(255)",  "pk": False, "description": "FK → sf_contacts.sfid — the primary donor contact"},
            "accountid":          {"type": "VARCHAR(255)",  "pk": False, "description": "Salesforce account ID"},
            "webcontactid":       {"type": "VARCHAR(255)",  "pk": False, "description": "Web contact ID"},
            "amount":             {"type": "DECIMAL(15,2)", "pk": False, "description": "Total pledged/expected donation amount"},
            "amountoutstanding":  {"type": "DECIMAL(15,2)", "pk": False, "description": "Remaining unpaid amount"},
            "paymentsmade":       {"type": "DECIMAL(15,2)", "pk": False, "description": "Total payments already received"},
            "stagename":          {"type": "VARCHAR(100)",  "pk": False, "description": "Pipeline stage e.g. Pledged, Closed Won, Posted"},
            "type":               {"type": "VARCHAR(100)",  "pk": False, "description": "Opportunity type e.g. One-Time, Recurring"},
            "closedate":          {"type": "DATE",          "pk": False, "description": "Expected or actual close date"},
            "recurringdonation":  {"type": "TINYINT(1)",    "pk": False, "description": "1 = this is a recurring donation"},
            "matchinggift":       {"type": "TINYINT(1)",    "pk": False, "description": "1 = employer matching gift"},
            "receiptnumber":      {"type": "VARCHAR(100)",  "pk": False, "description": "Receipt number for this opportunity"},
            "qboid":              {"type": "VARCHAR(100)",  "pk": False, "description": "QuickBooks Online ID"},
            "isqbo":              {"type": "TINYINT(1)",    "pk": False, "description": "1 = synced to QuickBooks Online"},
            "admin_fee":          {"type": "DECIMAL(15,2)", "pk": False, "description": "Administrative fee applied"},
            "bank_accounts":      {"type": "VARCHAR(255)",  "pk": False, "description": "Bank account reference"},
            "created_at":         {"type": "DATETIME",      "pk": False, "description": "Record creation timestamp"},
            "updated_at":         {"type": "DATETIME",      "pk": False, "description": "Last update timestamp"},
            "created_by":         {"type": "VARCHAR(100)",  "pk": False, "description": "User who created the record"},
            "updated_by":         {"type": "VARCHAR(100)",  "pk": False, "description": "User who last updated the record"},
        },
    },

    "sfpayments": {
        "description": "Records actual payments received. Links opportunities to ministries.",
        "columns": {
            "id":                  {"type": "INT",           "pk": True,  "description": "Primary key"},
            "sfid":                {"type": "VARCHAR(255)",  "pk": False, "description": "Salesforce ID of this payment"},
            "oppsfid":             {"type": "VARCHAR(255)",  "pk": False, "description": "FK → sf_opportunities.sfid — the opportunity this payment belongs to"},
            "ministryid":          {"type": "VARCHAR(255)",  "pk": False, "description": "FK → sf_ministries.sfid — the ministry receiving payment"},
            "webopportunityid":    {"type": "VARCHAR(255)",  "pk": False, "description": "Web opportunity reference ID"},
            "paymentamount":       {"type": "DECIMAL(15,2)", "pk": False, "description": "Actual amount paid"},
            "paymentdate":         {"type": "DATE",          "pk": False, "description": "Date payment was received"},
            "payment_method":      {"type": "VARCHAR(100)",  "pk": False, "description": "Payment method e.g. Card, Check, Cash, Transfer"},
            "cardlast4":           {"type": "VARCHAR(4)",    "pk": False, "description": "Last 4 digits of card used"},
            "paid":                {"type": "TINYINT(1)",    "pk": False, "description": "1 = payment is confirmed paid"},
            "special_designation": {"type": "VARCHAR(255)",  "pk": False, "description": "Special fund or designation note"},
            "created_at":          {"type": "DATETIME",      "pk": False, "description": "Record creation timestamp"},
            "updated_at":          {"type": "DATETIME",      "pk": False, "description": "Last update timestamp"},
            "created_by":          {"type": "VARCHAR(100)",  "pk": False, "description": "User who created the record"},
            "updated_by":          {"type": "VARCHAR(100)",  "pk": False, "description": "User who last updated the record"},
        },
    },
}

RELATIONSHIPS: list[dict] = [
    {
        "from_table":  "sf_contacts",
        "from_column": "sfid",
        "to_table":    "sf_opportunities",
        "to_column":   "primarycontact",
        "join_type":   "LEFT JOIN",
        "description": "A contact is the primary donor on an opportunity",
    },
    {
        "from_table":  "sf_opportunities",
        "from_column": "sfid",
        "to_table":    "sfpayments",
        "to_column":   "oppsfid",
        "join_type":   "LEFT JOIN",
        "description": "An opportunity has one or more payments made against it",
    },
    {
        "from_table":  "sfpayments",
        "from_column": "ministryid",
        "to_table":    "sf_ministries",
        "to_column":   "sfid",
        "join_type":   "LEFT JOIN",
        "description": "A payment is designated to a specific ministry",
    },
]

def get_allowed_table_names() -> list[str]:
    return list(ALLOWED_TABLES.keys())


def get_schema_prompt_block() -> str:
    """Renders the schema as a clean text block for injection into LLM prompts."""
    lines = ["### Allowed Tables & Schema\n"]
    for table, meta in ALLOWED_TABLES.items():
        lines.append(f"**{table}** — {meta['description']}")
        for col, col_meta in meta["columns"].items():
            pk_flag = " [PK]" if col_meta["pk"] else ""
            lines.append(f"  - {col} ({col_meta['type']}){pk_flag}: {col_meta['description']}")
        lines.append("")

    lines.append("### Relationships (always use these for JOINs)")
    for rel in RELATIONSHIPS:
        lines.append(
            f"  - {rel['from_table']}.{rel['from_column']} → "
            f"{rel['to_table']}.{rel['to_column']}  [{rel['join_type']}]  # {rel['description']}"
        )
    return "\n".join(lines)