"""
dynamic/schema_renderer.py
───────────────────────────
Renders the dynamically fetched schema into the same prompt block format
that get_schema_prompt_block() produces in context_enrichment.py.

This means prompts.py works identically whether schema comes from:
  - context_enrichment.py  (USE_DYNAMIC_SCHEMA=false)
  - dynamic fetch     (USE_DYNAMIC_SCHEMA=true)
"""

import logging

logger = logging.getLogger(__name__)


def render_schema_prompt_block(
    allowed_tables: dict,
    relationships: list,
) -> str:
    """
    Converts the dynamically fetched schema into a prompt-ready text block.
    Output format is identical to context_enrichment.get_schema_prompt_block().
    """
    lines = ["### Allowed Tables & Schema\n"]

    for table, meta in allowed_tables.items():
        lines.append(f"**{table}** — {meta['description']}")
        for col, col_meta in meta["columns"].items():
            pk_flag = " [PK]" if col_meta.get("pk") else ""
            lines.append(
                f"  - {col} ({col_meta['type']}){pk_flag}: {col_meta['description']}"
            )
        lines.append("")

    lines.append("### Relationships (always use these for JOINs)")
    if relationships:
        for rel in relationships:
            lines.append(
                f"  - {rel['from_table']}.{rel['from_column']} → "
                f"{rel['to_table']}.{rel['to_column']}  "
                f"[{rel['join_type']}]  # {rel['description']}"
            )
    else:
        lines.append("  - No foreign key relationships detected in DB.")
        lines.append("  - Use common sense column matching for JOINs.")

    return "\n".join(lines)


def render_intent_keywords(allowed_tables: dict) -> dict[str, list[str]]:
    """
    Auto-generates intent keywords from column names and table names.
    Used to update domain_guard dynamically when USE_DYNAMIC_SCHEMA=true.

    Returns a dict of { intent_name: [keywords] } built from actual column names.
    """
    keywords: dict[str, list[str]] = {}

    for table, meta in allowed_tables.items():
        # Use table name as an intent group
        intent_name = table.lower().replace("sf_", "").replace("s", "", 1) \
            if table.startswith("sf_") else table.lower()

        table_keywords = [table.lower()]

        # Add all column names as keywords
        for col in meta["columns"]:
            col_clean = col.lower().replace("_", " ")
            table_keywords.append(col.lower())
            if " " in col_clean:
                table_keywords.append(col_clean)

        keywords[intent_name] = list(set(table_keywords))

    # Always include aggregate keywords
    keywords["aggregate"] = [
        "total", "sum", "count", "how many", "how much", "average", "avg",
        "max", "min", "top", "bottom", "highest", "lowest", "most", "least",
        "list", "show", "get", "find", "fetch", "all", "recent", "latest",
        "this month", "this year", "last month", "last year", "report",
        "summary", "overview", "breakdown", "rank", "ranking",
    ]

    logger.info(f"Generated intent keywords for {len(keywords)} intents from dynamic schema")
    return keywords