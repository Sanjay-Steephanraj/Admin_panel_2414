"""
dynamic/schema_fetcher.py
──────────────────────────
Fetches real table schema (columns + FK relationships) directly from MySQL
information_schema. Builds the same data structure as ment.py so
the rest of the system works identically regardless of which mode is active.

Only runs when USE_DYNAMIC_SCHEMA=true in .env.
"""

import logging
from tools.db_tool import get_db_connection

logger = logging.getLogger(__name__)



def fetch_columns_for_table(table_name: str, database: str) -> dict[str, dict]:
    """
    Queries information_schema.COLUMNS for a given table.
    Returns a dict matching the ALLOWED_TABLES column format in context_enrichment.py.

    Example return:
    {
        "id":         {"type": "int",          "pk": True,  "description": "Primary key"},
        "firstname":  {"type": "varchar(100)", "pk": False, "description": "firstname column"},
        ...
    }
    """
    query = """
        SELECT
            COLUMN_NAME,
            COLUMN_TYPE,
            COLUMN_KEY,
            IS_NULLABLE,
            COLUMN_COMMENT
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s
          AND TABLE_NAME   = %s
        ORDER BY ORDINAL_POSITION
    """

    columns = {}
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, (database, table_name))
            rows = cursor.fetchall()
            cursor.close()

        for row in rows:
            col_name    = row["COLUMN_NAME"]
            col_type    = row["COLUMN_TYPE"]
            is_pk       = row["COLUMN_KEY"] == "PRI"
            comment     = row.get("COLUMN_COMMENT", "").strip()
            description = comment if comment else f"{col_name} column"

            columns[col_name] = {
                "type":        col_type,
                "pk":          is_pk,
                "description": description,
            }

        logger.info(f"Fetched {len(columns)} columns for table: {table_name}")

    except Exception as e:
        logger.error(f"Failed to fetch columns for {table_name}: {e}")

    return columns


def fetch_foreign_keys(allowed_tables: list[str], database: str) -> list[dict]:
    """
    Queries information_schema.KEY_COLUMN_USAGE for FK relationships
    between the allowed tables only.

    Returns a list matching the RELATIONSHIPS format in context_enrichment.py.

    Example return:
    [
        {
            "from_table":  "sfpayments",
            "from_column": "ministryid",
            "to_table":    "sf_ministries",
            "to_column":   "sfid",
            "join_type":   "LEFT JOIN",
            "description": "sfpayments.ministryid → sf_ministries.sfid"
        },
        ...
    ]
    """
    if not allowed_tables:
        return []

    # Build placeholders for IN clause
    placeholders = ", ".join(["%s"] * len(allowed_tables))

    query = f"""
        SELECT
            kcu.TABLE_NAME        AS from_table,
            kcu.COLUMN_NAME       AS from_column,
            kcu.REFERENCED_TABLE_NAME  AS to_table,
            kcu.REFERENCED_COLUMN_NAME AS to_column
        FROM information_schema.KEY_COLUMN_USAGE kcu
        JOIN information_schema.TABLE_CONSTRAINTS tc
          ON  kcu.CONSTRAINT_NAME   = tc.CONSTRAINT_NAME
          AND kcu.TABLE_SCHEMA      = tc.TABLE_SCHEMA
          AND kcu.TABLE_NAME        = tc.TABLE_NAME
        WHERE kcu.TABLE_SCHEMA             = %s
          AND tc.CONSTRAINT_TYPE           = 'FOREIGN KEY'
          AND kcu.TABLE_NAME               IN ({placeholders})
          AND kcu.REFERENCED_TABLE_NAME    IN ({placeholders})
    """

    params = [database] + allowed_tables + allowed_tables
    relationships = []

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            cursor.close()

        for row in rows:
            relationships.append({
                "from_table":  row["from_table"],
                "from_column": row["from_column"],
                "to_table":    row["to_table"],
                "to_column":   row["to_column"],
                "join_type":   "LEFT JOIN",
                "description": (
                    f"{row['from_table']}.{row['from_column']} → "
                    f"{row['to_table']}.{row['to_column']}"
                ),
            })

        logger.info(f"Fetched {len(relationships)} FK relationships from DB")

    except Exception as e:
        logger.error(f"Failed to fetch FK relationships: {e}")

    return relationships


def fetch_table_description(table_name: str, database: str) -> str:
    """
    Tries to get TABLE_COMMENT from information_schema.TABLES.
    Falls back to a generic description if no comment is set.
    """
    query = """
        SELECT TABLE_COMMENT
        FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = %s
          AND TABLE_NAME   = %s
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, (database, table_name))
            row = cursor.fetchone()
            cursor.close()

        comment = row.get("TABLE_COMMENT", "").strip() if row else ""
        return comment if comment else f"{table_name} table"

    except Exception as e:
        logger.error(f"Failed to fetch description for {table_name}: {e}")
        return f"{table_name} table"


def build_dynamic_schema(
    allowed_tables: list[str],
    database: str,
) -> tuple[dict, list]:
    """
    Builds the full schema dict + relationships list by querying MySQL directly.

    Returns:
        (ALLOWED_TABLES dict, RELATIONSHIPS list)
        — same structure as context_enrichment.py

    Usage:
        tables, rels = build_dynamic_schema(["sf_contacts", "sfpayments"], "mydb")
    """
    logger.info(f"Building dynamic schema for tables: {allowed_tables}")

    allowed_tables_dict = {}

    for table in allowed_tables:
        description = fetch_table_description(table, database)
        columns     = fetch_columns_for_table(table, database)

        if not columns:
            logger.warning(f"Table '{table}' has no columns — skipping")
            continue

        allowed_tables_dict[table] = {
            "description": description,
            "columns":     columns,
        }

    relationships = fetch_foreign_keys(allowed_tables, database)

    logger.info(
        f"Dynamic schema built: {len(allowed_tables_dict)} tables, "
        f"{len(relationships)} relationships"
    )

    return allowed_tables_dict, relationships