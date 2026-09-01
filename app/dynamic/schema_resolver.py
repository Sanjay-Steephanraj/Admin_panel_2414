"""
dynamic/schema_resolver.py
───────────────────────────
The single entry point the rest of the system uses to get schema data.

Acts as a smart router:
  - USE_DYNAMIC_SCHEMA=false → reads from config/context_enrichment.py (static, fast)
  - USE_DYNAMIC_SCHEMA=true  → fetches from MySQL at startup, caches in memory

This means the rest of the system (prompts.py, domain_guard.py) NEVER needs
to know which mode is active — they always call get_active_schema().

Boot sequence when dynamic mode is ON:
  1. App starts
  2. schema_resolver.initialize() is called from lifespan in main.py
  3. Fetches columns + FK relations from MySQL
  4. Caches result in memory for the entire session
  5. Exposes same interface as context_enrichment.py

To refresh the schema (e.g. after DB changes): call schema_resolver.refresh()
or hit DELETE /schema-cache endpoint.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

#  Internal cache 
_cached_tables:        Optional[dict]  = None
_cached_relationships: Optional[list]  = None
_cached_prompt_block:  Optional[str]   = None
_dynamic_mode_active:  bool            = False


def initialize() -> bool:
    """
    Called once at app startup from main.py lifespan.
    Checks USE_DYNAMIC_SCHEMA flag and fetches schema if enabled.

    Returns True if dynamic mode initialized successfully, False otherwise.
    """
    global _dynamic_mode_active

    from config.settings import get_settings
    settings = get_settings()

    if not settings.use_dynamic_schema:
        logger.info(" Schema mode: STATIC (context_enrichment.py)")
        _dynamic_mode_active = False
        return False

    # Dynamic mode — fetch from DB
    allowed_tables = settings.get_dynamic_table_list()

    if not allowed_tables:
        logger.warning(
            "USE_DYNAMIC_SCHEMA=true but DYNAMIC_ALLOWED_TABLES is empty. "
            "Falling back to static context_enrichment.py."
        )
        _dynamic_mode_active = False
        return False

    logger.info(f" Schema mode: DYNAMIC | tables={allowed_tables}")
    return refresh(allowed_tables=allowed_tables, database=settings.mysql_database)


def refresh(
    allowed_tables: Optional[list[str]] = None,
    database: Optional[str] = None,
) -> bool:
    """
    Fetches fresh schema from MySQL and updates the in-memory cache.
    Can be called at runtime to pick up DB schema changes.
    """
    global _cached_tables, _cached_relationships, _cached_prompt_block, _dynamic_mode_active

    from config.settings import get_settings
    settings = get_settings()

    allowed_tables = allowed_tables or settings.get_dynamic_table_list()
    database       = database or settings.mysql_database

    try:
        from dynamic.schema_fetcher import build_dynamic_schema
        from dynamic.schema_render import render_schema_prompt_block

        tables, relationships = build_dynamic_schema(allowed_tables, database)

        if not tables:
            logger.error("Dynamic schema fetch returned empty tables — falling back to static")
            _dynamic_mode_active = False
            return False

        _cached_tables        = tables
        _cached_relationships = relationships
        _cached_prompt_block  = render_schema_prompt_block(tables, relationships)
        _dynamic_mode_active  = True

        logger.info(
            f" Dynamic schema loaded: {len(tables)} tables, "
            f"{len(relationships)} relationships"
        )

        # Log what was fetched for visibility
        for table, meta in tables.items():
            col_count = len(meta["columns"])
            logger.info(f"   └─ {table}: {col_count} columns")

        return True

    except Exception as e:
        logger.error(f"Dynamic schema initialization failed: {e}", exc_info=True)
        _dynamic_mode_active = False
        return False


def get_active_schema_prompt_block() -> str:
    """
    Returns the schema prompt block string.
    Routes to dynamic cache or static context_enrichment based on mode.
    """
    if _dynamic_mode_active and _cached_prompt_block:
        return _cached_prompt_block

    # Fallback to static
    from components.context_enrichment import get_schema_prompt_block
    return get_schema_prompt_block()


def get_active_allowed_tables() -> dict:
    """
    Returns the ALLOWED_TABLES dict.
    Routes to dynamic cache or static context_enrichment based on mode.
    """
    if _dynamic_mode_active and _cached_tables:
        return _cached_tables

    from components.context_enrichment import ALLOWED_TABLES
    return ALLOWED_TABLES


def get_active_table_names() -> list[str]:
    """
    Returns list of allowed table names.
    Used by domain_guard for SQL validation.
    """
    if _dynamic_mode_active and _cached_tables:
        return list(_cached_tables.keys())

    from components.context_enrichment import get_allowed_table_names
    return get_allowed_table_names()


def get_active_relationships() -> list:
    """
    Returns the relationships list.
    """
    if _dynamic_mode_active and _cached_relationships is not None:
        return _cached_relationships

    from components.context_enrichment import RELATIONSHIPS
    return RELATIONSHIPS


def get_active_intent_keywords() -> dict[str, list[str]]:
    """
    Returns intent keywords dict for domain_guard.
    In dynamic mode: auto-generated from column names.
    In static mode: returns the hardcoded INTENT_KEYWORDS from domain_guard.
    """
    if _dynamic_mode_active and _cached_tables:
        from dynamic.schema_render import render_intent_keywords
        return render_intent_keywords(_cached_tables)

    from security.domain_guard import INTENT_KEYWORDS
    return INTENT_KEYWORDS


def is_dynamic_active() -> bool:
    return _dynamic_mode_active


def get_schema_status() -> dict:
    """Returns current schema mode status — used in /health endpoint."""
    if _dynamic_mode_active and _cached_tables:
        return {
            "mode":         "dynamic",
            "tables":       list(_cached_tables.keys()),
            "table_count":  len(_cached_tables),
            "rel_count":    len(_cached_relationships or []),
        }
    return {
        "mode":        "static",
        "source":      "components/context_enrichment.py",
        "table_count": len(__import__("components.context_enrichment", fromlist=["ALLOWED_TABLES"]).ALLOWED_TABLES),
    }