"""
tools/db_tool.py
──────────────────
MySQL access layer.

Fixes applied:
  1. Connection pooling via mysql.connector.pooling.MySQLConnectionPool.
     The pool is created once at module load and reused for the lifetime
     of the process — prevents exhausting MySQL's connection limit under load.
  2. execute_query / get_db_connection now draw from the pool and return
     connections on exit, so no connection leaks occur.
  3. @lru_cache on _get_connection_config() is kept (it only caches the
     config dict, which is cheap and correct).
"""
import logging
from contextlib import contextmanager
from datetime import date, datetime, time
from decimal import Decimal
from functools import lru_cache

from mysql.connector import Error as MySQLError
from mysql.connector.pooling import MySQLConnectionPool

from config.settings import get_settings

logger = logging.getLogger(__name__)


def _jsonify(value):
    """Recursively convert mysql-connector-python types to JSON-safe Python types.

    mysql-connector-python returns ``Decimal`` (DECIMAL/NUMERIC columns, e.g.
    ``paymentamount``) and ``datetime``/``date``/``time`` (DATE/DATETIME/TIME
    columns) which are NOT JSON-serializable.  The LangGraph MemorySaver
    checkpointer (REDIS_ENABLED=false) JSON-serializes the full NLSQLState at
    the end of every node, so these values must be normalized at the DB
    boundary — every downstream consumer (nodes, checkpointer, response, PII
    masker) then sees plain JSON-safe types.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        # Preserve precision as string (safer than float for money); callers can cast
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonify(v) for v in value]
    # Last resort: string repr (should not normally hit)
    return str(value)

# ── Pool config ───────────────────────────────────────────────────────────────
_POOL_SIZE = 10        # concurrent connections available to the app
_POOL_NAME = "donor_portal_pool"

# Pool singleton — created once on first import
_pool: MySQLConnectionPool | None = None


@lru_cache()
def _get_connection_config() -> dict:
    s = get_settings()
    return {
        "host":            s.mysql_host,
        "port":            s.mysql_port,
        "user":            s.mysql_user,
        "password":        s.mysql_password,
        "database":        s.mysql_database,
        "connect_timeout": 10,
    }


def _get_pool() -> MySQLConnectionPool:
    """
    Returns the global connection pool, creating it on first call.
    Thread-safe: Python's GIL ensures the assignment is atomic for
    the simple check-and-create pattern used here.
    """
    global _pool
    if _pool is None:
        config = _get_connection_config()
        logger.info(
            f"Creating MySQL connection pool | "
            f"host={config['host']} pool_size={_POOL_SIZE}"
        )
        _pool = MySQLConnectionPool(
            pool_name=_POOL_NAME,
            pool_size=_POOL_SIZE,
            pool_reset_session=True,
            **config,
        )
        logger.info("MySQL connection pool created")
    return _pool


@contextmanager
def get_db_connection():
    """
    Context manager that checks out a connection from the pool and
    returns it on exit.  Raises MySQLError on connection failure.
    """
    conn = None
    try:
        conn = _get_pool().get_connection()
        yield conn
    except MySQLError as e:
        logger.error(f"DB connection error: {e}")
        raise
    finally:
        if conn is not None:
            try:
                conn.close()   # returns connection to the pool
            except Exception:
                pass


def execute_query(sql: str) -> tuple[list[dict], str | None]:
    """
    Executes a SELECT SQL query safely.
    Returns (rows_as_dicts, error_message).
    On success: (rows, None)
    On failure: ([], error_string)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(sql)
            rows = [_jsonify(row) for row in cursor.fetchall()]
            cursor.close()
            logger.info(f"Query returned {len(rows)} rows")
            return rows, None
    except MySQLError as e:
        error_msg = str(e)
        logger.error(f"Query execution error: {error_msg}")
        return [], error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return [], error_msg


def ping_db() -> bool:
    """Health check — returns True if DB is reachable."""
    try:
        with get_db_connection() as conn:
            return conn.is_connected()
    except Exception:
        return False