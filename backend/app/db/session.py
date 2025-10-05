from __future__ import annotations
import logging
from psycopg_pool import ConnectionPool
from app.db.config import settings

logger = logging.getLogger("rag.db")

pool: ConnectionPool | None = None

def init_pool() -> ConnectionPool:
    global pool
    if pool is None:
        logger.info("Creating psycopg3 pool to %s:%s/%s",
                    settings.db_host, settings.db_port, settings.db_name)
        pool = ConnectionPool(
            conninfo=settings.dsn,
            min_size=1,
            max_size=10,
            kwargs={"autocommit": True},
        )
        # Smoke test
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
        logger.info("DB pool ready.")
    return pool

def close_pool() -> None:
    global pool
    if pool is not None:
        pool.close()
        pool = None
        logger.info("DB pool closed.")

def ping_db() -> tuple[bool, str]:
    """
    Try a simple SELECT 1; using the pool if available (preferred),
    or a one-off connection if the pool isn't ready yet.
    Returns (ok, message).
    """
    import psycopg

    try:
        if pool is not None:
            with pool.connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        else:
            with psycopg.connect(settings.dsn) as conn, conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()

        msg = f"Database connection successful: {settings.db_host}:{settings.db_port}/{settings.db_name}"
        logger.info(msg)
        print(msg)  # explicit print as requested
        return True, msg
    except Exception as e:
        msg = f"Database connection failed: {e}"
        logger.error(msg)
        print(msg)
        return False, msg