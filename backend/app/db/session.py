# backend/app/db/session.py
from __future__ import annotations
import psycopg
from psycopg_pool import ConnectionPool
from app.db.config import settings
import logging

logger = logging.getLogger("rag.db")

class DatabasePool:
    """Global psycopg3 connection pool."""
    pool: ConnectionPool | None = None

    @classmethod
    def init(cls):
        if cls.pool:
            logger.info("Database pool already initialized.")
            return

        dsn = settings.database_url
        cls.pool = ConnectionPool(
            conninfo=dsn,
            min_size=1,
            max_size=10,
            num_workers=2,
            timeout=30,
        )
        logger.info("✅ Database connection pool initialized.")

    @classmethod
    def close(cls):
        if cls.pool:
            cls.pool.close()
            cls.pool = None
            logger.info("🧹 Database pool closed.")

def ping_db() -> tuple[bool, str]:
    """Check DB connectivity."""
    try:
        if not DatabasePool.pool:
            return False, "Pool not initialized"
        with DatabasePool.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                res = cur.fetchone()
                return True, f"Database connection successful: {settings.db_host}:{settings.db_port}/{settings.db_name}"
    except Exception as e:
        return False, str(e)
