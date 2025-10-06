# backend/app/db/deps.py
from __future__ import annotations
from app.db.session import DatabasePool
import logging

logger = logging.getLogger("rag.db")

def get_db():
    """FastAPI dependency that yields a psycopg connection from the pool."""
    if not DatabasePool.pool:
        logger.warning("DatabasePool not initialized; initializing automatically.")
        DatabasePool.init()

    # psycopg_pool gives a real psycopg.Connection via context manager
    with DatabasePool.pool.connection() as conn:
        yield conn
