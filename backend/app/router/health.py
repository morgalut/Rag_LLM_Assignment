# backend/app/router/health.py
from __future__ import annotations

import logging
import psycopg
import sys
from fastapi import APIRouter, Depends, HTTPException

from app.db.deps import get_db
from app.db.session import DatabasePool, ping_db
from app.models.schemas import HealthResponse

# -----------------------------------------------------------
# Router setup
# -----------------------------------------------------------
router = APIRouter(prefix="/health", tags=["health"])

# Unified logging (avoid using fastapi.logger)
logger = logging.getLogger("app.health")


# -----------------------------------------------------------
# Primary health endpoint
# -----------------------------------------------------------
@router.get("", response_model=HealthResponse)
def health(conn: psycopg.Connection = Depends(get_db)):
    """
    ✅ Health check endpoint
    ------------------------
    Performs:
      • Database connectivity check
      • pgvector extension verification
      • Papers table availability + record count

    Returns:
      - "ok"          → all systems good
      - "recovering"  → database restarting (AdminShutdown)
      - raises 500    → other failures
    """
    try:
        with conn.cursor() as cur:
            # Ensure vector extension exists
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
            vector_ok = bool(cur.fetchone())

            # Ensure table and row count exist
            cur.execute("SELECT COUNT(*) FROM papers;")
            count_papers = cur.fetchone()[0]

        return HealthResponse(
            status="ok",
            vector_extension=vector_ok,
            tables_ok=True,
            count_papers=count_papers,
        )

    except psycopg.errors.AdminShutdown:
        logger.warning("⚠️ Postgres temporarily unavailable (AdminShutdown). Retrying soon.")
        return HealthResponse(
            status="recovering",
            vector_extension=False,
            tables_ok=False,
            count_papers=None,
        )

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


# -----------------------------------------------------------
# Additional debug endpoints
# -----------------------------------------------------------
@router.get("/db-ping")
def db_ping():
    """Lightweight database connection test."""
    ok, message = ping_db()
    return {"ok": ok, "message": message}


@router.get("/debug/pool-status")
def pool_status():
    """
    Return whether the DatabasePool has been initialized.
    Checks the actual pool reference used by psycopg_pool.
    """
    return {
        "initialized": DatabasePool.pool is not None,
        "pool_class": str(type(DatabasePool.pool)) if DatabasePool.pool else None,
    }



@router.get("/debug/import-check")
def import_check():
    """List imported modules containing 'session' (diagnostic helper)."""
    modules = [m for m in sys.modules.keys() if "session" in m]
    return {"session_modules": modules}
