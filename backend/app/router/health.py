from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends
from app.models.schemas import HealthResponse  # keep as in your tree
from app.db.deps import get_db

router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthResponse)
def health(conn: psycopg.Connection = Depends(get_db)):
    vector_ok = False
    tables_ok = False
    count = None

    with conn.cursor() as cur:
        # Check pgvector extension
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
        vector_ok = cur.fetchone() is not None

        # Check required tables
        cur.execute("""
            SELECT to_regclass('public.papers') IS NOT NULL,
                   to_regclass('public.corpus_state') IS NOT NULL;
        """)
        row = cur.fetchone()
        tables_ok = bool(row and all(row))

        # Optional: count rows for visibility
        if tables_ok:
            cur.execute("SELECT COUNT(*) FROM public.papers;")
            count = cur.fetchone()[0]

    return HealthResponse(
        status="ok" if (vector_ok and tables_ok) else "degraded",
        vector_extension=vector_ok,
        tables_ok=tables_ok,
        count_papers=count,
    )


@router.get("/db/ping")
def db_ping(conn: psycopg.Connection = Depends(get_db)):
    """
    Simple connectivity check that also prints/logs the result.
    Using get_db ensures the pool is initialized before pinging.
    """
    from app.db.session import ping_db  # local import to avoid cycles

    ok, message = ping_db()
    return {"ok": ok, "message": message}
