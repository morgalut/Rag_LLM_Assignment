from __future__ import annotations
import psycopg
import sys
from fastapi import APIRouter, Depends
from app.db.deps import get_db
from app.db.session import DatabasePool, ping_db
from app.models.schemas import HealthResponse  # keep or remove depending on your project

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(conn: psycopg.Connection = Depends(get_db)):
    vector_ok = False
    tables_ok = False
    count = None

    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
        vector_ok = cur.fetchone() is not None

        cur.execute("""
            SELECT to_regclass('public.papers') IS NOT NULL,
                   to_regclass('public.corpus_state') IS NOT NULL;
        """)
        row = cur.fetchone()
        tables_ok = bool(row and all(row))

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
def db_ping():
    ok, message = ping_db()
    return {"ok": ok, "message": message}


@router.get("/debug/pool-status")
def pool_status():
    return {"initialized": DatabasePool._instance is not None}


@router.get("/debug/import-check")
def import_check():
    modules = [m for m in sys.modules.keys() if "session" in m]
    return {"session_modules": modules}
