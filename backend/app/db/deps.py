from __future__ import annotations
import psycopg
from fastapi import HTTPException, status
from app.db.session import pool

def get_db() -> psycopg.Connection:
    if pool is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DB pool not initialized."
        )
    return pool.connection()
