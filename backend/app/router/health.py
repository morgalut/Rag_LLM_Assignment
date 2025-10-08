# backend/app/router/health.py
from __future__ import annotations
import logging
import sys
import httpx
from fastapi import APIRouter, HTTPException
import time

from app.db.session import DatabasePool, ping_db
from app.db.deps import get_db
from app.models.schemas import HealthResponse

router = APIRouter()
logger = logging.getLogger(__name__)

router = APIRouter()
startup_time = time.time()
startup_complete = False

@router.get("/health")
async def health_check():
    """Basic health check - service is running"""
    global startup_complete
    if time.time() - startup_time > 300:  # 5 minutes max startup time
        startup_complete = True
    return {"status": "ok" if startup_complete else "starting"}

@router.get("/health/ready")
async def readiness_check():
    """Readiness check - verify all services"""
    global startup_complete
    
    if not startup_complete:
        if time.time() - startup_time > 300:
            startup_complete = True
        else:
            raise HTTPException(status_code=503, detail="Service starting up")


@router.get("/db-ping")
def db_ping():
    """Simple DB connectivity test."""
    ok, message = ping_db()
    return {"ok": ok, "message": message}


@router.get("/debug/pool-status")
def pool_status():
    """Show connection pool diagnostics."""
    return {
        "initialized": DatabasePool.pool is not None,
        "pool_class": str(type(DatabasePool.pool)) if DatabasePool.pool else None,
    }


@router.get("/debug/import-check")
def import_check():
    """Debug module imports related to session handling."""
    modules = [m for m in sys.modules.keys() if "session" in m]
    return {"session_modules": modules}
