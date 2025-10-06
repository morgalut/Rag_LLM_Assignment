# backend/app/router/ingest_router.py
from __future__ import annotations

import sys
import traceback
import psycopg
from fastapi import APIRouter, Depends, HTTPException, status

from app.db.deps import get_db
from app.core.services.ingest_service import ingest_json_service
from app.models.ingest_model import IngestRequest, IngestResult

import logging
logger = logging.getLogger("app.ingest")

router = APIRouter(prefix="/db", tags=["ingest"])

@router.post("/ingest-json", response_model=IngestResult)
def ingest_json(request: IngestRequest, conn: psycopg.Connection = Depends(get_db)):
    try:
        result = ingest_json_service(request, conn)
        logger.info(
            "✅ Ingestion done | file=%s | processed=%d | inserted=%d | updated=%d | skipped=%d",
            result.file, result.rows_processed, result.rows_inserted, result.rows_updated, result.rows_skipped
        )
        return result

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception as e:
        exc_type, exc_obj, tb = sys.exc_info()
        tb_frame = traceback.extract_tb(tb)[-1] if tb else None
        filename = tb_frame.filename if tb_frame else "unknown"
        line_no = tb_frame.lineno if tb_frame else "?"
        func_name = tb_frame.name if tb_frame else "?"
        error_type = exc_type.__name__ if exc_type else "UnknownError"

        logger.error(
            f"Unhandled Exception [{error_type}] in {func_name}() at {filename}:{line_no}\n"
            f"Message: {e}\n"
            f"Traceback:\n{''.join(traceback.format_exception(exc_type, exc_obj, tb))}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {error_type} in {func_name}() line {line_no}",
        )
