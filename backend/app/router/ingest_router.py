from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, HTTPException, status

from app.db.deps import get_db
from app.core.services.ingest_service import ingest_json_service
from app.models.ingest_model import IngestRequest, IngestResult

router = APIRouter(prefix="/db", tags=["ingest"])

@router.post("/ingest-json", response_model=IngestResult)
def ingest_json(request: IngestRequest, conn: psycopg.Connection = Depends(get_db)):
    try:
        return ingest_json_service(request, conn)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {e}"
        )
