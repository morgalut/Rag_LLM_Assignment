# backend/app/models/ingest_model.py
from pydantic import BaseModel
from typing import Optional

class IngestRequest(BaseModel):
    path: str
    batch_size: int = 256
    embedding_mode: str = "hash"
    corpus_name: Optional[str] = None
    update_fingerprint: bool = True


class IngestResult(BaseModel):
    file: str
    corpus_name: str
    fingerprint: Optional[str] = None

    rows_processed: int
    rows_inserted: int
    rows_updated: int
    rows_skipped: int
    duplicates_found: int
    duration_sec: float
