from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional

# ---------- Request / Response models ----------
class IngestRequest(BaseModel):
    path: Optional[str] = Field(
        default=None,
        description="Path to JSON/JSONL corpus on the server/container. Defaults to DATA_PATH."
    )
    batch_size: int = Field(default=256, ge=1, le=5000)
    embedding_mode: str = Field(
        default="hash",
        description="Embedding strategy: 'hash' (deterministic fallback) or 'zeros'."
    )
    update_fingerprint: bool = Field(default=True)
    corpus_name: Optional[str] = Field(
        default=None,
        description="Optional label for this corpus (defaults to file basename)."
    )


class IngestResult(BaseModel):
    file: str
    corpus_name: str
    fingerprint: Optional[str]
    rows_processed: int
    rows_inserted: int
    rows_updated: int
    duration_sec: float
