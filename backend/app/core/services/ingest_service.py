from __future__ import annotations

import io
import json
import math
import time
import hashlib
from pathlib import Path
from typing import List, Tuple, Iterator

import psycopg
from app.db.config import settings
from app.models.ingest_model import IngestRequest, IngestResult


# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def _vector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def _hash_embedding(text: str, dim: int) -> List[float]:
    """Deterministic offline fallback embedding (not semantic)."""
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=64).digest()
    buf = io.BytesIO(h)
    vals: List[float] = []
    need = dim
    counter = 0
    while need > 0:
        chunk = buf.read(8)
        if len(chunk) < 8:
            counter += 1
            buf = io.BytesIO(hashlib.blake2b(h + counter.to_bytes(4, "little"), digest_size=64).digest())
            continue
        x = int.from_bytes(chunk, "little")
        vals.append((x % 10_000_000) / 10_000_000.0)
        need -= 1
    mean = sum(vals) / dim
    vec = [v - mean for v in vals]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _zeros_embedding(dim: int) -> List[float]:
    return [0.0] * dim


def _fingerprint_file(path: Path, chunk_size: int = 1 << 20) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest()


def _iter_json_records(path: Path) -> Iterator[dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict):
                        yield obj


def _embedding_for(text: str, mode: str, dim: int) -> List[float]:
    if mode == "zeros":
        return _zeros_embedding(dim)
    return _hash_embedding(text or "", dim)


def _upsert_batch(cur: psycopg.Cursor, rows: List[Tuple[str, str, str, str]]) -> int:
    """Upsert a batch of rows into the papers table."""
    q = """
    INSERT INTO papers (doc_id, title, abstract, embedding)
    VALUES (%s, %s, %s, %s::vector)
    ON CONFLICT (doc_id) DO UPDATE
      SET title = EXCLUDED.title,
          abstract = EXCLUDED.abstract,
          embedding = EXCLUDED.embedding;
    """
    cur.executemany(q, rows)
    return cur.rowcount if cur.rowcount is not None else len(rows)


# --------------------------------------------------
# Service logic
# --------------------------------------------------
def ingest_json_service(request: IngestRequest, conn: psycopg.Connection) -> IngestResult:
    start_time = time.time()

    data_path = Path(request.path or (settings.data_path or ""))
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    corpus_name = request.corpus_name or data_path.name
    dim = int(settings.embedding_dim)

    rows_processed = 0
    affected_total = 0

    with conn.transaction():
        with conn.cursor() as cur:
            batch: List[Tuple[str, str, str, str]] = []
            for obj in _iter_json_records(data_path):
                doc_id = (obj.get("id") or obj.get("doc_id") or "").strip()
                title = (obj.get("title") or "").strip()
                abstract = (obj.get("abstract") or "").strip()
                if not (doc_id and title and abstract):
                    continue

                emb = _embedding_for(abstract, request.embedding_mode, dim)
                batch.append((doc_id, title, abstract, _vector_literal(emb)))
                rows_processed += 1

                if len(batch) >= request.batch_size:
                    affected_total += _upsert_batch(cur, batch)
                    batch.clear()

            if batch:
                affected_total += _upsert_batch(cur, batch)

            # Update fingerprint if requested
            fp = None
            if request.update_fingerprint:
                fp = _fingerprint_file(data_path)
                cur.execute(
                    """
                    INSERT INTO corpus_state (corpus_name, fingerprint)
                    VALUES (%s, %s)
                    ON CONFLICT (corpus_name) DO UPDATE
                      SET fingerprint = EXCLUDED.fingerprint,
                          updated_at = now();
                    """,
                    (corpus_name, fp),
                )

            cur.execute("ANALYZE papers;")

    dt = time.time() - start_time
    return IngestResult(
        file=str(data_path),
        corpus_name=corpus_name,
        fingerprint=fp if request.update_fingerprint else None,
        rows_processed=rows_processed,
        rows_inserted=affected_total,
        rows_updated=0,
        duration_sec=round(dt, 3),
    )
