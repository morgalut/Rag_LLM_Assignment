from __future__ import annotations

import io
import json
import math
import time
import hashlib
from pathlib import Path
from typing import List, Tuple, Iterator, Optional

import psycopg
from app.db.config import settings
from app.db.session import DatabasePool
from app.models.ingest_model import IngestRequest, IngestResult
import logging

log = logging.getLogger("app.ingest")

# ================================================================
# Utility functions
# ================================================================
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
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception as e:
                    log.warning("⚠️ Bad JSON on line %d: %s", i, e)
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


def _pick(obj: dict, keys: List[str]) -> str:
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _map_record(obj: dict) -> tuple[str, str, str]:
    """Map varied JSON schemas to (doc_id, title, abstract)."""
    doc_id = _pick(obj, ["id", "doc_id", "paperId", "uid", "uuid"])
    title = _pick(obj, ["title", "paper_title", "name"])
    abstract = _pick(obj, ["abstract", "summary", "abstract_text", "text", "description"])
    return doc_id, title, abstract


def _upsert_batch(cur: psycopg.Cursor, rows: List[Tuple[str, str, str, str]]) -> Tuple[int, int]:
    """Upsert a batch of rows into the papers table."""
    inserted = updated = 0
    for doc_id, title, abstract, emb in rows:
        cur.execute(
            """
            INSERT INTO papers (doc_id, title, abstract, embedding)
            VALUES (%s, %s, %s, %s::vector)
            ON CONFLICT (doc_id) DO UPDATE
              SET title = EXCLUDED.title,
                  abstract = EXCLUDED.abstract,
                  embedding = EXCLUDED.embedding
              RETURNING (xmax = 0) AS inserted;
            """,
            (doc_id, title, abstract, emb),
        )
        if cur.fetchone()[0]:
            inserted += 1
        else:
            updated += 1
    return inserted, updated


# ================================================================
# Resilient ingestion wrapper
# ================================================================
def ingest_json_service(request: IngestRequest, conn: psycopg.Connection) -> IngestResult:
    """
    Ingest JSON/JSONL data into the `papers` table.
    Automatically retries on transient DB restarts (AdminShutdown / closed connection).
    """
    for attempt in range(3):
        try:
            # If the connection is closed, reinitialize it
            if conn.closed:
                log.warning("⚠️ Connection closed — reacquiring from pool...")
                if not DatabasePool.pool:
                    DatabasePool.init()
                conn = DatabasePool.pool.getconn()

            return _perform_ingest(request, conn)

        except psycopg.errors.AdminShutdown:
            log.warning(f"⚠️ Database restarted (AdminShutdown). Retrying ingestion ({attempt+1}/3)...")
            time.sleep(5)

        except psycopg.OperationalError as e:
            if "closed" in str(e).lower():
                log.warning(f"⚠️ Connection lost, retrying ({attempt+1}/3)...")
                time.sleep(5)
                if not DatabasePool.pool:
                    DatabasePool.init()
                conn = DatabasePool.pool.getconn()
                continue
            else:
                log.error(f"❌ OperationalError: {e}")
                raise

        except Exception as e:
            log.error(f"❌ Unhandled error during ingestion attempt {attempt+1}: {e}")
            raise

    raise RuntimeError("Database kept restarting or connection failed after 3 retries. Aborting.")


# ================================================================
# Core ingestion logic
# ================================================================
def _perform_ingest(request: IngestRequest, conn: psycopg.Connection) -> IngestResult:
    start_time = time.time()

    data_path = Path(request.path or (settings.data_path or ""))
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    corpus_name = request.corpus_name or data_path.name
    dim = int(settings.embedding_dim)
    if dim <= 0:
        raise ValueError(f"Invalid embedding_dim: {dim}")

    rows_processed = rows_inserted = rows_updated = rows_skipped = 0
    fp: Optional[str] = None

    with conn.cursor() as cur:
        batch: List[Tuple[str, str, str, str]] = []

        for idx, obj in enumerate(_iter_json_records(data_path), 1):
            doc_id, title, abstract = _map_record(obj)
            if not (doc_id and title and abstract):
                rows_skipped += 1
                if rows_skipped <= 10:
                    log.debug("Skipping record #%d - missing fields. Keys: %s", idx, list(obj.keys()))
                continue

            emb = _embedding_for(abstract, request.embedding_mode, dim)
            batch.append((doc_id, title, abstract, _vector_literal(emb)))
            rows_processed += 1

            if len(batch) >= request.batch_size:
                ins, upd = _upsert_batch(cur, batch)
                rows_inserted += ins
                rows_updated += upd
                batch.clear()

        # Final batch
        if batch:
            ins, upd = _upsert_batch(cur, batch)
            rows_inserted += ins
            rows_updated += upd

        # Fingerprint tracking
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

        conn.commit()

    duration = round(time.time() - start_time, 3)
    log.info(
        "📥 Ingest summary | file=%s | processed=%d | inserted=%d | updated=%d | skipped=%d | took=%.3fs",
        str(data_path), rows_processed, rows_inserted, rows_updated, rows_skipped, duration,
    )

    return IngestResult(
        file=str(data_path),
        corpus_name=corpus_name,
        fingerprint=fp,
        rows_processed=rows_processed,
        rows_inserted=rows_inserted,
        rows_updated=rows_updated,
        rows_skipped=rows_skipped,
        duplicates_found=rows_updated,
        duration_sec=duration,
    )
