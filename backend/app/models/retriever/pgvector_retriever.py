# backend/app/models/retriever/pgvector_retriever.py

from __future__ import annotations
from typing import List
import os, logging
import numpy as np
import psycopg
from psycopg import sql
from app.core.entities import Hit, RetrievalResult
from app.core.ports.embeddings import IEmbeddingModel
from app.db.session import DatabasePool

logger = logging.getLogger("app.retriever.pgvector")

# -----------------------------
# Tunables (env-overridable)
# -----------------------------
IVF_PROBES      = int(os.getenv("RAG_IVF_PROBES", "50"))
CANDIDATES      = int(os.getenv("RAG_CANDIDATES", "50"))
MIN_STRONG_SIM  = float(os.getenv("RAG_MIN_SIMILARITY", "0.05"))
TOP_FALLBACK_N  = int(os.getenv("RAG_TOP_FALLBACK", "3"))

# -----------------------------
# Helpers
# -----------------------------
def _normalize(vec: List[float]) -> List[float]:
    v = np.asarray(vec, dtype=float)
    n = np.linalg.norm(v)
    return (v / n).tolist() if n > 0 else [0.0] * len(v)

def _to_vector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"

def _adaptive_cutoff(scores: List[float]) -> float:
    top = scores[0]
    med = np.median(scores) if len(scores) > 1 else top
    return float(max(MIN_STRONG_SIM, 0.35 * top, 0.50 * med))

# -----------------------------
# Retriever
# -----------------------------
class PgVectorRetriever:
    """Robust retriever for pgvector-based RAG pipelines."""

    def __init__(self, embedder: IEmbeddingModel, table: str = "papers"):
        self.embedder = embedder
        self.table = table

    def _prepare_session(self, cur: psycopg.Cursor) -> None:
        try:
            # ❗ Directly inline integer (no placeholder)
            cur.execute(f"SET ivfflat.probes = {IVF_PROBES};")
        except Exception as e:
            logger.warning(f"⚠️ Could not set ivfflat.probes={IVF_PROBES}: {e}")

        try:
            cur.execute("SET application_name = 'rag_pgvector_retriever';")
        except Exception:
            pass

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        if not DatabasePool.pool:
            raise RuntimeError("Database pool not initialized")

        # 1️⃣ Embed + normalize
        qv = self.embedder.embed(query)
        if not qv:
            logger.error("❌ Query embedding is empty.")
            return RetrievalResult(hits=[], contexts=[])
        qv = _normalize(qv)
        qv_txt = _to_vector_literal(qv)

        # 2️⃣ Query database
        with DatabasePool.pool.connection() as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                self._prepare_session(cur)

                sql_query = sql.SQL("""
                    SELECT doc_id, title, abstract, embedding
                    FROM {table}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """).format(table=sql.Identifier(self.table))

                try:
                    cur.execute(sql_query, (qv_txt, CANDIDATES))
                    rows = cur.fetchall()
                except Exception as e:
                    logger.exception(f"❌ Candidate retrieval failed: {e}")
                    return RetrievalResult(hits=[], contexts=[])

        if not rows:
            logger.warning("⚠️ No candidates returned from pgvector.")
            return RetrievalResult(hits=[], contexts=[])

        # 3️⃣ Re-rank with cosine similarity
        hits: List[Hit] = []
        qv_np = np.asarray(qv, dtype=float)
        for doc_id, title, abstract, emb in rows:
            # Convert string vector to numpy safely
            if isinstance(emb, str):
                emb = np.fromstring(emb.strip("[]"), sep=",")
            else:
                emb = np.asarray(emb, dtype=float)

            en = np.linalg.norm(emb)
            sim = float(np.dot(qv_np, emb) / en) if en > 0 else -1.0
            hits.append(Hit(doc_id=doc_id, title=title, score=sim, chunk=abstract))

        hits.sort(key=lambda h: h.score, reverse=True)
        top_scores = [h.score for h in hits[:k]]
        logger.info("🔍 Retrieved %d candidates; top-%d scores: %s",
                    len(hits), k, [round(s, 3) for s in top_scores])

        # 4️⃣ Adaptive filtering + fallback
        cutoff = _adaptive_cutoff(top_scores)
        strong = [h for h in hits if h.score >= cutoff][:k]

        if not strong:
            logger.warning("⚠️ No strong hits above threshold (%.2f); using top-%d fallback.",
                           cutoff, max(TOP_FALLBACK_N, k))
            strong = hits[:max(TOP_FALLBACK_N, k)]

        contexts = [h.chunk for h in strong]
        return RetrievalResult(hits=strong, contexts=contexts)
