from __future__ import annotations
from typing import List
import psycopg
from app.core.entities import Hit, RetrievalResult
from app.core.ports.embeddings import IEmbeddingModel
from app.db.session import DatabasePool

def _vector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

class PgVectorRetriever:
    """
    Retrieval directly from Postgres using pgvector.
    Requires papers(embedding vector(dim)), title, abstract columns.
    """
    def __init__(self, embedder: IEmbeddingModel, table: str = "papers"):
        self.embedder = embedder
        self.table = table

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        qvec = self.embedder.embed(query)
        vlit = _vector_literal(qvec)
        hits: List[Hit] = []
        contexts: List[str] = []

        if not DatabasePool.pool:
            raise RuntimeError("Database pool is not initialized")

        with DatabasePool.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT doc_id, title, abstract,
                           (1.0 - (embedding <=> %s::vector)) AS score
                    FROM {self.table}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (vlit, vlit, k),
                )
                for doc_id, title, abstract, score in cur.fetchall():
                    hits.append(Hit(doc_id=doc_id, title=title, score=float(score), chunk=abstract))
                    contexts.append(abstract)

        return RetrievalResult(hits=hits, contexts=contexts)
