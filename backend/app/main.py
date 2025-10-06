from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.db.session import DatabasePool, ping_db
from app.db.config import settings
from app.router.health import router as health_router
from app.router.answer import router as answer_router
from app.router.ingest_router import router as ingest_router
from app.container import build_container


# ============================================================
# Unified Logger Setup
# ============================================================
logger = logging.getLogger("rag.app")

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)


# ============================================================
# Global container reference
# ============================================================
container = None


# ============================================================
# Lifespan startup / shutdown logic
# ============================================================
@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Application startup sequence:
    - Initialize DB pool
    - Build DI container
    - Initialize QA / indexing services
    - Wire QA service into router
    """
    global container
    logger.info("🚀 Initializing application...")

    # --- Database initialization ---
    try:
        DatabasePool.init()
        ok, msg = ping_db()
        if ok:
            logger.info(f"✅ DB OK: {msg}")
        else:
            logger.warning(f"⚠️ DB ping failed: {msg}")
    except Exception as e:
        logger.warning(f"⚠️ Database init skipped or failed: {e}")

    # --- Dependency injection container ---
    try:
        container = build_container(settings)
        from app.router import answer as answer_router_module
        answer_router_module.qa_service = container.qa_service

        # --- Index startup only for local mode ---
        if container.indexing_service is not None:
            idx_info = container.indexing_service.startup()
            logger.info(
                f"📚 Index ready | rebuilt={idx_info['rebuilt']} | count={idx_info['count']}"
            )
        else:
            logger.info("🔗 Using pgvector in Postgres for retrieval; no local index required.")
    except Exception as e:
        logger.error(f"❌ Container build failed: {e}", exc_info=True)
        raise

    try:
        yield
    finally:
        try:
            DatabasePool.close()
            logger.info("🧹 Database pool closed.")
        except Exception:
            pass


# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title="On-Prem RAG (Ollama + Postgres)",
    version="0.4.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Routers
app.include_router(health_router)
app.include_router(answer_router)
app.include_router(ingest_router)


# ============================================================
# Root Endpoint
# ============================================================
@app.get("/")
def root():
    """
    Root route: show available endpoints and configuration summary.
    """
    return {
        "app": "On-Prem RAG (Ollama + Postgres)",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "db_ping": "/db/ping",
        "ingest_json": {
            "method": "POST",
            "path": "/db/ingest-json",
            "body_example": {
                "path": "data/arxiv_2.9k.jsonl",
                "batch_size": 512,
                "embedding_mode": "hash",
            },
        },
        "qa": {
            "method": "POST",
            "path": "/answer",
            "body_example": {
                "query": "How do transformers handle long context?",
                "k": 5,
            },
        },
        "retriever_backend": "pgvector (Postgres)" if container and container.indexing_service is None else "local file index",
    }
