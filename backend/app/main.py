from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.db.config import settings
from app.db.session import DatabasePool, ping_db
from app.router.health import router as health_router
from app.router.answer import router as answer_router
from app.router.ingest_router import router as ingest_router

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
# Lifespan startup / shutdown logic
# ============================================================
@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Initializing application and database pool...")
    DatabasePool.init()
    ok, msg = ping_db()
    if ok:
        logger.info(f"Database connected successfully: {msg}")
    else:
        logger.warning(f"Database ping failed: {msg}")
    try:
        yield
    finally:
        DatabasePool.close()
        logger.info("Database pool closed.")

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title="Generative-AI Paper QA (Offline, pgvector)",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ============================================================
# Routers
# ============================================================
app.include_router(health_router)
app.include_router(answer_router)
app.include_router(ingest_router)

# ============================================================
# Root Endpoint
# ============================================================
@app.get("/")
def root():
    return {
        "app": "Generative-AI Paper QA (Offline, pgvector)",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "db_ping": "/db/ping",
        "ingest": {
            "method": "POST",
            "path": "/db/ingest-json",
            "body_example": {
                "path": settings.data_path or "/data/arxiv_2.9k.jsonl",
                "batch_size": 512,
                "embedding_mode": "hash",
            },
        },
        "answer": {
            "method": "POST",
            "path": "/answer",
            "body_example": {
                "query": "How do transformers handle long context?",
                "k": settings.k_default,
            },
        },
    }


