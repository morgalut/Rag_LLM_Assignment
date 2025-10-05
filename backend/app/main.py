from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.db.config import settings
from app.db.session import init_pool, close_pool
from app.router.health import router as health_router
from app.router.answer import router as answer_router
from app.router.ingest_router import router as ingest_router  

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logger = logging.getLogger("rag.app")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    init_pool()
    try:
        from app.db.session import ping_db
        ping_db()
    except Exception:
        logger.exception("DB ping on startup failed.")
    try:
        yield
    finally:
        close_pool()

# ------------------------------------------------------------
# App
# ------------------------------------------------------------
app = FastAPI(
    title="Generative-AI Paper QA (Offline, pgvector)",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Routers
app.include_router(health_router)
app.include_router(answer_router)
app.include_router(ingest_router)  # <-- NEW

# Root
@app.get("/")
def root():
    return {
        "name": "Generative-AI Paper QA (Offline, pgvector)",
        "docs": "/docs",
        "health": "/health",
        "db_ping": "/db/ping",
        "ingest": {
            "method": "POST",
            "path": "/db/ingest-json",
            "body": {"path": settings.data_path, "batch_size": 256, "embedding_mode": "hash"},
        },
        "answer": {
            "method": "POST",
            "path": "/answer",
            "body": {"query": "...", "k": settings.k_default},
        },
    }
