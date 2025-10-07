from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# 📦 Database & DI Imports
# ============================================================
from app.db.session import DatabasePool, ping_db
from app.db.config import settings
from app.container import build_container

# ============================================================
# 🤖 Local RAG Components (for on-prem mode)
# ============================================================
from app.models.store.inmemory_store import InMemoryStore
from app.models.index.numpy_index import BasicNumpyIndex
from app.models.fingerprint.file_fingerprint import FileFingerprint
from app.models.embedding.hybrid_embedding import HybridEmbedding
from app.models.llm.ollama_generator import OllamaAnswerGenerator
from app.models.llm.extractive_generator import ExtractiveAnswerGenerator
from app.core.services.qa_service import QARetriever, QAService
from app.core.services.indexing_service import IndexingService

# ============================================================
# 🌐 Routers
# ============================================================
from app.router.health import router as health_router
from app.router.answer import router as answer_router
from app.router.ingest_router import router as ingest_router


# ============================================================
# 🪵 Unified Logger Setup
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
# 🌍 Global Container / Service References
# ============================================================
container = None
local_indexing_service = None


# ============================================================
# 🚀 Lifespan Startup / Shutdown Logic
# ============================================================
@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Application startup sequence:
    1️⃣ Initialize DB (if available)
    2️⃣ Build DI container
    3️⃣ Initialize RAG components (pgvector or local)
    4️⃣ Wire QA service into router
    """
    global container, local_indexing_service
    logger.info("🚀 Initializing On-Prem RAG Application...")

    # --- Database initialization ---
    try:
        DatabasePool.init()
        ok, msg = ping_db()
        if ok:
            logger.info(f"✅ Database OK: {msg}")
        else:
            logger.warning(f"⚠️ DB ping failed: {msg}")
    except Exception as e:
        logger.warning(f"⚠️ Database initialization skipped or failed: {e}")

    # --- Dependency Injection Container ---
    try:
        container = build_container(settings)
        from app.router import answer as answer_router_module
        answer_router_module.qa_service = container.qa_service

        # ✅ Initialize local index (if configured for file mode)
        DATA_PATH = os.getenv("DATA_PATH")
        if DATA_PATH and os.path.exists(DATA_PATH):
            logger.info(f"📂 Local dataset detected → building local index for {DATA_PATH}")

            INDEX_DIR = os.getenv("INDEX_DIR", "/app/index")
            INDEX_PATH = os.path.join(INDEX_DIR, "vectors.npz")
            FP_PATH = os.path.join(INDEX_DIR, "fp.json")
            EMB_DIM = int(os.getenv("EMBED_DIM", "768"))
            OLLAMA_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
            CHUNK_LEN = int(os.getenv("CHUNK_LEN", "0"))

            store = InMemoryStore()
            embedder = HybridEmbedding(dim=EMB_DIM, ollama_model=OLLAMA_MODEL)
            index = BasicNumpyIndex()
            fp = FileFingerprint(FP_PATH, DATA_PATH, {"dim": EMB_DIM, "model": OLLAMA_MODEL})
            local_indexing_service = IndexingService(store, embedder, index, fp, DATA_PATH, INDEX_PATH, chunk_len=CHUNK_LEN)

            info = local_indexing_service.startup()
            logger.info(f"📚 Local index ready | rebuilt={info['rebuilt']} | count={info['count']}")
            
            # Build local QA service
            ollama_gen = OllamaAnswerGenerator()
            extractive_gen = ExtractiveAnswerGenerator()

            class CompositeGen:
                def generate(self, query, contexts, citations_schema):
                    try:
                        return ollama_gen.generate(query, contexts, citations_schema)
                    except Exception as e:
                        logger.warning(f"Ollama gen failed; fallback to extractive: {e}")
                        return extractive_gen.generate(query, contexts, citations_schema)
                def generate_stream(self, query, contexts, citations_schema):
                    return ollama_gen.generate_stream(query, contexts, citations_schema)

            retriever = QARetriever(store, embedder, index)
            qa_local = QAService(retriever, CompositeGen(), cite_top_k=5)
            answer_router_module.qa_service = qa_local
            answer_router_module.streaming_generator = ollama_gen
        else:
            logger.info("🔗 Using pgvector (Postgres) as retriever backend.")
    except Exception as e:
        logger.error(f"❌ Container build or RAG init failed: {e}", exc_info=True)
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
# ⚙️ FastAPI App Initialization
# ============================================================
app = FastAPI(
    title="On-Prem RAG (Ollama + Postgres)",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS for UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health_router)
app.include_router(answer_router)
app.include_router(ingest_router)


# ============================================================
# 🏠 Root Endpoint
# ============================================================
@app.get("/")
def root():
    """Root route: overview + API guide"""
    mode = (
        "pgvector (Postgres)"
        if container and container.indexing_service is None
        else "local file index"
    )
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
        "retriever_backend": mode,
        "local_index_dir": os.getenv("INDEX_DIR", "/app/index"),
        "data_path": os.getenv("DATA_PATH"),
    }
