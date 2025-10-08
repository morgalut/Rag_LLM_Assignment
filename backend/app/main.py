from __future__ import annotations
import os
import logging
import asyncio
import time
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# 🪵 Logging Setup
# ============================================================
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("rag.app")

# ============================================================
# 📦 Core Imports (DB + Dependency Injection)
# ============================================================
from app.db.session import DatabasePool, ping_db
from app.db.config import settings
from app.container import build_container, create_local_index_components

# ============================================================
# 🌐 Routers
# ============================================================
from app.router.health import router as health_router
from app.router.answer import router as answer_router
from app.router.ingest_router import router as ingest_router

# ============================================================
# ⚙️ Global State & Application Status
# ============================================================
class AppState:
    def __init__(self):
        self.container = None
        self.local_indexing_service = None
        self.index_build_task = None
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag_index")
        self.startup_time = time.time()
        self.startup_complete = False
        self.index_build_complete = False
        self.index_build_progress = 0.0
        self.index_build_status = "not_started"  # not_started, building, complete, failed
        self.index_build_error = None
        self.index_stats = None
        self.lock = Lock()
        self.mode = "unknown"
        
    def update_progress(self, progress: float, status: str = None):
        with self.lock:
            self.index_build_progress = max(0.0, min(1.0, progress))
            if status:
                self.index_build_status = status
            if progress >= 1.0:
                self.index_build_complete = True
                self.startup_complete = True
    
    def set_index_stats(self, stats: dict):
        with self.lock:
            self.index_stats = stats
    
    def set_error(self, error: str):
        with self.lock:
            self.index_build_status = "failed"
            self.index_build_error = error
            self.startup_complete = True  # Still mark startup complete to avoid blocking health checks
    
    def set_mode(self, mode: str):
        with self.lock:
            self.mode = mode
    
    def get_status(self):
        with self.lock:
            return {
                "startup_complete": self.startup_complete,
                "index_build_complete": self.index_build_complete,
                "index_build_status": self.index_build_status,
                "index_build_progress": self.index_build_progress,
                "index_build_error": self.index_build_error,
                "index_stats": self.index_stats,
                "mode": self.mode,
                "uptime_seconds": time.time() - self.startup_time
            }

app_state = AppState()

# ============================================================
# 🚀 Startup / Shutdown Lifecycle
# ============================================================
@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Enhanced application startup lifecycle with background indexing
    """
    global app_state
    
    logger.info("🚀 Initializing Enhanced RAG Backend API...")

    # --- Database connection ---
    try:
        DatabasePool.init()
        ok, msg = ping_db()
        if ok:
            logger.info(f"✅ Database OK: {msg}")
        else:
            logger.warning(f"⚠️ DB ping failed: {msg}")
    except Exception as e:
        logger.warning(f"⚠️ Database init skipped or failed: {e}")

    # --- Build container and determine mode ---
    try:
        app_state.container = build_container(settings)
        from app.router import answer as answer_router_module

        # Determine operation mode
        retriever_backend = os.getenv("RETRIEVER_BACKEND", "pgvector")
        data_path = os.getenv("DATA_PATH")
        use_local_index = retriever_backend == "local" and data_path and os.path.isfile(data_path)
        
        if use_local_index:
            app_state.set_mode("local_index")
            logger.info(f"📂 Local index mode detected - data: {data_path}")
            
            # Start background index building
            def build_index_with_progress():
                try:
                    app_state.update_progress(0.1, "building")
                    logger.info("🔄 Starting background index building...")
                    
                    # Use the indexing service from container
                    if app_state.container and app_state.container.indexing_service:
                        info = app_state.container.indexing_service.startup()
                        
                        app_state.update_progress(1.0, "complete")
                        app_state.set_index_stats(info)
                        logger.info(f"✅ Background index building complete | rebuilt={info['rebuilt']} | count={info['count']}")
                    else:
                        raise RuntimeError("Indexing service not available in container")
                        
                except Exception as e:
                    error_msg = f"Index building failed: {str(e)}"
                    logger.error(f"❌ {error_msg}")
                    app_state.set_error(error_msg)

            # Submit background task
            app_state.index_build_task = app_state.executor.submit(build_index_with_progress)
            
            # Set QA service for local index mode
            answer_router_module.qa_service = app_state.container.qa_service
            answer_router_module.streaming_generator = app_state.container.generator.ollama_gen
            
        else:
            app_state.set_mode("pgvector")
            logger.info("🔗 Using pgvector (Postgres) as retriever backend.")
            answer_router_module.qa_service = app_state.container.qa_service
        
        # Mark startup as complete (API is usable)
        app_state.startup_complete = True
        logger.info("🎯 API is ready and accepting requests")
            
    except Exception as e:
        logger.error(f"❌ Container or RAG init failed: {e}", exc_info=True)
        app_state.set_error(f"Startup failed: {str(e)}")
        app_state.startup_complete = True  # Still allow API to start
        raise

    # --- Startup completed ---
    try:
        yield
    finally:
        # --- Cleanup ---
        try:
            if app_state.index_build_task and not app_state.index_build_task.done():
                app_state.index_build_task.cancel()
                logger.info("🛑 Cancelled background index building")
            
            app_state.executor.shutdown(wait=False)
            DatabasePool.close()
            logger.info("🧹 Application shutdown complete")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")

# ============================================================
# 🌍 FastAPI App Definition
# ============================================================
app = FastAPI(
    title="RAG Backend API",
    description="Enhanced RAG Backend with background indexing and improved health checks",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health_router)
app.include_router(answer_router)
app.include_router(ingest_router)

# ============================================================
# 🆕 Enhanced Status Endpoints
# ============================================================
@app.get("/status")
async def get_app_status():
    """Enhanced application status with indexing progress"""
    status = app_state.get_status()
    
    # Determine overall system status
    if status["index_build_status"] == "failed":
        system_status = "degraded"
    elif not status["index_build_complete"] and status["index_build_status"] == "building":
        system_status = "initializing"
    else:
        system_status = "healthy"
    
    return {
        "system_status": system_status,
        "timestamp": time.time(),
        **status
    }

@app.get("/status/index")
async def get_index_status():
    """Detailed index status and statistics"""
    status = app_state.get_status()
    
    response = {
        "index_status": status["index_build_status"],
        "progress": status["index_build_progress"],
        "error": status["index_build_error"],
        "stats": status["index_stats"],
        "mode": status["mode"],
        "timestamp": time.time()
    }
    
    # Add estimated time remaining if building
    if status["index_build_status"] == "building" and status["index_build_progress"] > 0:
        elapsed = status["uptime_seconds"]
        estimated_total = elapsed / status["index_build_progress"]
        remaining = max(0, estimated_total - elapsed)
        response["estimated_seconds_remaining"] = int(remaining)
    
    return response

# ============================================================
# 🏠 Root Endpoint
# ============================================================
@app.get("/")
def root():
    """Enhanced root route with system status"""
    status = app_state.get_status()
    
    system_status = "healthy"
    if status["index_build_status"] == "failed":
        system_status = "degraded"
    elif status["index_build_status"] == "building":
        system_status = "initializing"
    
    return {
        "app": "RAG Backend API (Ollama + Postgres)",
        "version": "2.0.0",
        "status": system_status,
        "mode": status["mode"],
        "index_status": status["index_build_status"],
        "index_progress": f"{status['index_build_progress']*100:.1f}%" if status['index_build_progress'] > 0 else "0%",
        "data_path": os.getenv("DATA_PATH"),
        "index_dir": os.getenv("INDEX_DIR", "/app/index"),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "readiness": "/health/ready", 
            "status": "/status",
            "index_status": "/status/index"
        },
        "examples": {
            "ingest": {
                "method": "POST",
                "path": "/db/ingest-json",
                "body": {"path": "data/arxiv_2.9k.jsonl", "batch_size": 512, "embedding_mode": "hash"},
            },
            "qa": {
                "method": "POST", 
                "path": "/answer",
                "body": {"query": "Explain transformer architecture.", "k": 5},
            },
        },
    }

# ============================================================
# 🏁 Entrypoint
# ============================================================
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Enhanced RAG Backend API on port 8080...")
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8080, 
        reload=False,
        log_config=None
    )