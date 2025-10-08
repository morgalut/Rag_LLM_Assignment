from __future__ import annotations
import os
import logging
from dataclasses import dataclass
from typing import Optional

from app.models.embedding.hash_embedding import HashEmbedding
from app.models.embedding.ollama_embedding import OllamaEmbedding
from app.models.embedding.hybrid_embedding import HybridEmbedding
from app.models.llm.ollama_generator import OllamaAnswerGenerator
from app.models.llm.extractive_generator import ExtractiveAnswerGenerator
from app.models.index.np_index import NpCosineIndex
from app.models.index.numpy_index import BasicNumpyIndex
from app.models.store.inmemory_store import InMemoryStore
from app.models.fingerprint.file_fingerprint import FileFingerprint
from app.models.retriever.pgvector_retriever import PgVectorRetriever

from app.core.services.indexing_service import IndexingService
from app.core.services.qa_service import QARetriever, QAService

logger = logging.getLogger("rag.container")

@dataclass
class AppContainer:
    indexing_service: IndexingService | None
    qa_service: QAService
    embedder: object
    generator: object
    retriever: object

class CompositeGenerator:
    """Composite generator that falls back to extractive if Ollama fails"""
    def __init__(self):
        self.ollama_gen = OllamaAnswerGenerator()
        self.extractive_gen = ExtractiveAnswerGenerator()
        self.logger = logging.getLogger("rag.generator")

    def generate(self, query, contexts, citations_schema):
        try:
            return self.ollama_gen.generate(query, contexts, citations_schema)
        except Exception as e:
            self.logger.warning(f"Ollama generation failed; fallback to extractive: {e}")
            return self.extractive_gen.generate(query, contexts, citations_schema)

    def generate_stream(self, query, contexts, citations_schema):
        return self.ollama_gen.generate_stream(query, contexts, citations_schema)

def build_container(settings) -> AppContainer:
    """
    Build the application container with enhanced configuration.
    Supports both pgvector and local index modes with improved error handling.
    """
    # Determine retriever backend
    retriever_backend = os.getenv("RETRIEVER_BACKEND", "pgvector")  # "pgvector" | "local"
    data_path = os.getenv("DATA_PATH")
    use_local_index = retriever_backend == "local" and data_path and os.path.isfile(data_path)
    
    logger.info(f"🔧 Building container - Backend: {retriever_backend}, Local Index: {use_local_index}")

    # Embedding configuration
    embed_mode = os.getenv("EMBEDDING_BACKEND", "ollama")  # "ollama" | "hash" | "hybrid"
    embed_dim = int(os.getenv("EMBEDDING_DIM", "768"))
    ollama_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    
    if embed_mode == "hybrid":
        embedder = HybridEmbedding(dim=embed_dim, ollama_model=ollama_model)
        logger.info(f"🔌 Using hybrid embedding: dim={embed_dim}, model={ollama_model}")
    elif embed_mode == "hash":
        embedder = HashEmbedding(dim=embed_dim)
        logger.info(f"🔌 Using hash embedding: dim={embed_dim}")
    else:  # ollama
        embedder = OllamaEmbedding(model=ollama_model)
        logger.info(f"🔌 Using Ollama embedding: model={ollama_model}")

    # Generator configuration
    generator = CompositeGenerator()
    logger.info("🔌 Using composite generator with Ollama + extractive fallback")

    if use_local_index:
        # ---- Local file index mode ----
        index_dir = os.getenv("INDEX_DIR", "/app/index")
        index_path = os.path.join(index_dir, "vectors.npz")
        fp_path = os.path.join(index_dir, "fp.json")
        chunk_len = int(os.getenv("CHUNK_LEN", "0"))
        
        # Ensure index directory exists
        os.makedirs(index_dir, exist_ok=True)
        
        logger.info(f"📂 Local index mode - Data: {data_path}, Index: {index_dir}")

        # Initialize local components
        store = InMemoryStore()
        
        # Use appropriate index based on configuration
        index_type = os.getenv("INDEX_TYPE", "numpy")  # "numpy" | "cosine"
        if index_type == "cosine":
            index = NpCosineIndex()
            logger.info("📊 Using cosine similarity index")
        else:
            index = BasicNumpyIndex()
            logger.info("📊 Using basic numpy index")
        
        fp = FileFingerprint(
            fp_path, 
            data_path, 
            {
                "dim": embed_dim, 
                "model": ollama_model,
                "embed_mode": embed_mode,
                "chunk_len": chunk_len
            }
        )

        indexing_service = IndexingService(
            store=store,
            embedder=embedder,
            index=index,
            fp=fp,
            data_path=data_path,
            index_path=index_path,
            chunk_len=chunk_len
        )
        
        retriever = QARetriever(store=store, embedder=embedder, index=index)
        qa_service = QAService(retriever=retriever, generator=generator, cite_top_k=5)
        
        logger.info("✅ Local index container built successfully")
        return AppContainer(
            indexing_service=indexing_service,
            qa_service=qa_service,
            embedder=embedder,
            generator=generator,
            retriever=retriever
        )
    else:
        # ---- pgVector mode ----
        logger.info("🔗 Using pgvector retriever backend")
        
        retriever = PgVectorRetriever(
            embedder=embedder, 
            table=os.getenv("PAPERS_TABLE", "papers")
        )
        qa_service = QAService(
            retriever=retriever, 
            generator=generator, 
            cite_top_k=5
        )
        
        logger.info("✅ pgVector container built successfully")
        return AppContainer(
            indexing_service=None,
            qa_service=qa_service,
            embedder=embedder,
            generator=generator,
            retriever=retriever
        )


def get_embedder() -> object:
    """Get embedder instance based on configuration"""
    backend = os.getenv("EMBEDDING_BACKEND", "ollama")
    embed_dim = int(os.getenv("EMBEDDING_DIM", "768"))
    ollama_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    
    if backend == "hash":
        return HashEmbedding(dim=embed_dim)
    elif backend == "hybrid":
        return HybridEmbedding(dim=embed_dim, ollama_model=ollama_model)
    else:  # ollama
        return OllamaEmbedding(model=ollama_model)


def get_generator() -> object:
    """Get generator instance with fallback support"""
    return CompositeGenerator()


def create_local_index_components(data_path: str, index_dir: str = None) -> tuple:
    """
    Create local index components for manual initialization.
    Useful for background indexing and testing.
    """
    if index_dir is None:
        index_dir = os.getenv("INDEX_DIR", "/app/index")
    
    index_path = os.path.join(index_dir, "vectors.npz")
    fp_path = os.path.join(index_dir, "fp.json")
    embed_dim = int(os.getenv("EMBEDDING_DIM", "768"))
    ollama_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    chunk_len = int(os.getenv("CHUNK_LEN", "0"))
    
    embedder = get_embedder()
    store = InMemoryStore()
    
    index_type = os.getenv("INDEX_TYPE", "numpy")
    if index_type == "cosine":
        index = NpCosineIndex()
    else:
        index = BasicNumpyIndex()
    
    fp = FileFingerprint(
        fp_path, 
        data_path, 
        {
            "dim": embed_dim, 
            "model": ollama_model,
            "chunk_len": chunk_len
        }
    )
    
    indexing_service = IndexingService(
        store=store,
        embedder=embedder,
        index=index,
        fp=fp,
        data_path=data_path,
        index_path=index_path,
        chunk_len=chunk_len
    )
    
    retriever = QARetriever(store=store, embedder=embedder, index=index)
    generator = get_generator()
    qa_service = QAService(retriever=retriever, generator=generator, cite_top_k=5)
    
    return indexing_service, qa_service, retriever, embedder, generator


def get_container_mode() -> str:
    """Get the current container mode"""
    retriever_backend = os.getenv("RETRIEVER_BACKEND", "pgvector")
    data_path = os.getenv("DATA_PATH")
    
    if retriever_backend == "local" and data_path and os.path.isfile(data_path):
        return "local_index"
    else:
        return "pgvector"