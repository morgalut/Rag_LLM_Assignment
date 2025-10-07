from __future__ import annotations
import os
from dataclasses import dataclass

from app.models.embedding.hash_embedding import HashEmbedding
from app.models.embedding.ollama_embedding import OllamaEmbedding
from app.models.llm.ollama_generator import OllamaAnswerGenerator
from app.models.index.np_index import NpCosineIndex
from app.models.store.inmemory_store import InMemoryStore
from app.models.fingerprint.file_fingerprint import FileFingerprint
from app.models.retriever.pgvector_retriever import PgVectorRetriever

from app.core.services.indexing_service import IndexingService
from app.core.services.qa_service import QARetriever, QAService

@dataclass
class AppContainer:
    indexing_service: IndexingService | None
    qa_service: QAService

def build_container(settings) -> AppContainer:
    # Embedding backend (used both for local index & pgvector queries)
    embed_mode = os.getenv("EMBEDDING_BACKEND", "ollama")  # "ollama" | "hash"
    embedder = (
        OllamaEmbedding(model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"))
        if embed_mode == "ollama"
        else HashEmbedding(dim=int(os.getenv("EMBEDDING_DIM", "768")))
    )

    # ⚠️ Do NOT pass 'model=' here; OllamaAnswerGenerator reads from env
    generator = OllamaAnswerGenerator()

    retriever_backend = os.getenv("RETRIEVER_BACKEND", "pgvector")  # "pgvector" | "local"

    if retriever_backend == "pgvector":
        # ✅ Pure DB mode, no filesystem artifacts
        retriever = PgVectorRetriever(embedder=embedder, table=os.getenv("PAPERS_TABLE", "papers"))
        qa_service = QAService(retriever=retriever, generator=generator, cite_top_k=5)
        return AppContainer(indexing_service=None, qa_service=qa_service)

    # ---- Local file index mode ----
    data_path = os.getenv("DATA_PATH", settings.data_path or "./data/dataset.jsonl")
    index_dir = os.getenv("INDEX_DIR", "./.cache/index")
    index_path = os.path.join(index_dir, "index.npy")

    store = InMemoryStore()
    index = NpCosineIndex()
    fp = FileFingerprint(state_dir=index_dir)

    indexing_service = IndexingService(
        store=store,
        embedder=embedder,
        index=index,
        fp=fp,
        data_path=data_path,
        index_path=index_path,
    )
    retriever = QARetriever(store=store, embedder=embedder, index=index)
    qa_service = QAService(retriever=retriever, generator=generator, cite_top_k=5)
    return AppContainer(indexing_service=indexing_service, qa_service=qa_service)


def get_embedder():
    backend = os.getenv("EMBEDDING_BACKEND", "ollama")
    if backend == "hash":
        return HashEmbedding(dim=int(os.getenv("EMBEDDING_DIM", "768")))
    return OllamaEmbedding(model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"))
