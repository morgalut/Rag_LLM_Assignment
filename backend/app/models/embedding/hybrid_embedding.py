from __future__ import annotations
from typing import List
import logging
from app.models.embedding.ollama_embedding import OllamaEmbedding
from app.models.embedding.hash_embedding import HashEmbedding
from app.core.ports.embeddings import IEmbeddingModel

log = logging.getLogger("app.embed")

class HybridEmbedding(IEmbeddingModel):
    """
    Try Ollama embeddings first; if it fails, fallback to deterministic HashEmbedding.
    """
    def __init__(self, dim: int = 768, ollama_model: str = "nomic-embed-text"):
        self.primary = OllamaEmbedding(model=ollama_model)
        self.fallback = HashEmbedding(dim=dim)
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        try:
            return self.primary.embed(text)
        except Exception as e:
            log.warning("Ollama embed failed; falling back. %s", e)
            return self.fallback.embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            return self.primary.embed_batch(texts)
        except Exception as e:
            log.warning("Ollama batch embed failed; falling back. %s", e)
            return self.fallback.embed_batch(texts)
