from __future__ import annotations
from typing import List
import os, requests
from app.core.ports.embeddings import IEmbeddingModel


class OllamaEmbedding(IEmbeddingModel):
    """
    Ollama-based embedding model.
    Calls the local Ollama API to generate vector embeddings.
    """

    def __init__(self, host: str | None = None, model: str = "nomic-embed-text", timeout: int = 120):
        self.host = host or os.getenv("OLLAMA_HOST", "http://ollama:11434")
        self.model = os.getenv("EMBEDDING_MODEL", model)
        self.timeout = timeout

    def _one(self, text: str) -> List[float]:
        url = f"{self.host}/api/embeddings"
        payload = {"model": self.model, "input": text}
        try:
            r = requests.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()

            # Handle both new and legacy API formats
            if "embedding" in data:
                return data["embedding"]
            if "data" in data and data["data"] and "embedding" in data["data"][0]:
                return data["data"][0]["embedding"]

            raise RuntimeError(f"Unexpected embedding response: {data}")
        except Exception as e:
            raise RuntimeError(f"Ollama embedding failed: {e}")

    def embed(self, text: str) -> List[float]:
        return self._one(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._one(t) for t in texts]
