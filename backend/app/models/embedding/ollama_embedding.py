# backend/app/models/embedding/ollama_embedding.py
from __future__ import annotations
from typing import List
import os, requests, time, math, logging

from app.core.ports.embeddings import IEmbeddingModel

logger = logging.getLogger("app.embedding.ollama")


def _resolve_host() -> str:
    """Resolve Ollama host inside/outside Docker with env override."""
    env_host = os.getenv("OLLAMA_HOST")
    if env_host:
        return env_host.rstrip("/")
    if os.path.exists("/.dockerenv"):
        return "http://ollama:11434"
    return "http://127.0.0.1:11434"


def _l2_normalize(vec: List[float]) -> List[float]:
    s = sum(x * x for x in vec)
    if s <= 0.0:
        return vec
    inv = 1.0 / math.sqrt(s)
    return [x * inv for x in vec]


class OllamaEmbedding(IEmbeddingModel):
    """
    Embedding model using Ollama's /api/embed endpoint.
    Supports batch processing, retries, and configurable timeout/batch size.
    """

    def __init__(self, host: str | None = None, model: str = "nomic-embed-text:latest"):
        self.host = host or _resolve_host()
        self.model = os.getenv("EMBEDDING_MODEL", model).strip()
        if ":" not in self.model:
            self.model += ":latest"
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT", "600"))
        self.batch_size = int(os.getenv("OLLAMA_EMBED_BATCH", "16"))
        self._check_models()

    # ----------------------------------------------------------
    # ✅ Model availability check
    # ----------------------------------------------------------
    def _check_models(self):
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=10)
            r.raise_for_status()
            models = [m.get("model") or m.get("name") for m in r.json().get("models", [])]
            if self.model not in models:
                logger.warning(f"⚠️ Embedding model '{self.model}' not registered. Available: {models}")
            else:
                logger.info(f"✅ Embedding model '{self.model}' available.")
        except Exception as e:
            logger.warning(f"⚠️ Could not verify Ollama models at {self.host}: {e}")

    # ----------------------------------------------------------
    # 🧠 Single embedding
    # ----------------------------------------------------------
    def _one(self, text: str) -> List[float]:
        if not (text := (text or "").strip()):
            return []
        payload = {"model": self.model, "input": [text]}
        url = f"{self.host}/api/embed"

        for attempt in range(3):
            try:
                r = requests.post(url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                emb = (data.get("embeddings") or [[]])[0]
                if emb and all(isinstance(x, (int, float)) for x in emb):
                    return _l2_normalize([float(x) for x in emb])
            except Exception as e:
                logger.warning(f"⚠️ Single embed failed: {e} (Attempt {attempt + 1}/3)")
                time.sleep(2 * (attempt + 1))
        return []

    # ----------------------------------------------------------
    # 🧩 Batch embedding
    # ----------------------------------------------------------
    def _batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embeddings with retries, chunking, and timeout safety."""
        clean = [(t or "").strip() for t in texts]
        idxs = [i for i, t in enumerate(clean) if t]
        out = [[] for _ in texts]
        if not idxs:
            return out

        url = f"{self.host}/api/embed"
        for start in range(0, len(idxs), self.batch_size):
            sub_idxs = idxs[start:start + self.batch_size]
            batch_inputs = [clean[i] for i in sub_idxs]
            payload = {"model": self.model, "input": batch_inputs}

            for attempt in range(3):
                try:
                    r = requests.post(url, json=payload, timeout=self.timeout)
                    r.raise_for_status()
                    data = r.json()
                    embs = data.get("embeddings") or []
                    if len(embs) != len(batch_inputs):
                        raise ValueError(f"Embedding batch mismatch: {len(embs)} vs {len(batch_inputs)}")
                    for slot, vec in zip(sub_idxs, embs):
                        if vec and all(isinstance(x, (int, float)) for x in vec):
                            out[slot] = _l2_normalize([float(x) for x in vec])
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Batch embed failed ({len(batch_inputs)} items): {e} (Attempt {attempt + 1}/3)")
                    time.sleep(2 * (attempt + 1))
        return out

    # ----------------------------------------------------------
    # 🧩 Public API
    # ----------------------------------------------------------
    def embed(self, text: str) -> List[float]:
        return self._one(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embs = self._batch(texts)
        # Fallback for missing embeddings
        for i, v in enumerate(embs):
            if not v and texts[i].strip():
                embs[i] = self._one(texts[i])
        return embs
