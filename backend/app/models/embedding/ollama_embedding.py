# backend/app/models/embedding/ollama_embedding.py
from __future__ import annotations
from typing import List
import os, requests, time, logging
import math

from app.core.ports.embeddings import IEmbeddingModel

logger = logging.getLogger("app.embedding.ollama")

def _resolve_host() -> str:
    if os.path.exists("/.dockerenv"):
        return "http://ollama:11434"
    return os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()

def _l2_normalize(vec: List[float]) -> List[float]:
    s = sum(x*x for x in vec)
    if s <= 0.0:
        return vec
    inv = 1.0 / math.sqrt(s)
    return [x * inv for x in vec]

class OllamaEmbedding(IEmbeddingModel):
    """
    Uses Ollama /api/embed (new format) with batching, retries, validation, and normalization.
    """

    def __init__(self, host: str | None = None, model: str = "nomic-embed-text:latest", timeout: int = 120):
        self.host = host or _resolve_host()
        self.model = os.getenv("EMBEDDING_MODEL", model).strip()
        if ":" not in self.model:
            self.model += ":latest"
        self.timeout = timeout
        self._check_models()

    def _check_models(self):
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m.get("model") or m.get("name") for m in r.json().get("models", [])]
            if self.model not in models:
                logger.warning(f"⚠️ Embedding model '{self.model}' not registered in Ollama. Available: {models}")
            else:
                logger.info(f"✅ Embedding model '{self.model}' available.")
        except Exception as e:
            logger.warning(f"⚠️ Could not verify Ollama models at {self.host}: {e}")

    # ---------- single ----------
    def _one(self, text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            return []

        url = f"{self.host}/api/embed"
        payload = {"model": self.model, "input": [text]}
        for attempt in range(2):
            try:
                r = requests.post(url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                emb = (data.get("embeddings") or [[]])[0]
                if isinstance(emb, list) and emb and all(isinstance(x, (int, float)) for x in emb):
                    return _l2_normalize([float(x) for x in emb])
                logger.warning(f"⚠️ Empty/invalid embedding for input (len={len(text)}). Attempt {attempt+1}")
            except requests.RequestException as e:
                logger.warning(f"⚠️ Ollama embed request failed: {e}. Attempt {attempt+1}")
            time.sleep(0.8)
        return []  # signal failure

    # ---------- batch ----------
    def _batch(self, texts: List[str]) -> List[List[float]]:
        clean = [(t or "").strip() for t in texts]
        idxs = [i for i, t in enumerate(clean) if t]
        if not idxs:
            return [[] for _ in texts]

        inputs = [clean[i] for i in idxs]
        url = f"{self.host}/api/embed"
        payload = {"model": self.model, "input": inputs}

        for attempt in range(2):
            try:
                r = requests.post(url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                embs = data.get("embeddings") or []
                if len(embs) != len(inputs):
                    raise ValueError(f"Embedding batch length mismatch: {len(embs)} vs {len(inputs)}")
                out = [[] for _ in texts]
                for slot, vec in zip(idxs, embs):
                    if isinstance(vec, list) and vec and all(isinstance(x, (int, float)) for x in vec):
                        out[slot] = _l2_normalize([float(x) for x in vec])
                return out
            except Exception as e:
                logger.warning(f"⚠️ Batch embed failed: {e}. Attempt {attempt+1}")
                time.sleep(1.2)

        return [[] for _ in texts]  # all failed

    # ---------- public ----------
    def embed(self, text: str) -> List[float]:
        return self._one(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Prefer batch; will fall back to per-item if some are empty
        embs = self._batch(texts)
        if any(not v for v in embs):
            # retry failed ones individually just once
            for i, v in enumerate(embs):
                if not v and texts[i].strip():
                    embs[i] = self._one(texts[i])
        return embs
