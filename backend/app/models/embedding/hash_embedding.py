from __future__ import annotations
from typing import List
import io, math, hashlib, struct
from app.core.ports.embeddings import IEmbeddingModel


class HashEmbedding(IEmbeddingModel):
    """
    Deterministic hash-based embedding for offline / fallback mode.
    Generates pseudo-random unit vectors based on text hash.
    """

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _hash_vec(self, text: str) -> List[float]:
        h = hashlib.blake2b(text.encode("utf-8"), digest_size=64).digest()
        buf = io.BytesIO(h)
        vals: List[float] = []

        while len(vals) < self.dim:
            chunk = buf.read(8)
            if len(chunk) < 8:
                # Re-hash the digest to continue filling the vector
                h = hashlib.blake2b(h, digest_size=64).digest()
                buf = io.BytesIO(h)
                continue

            x = struct.unpack("<Q", chunk)[0]
            vals.append((x % 10_000_000) / 10_000_000.0)

        mean = sum(vals) / self.dim
        vec = [v - mean for v in vals]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, text: str) -> List[float]:
        return self._hash_vec(text or "")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_vec(t or "") for t in texts]
