from __future__ import annotations
from typing import List, Tuple
import numpy as np, os
from app.core.ports.index import IVectorIndex

class NpCosineIndex(IVectorIndex):
    """
    Simple cosine-similarity index (no FAISS dependency).
    Suitable for up to ~100k items depending on RAM/latency SLAs.
    """
    def __init__(self):
        self.mat: np.ndarray | None = None
        self.norms: np.ndarray | None = None

    def build(self, embeddings: List[List[float]]) -> None:
        self.mat = np.asarray(embeddings, dtype=np.float32)
        # L2-normalize upfront
        n = np.linalg.norm(self.mat, axis=1, keepdims=True)
        n[n == 0] = 1.0
        self.mat = self.mat / n
        self.norms = None  # not used

    def save(self, path: str) -> None:
        if self.mat is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.mat)

    def load(self, path: str) -> None:
        self.mat = np.load(path)
        # already normalized at build time

    def search(self, query_vec: List[float], k: int) -> List[Tuple[int, float]]:
        if self.mat is None:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        qn = q / (np.linalg.norm(q) + 1e-9)
        sims = self.mat @ qn
        if k >= sims.shape[0]:
            idx = np.argsort(-sims)
        else:
            idx = np.argpartition(-sims, k)[:k]
            idx = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i])) for i in idx[:k]]
