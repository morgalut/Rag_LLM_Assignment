from __future__ import annotations
from typing import List, Tuple
import os, json
import numpy as np
from app.core.ports.index import IVectorIndex

class BasicNumpyIndex(IVectorIndex):
    """
    Lightweight cosine-similarity index using NumPy only.
    Saves to .npz with L2-normalized vectors.
    """
    def __init__(self) -> None:
        self._mat: np.ndarray | None = None  # shape (N, D)

    def build(self, embeddings: List[List[float]]) -> None:
        M = np.asarray(embeddings, dtype=np.float32)
        eps = 1e-12
        norms = np.linalg.norm(M, axis=1, keepdims=True) + eps
        self._mat = M / norms

    def search(self, qv: List[float], k: int) -> List[Tuple[int, float]]:
        assert self._mat is not None, "Index not built/loaded"
        q = np.asarray(qv, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = (self._mat @ q)  # cosine
        idx = np.argpartition(-sims, kth=min(k, sims.size-1))[:k]
        idx_sorted = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i])) for i in idx_sorted]

    def save(self, path: str) -> None:
        assert self._mat is not None, "Nothing to save"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, mat=self._mat)

        meta = {"rows": int(self._mat.shape[0]), "dim": int(self._mat.shape[1])}
        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def load(self, path: str) -> None:
        data = np.load(path)
        self._mat = data["mat"].astype(np.float32)
