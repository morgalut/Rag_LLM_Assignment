from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple

class IVectorIndex(ABC):
    @abstractmethod
    def build(self, embeddings: List[List[float]]) -> None: ...
    @abstractmethod
    def save(self, path: str) -> None: ...
    @abstractmethod
    def load(self, path: str) -> None: ...
    @abstractmethod
    def search(self, query_vec: List[float], k: int) -> List[Tuple[int, float]]:
        """Return list of (row_index, score)"""
        ...
