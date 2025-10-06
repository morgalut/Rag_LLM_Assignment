from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

class IEmbeddingModel(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        ...

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        ...
