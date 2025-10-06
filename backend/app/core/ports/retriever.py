from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from app.core.entities import RetrievalResult, Document

class IRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        ...
