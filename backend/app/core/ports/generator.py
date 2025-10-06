from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

class IAnswerGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, contexts: List[str], citations_schema: str) -> str:
        ...
