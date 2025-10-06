from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Iterable
from app.core.entities import Document

class IMetadataStore(ABC):
    @abstractmethod
    def load(self, path: str) -> List[Document]: ...
    @abstractmethod
    def iter_documents(self) -> Iterable[Document]: ...
    @abstractmethod
    def get(self, idx: int) -> Document: ...
    @abstractmethod
    def __len__(self) -> int: ...
