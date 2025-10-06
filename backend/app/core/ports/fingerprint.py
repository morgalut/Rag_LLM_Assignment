from __future__ import annotations
from abc import ABC, abstractmethod

class IFingerprint(ABC):
    @abstractmethod
    def fingerprint(self, path: str) -> str: ...
    @abstractmethod
    def read_last(self) -> str | None: ...
    @abstractmethod
    def write(self, value: str) -> None: ...
