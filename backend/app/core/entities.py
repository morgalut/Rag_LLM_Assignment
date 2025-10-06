from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str  # abstract/content/body

@dataclass(frozen=True)
class Hit:
    doc_id: str
    title: str
    score: float
    chunk: str  # retrieved snippet / text

@dataclass(frozen=True)
class RetrievalResult:
    hits: List[Hit]
    contexts: List[str]  # extracted text chunks

@dataclass(frozen=True)
class Answer:
    text: str
    citations: List[Hit]
    contexts: List[str]
