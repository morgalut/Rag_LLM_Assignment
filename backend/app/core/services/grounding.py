from __future__ import annotations
from typing import List
from app.core.entities import Hit

def enforce_citations(hits: List[Hit], k: int) -> List[Hit]:
    # Here we simply pass through top-k; could add reranking or dedup logic later.
    return hits[:k]
