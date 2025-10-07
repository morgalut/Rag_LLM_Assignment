from __future__ import annotations
from typing import List
import re
from collections import Counter
from app.core.ports.generator import IAnswerGenerator

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\w+", re.U)

def _sentences(s: str) -> List[str]:
    return [x.strip() for x in _SENT_SPLIT.split(s) if x.strip()]

def _tokens(s: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(s)]

class ExtractiveAnswerGenerator(IAnswerGenerator):
    """
    Pure-offline fallback: choose 3 sentences from the retrieved contexts
    with highest token overlap with the query.
    """
    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def generate(self, query: str, contexts: List[str], citations_schema: str) -> str:
        qtok = Counter(_tokens(query))
        cand: List[tuple[float, str]] = []
        for ctx in contexts:
            for s in _sentences(ctx):
                st = Counter(_tokens(s))
                overlap = sum(min(qtok[w], st[w]) for w in set(qtok) & set(st))
                cand.append((float(overlap), s))
        cand.sort(key=lambda x: x[0], reverse=True)
        chosen = [s for _, s in cand[: self.max_sentences]] or (contexts[:1] if contexts else ["I don't know."])
        return " ".join(chosen)
