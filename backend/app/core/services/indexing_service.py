from __future__ import annotations
from typing import List
import os
from app.core.ports.embeddings import IEmbeddingModel
from app.core.ports.index import IVectorIndex
from app.core.ports.store import IMetadataStore
from app.core.ports.fingerprint import IFingerprint

def _maybe_chunk(text: str, max_len: int = 0) -> List[str]:
    if not max_len or len(text) <= max_len:
        return [text]
    # naive chunker on sentence boundaries
    import re
    sents = re.split(r'(?<=[.!?])\s+', text)
    chunks: List[str] = []
    buf = []
    cur = 0
    for s in sents:
        if cur + len(s) + 1 > max_len and buf:
            chunks.append(" ".join(buf))
            buf, cur = [s], len(s)
        else:
            buf.append(s); cur += len(s) + 1
    if buf:
        chunks.append(" ".join(buf))
    return chunks

class IndexingService:
    def __init__(
        self,
        store: IMetadataStore,
        embedder: IEmbeddingModel,
        index: IVectorIndex,
        fp: IFingerprint,
        data_path: str,
        index_path: str,
        chunk_len: int = 0,
    ):
        self.store = store
        self.embedder = embedder
        self.index = index
        self.fp = fp
        self.data_path = data_path
        self.index_path = index_path
        self.chunk_len = chunk_len

    def startup(self) -> dict:
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        current = self.fp.fingerprint(self.data_path)
        last = self.fp.read_last()
        if last != current or not os.path.exists(self.index_path):
            docs = self.store.load(self.data_path)
            # optionally chunk (for abstracts this is probably unnecessary)
            texts: List[str] = []
            for d in docs:
                texts.extend(_maybe_chunk(d.text, self.chunk_len))
            embeddings = self.embedder.embed_batch(texts)
            self.index.build(embeddings)
            self.index.save(self.index_path)
            self.fp.write(current)
            return {"rebuilt": True, "count": len(texts)}
        self.store.load(self.data_path)   # ensure docs are in memory
        self.index.load(self.index_path)
        return {"rebuilt": False, "count": len(self.store)}
