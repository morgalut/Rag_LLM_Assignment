from __future__ import annotations
from typing import List
import os
from app.core.ports.embeddings import IEmbeddingModel
from app.core.ports.index import IVectorIndex
from app.core.ports.store import IMetadataStore
from app.core.ports.fingerprint import IFingerprint

class IndexingService:
    def __init__(
        self,
        store: IMetadataStore,
        embedder: IEmbeddingModel,
        index: IVectorIndex,
        fp: IFingerprint,
        data_path: str,
        index_path: str,
    ):
        self.store = store
        self.embedder = embedder
        self.index = index
        self.fp = fp
        self.data_path = data_path
        self.index_path = index_path

    def startup(self) -> dict:
        """Build or load the vector index based on dataset fingerprint."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        current = self.fp.fingerprint(self.data_path)
        last = self.fp.read_last()

        if last != current or not os.path.exists(self.index_path):
            # Rebuild
            docs = self.store.load(self.data_path)
            embeddings = self.embedder.embed_batch([d.text for d in docs])
            self.index.build(embeddings)
            self.index.save(self.index_path)
            self.fp.write(current)
            return {"rebuilt": True, "count": len(docs)}
        # Load
        self.store.load(self.data_path)  # ensure docs in memory
        self.index.load(self.index_path)
        return {"rebuilt": False, "count": len(self.store)}
