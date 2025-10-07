from __future__ import annotations
from typing import List
from app.core.ports.embeddings import IEmbeddingModel
from app.core.ports.index import IVectorIndex
from app.core.ports.store import IMetadataStore
from app.core.entities import Hit, RetrievalResult, Answer
from app.core.services.grounding import enforce_citations

class QARetriever:
    def __init__(self, store: IMetadataStore, embedder: IEmbeddingModel, index: IVectorIndex):
        self.store = store
        self.embedder = embedder
        self.index = index

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        qv = self.embedder.embed(query)
        results = self.index.search(qv, k)
        hits: List[Hit] = []
        contexts: List[str] = []
        for row_idx, score in results:
            doc = self.store.get(row_idx)
            hits.append(Hit(doc_id=doc.doc_id, title=doc.title, score=score, chunk=doc.text))
            contexts.append(doc.text)
        return RetrievalResult(hits=hits, contexts=contexts)

class QAService:
    def __init__(self, retriever: QARetriever, generator, cite_top_k: int = 5):
        self.retriever = retriever
        self.generator = generator
        self.cite_top_k = cite_top_k

    def answer(self, query: str, k: int = 5) -> Answer:
        rr = self.retriever.retrieve(query, k=k)
        chosen = enforce_citations(rr.hits, self.cite_top_k)
        answer_text = self.generator.generate(
            query=query,
            contexts=[h.chunk for h in chosen],
            citations_schema="doc_id + title",
        )
        return Answer(
            text=answer_text.strip(),
            citations=chosen,
            contexts=[h.chunk for h in chosen],
        )
