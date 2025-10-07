from __future__ import annotations
from typing import List, Tuple
import os, logging, statistics

from app.core.ports.embeddings import IEmbeddingModel
from app.core.ports.index import IVectorIndex
from app.core.ports.store import IMetadataStore
from app.core.entities import Hit, RetrievalResult, Answer

logger = logging.getLogger("app.qa")

# ==========================================================
# ⚙️ Configurable thresholds
# ==========================================================
MIN_SIM = float(os.getenv("RAG_MIN_SIMILARITY", "0.05"))   # soft floor for filtering
MIN_CTX = int(os.getenv("RAG_MIN_CONTEXTS", "1"))          # minimal contexts to send
FORCE_TOP_K = int(os.getenv("RAG_FORCE_TOP_K", "5"))       # fallback top-k if weak
DEFAULT_FALLBACK = (
    "No relevant passages were strongly matched, "
    "but here’s the best possible answer based on the available database context."
)


# ==========================================================
# 🔍 Retriever Wrapper
# ==========================================================
class QARetriever:
    """Retrieves candidate documents using pgvector or local index."""

    def __init__(self, store: IMetadataStore, embedder: IEmbeddingModel, index: IVectorIndex):
        self.store = store
        self.embedder = embedder
        self.index = index

    def retrieve(self, query: str, k: int = 8) -> RetrievalResult:
        """Return raw retrieval hits (no filtering)."""
        qv = self.embedder.embed(query)
        results: List[Tuple[int, float]] = self.index.search(qv, k)
        hits, contexts = [], []

        for row_idx, score in results:
            try:
                doc = self.store.get(row_idx)
                hits.append(Hit(doc_id=doc.doc_id, title=doc.title, score=score, chunk=doc.text))
                contexts.append(doc.text)
            except Exception as e:
                logger.warning(f"⚠️ Retrieval skip for row {row_idx}: {e}")

        top_score = max([h.score for h in hits], default=0.0)
        logger.info(f"🔍 Retrieved {len(hits)} hits (top score={top_score:.3f})")
        return RetrievalResult(hits=hits, contexts=contexts)


# ==========================================================
# 🧠 QA Service: Dynamic RAG pipeline
# ==========================================================
class QAService:
    """
    Retrieval-Augmented Generation (RAG) service that always returns an answer.
    Key features:
      - Adaptive dynamic cutoff for context selection
      - Confidence-aware generation (logs weak retrieval)
      - Never abstains completely: will use fallback if needed
      - Emits clear logs on retrieval strength
    """

    def __init__(self, retriever: QARetriever, generator, cite_top_k: int = 5):
        self.retriever = retriever
        self.generator = generator
        self.cite_top_k = cite_top_k

    # ------------------------------------------------------
    # 🔎 Dynamic context selection
    # ------------------------------------------------------
    def _select_strong(self, hits: List[Hit]) -> List[Hit]:
        """
        Dynamically select the strongest contexts:
          - Always include the top result
          - Compute an adaptive threshold based on top score
          - Relax filtering if results are generally weak
        """
        if not hits:
            return []

        hits.sort(key=lambda h: h.score or 0.0, reverse=True)
        max_score = hits[0].score or 0.0
        median_score = statistics.median([h.score for h in hits]) if len(hits) > 1 else max_score

        # Dynamic cutoff: proportional to top score but never below MIN_SIM
        dynamic_min = max(MIN_SIM, 0.35 * max_score, 0.5 * median_score)
        strong = [h for h in hits if (h.score or 0.0) >= dynamic_min]

        if not strong:
            # Fallback: use top-k anyway
            strong = hits[: self.cite_top_k]

        logger.debug(
            f"🧩 Adaptive cutoff={dynamic_min:.3f} | top={max_score:.3f}, "
            f"median={median_score:.3f} | kept={len(strong)}/{len(hits)}"
        )

        return strong[: self.cite_top_k]

    # ------------------------------------------------------
    # 🧠 Main RAG pipeline
    # ------------------------------------------------------
    def answer(self, query: str, k: int = 8) -> Answer:
        """Retrieve → select → generate → respond."""
        rr = self.retriever.retrieve(query, k=k)

        # No hits at all (empty DB or embedder failure)
        if not rr.hits:
            logger.warning("⚠️ No retrieval hits at all — database may be empty or embeddings invalid.")
            return Answer(
                text="No relevant passages were found in the database.",
                citations=[],
                contexts=[],
            )

        # Step 1: Choose contexts adaptively
        selected = self._select_strong(rr.hits)
        avg_sim = sum(h.score or 0 for h in selected) / max(1, len(selected))
        logger.info(f"🧠 Selected {len(selected)} contexts (avg similarity={avg_sim:.3f})")

        # Step 2: Construct model context text
        context_texts = [h.chunk for h in selected]
        if not context_texts:
            context_texts = [DEFAULT_FALLBACK]

        # Step 3: Log the retrieved context titles for visibility
        context_titles = [h.title for h in selected]
        logger.debug(f"📚 Context titles: {context_titles}")
        logger.debug(f"🪶 Sending {len(context_texts)} contexts to generator for query: {query}")

        # Step 4: Generate answer from LLM
        try:
            answer_text = self.generator.generate(
                query=query,
                contexts=context_texts,
                citations_schema="doc_id + title",
            )
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            answer_text = DEFAULT_FALLBACK

        # Step 5: Fallback if model returns nothing
        if not answer_text or not answer_text.strip():
            logger.warning("⚠️ Model returned empty response — using fallback.")
            answer_text = DEFAULT_FALLBACK

        # Step 6: Package result
        return Answer(
            text=answer_text.strip(),
            citations=selected,
            contexts=[h.chunk for h in selected],
        )
