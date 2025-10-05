
# Core domain (entities + ports)

## Entities / DTOs (pure data)

* `Document(doc_id: str, title: str, abstract: str)`
* `Query(text: str)`
* `Hit(doc_id: str, score: float, snippet: str)`
* `Answer(text: str, citations: list[Hit], contexts: list[str])`

## Ports (interfaces)

* `IEmbeddingModel.embed_texts(list[str]) -> ndarray`
* `IVectorIndex.build(vectors, ids)`, `search(query_vec, k) -> list[(id, score)]`, `persist(path)`, `load(path)`
* `IRetriever.retrieve(query: Query, k: int) -> list[Hit]`
* `IAnswerGenerator.generate(query: Query, contexts: list[str]) -> str`
* `IMetadataStore.iter_documents() -> Iterator[Document]`, `get(doc_id) -> Document`
* `IFingerprint.current(DATA_PATH) -> str`, `was_changed(prev: str) -> bool`

**Why:** the **core services** depend only on these ports, letting you switch FAISS ↔ HNSW, embedding models, or LLMs **without touching** business logic.

---

# Core services (application layer)

## IndexingService (Single Responsibility)

* **Input:** `IMetadataStore`, `IEmbeddingModel`, `IVectorIndex`, `IFingerprint`
* **Flow:**

  1. Check fingerprint → rebuild if changed
  2. Stream documents → batch embed abstracts
  3. Build vector index, persist index + metadata map
* **Open/Closed:** add new index/embedding implementations; no changes here

## QAService (Use-case orchestration)

* **Input:** `IRetriever`, `IAnswerGenerator`, `GroundingPolicy`
* **Flow:**

  1. Retrieve top-k hits
  2. Select snippets (Policy: cap tokens/bytes)
  3. Generate answer from **only** those snippets
  4. **Grounding check:** ensure citations ⊆ retrieved hits
  5. Return `Answer`

## GroundingPolicy (Strategy)

* Enforces similarity threshold, snippet budgeting, and citation validation.

---

# Adapters (infrastructure layer)

Concrete, swappable implementations of ports:

* **Embeddings:** ONNX sentence model (`SentenceONNXEmbedding`)
* **Index:** FAISS Flat / HNSW (`FaissFlatIndex`, `HnswlibIndex`)
* **Retriever:** `VectorRetriever` (optionally composes `CrossEncoderReranker`)
* **Generator:** `LocalLLMGenerator` (prompt template injected)
* **Store:** `JsonlStore` streaming the corpus
* **Fingerprint:** `FileFingerprint` using SHA256 or mtime+size

All adapters are **replaceable** and only imported in the DI container.

---

# Dependency Injection (container.py)

* Bind interfaces → chosen implementations based on `settings.py`

  * `IEmbeddingModel` → `SentenceONNXEmbedding(models/embedding/...)`
  * `IVectorIndex` → `FaissFlatIndex(.cache/index)` (or HNSW via env)
  * `IRetriever` → `VectorRetriever(index, store, embedding, policy)`
  * `IAnswerGenerator` → `LocalLLMGenerator(models/llm/...)`
  * `IMetadataStore` → `JsonlStore(DATA_PATH)`
  * `IFingerprint` → `FileFingerprint(DATA_PATH)`
* Expose factories for `IndexingService` and `QAService`.

**Benefit:** tests can swap implementations (e.g., `FakeEmbeddingModel`) effortlessly.

---

# Request flow (end-to-end)

```
HTTP /answer (FastAPI)
   ↓  (app/http_server.py)
QAService.answer(query)
   ↓
Retriever.retrieve(query)     # uses Embedding + VectorIndex + Store
   ↓
GroundingPolicy.select(contexts)  # snippet budget, threshold
   ↓
AnswerGenerator.generate(query, contexts)
   ↓
GroundingPolicy.validate_citations(answer, hits)
   ↓
return Answer
```

---

# SOLID mapping (quick checklist)

* **S**: Each class has one reason to change (e.g., `IndexingService` vs `QAService`).
* **O**: Add a new embedding or index by implementing the port; no core changes.
* **L**: Any `IVectorIndex` impl can replace another without breaking callers.
* **I**: Separate small interfaces (`IRetriever`, `IAnswerGenerator`, etc.), no fat API.
* **D**: Core depends on **interfaces**; adapters depend on concrete libs.

---

# Configuration & policies

* `settings.py` (Pydantic):

  * `DATA_PATH`, `K`, `TEMPERATURE`, `RERANK`, `SIM_THRESHOLD`, `MAX_CONTEXT_TOKENS`
  * `INDEX_BACKEND=faiss|hnsw`, `EMBEDDING_MODEL_NAME`, `LLM_MODEL_PATH`
* `policies/selection.py`:

  * Strategy to limit contexts by tokens/chars, deduplicate overlaps, keep top-N.
* `policies/thresholds.py`:

  * Cosine similarity cutoffs, minimum retrieved size to answer; fallback message.

---

# Testing strategy

* **Unit** (pure core):

  * `IndexingService`: builds index from tiny fixture; persists & reloads
  * `QAService`: returns grounded `Answer` and rejects hallucinated citations
  * `GroundingPolicy`: threshold and budgeting
* **Integration** (adapters wired):

  * Start with FAISS + ONNX models on tiny corpus; assert latency bounds
* **Contract** (API):

  * `/health` returns ok after startup
  * `/answer` schema and behavior (at least one citation when relevant)

Use fakes/mocks for embeddings/LLM to keep unit tests fast and deterministic.

---

# Local → Docker workflow (keeps everything offline)

1. **Local dev:** run with real or fake models; pass tests.
2. **Package models locally** under `app/models/*`.
3. **Dockerfile** copies the app + models; no runtime downloads.
4. **Entrypoint** runs `startup.py`:

   * Wires DI, builds/loads index if fingerprint changed, then starts HTTP server.
5. **Run** with `-e DATA_PATH=/data/arxiv.jsonl -v /host/arxiv.jsonl:/data/arxiv.jsonl:ro`

---



---

# Evolution paths (future-proof)

* Add **BM25 hybrid**: implement `IRetriever` that merges BM25 + vectors.
* Swap **LLM**: new `IAnswerGenerator` impl (GGUF → ONNX → vLLM) with the same interface.
* Add **reranker**: Strategy injected into `VectorRetriever` constructor.
* Move metadata to **SQLite**: plug a new `IMetadataStore`.

