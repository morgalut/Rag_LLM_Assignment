-- ============================================================
-- 🧠 backend/app/db/init/010_schema.sql
-- RAG-ready Postgres schema for pgvector retrieval
-- ============================================================
-- This file initializes a semantic-search-ready Postgres database using pgvector.
-- It defines core tables, indexes, and utility functions to validate embeddings.
-- ============================================================

-- ------------------------------------------------------------
-- 🧩 Enable required extensions
-- ------------------------------------------------------------
-- Ensure all required extensions exist before proceeding.
CREATE EXTENSION IF NOT EXISTS vector;      -- For vector storage and similarity search
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- For trigram text search
CREATE EXTENSION IF NOT EXISTS btree_gin;   -- For efficient multi-column GIN indexes

-- ------------------------------------------------------------
-- 🧮 Table: papers
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS papers (
  doc_id        TEXT PRIMARY KEY,
  title         TEXT NOT NULL,
  abstract      TEXT NOT NULL,
  embedding     vector(768) NOT NULL CHECK (vector_dims(embedding) = 768),
  created_at    TIMESTAMPTZ DEFAULT now(),
  updated_at    TIMESTAMPTZ DEFAULT now(),
  category      TEXT,
  author        TEXT,
  source_url    TEXT
);

COMMENT ON TABLE papers IS
  'Main corpus table: one row per document containing metadata and a 768-dim pgvector embedding.';
COMMENT ON COLUMN papers.embedding IS
  'pgvector(768) embedding used for semantic search operations.';

-- ------------------------------------------------------------
-- 🧾 Table: corpus_state
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS corpus_state (
  corpus_name   TEXT PRIMARY KEY,
  fingerprint   TEXT NOT NULL,
  rows_count    INTEGER DEFAULT 0,
  last_ingested TIMESTAMPTZ DEFAULT now(),
  updated_at    TIMESTAMPTZ DEFAULT now()
);

COMMENT ON TABLE corpus_state IS
  'Tracks corpus fingerprint and ingestion metadata for versioning.';

-- ------------------------------------------------------------
-- 🔍 Indexes
-- ------------------------------------------------------------
-- IVF indexes improve retrieval but have low recall on very small datasets.
CREATE INDEX IF NOT EXISTS papers_embedding_ivf
  ON papers
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 200);

CREATE INDEX IF NOT EXISTS papers_title_trgm
  ON papers
  USING gin (title gin_trgm_ops);

CREATE INDEX IF NOT EXISTS papers_doc_id_idx   ON papers (doc_id);
CREATE INDEX IF NOT EXISTS papers_category_idx ON papers (category);
CREATE INDEX IF NOT EXISTS papers_created_idx  ON papers (created_at DESC);

CREATE OR REPLACE FUNCTION validate_embeddings()
RETURNS TABLE (
  total_rows   BIGINT,
  avg_norm     DOUBLE PRECISION,
  min_norm     DOUBLE PRECISION,
  max_norm     DOUBLE PRECISION,
  invalid_rows BIGINT
)
LANGUAGE SQL AS $$
  SELECT
    COUNT(*) AS total_rows,
    AVG(sqrt(inner_product(p.embedding, p.embedding))) AS avg_norm,
    MIN(sqrt(inner_product(p.embedding, p.embedding))) AS min_norm,
    MAX(sqrt(inner_product(p.embedding, p.embedding))) AS max_norm,
    COUNT(*) FILTER (
      WHERE sqrt(inner_product(p.embedding, p.embedding)) < 0.1
         OR sqrt(inner_product(p.embedding, p.embedding)) > 2.0
    ) AS invalid_rows
  FROM papers AS p;
$$;

COMMENT ON FUNCTION validate_embeddings() IS
  'Checks embedding vector norms using pg_catalog.l2_norm() from the pgvector extension.';

-- ------------------------------------------------------------
-- 🧹 Trigger: auto-update updated_at
-- ------------------------------------------------------------
-- Automatically updates the "updated_at" timestamp on every row update.
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop old trigger if it exists (idempotent initialization)
DROP TRIGGER IF EXISTS trg_papers_updated_at ON papers;

CREATE TRIGGER trg_papers_updated_at
BEFORE UPDATE ON papers
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

-- ------------------------------------------------------------
-- ✅ Manual Validation (Optional)
-- ------------------------------------------------------------
-- After container startup, you can verify correct initialization via:
--   docker exec -it rag_pgvector_db psql -U raguser -d ragdb
--   SELECT * FROM validate_embeddings();
--
-- Expected output (if no embeddings yet):
--   total_rows | avg_norm | min_norm | max_norm | invalid_rows
--  ------------+-----------+-----------+-----------+--------------
--             0 |           |           |           |            0
-- ------------------------------------------------------------
