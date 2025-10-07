-- ============================================================
-- backend/app/db/init/010_schema.sql
-- 🧠 Schema for RAG-ready Postgres (pgvector) database
-- ============================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- optional, improves fuzzy search
CREATE EXTENSION IF NOT EXISTS btree_gin; -- for hybrid metadata filters

-- ------------------------------------------------------------
-- 🧩 Table: papers
-- Stores document metadata, text, and embeddings
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS papers (
  doc_id       TEXT PRIMARY KEY,
  title        TEXT NOT NULL,
  abstract     TEXT NOT NULL,
  embedding    VECTOR(768) NOT NULL CHECK (vector_dims(embedding) = 768),
  created_at   TIMESTAMPTZ DEFAULT now(),
  updated_at   TIMESTAMPTZ DEFAULT now(),

  -- Optional metadata columns for ranking / filters
  category     TEXT,
  author       TEXT,
  source_url   TEXT
);

COMMENT ON TABLE papers IS 'Main corpus table: one row per paper or document.';
COMMENT ON COLUMN papers.embedding IS 'pgvector(768) embedding used for semantic search.';

-- ------------------------------------------------------------
-- 🧩 Table: corpus_state
-- Stores ingestion fingerprint metadata (for reingestion detection)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS corpus_state (
  corpus_name  TEXT PRIMARY KEY,
  fingerprint  TEXT NOT NULL,
  rows_count   INTEGER DEFAULT 0,
  last_ingested TIMESTAMPTZ DEFAULT now(),
  updated_at   TIMESTAMPTZ DEFAULT now()
);

COMMENT ON TABLE corpus_state IS 'Tracks corpus fingerprint and ingestion metadata.';

-- ------------------------------------------------------------
-- 🔍 Indexes for hybrid retrieval
-- ------------------------------------------------------------

-- Vector index for fast semantic similarity search
CREATE INDEX IF NOT EXISTS papers_embedding_ivf
  ON papers
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 200);  -- increase lists for more fine-grained recall

-- Text-based fuzzy search (for fallback / keyword hybrid search)
CREATE INDEX IF NOT EXISTS papers_title_trgm
  ON papers
  USING gin (title gin_trgm_ops);

-- Metadata indexes (useful for filtering / deduplication)
CREATE INDEX IF NOT EXISTS papers_doc_id_idx ON papers (doc_id);
CREATE INDEX IF NOT EXISTS papers_category_idx ON papers (category);
CREATE INDEX IF NOT EXISTS papers_created_idx  ON papers (created_at DESC);

-- ------------------------------------------------------------
-- 🧮 Sanity Checks and Statistics Helpers
-- ------------------------------------------------------------

-- Function: validate_embeddings()
-- Checks embedding norms and reports anomalies
CREATE OR REPLACE FUNCTION validate_embeddings()
RETURNS TABLE (
  total_rows BIGINT,
  avg_norm DOUBLE PRECISION,
  min_norm DOUBLE PRECISION,
  max_norm DOUBLE PRECISION,
  invalid_rows BIGINT
)
LANGUAGE SQL AS $$
  SELECT
    COUNT(*) AS total_rows,
    AVG(l2_norm(embedding)) AS avg_norm,
    MIN(l2_norm(embedding)) AS min_norm,
    MAX(l2_norm(embedding)) AS max_norm,
    COUNT(*) FILTER (WHERE l2_norm(embedding) < 0.1 OR l2_norm(embedding) > 2.0) AS invalid_rows
  FROM papers;
$$;

COMMENT ON FUNCTION validate_embeddings() IS 'Diagnostic function to check average embedding norms and detect anomalies.';

-- ------------------------------------------------------------
-- 🧹 Housekeeping Trigger
-- Automatically update timestamps on row update
-- ------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_papers_updated_at ON papers;
CREATE TRIGGER trg_papers_updated_at
BEFORE UPDATE ON papers
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

-- ------------------------------------------------------------
-- ✅ Sanity check query suggestion (manual)
-- Run to ensure embeddings are valid:
-- SELECT vector_dims(embedding), AVG(l2_norm(embedding)), COUNT(*) FROM papers;
-- Expected: (768, ~0.6–0.8, >0)
-- ------------------------------------------------------------
